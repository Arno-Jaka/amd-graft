"""Drive a remote AMD instance through a Jupyter server's HTTP API.

Needed because many cloud GPU providers expose port 80 and block SSH to the
public internet, leaving the web API as the only reachable control surface.
The same is true of hosted notebooks generally.

Commands run in a terminal session over the Jupyter websocket, which is the
only channel that gives back stdout: the REST API can create and kill
terminals but not read them. Upload and download use the contents API.
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
from collections.abc import Callable
from threading import Event
from urllib.parse import quote as url_quote

from graft.target.spec import TargetSpec
from graft.target.transport import CommandResult, TransportError, clip, quote

try:
    import requests
except ImportError:  # pragma: no cover - exercised by absence, not by test
    requests = None

try:
    import websocket
except ImportError:  # pragma: no cover - exercised by absence, not by test
    websocket = None

TOKEN_HEADER = "Authorization"
CONTENT_TYPE = "application/json"

# Markers bracket the output of one command so it can be told apart from the
# shell's own noise. The exit code is glued to the front of the closing
# marker, which keeps it next to the output it describes.
BEGIN = "__GRAFT_BEGIN__"
END = "__GRAFT_END__"

# Exit status used when the requested working directory does not exist. Chosen
# to be far from anything a real command returns, so it is never mistaken for
# the command's own result.
EXIT_NO_SUCH_DIR = 97

# Used when a caller does not state its own budget. Mirrors the spec default so
# a direct run() call behaves like one made through the form.
DEFAULT_RUN_TIMEOUT_S = 900
# Ceiling on the websocket handshake, so a dead endpoint fails fast instead of
# inheriting the full command budget.
CONNECT_TIMEOUT_S = 60
# A live stream notices a target that vanished silently only if the websocket
# pings it: protocol pings fail fast when the peer is gone, where a plain read
# timeout would simply repeat. Chosen short enough to catch a dropped target,
# long enough not to look like abuse.
PING_INTERVAL_S = 10
PING_TIMEOUT_S = 5
MAX_FRAMES = 200_000

# A fresh server, a dropped packet, or a moment of load can refuse the very
# first connection. Those blips are retried a couple of times with a short
# backoff rather than failing the whole operation on one bad handshake.
# Read timeouts and HTTP errors are NOT retried: a stalled command has its own
# budget, and repeating a slow request only makes it slower.
CONNECT_RETRIES = 3
CONNECT_BACKOFF_S = 1.5


def supports_terminal_api(version: str) -> bool:
    """True when a Jupyter version has the terminal API Graft needs.

    The string is ambiguous because it comes from either product. A modern
    ``jupyter_server`` reports its own version (1.x, 2.x), where the terminal
    API has existed since 1.0. A legacy ``notebook`` server reports the
    notebook version, and only 5.0 added terminals, so 4.x is the one case
    that must be refused. Reading it as "5 or newer" rejected every current
    jupyter_server deployment.
    """
    major_text = version.split(".", 1)[0]
    if not major_text.isdigit():
        return False
    major = int(major_text)
    if major == 4:
        return False
    return major >= 1


def _timeout_message(spec: TargetSpec, budget: float) -> str:
    return (
        f"Command timed out on {spec.safe_label()} after {budget:.0f}s. "
        "Raise the command timeout if the work is legitimately slow."
    )


def _require_deps() -> None:
    missing = [
        name
        for name, module in (("requests", requests), ("websocket-client", websocket))
        if module is None
    ]
    if not missing:
        return
    joined = " and ".join(missing)
    raise TransportError(
        f"Jupyter support needs {joined}. Install "
        f"{'them' if len(missing) > 1 else 'it'} with "
        f"'pip install {' '.join(missing)}', or use the SSH option instead."
    )


class JupyterTransport:
    """A Jupyter-backed connection to a target.

    One terminal is created per command and deleted afterwards. The wrapper
    script ends with ``exit``, so the shell is gone once the command finishes
    and nothing leaks into the next command's output.
    """

    def __init__(self, spec: TargetSpec) -> None:
        _require_deps()
        self.spec = spec
        # The scheme comes from the spec, not a hardcoded http://. Every hosted
        # notebook is https, and silently speaking plaintext to port 443 simply
        # fails to connect.
        scheme = spec.scheme or "http"
        self.base = f"{scheme}://{spec.host}:{spec.port}"
        self.ws_scheme = "wss" if scheme == "https" else "ws"
        self.session = requests.Session()
        # The server's root directory, discovered on first use. The contents
        # API resolves paths under it, and it is not always "/".
        self._server_root: str | None = None

    # -- plumbing ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": CONTENT_TYPE}
        if self.spec.token:
            headers[TOKEN_HEADER] = f"token {self.spec.token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _check(self, response, what: str) -> None:
        if response.status_code in (401, 403):
            raise TransportError(
                f"Jupyter refused the request while {what}. Check the token."
            )
        if response.status_code == 404:
            raise TransportError(
                f"Jupyter has no endpoint for {what}. Confirm this is a "
                "Jupyter server and that the path is right. Note that a "
                "Jupyter contents path is resolved relative to the server's "
                "root directory, so a remote directory outside that root will "
                "not resolve."
            )
        if not response.ok:
            raise TransportError(
                f"Jupyter returned HTTP {response.status_code} while {what}."
            )

    def _request(self, method: str, path: str, *, what: str, timeout: float, **kwargs):
        # A connection that never established (ConnectTimeout is a subclass of
        # ConnectionError) is safe and worth to retry: nothing was sent, so no
        # server-side effect can be duplicated. Anything else fails at once.
        last_error: requests.exceptions.ConnectionError | None = None
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                response = self.session.request(
                    method,
                    self._url(path),
                    headers=self._headers(),
                    timeout=timeout,
                    **kwargs,
                )
            except requests.exceptions.ConnectionError as error:
                last_error = error
                if attempt < CONNECT_RETRIES:
                    time.sleep(CONNECT_BACKOFF_S * attempt)
                    continue
                raise TransportError(
                    f"Could not reach {self.spec.safe_label()}. Check the host and port."
                ) from error
            except requests.exceptions.Timeout as error:
                raise TransportError(
                    f"Timed out reaching {self.spec.safe_label()} while {what}."
                ) from error
            except requests.exceptions.RequestException as error:
                raise TransportError(
                    f"Jupyter request failed while {what}: {error}"
                ) from error
            self._check(response, what)
            return response
        # Unreachable: the loop either returns or raises. Kept for the type
        # checker, which cannot see that every path above is terminal.
        raise TransportError(
            f"Could not reach {self.spec.safe_label()}. Check the host and port."
        ) from last_error

    def _contents_path(self, remote_path: str) -> str:
        """Map an absolute remote path to a Jupyter contents API path.

        The contents API resolves paths under the server's root directory,
        which is not always ``/``: an image that starts Jupyter in
        ``/shared-docker`` serves its files from there. Strip that root, then
        reduce what is left to the components below it, so an absolute path a
        shell command can use also names the same file to the contents API.
        """
        root = self.server_root.rstrip("/")
        path = remote_path
        if root and (path == root or path.startswith(root + "/")):
            path = path[len(root) :]
        parts = [part for part in path.split("/") if part]
        return "/api/contents" + ("/" + "/".join(parts) if parts else "")

    @property
    def server_root(self) -> str:
        """The filesystem root the Jupyter server serves files from."""
        if self._server_root is None:
            self._server_root = self._discover_server_root()
        return self._server_root

    def _discover_server_root(self) -> str:
        """Ask the server for its root directory, defaulting to ``/``.

        ``jupyter server list`` prints ``<url> :: <root>``. The listing is
        parsed here rather than only in the remote command, so a server that
        answers without the ``sed`` in place is still understood. When the CLI
        is unavailable the server is assumed to serve from the filesystem root,
        which is the common case and the previous behaviour.
        """
        result = self.run(
            "jupyter server list 2>/dev/null | sed -n 's/.* :: //p' | head -1",
            timeout_s=30,
        )
        text = result.stdout.strip()
        first = text.splitlines()[0].strip() if text else ""
        if "::" in first:
            first = first.rsplit("::", 1)[1].strip()
        return first if first.startswith("/") else "/"

    def resolve_remote_dir(self, remote_dir: str) -> str:
        """Place a configured directory under the server root.

        The terminal and the contents API share one filesystem, but only the
        contents API is confined to the server root. Joining the configured
        directory onto that root is what makes an upload land where a shell
        command will later look for it.
        """
        root = self.server_root.rstrip("/")
        if not root:
            return f"/{remote_dir.lstrip('/')}"
        if remote_dir == root or remote_dir.startswith(root + "/"):
            return remote_dir
        return f"{root}/{remote_dir.lstrip('/')}"

    def _ws_url(self, name: str) -> str:
        # Note the missing /api segment: the terminal websocket lives at the
        # site root, unlike every REST endpoint on the same server.
        return (
            f"{self.ws_scheme}://{self.spec.host}:{self.spec.port}"
            f"/terminals/websocket/{url_quote(name)}"
        )

    # -- lifecycle --------------------------------------------------------

    def probe(self) -> None:
        self._request(
            "GET", "/api/status", what="checking server status", timeout=20
        )
        # The server version comes from the API root, not /api/status, which
        # only reports live connection counts. Ask the root for it, but treat a
        # failure to read it as "unknown" and let the terminal test below be
        # the real gate rather than refusing a server we simply can't identify.
        version = "unknown"
        with contextlib.suppress(ValueError, AttributeError, TransportError):
            info = self._request("GET", "/api", what="checking server version", timeout=20)
            version = (info.json().get("version") or "unknown").lower()
        # /api/status is unauthenticated on many deployments, so success there
        # does not prove the token works. Ask for something that is gated.
        self._request("GET", "/api/contents/", what="listing files", timeout=20)
        if version != "unknown" and not supports_terminal_api(version):
            raise TransportError(
                f"Jupyter version {version} is not supported. Graft needs the "
                "terminal API, which arrived in notebook 5 / jupyter_server 1."
            )
        # Prove the terminal path works too, which is what every command needs.
        result = self.run("echo graft-probe-ok")
        if not result.ok or "graft-probe-ok" not in result.stdout:
            raise TransportError(
                f"Reached {self.spec.safe_label()} but could not run a command "
                "in a terminal there."
            )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> JupyterTransport:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- work -------------------------------------------------------------

    def _script(self, command: str, cwd: str | None) -> str:
        """The shell script that runs one command and reports its status."""
        # A Jupyter terminal is an interactive login shell: it prints a PS1
        # prompt before every line and a PS2 continuation prompt for a
        # multi-line block. Those prompts land inside the transcript, and
        # between the output lines the caller parses, so a marker line no
        # longer starts the line. Blank the prompts first (a bare assignment
        # persists), then quiet the echo, so nothing but the command's own
        # output sits between the markers.
        parts = [
            "PS1= PS2= PROMPT_COMMAND=; stty -echo 2>/dev/null",
            f"printf '%s\\n' {BEGIN}",
        ]
        if cwd:
            parts.append(f"cd {quote(cwd)} || exit {EXIT_NO_SUCH_DIR}")
        parts.append(f"{{ {command}\n}}; __graft_rc=$?")
        parts.append(f"printf '%s%s\\n' \"$__graft_rc\" {END}")
        parts.append("exit $__graft_rc")
        return "\n".join(parts)

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
    ) -> CommandResult:
        name = self._create_terminal()
        buffer = ""
        try:
            buffer = self._drive(name, self._script(command, cwd), timeout_s)
        finally:
            self._delete_terminal(name)
        body, code = _split_markers(buffer)
        return CommandResult(
            command=f"cd {cwd} && {command}" if cwd else command,
            returncode=code,
            stdout=clip(body.strip("\r\n")),
            stderr="",
        )

    def _open_socket(self, name: str, budget: float):
        """Open the terminal websocket, mapping transport failures for the UI.

        A refused or dropped handshake is retried briefly: the terminal itself
        already exists, so re-attempting the connection cannot create anything
        twice. Only after the retries does the failure surface.
        """
        last_error: Exception | None = None
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                return websocket.create_connection(
                    self._ws_url(name),
                    header=[f"{TOKEN_HEADER}: token {self.spec.token}"],
                    timeout=max(1.0, min(budget, CONNECT_TIMEOUT_S)),
                    # Jupyter's own origin check rejects the Host header a raw
                    # socket sends unless an origin is supplied.
                    origin=self.base,
                    # Ping the peer so a silently dropped server raises instead
                    # of leaving the reader waiting forever.
                    ping_interval=PING_INTERVAL_S,
                    ping_timeout=PING_TIMEOUT_S,
                )
            except (OSError, websocket.WebSocketException) as error:
                last_error = error
                if attempt < CONNECT_RETRIES:
                    time.sleep(CONNECT_BACKOFF_S * attempt)
                    continue
                raise TransportError(
                    f"Could not open a terminal on {self.spec.safe_label()}: {error}"
                ) from error
        raise TransportError(
            f"Could not open a terminal on {self.spec.safe_label()}: {last_error}"
        ) from last_error

    def _drive(self, name: str, script: str, timeout_s: int | None) -> str:
        """Send the script and read the transcript back under a real deadline.

        The budget is enforced against the clock rather than by a fixed
        per-read timeout. A per-read cap silently truncated long jobs: a remote
        hipcc compile can print nothing for minutes, and the caller's stated
        timeout was never consulted at all.
        """
        budget = float(timeout_s) if timeout_s else DEFAULT_RUN_TIMEOUT_S
        deadline = time.monotonic() + budget
        socket_ = self._open_socket(name, budget)

        buffer = ""
        frames = 0
        try:
            socket_.send(_stdin_frame(script))
            while END not in buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError(_timeout_message(self.spec, budget))
                # Re-arm the read each pass so the socket cannot outlive the
                # budget between frames.
                socket_.settimeout(max(1.0, remaining))
                try:
                    frame = socket_.recv()
                except (TimeoutError, websocket.WebSocketTimeoutException) as error:
                    raise TransportError(_timeout_message(self.spec, budget)) from error
                if not frame:
                    break
                buffer += _decode_frame(frame)
                frames += 1
                if frames > MAX_FRAMES:  # pragma: no cover - backstop
                    raise TransportError("Terminal stream did not terminate.")
        except websocket.WebSocketException as error:
            raise TransportError(f"Terminal session failed: {error}") from error
        finally:
            socket_.close()
        return buffer

    def stream(
        self,
        command: str,
        on_line: Callable[[str], None],
        *,
        stop: Event,
    ) -> None:
        """Stream a long-lived command's output over one websocket.

        A sentinel-terminated shell loop on the remote prints each reading;
        this reads the frames and forwards whole lines. The websocket is the
        same channel ``run`` uses, just held open instead of polled.
        """
        name = self._create_terminal()
        socket_ = self._open_socket(name, CONNECT_TIMEOUT_S)
        buffer = ""
        frames = 0
        try:
            # stty first so the loop's own source text is not echoed back; the
            # metrics parser ignores lines without a key anyway, but this keeps
            # the transcript clean. No markers: this command does not end.
            socket_.send(_stdin_frame(f"stty -echo 2>/dev/null\n{command}\n"))
            while not stop.is_set():
                socket_.settimeout(1.0)
                try:
                    frame = socket_.recv()
                except (TimeoutError, websocket.WebSocketTimeoutException):
                    continue
                if not frame:
                    if stop.is_set():
                        break
                    raise TransportError(
                        f"The live connection to {self.spec.safe_label()} "
                        "was closed by the target."
                    )
                buffer += _decode_frame(frame)
                frames += 1
                if frames > MAX_FRAMES:  # pragma: no cover - backstop
                    raise TransportError("Terminal stream did not terminate.")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    on_line(line.rstrip("\r"))
        except websocket.WebSocketException as error:
            raise TransportError(f"Terminal session failed: {error}") from error
        finally:
            socket_.close()
            self._delete_terminal(name)

    def _create_terminal(self) -> str:
        # The REST terminal endpoints live under /api, unlike the websocket the
        # commands actually run over. Posting to /terminals returns 404 on a
        # real server, which read as "no terminal API" and refused the target.
        response = self._request(
            "POST",
            "/api/terminals",
            what="starting a terminal",
            timeout=30,
            data=json.dumps({}),
        )
        try:
            name = response.json().get("name")
        except ValueError as error:
            raise TransportError(
                "Jupyter returned an unreadable terminal reply."
            ) from error
        if not name:
            raise TransportError("Jupyter did not return a terminal name.")
        return name

    def _delete_terminal(self, name: str) -> None:
        """Best-effort cleanup so the server does not accumulate sessions."""
        # The shell already exited, so a failure here is cosmetic: it would
        # leave one stale terminal on the server. Not worth failing the
        # command over, and not worth telling the user about.
        with contextlib.suppress(requests.exceptions.RequestException):
            self.session.delete(
                self._url(f"/api/terminals/{url_quote(name)}"),
                headers=self._headers(),
                timeout=15,
            )

    # -- file transfer ----------------------------------------------------

    def push(self, local: bytes, remote_path: str) -> None:
        parts = [part for part in remote_path.split("/") if part]
        if not parts:
            raise TransportError("Upload needs a filename.")
        directory, name = "/".join(parts[:-1]), parts[-1]
        parent = self._contents_path(f"/{directory}" if directory else "")
        body = {
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(local).decode("ascii"),
        }
        self._request(
            "PUT",
            f"{parent}/{url_quote(name)}",
            what=f"uploading {name}",
            timeout=300,
            data=json.dumps(body),
        )

    def pull(self, remote_path: str) -> bytes:
        path = f"{self._contents_path(remote_path)}?content=1"
        response = self._request(
            "GET",
            path,
            what=f"downloading {remote_path.rsplit('/', 1)[-1]}",
            timeout=300,
        )
        try:
            content = response.json().get("content", "")
        except ValueError as error:
            raise TransportError(
                "Jupyter returned an unreadable file reply."
            ) from error
        return base64.b64decode(content)


def _stdin_frame(text: str) -> str:
    """Wrap terminal input the way jupyter_server's websocket expects.

    The terminal websocket is not a raw byte pipe: every message is a JSON
    frame, and input is the two-element array ``["stdin", text]``. Sending the
    script as a bare string is accepted by the socket and then quietly ignored
    by jupyter_server 2.x, so the command never runs and the reader blocks on
    output that will never arrive. Sending a stray ``{}`` first corrupts the
    channel the same way.
    """
    return json.dumps(["stdin", text])


# A PTY wraps output in terminal control sequences: bracketed-paste toggles,
# the shell's window-title OSC, and cursor moves. None of it belongs in a
# command's stdout, and left in place it stops a marker line from matching at
# the start of the line. Covers CSI (``ESC [ … final``), OSC (``ESC ] …
# BEL/ST``), and the two-character escapes.
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[@-Z\\-_]"
)

# Server control frames carry no terminal output. Anything else that parses as
# JSON but is not a stdout frame is passed through untouched, so a genuinely
# old server that sends raw text frames still works.
_CONTROL_FRAMES = {"setup", "disconnect", "error"}


def _decode_frame(frame: str) -> str:
    """Reduce one websocket frame to the terminal text it carries.

    jupyter_server sends JSON arrays: ``["stdout", text]`` for output and
    ``["setup", {}]`` / ``["disconnect", n]`` around it. Concatenating the raw
    frames leaves the JSON syntax and escapes in the middle of the output, so
    the marker parser never sees a clean line.
    """
    try:
        message = json.loads(frame)
    except (ValueError, TypeError):
        return frame
    if isinstance(message, list) and len(message) == 2 and isinstance(message[0], str):
        kind, payload = message
        if kind == "stdout" and isinstance(payload, str):
            return _ANSI_RE.sub("", payload)
        if kind in _CONTROL_FRAMES:
            return ""
    return frame


def _split_markers(buffer: str) -> tuple[str, int]:
    """Extract the command output and exit code from a terminal transcript."""
    begin = buffer.find(BEGIN)
    end = buffer.rfind(END)
    if begin == -1 or end == -1 or end < begin:
        # No markers means the command never completed or the shell died.
        # Hand back the transcript so the caller can show what arrived.
        return buffer, -1
    body = buffer[begin + len(BEGIN) : end]
    # The wrapper prints the status immediately before the closing marker, so
    # the trailing digits of the body are the exit code.
    code = -1
    stripped = body.rstrip("\r\n")
    digits = ""
    while stripped and stripped[-1].isdigit():
        digits = stripped[-1] + digits
        stripped = stripped[:-1]
    if digits:
        code = int(digits)
        body = stripped
    return body, code
