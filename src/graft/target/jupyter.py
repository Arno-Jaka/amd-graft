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
import time
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
MAX_FRAMES = 200_000


def supports_terminal_api(version: str) -> bool:
    """True when a Jupyter major version has the terminal API Graft needs.

    Compared as a number rather than against a list of prefixes: 8 is the
    current release and its terminal API is unchanged, so a prefix test
    refused most real deployments.
    """
    major_text = version.split(".", 1)[0]
    return major_text.isdigit() and int(major_text) >= 5


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
            raise TransportError(f"Jupyter refused the request while {what}. Check the token.")
        if response.status_code == 404:
            raise TransportError(
                f"Jupyter has no endpoint for {what}. Confirm this is a "
                "Jupyter server and that the path is right. Note that a "
                "Jupyter contents path is resolved relative to the server's "
                "root directory, so a remote directory outside that root will "
                "not resolve."
            )
        if not response.ok:
            raise TransportError(f"Jupyter returned HTTP {response.status_code} while {what}.")

    def _request(self, method: str, path: str, *, what: str, timeout: float, **kwargs):
        try:
            response = self.session.request(
                method,
                self._url(path),
                headers=self._headers(),
                timeout=timeout,
                **kwargs,
            )
        except requests.exceptions.ConnectionError as error:
            raise TransportError(
                f"Could not reach {self.spec.safe_label()}. Check the host and port."
            ) from error
        except requests.exceptions.Timeout as error:
            raise TransportError(f"Timed out reaching {self.spec.safe_label()} while {what}.") from error
        except requests.exceptions.RequestException as error:
            raise TransportError(f"Jupyter request failed while {what}: {error}") from error
        self._check(response, what)
        return response

    def _contents_path(self, remote_path: str) -> str:
        """Map an absolute remote path to a Jupyter contents API path.

        Jupyter serves its contents API relative to the server root, so an
        absolute filesystem path is reduced to the components below that root.
        """
        parts = [part for part in remote_path.split("/") if part]
        return "/api/contents" + ("/" + "/".join(parts) if parts else "")

    def _ws_url(self, name: str) -> str:
        # Note the missing /api segment: the terminal websocket lives at the
        # site root, unlike every REST endpoint on the same server.
        return (
            f"{self.ws_scheme}://{self.spec.host}:{self.spec.port}"
            f"/terminals/websocket/{url_quote(name)}"
        )

    # -- lifecycle --------------------------------------------------------

    def probe(self) -> None:
        status = self._request("GET", "/api/status", what="checking server status", timeout=20)
        version = "unknown"
        with contextlib.suppress(ValueError, AttributeError):
            version = (status.json().get("version") or "unknown").lower()
        # /api/status is unauthenticated on many deployments, so success here
        # does not prove the token works. Ask for something that is gated.
        self._request("GET", "/api/contents/", what="listing files", timeout=20)
        if not supports_terminal_api(version):
            raise TransportError(
                f"Jupyter version {version} is not supported. Graft needs 5 or "
                "newer for the terminal API."
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
        # stty first, so the transcript does not echo the rest of the script
        # back into what the command printed. This line still echoes, but it
        # lands before the opening marker and is discarded with the preamble.
        parts = [
            "stty -echo 2>/dev/null",
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

    def _drive(self, name: str, script: str, timeout_s: int | None) -> str:
        """Send the script and read the transcript back under a real deadline.

        The budget is enforced against the clock rather than by a fixed
        per-read timeout. A per-read cap silently truncated long jobs: a remote
        hipcc compile can print nothing for minutes, and the caller's stated
        timeout was never consulted at all.
        """
        budget = float(timeout_s) if timeout_s else DEFAULT_RUN_TIMEOUT_S
        deadline = time.monotonic() + budget
        try:
            socket_ = websocket.create_connection(
                self._ws_url(name),
                header=[f"{TOKEN_HEADER}: token {self.spec.token}"],
                timeout=max(1.0, min(budget, CONNECT_TIMEOUT_S)),
                # Jupyter's own origin check rejects the Host header a raw
                # socket sends unless an origin is supplied.
                origin=self.base,
            )
        except (OSError, websocket.WebSocketException) as error:
            raise TransportError(
                f"Could not open a terminal on {self.spec.safe_label()}: {error}"
            ) from error

        buffer = ""
        frames = 0
        try:
            socket_.send(json.dumps({}))
            socket_.send(script)
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
                buffer += frame
                frames += 1
                if frames > MAX_FRAMES:  # pragma: no cover - backstop
                    raise TransportError("Terminal stream did not terminate.")
        except websocket.WebSocketException as error:
            raise TransportError(f"Terminal session failed: {error}") from error
        finally:
            socket_.close()
        return buffer

    def _create_terminal(self) -> str:
        response = self._request(
            "POST",
            "/terminals",
            what="starting a terminal",
            timeout=30,
            data=json.dumps({}),
        )
        try:
            name = response.json().get("name")
        except ValueError as error:
            raise TransportError("Jupyter returned an unreadable terminal reply.") from error
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
                self._url(f"/terminals/{url_quote(name)}"),
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
            "GET", path, what=f"downloading {remote_path.rsplit('/', 1)[-1]}", timeout=300
        )
        try:
            content = response.json().get("content", "")
        except ValueError as error:
            raise TransportError("Jupyter returned an unreadable file reply.") from error
        return base64.b64decode(content)


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
