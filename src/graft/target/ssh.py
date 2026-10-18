"""Drive a remote AMD instance over SSH.

Uses paramiko rather than the ``ssh`` binary so a passphrase can be handled in
process. Shelling out would put the passphrase somewhere it tends to leak:
in argv, where any local process can read it, or in an ``SSH_ASKPASS``
environment variable inherited by children.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from threading import Event

from graft.target.spec import TargetSpec
from graft.target.transport import (
    CommandResult,
    TransportError,
    clip,
    quote,
)

try:
    import paramiko
except ImportError:
    paramiko = None

CONNECT_TIMEOUT_S = 20
BANNER_TIMEOUT_S = 30
AUTH_TIMEOUT_S = 30
KEEPALIVE_S = 10


class SSHTransport:
    """An SSH connection to a target, kept open for the life of one request."""

    def __init__(self, spec: TargetSpec) -> None:
        if paramiko is None:
            raise TransportError(
                "SSH support needs the paramiko package. Install it with "
                "'pip install paramiko', or use the Jupyter option instead."
            )
        self.spec = spec
        self._client: paramiko.SSHClient | None = None


    def _auth_kwargs(self) -> dict:
        """Build auth arguments, without ever logging them."""
        kwargs: dict = {
            "hostname": self.spec.host,
            "port": self.spec.port,
            "username": self.spec.username or "root",
            "timeout": CONNECT_TIMEOUT_S,
            "banner_timeout": BANNER_TIMEOUT_S,
            "auth_timeout": AUTH_TIMEOUT_S,
            "allow_agent": False,
            "look_for_keys": False,
        }
        kwargs["key_filename"] = str(_expand(self.spec.key_path))
        if self.spec.key_passphrase:
            kwargs["passphrase"] = self.spec.key_passphrase
        return kwargs

    def probe(self) -> None:
        self._ensure_connected()
        result = self.run("echo graft-probe-ok")
        if not result.ok or "graft-probe-ok" not in result.stdout:
            raise TransportError(
                f"Connected to {self.spec.safe_label()} but could not run a "
                "command there. Check that the login shell is usable."
            )

    def resolve_remote_dir(self, remote_dir: str) -> str:
        """An SSH login sees the whole filesystem, so the path is unchanged."""
        return remote_dir

    def _ensure_connected(self) -> paramiko.SSHClient:
        if self._client is not None:
            return self._client
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(**self._auth_kwargs())
        except FileNotFoundError as error:
            raise TransportError(
                "The private key file was not found. Check the key path."
            ) from error
        except paramiko.AuthenticationException as error:
            raise TransportError(
                "Authentication failed. Check the username, key path, and passphrase."
            ) from error
        except paramiko.BadHostKeyException as error:
            raise TransportError(
                f"The host key presented by {self.spec.host} did not match "
                "the one already known. This can mean the instance was "
                "rebuilt, or that the connection is not going where you "
                "expect."
            ) from error
        except TimeoutError as error:
            raise TransportError(
                f"Timed out connecting to {self.spec.safe_label()} on port "
                f"{self.spec.port}. Some providers expose SSH only on a "
                "private network; the Jupyter option may reach the same "
                "machine."
            ) from error
        except OSError as error:
            raise TransportError(
                f"Could not reach {self.spec.safe_label()}: {error.strerror or error}"
            ) from error
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(KEEPALIVE_S)
        self._client = client
        return client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def __enter__(self) -> SSHTransport:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
    ) -> CommandResult:
        client = self._ensure_connected()
        full = f"cd {quote(cwd)} && {command}" if cwd else command
        try:
            _stdin, stdout, stderr = client.exec_command(
                full,
                timeout=timeout_s or self.spec.command_timeout_s,
            )
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            code = stdout.channel.recv_exit_status()
        except TimeoutError as error:
            raise TransportError(
                f"Command timed out on {self.spec.safe_label()}: {command}"
            ) from error
        except OSError as error:
            raise TransportError(
                f"Command failed on {self.spec.safe_label()}: {error.strerror or error}"
            ) from error
        return CommandResult(
            command=full,
            returncode=code,
            stdout=clip(out),
            stderr=clip(err),
        )

    def stream(
        self,
        command: str,
        on_line: Callable[[str], None],
        *,
        stop: Event,
    ) -> None:
        """Read a long-lived command line by line over one SSH channel.

        The channel stays open for the life of the stream, so a live panel
        costs one channel and no reconnects, rather than a fresh command every
        refresh.
        """
        client = self._ensure_connected()
        try:
            _stdin, stdout, _stderr = client.exec_command(command)
        except OSError as error:
            raise TransportError(
                f"Could not start a stream on {self.spec.safe_label()}: "
                f"{error.strerror or error}"
            ) from error
        channel = stdout.channel
        channel.settimeout(1.0)
        buffer = ""
        try:
            while not stop.is_set():
                try:
                    chunk = channel.recv(4096)
                except TimeoutError:
                    continue
                except OSError as error:
                    raise TransportError(
                        f"Stream failed on {self.spec.safe_label()}: "
                        f"{error.strerror or error}"
                    ) from error
                if not chunk:
                    if stop.is_set():
                        break
                    raise TransportError(
                        f"The live connection to {self.spec.safe_label()} "
                        "was closed by the target."
                    )
                buffer += chunk.decode("utf-8", "replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    on_line(line.rstrip("\r"))
            if buffer.strip():
                on_line(buffer.rstrip("\r"))
        finally:
            channel.close()

    def push(self, local: bytes, remote_path: str) -> None:
        client = self._ensure_connected()
        payload = io.BytesIO(local)
        try:
            with client.open_sftp() as sftp:
                sftp.putfo(payload, remote_path)
        except OSError as error:
            raise TransportError(
                f"Could not upload to {remote_path}: {error.strerror or error}"
            ) from error

    def pull(self, remote_path: str) -> bytes:
        client = self._ensure_connected()
        buffer = io.BytesIO()
        try:
            with client.open_sftp() as sftp:
                sftp.getfo(remote_path, buffer)
        except OSError as error:
            raise TransportError(
                f"Could not download {remote_path}: {error.strerror or error}"
            ) from error
        return buffer.getvalue()


def _expand(key_path: str) -> Path:
    """Resolve a leading ``~`` the way the form's placeholder implies.

    ``Path("~/.ssh/id_ed25519")`` does not exist, so a form that suggests the
    tilde form would report a missing key for a key that is really there.
    """
    if key_path.startswith("~"):
        return Path.home() / key_path[1:].lstrip("/")
    return Path(key_path)


def local_key_exists(spec: TargetSpec) -> bool:
    """True when the spec names a key file that is actually present."""
    return _expand(spec.key_path).is_file()
