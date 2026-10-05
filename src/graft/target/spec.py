"""Connection details for a remote AMD target.

This module is deliberately strict about two things, because the values it
carries are credentials:

1. Secrets are never rendered. :meth:`TargetSpec.redacted` returns a form
   that is safe to log, embed in HTML, put in a report, or return from an API
   route. Nothing in this package calls ``str(spec)``, ``repr(spec)``, or any
   ``__format__`` that would expose a secret, so an accidental interpolation
   cannot leak one.
2. Secrets are never persisted. The UI collects them per request and hands
   them to a transport in memory. Nothing here writes them to disk, a config
   file, or a cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlsplit

KIND_SSH = "ssh"
KIND_JUPYTER = "jupyter"
KINDS = (KIND_SSH, KIND_JUPYTER)

# Sentinel written in place of a secret. Long enough that a reader can see
# something was removed, short enough not to be mistaken for real output.
REDACTED = "********"

DEFAULT_SSH_PORT = 22
DEFAULT_JUPYTER_PORT = 80
DEFAULT_HTTPS_PORT = 443

SCHEME_HTTP = "http"
SCHEME_HTTPS = "https"
DEFAULT_REMOTE_DIR = "/root/graft"

# Hostnames and addresses only. Deliberately stricter than socket semantics:
# a target is typed by a human into a form, and anything exotic here is far
# more likely to be a paste accident than a real cloud endpoint.
_HOST_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,251}[A-Za-z0-9])?$")

# Remote paths we control. Rejecting everything else keeps a stray value from
# turning into an argument injection or a write outside the intended tree.
_REMOTE_DIR_RE = re.compile(r"^/[A-Za-z0-9._/-]{1,180}$")

_HOST_CHARS_FORBIDDEN = "@:/\\?#"


class TargetError(ValueError):
    """A target description was unusable. Safe to show to the user."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TargetError(message)


def _clean(value: Any) -> str:
    """Coerce a form value to a stripped string, treating None as empty."""
    if value is None:
        return ""
    return str(value).strip()


def _coerce_port(raw: Any, default: int) -> int:
    """Normalise a port to a usable int, raising on nonsense."""
    if raw in (None, "", 0):
        return default
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise TargetError("Port must be a number.") from None
    _require(1 <= port <= 65535, "Port must be between 1 and 65535.")
    return port


def _coerce_timeout(raw: Any) -> int:
    if raw in (None, ""):
        return 900
    try:
        seconds = int(raw)
    except (TypeError, ValueError):
        raise TargetError("Command timeout must be a number of seconds.") from None
    _require(
        10 <= seconds <= 86400,
        "Command timeout must be between 10 and 86400 seconds.",
    )
    return seconds


_BARE_HOST_PORT_RE = re.compile(r"^(?P<host>[^:/]+):(?P<port>\d{1,5})$")


def _split_jupyter_url(raw: str) -> tuple[str, int | None]:
    """Return (host, port-if-explicit) from a bare host, host:port, or URL.

    A Jupyter target is often pasted straight from the browser address bar,
    which brings along a scheme, an optional port, and sometimes a path.
    Reducing it here means the rest of the package only ever sees a host and
    a port.
    """
    if "://" not in raw and "/" not in raw:
        # Bare "host:port", which urlsplit cannot tell apart from a scheme.
        bare = _BARE_HOST_PORT_RE.match(raw)
        if bare:
            return bare.group("host"), int(bare.group("port")), SCHEME_HTTP
        return raw, None, SCHEME_HTTP

    candidate = raw if "://" in raw else f"http://{raw}"
    parts = urlsplit(candidate)
    _require(
        parts.scheme in ("http", "https"),
        "Jupyter URL must use http or https.",
    )
    _require(bool(parts.hostname), "Jupyter URL must include a host.")
    _require(
        not parts.username and not parts.password,
        "Do not put credentials in the URL. Use the token field.",
    )
    return parts.hostname, parts.port or None, parts.scheme


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """An immutable, validated description of a machine to run Graft on.

    Exactly one secret field is used per kind: ``key_passphrase`` for SSH,
    ``token`` for Jupyter. Both are excluded from :meth:`redacted`.
    """

    kind: Literal["ssh", "jupyter"]
    host: str
    port: int = 0
    # Transport security for Jupyter only; SSH decides this itself. Kept as a
    # field because the host string is reduced to a bare name, and dropping the
    # scheme here would silently downgrade an https:// target to plaintext.
    scheme: str = ""
    username: str = ""
    # Path on the local machine to a private key. This is a filesystem path
    # rather than a secret, so it is not a credential; it is still elided
    # from redacted output because it can reveal local directory layout.
    key_path: str = ""
    key_passphrase: str = field(default="", repr=False)
    token: str = field(default="", repr=False)
    remote_dir: str = DEFAULT_REMOTE_DIR
    command_timeout_s: int = 900

    def __post_init__(self) -> None:
        # Normalise before validating. The dataclass is frozen, so each
        # field is set once, here, and validation below then only has to
        # check final values. Doing it in this order is what keeps an empty
        # port from surviving as 0 and rendering as "host:0".
        object.__setattr__(self, "kind", _clean(self.kind).lower())
        object.__setattr__(self, "host", _clean(self.host))
        object.__setattr__(self, "username", _clean(self.username))
        object.__setattr__(self, "key_path", _clean(self.key_path))
        object.__setattr__(
            self, "remote_dir", _clean(self.remote_dir) or DEFAULT_REMOTE_DIR
        )
        object.__setattr__(self, "key_passphrase", _clean(self.key_passphrase))
        object.__setattr__(self, "token", _clean(self.token))
        object.__setattr__(
            self, "command_timeout_s", _coerce_timeout(self.command_timeout_s)
        )

        port_default = DEFAULT_SSH_PORT if self.kind == KIND_SSH else DEFAULT_JUPYTER_PORT
        port = _coerce_port(self.port, port_default)
        if self.kind == KIND_JUPYTER:
            host, explicit, scheme = _split_jupyter_url(self.host)
            if explicit is not None:
                port = _coerce_port(explicit, port_default)
            elif scheme == SCHEME_HTTPS:
                # An https:// URL with no port means 443, not 80. Defaulting
                # to 80 here is what made every hosted provider unreachable.
                port = _coerce_port(self.port, DEFAULT_HTTPS_PORT)
        else:
            host, scheme = self.host, ""
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "scheme", scheme)

        _validate(self)

    # -- secrets ---------------------------------------------------------

    @property
    def secret(self) -> str:
        """The credential for this kind. Empty when none was supplied."""
        return self.key_passphrase if self.kind == KIND_SSH else self.token

    @property
    def has_secret(self) -> bool:
        return bool(self.secret)

    def redacted(self) -> dict[str, Any]:
        """A dict safe to log, render, serialise, or return over HTTP."""
        return {
            "kind": self.kind,
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme,
            "username": self.username,
            "key_path": REDACTED if self.key_path else "",
            "secret": REDACTED if self.has_secret else "",
            "remote_dir": self.remote_dir,
            "command_timeout_s": self.command_timeout_s,
        }

    def safe_label(self) -> str:
        """Short human label for headings and status lines. Never a secret."""
        who = f"{self.username}@" if self.username else ""
        return f"{who}{self.host}:{self.port}"

    def with_changes(self, **changes: Any) -> TargetSpec:
        return replace(self, **changes)


def _validate(spec: TargetSpec) -> None:
    _require(spec.kind in KINDS, "Choose either SSH or Jupyter.")
    _validate_host(spec.host)

    if spec.kind == KIND_JUPYTER:
        _require(
            spec.scheme in (SCHEME_HTTP, SCHEME_HTTPS),
            "Jupyter URL must use http or https.",
        )

    if spec.kind == KIND_SSH:
        _require(len(spec.username) <= 64, "Username is too long.")
        _require(
            not any(character.isspace() for character in spec.username),
            "Username cannot contain spaces.",
        )
        # An explicit key is required. The alternative is letting paramiko fall
        # back to the local ssh-agent, which would silently borrow a standing
        # credential from the machine and make the form's promise ("used for
        # this request only") false.
        _require(bool(spec.key_path), "SSH needs a private key path.")
        _require(len(spec.key_path) <= 4096, "Key path is too long.")
        _require("\x00" not in spec.key_path, "Key path contains an invalid character.")

    _require(
        bool(_REMOTE_DIR_RE.match(spec.remote_dir)),
        "Remote directory must be an absolute path made of letters, digits, "
        "dots, hyphens, slashes, and underscores.",
    )
    _require(
        not spec.remote_dir.endswith("/"),
        "Remote directory must not end with a slash.",
    )
    _require(
        "//" not in spec.remote_dir,
        "Remote directory must not contain an empty path segment.",
    )
    _require(
        ".." not in spec.remote_dir.split("/"),
        "Remote directory must not contain '..'.",
    )

    if spec.secret:
        _require(len(spec.secret) <= 8192, "Credential is too long.")
        _require(
            not any(bad in spec.secret for bad in ("\x00", "\n", "\r")),
            "Credential contains an invalid character.",
        )


def _validate_host(raw: str) -> None:
    _require(bool(raw), "Host is required.")
    _require(len(raw) <= 253, "Host is too long.")
    _require(
        not any(character.isspace() for character in raw),
        "Host cannot contain spaces.",
    )
    _require(
        not any(character in raw for character in _HOST_CHARS_FORBIDDEN),
        "Host must be a bare hostname or IP address, without a scheme, "
        "port, path, or credentials.",
    )
    _require(
        bool(_HOST_RE.match(raw)),
        "Host must be a bare hostname or IP address (letters, digits, dots, "
        "hyphens, and underscores only).",
    )


def parse_form(fields: dict[str, Any]) -> TargetSpec:
    """Build a validated :class:`TargetSpec` from raw form values.

    Only the secret belonging to the chosen kind is read, so a form that
    carries both fields cannot leak the unused one into a transport. Unknown
    keys are ignored so a form can carry presentation-only fields.
    """
    _require(bool(fields), "No connection details were provided.")

    kind = _clean(fields.get("kind")).lower()
    _require(kind in KINDS, "Choose either SSH or Jupyter.")

    if kind == KIND_SSH:
        secret = _clean(fields.get("key_passphrase"))
    else:
        secret = _clean(fields.get("token"))

    return TargetSpec(
        kind=kind,
        host=_clean(fields.get("host")),
        # 0 means "not given". Deciding the default here instead would fix the
        # port at 80 and make an https:// URL with no port unreachable.
        port=_clean(fields.get("port")) or 0,
        username=_clean(fields.get("username")),
        key_path=_clean(fields.get("key_path")),
        key_passphrase=secret if kind == KIND_SSH else "",
        token=secret if kind == KIND_JUPYTER else "",
        remote_dir=_clean(fields.get("remote_dir")) or DEFAULT_REMOTE_DIR,
        command_timeout_s=_clean(fields.get("command_timeout_s")) or 900,
    )
