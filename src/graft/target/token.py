"""Find a Jupyter token on a target so the user does not have to paste it.

Cloud GPU images routinely start Jupyter *inside* a container and print the
token only to a log the user never sees. When SSH already reaches the host,
Graft can log in once, look where a Jupyter server records its token, and hand
it back to fill the form. The token is returned in memory only; it is never
written to disk.
"""

from __future__ import annotations

import re

from graft.target.spec import TargetError

# Look on the host and inside each running container, then fall back to a live
# process's own arguments. Every branch is best-effort: a missing tool or an
# unreadable file must not abort the search, so failures are swallowed and the
# caller simply sees no token rather than a broken command.
TOKEN_DISCOVERY_SCRIPT = r"""
set +e

runtime_files() {
  for f in "$HOME"/.local/share/jupyter/runtime/jpserver-*.json \
           "$HOME"/.local/share/jupyter/runtime/nbserver-*.json \
           /root/.local/share/jupyter/runtime/jpserver-*.json \
           /home/*/.local/share/jupyter/runtime/jpserver-*.json \
           /root/.jupyter/jupyter_server_config.py \
           "$HOME"/.jupyter/jupyter_server_config.py; do
    [ -r "$f" ] && cat "$f" 2>/dev/null
  done
}

if command -v jupyter >/dev/null 2>&1; then
  jupyter server list 2>/dev/null
  jupyter notebook list 2>/dev/null
fi
runtime_files

for runtime in docker podman; do
  command -v "$runtime" >/dev/null 2>&1 || continue
  for c in $("$runtime" ps --format '{{.Names}}' 2>/dev/null); do
    "$runtime" exec "$c" sh -lc 'jupyter server list 2>/dev/null' 2>/dev/null
    "$runtime" exec "$c" sh -lc 'cat /root/.local/share/jupyter/runtime/jpserver-*.json 2>/dev/null' 2>/dev/null
    "$runtime" exec "$c" sh -lc 'cat "$HOME"/.local/share/jupyter/runtime/jpserver-*.json 2>/dev/null' 2>/dev/null
  done
done

ps -eo args 2>/dev/null | grep -Eo '(ServerApp|NotebookApp)\.token=[^ ]+' 2>/dev/null
"""

# A token is opaque, but a real one is long and drawn from the URL-safe base64
# alphabet (which includes '+', '/', and '='). Six characters is the shortest
# worth trusting; shorter runs are almost always a placeholder or a variable
# name caught by the looser patterns below.
_TOKEN_RES = (
    re.compile(r'"token"\s*:\s*"([^"]{6,})"'),
    re.compile(r"'token'\s*:\s*'([^']{6,})'"),
    re.compile(r"\.token\s*=\s*['\"]?([A-Za-z0-9._~+/=%-]{6,})"),
    re.compile(r"\btoken\s*[=:]\s*['\"]?([A-Za-z0-9._~+/=%-]{6,})"),
)

# Values that parse like a token but never are: disabled auth, or an unset
# placeholder. Treating any of them as the answer would be worse than finding
# nothing, because the connection would then fail with a confusing auth error.
_NOT_A_TOKEN = {"disabled", "none", "null", "false", "true", "${token}"}


def parse_jupyter_token(text: str) -> str:
    """Return the first plausible Jupyter token in ``text``, or "".

    Order matters: the JSON and config patterns are tried before the loose
    ``token=`` scan, so a well-formed ``"token": "…"`` line wins over a
    substring of it.
    """
    haystack = text or ""
    for pattern in _TOKEN_RES:
        for match in pattern.finditer(haystack):
            value = match.group(1).strip()
            if value and value.lower() not in _NOT_A_TOKEN:
                return value
    return ""


def discover_jupyter_token(transport, *, timeout_s: int = 60) -> str:
    """Log in over ``transport`` and return the Jupyter token it finds.

    ``transport`` is any object with ``run(command, *, timeout_s)``; in
    practice an SSH transport, because token discovery is a filesystem and
    process task that has to run on the host.
    """
    result = transport.run(TOKEN_DISCOVERY_SCRIPT, timeout_s=timeout_s)
    token = parse_jupyter_token(result.stdout)
    if not token:
        raise TargetError(
            "No Jupyter token was found on the host or its containers. Start "
            "the Jupyter server (or paste the token) and try again."
        )
    return token


__all__ = [
    "TOKEN_DISCOVERY_SCRIPT",
    "discover_jupyter_token",
    "parse_jupyter_token",
]
