"""Troubleshooting tips shown after a connection is refused.

A remote GPU is most often unreachable for one of four reasons, and the
connection error alone does not say which: SSH may be refused because the
host's fail2ban banned this machine's address after a few bad key attempts,
the credentials may be wrong, the server may be slow, or the address may
simply be gone. Each of those has a different fix, so the raw error is
classified and the matching advice is shown rather than a generic retry.

The classification and the tip text are pure functions so they can be tested
without a terminal; the screen is only the presentation of their result.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

# Ordered from most to least specific. The first category whose keywords match
# wins, because a refused socket and a wrong key can both mention the host.
CATEGORIES = ("banned", "auth", "timeout", "unreachable", "other")

_BANNED_MARKERS = (
    "unable to connect",
    "connection refused",
    "connection reset",
    "connection aborted",
    "no route to host",
    "network is unreachable",
)

_AUTH_MARKERS = (
    "authentication failed",
    "permission denied",
    "refused the request",
    "check the token",
    "private key file was not found",
    "host key",
    "invalid key",
    "incorrect passphrase",
    "passphrase",
)

_TIMEOUT_MARKERS = (
    "timed out",
    "timeout",
    "timed-out",
)

_UNREACHABLE_MARKERS = (
    "could not reach",
    "name or service not known",
    "name resolution",
    "temporary failure in name resolution",
    "nodename nor servname",
)


def classify_connection_failure(error: str, kind: str = "ssh") -> str:
    """Return the category of a failed connection from its message.

    ``kind`` is the transport ("ssh" or "jupyter"); it only breaks ties, so a
    token complaint is read as an auth problem even if it never says so.
    """
    lowered = (error or "").lower()
    if any(marker in lowered for marker in _BANNED_MARKERS):
        return "banned"
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return "auth"
    if any(marker in lowered for marker in _TIMEOUT_MARKERS):
        return "timeout"
    if any(marker in lowered for marker in _UNREACHABLE_MARKERS):
        return "unreachable"
    return "other"


_TITLES = {
    "banned": "SSH is refusing connections",
    "auth": "The connection was authenticated but refused",
    "timeout": "The connection timed out",
    "unreachable": "The host could not be reached",
    "other": "The connection failed",
}

_SSH_BAN_TIPS = [
    "The host answered and then refused the connection. That is usually "
    "fail2ban banning this machine's address after a few failed key "
    "attempts, not a bad key.",
    "Open the provider's web or serial console and run: "
    "fail2ban-client status sshd",
    "Unban this machine with: fail2ban-client set sshd unbanip "
    "<your-address>  (the provider console shows the address fail2ban lists).",
    "If there is no fail2ban, check for a firewall rule: iptables -S | grep 22. "
    "A REJECT or DROP on port 22 blocks the connection the same way.",
    "Do not retry in a loop: every failed attempt can extend the ban.",
    "If the provider exposes SSH only on a VPN, the Jupyter transport may "
    "reach the same machine.",
]

_SSH_AUTH_TIPS = [
    "Authentication failed. Confirm the username (cloud images often use "
    "root or ubuntu) and that the key path is the one registered with the "
    "instance.",
    "A passphrase-protected key needs its passphrase, under Advanced "
    "connection settings.",
    "A host key mismatch means the instance was rebuilt; remove the stale "
    "entry from ~/.ssh/known_hosts and reconnect.",
    "The private key file must be readable by this process; check the path "
    "and its permissions.",
]

_JUPYTER_AUTH_TIPS = [
    "The server is reachable but rejected the request. The token is missing, "
    "wrong, or expired.",
    "Find the token on the host with: jupyter server list, or read "
    "~/.local/share/jupyter/runtime/jpserver-*.json.",
    "If the token was disabled and a password set instead, this token field "
    "will not authenticate.",
    "Confirm the URL and port match the one the browser uses: a hosted "
    "notebook is usually https on 443 or 80, not 8888.",
    "Graft needs the terminal API (Jupyter 5 or newer) to run commands. If "
    "/terminals is disabled on the server, the run cannot proceed.",
]

_TIMEOUT_TIPS = [
    "The connection timed out before the server answered.",
    "Confirm the port is right (22 for SSH; 80, 443, or 8888 for Jupyter) "
    "and that the instance is running.",
    "Some providers expose SSH only on a private network; the Jupyter "
    "transport may reach the same machine over the web API.",
]

_UNREACHABLE_TIPS = [
    "The host could not be reached at all.",
    "Double-check the public address: cloud instances often change it after "
    "a restart or stop.",
    "Confirm the port is open in the provider's firewall for this machine's "
    "address.",
]

_OTHER_TIPS = [
    "Check the host, port, and credentials, then try again.",
    "The provider's console usually shows the correct address and how to "
    "reach the instance.",
]


def tips_for(category: str, kind: str = "ssh") -> tuple[str, list[str]]:
    """Return (headline, tips) for a classified failure."""
    title = _TITLES.get(category, _TITLES["other"])
    if category == "banned":
        return title, _SSH_BAN_TIPS
    if category == "auth":
        return title, _JUPYTER_AUTH_TIPS if kind == "jupyter" else _SSH_AUTH_TIPS
    if category == "timeout":
        return title, _TIMEOUT_TIPS
    if category == "unreachable":
        return title, _UNREACHABLE_TIPS
    return title, _OTHER_TIPS


class TipsScreen(ModalScreen[None]):
    """A dismissible panel of advice for a failed connection."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss_tips", "Close"),
        ("enter", "dismiss_tips", "Close"),
    ]

    def __init__(self, category: str, kind: str, error: str) -> None:
        super().__init__()
        self.category = category
        self.kind = kind
        self.error = error
        self.headline, self.tips = tips_for(category, kind)

    def compose(self) -> ComposeResult:
        with Vertical(id="tips-card"):
            yield Static("GRAFT / CONNECTION HELP", id="tips-eyebrow", markup=False)
            yield Static(self.headline, id="tips-title", markup=False)
            if self.error:
                yield Static(self.error, id="tips-error", markup=False)
            with VerticalScroll(id="tips-list"):
                for tip in self.tips:
                    yield Static(f"• {tip}", classes="tip", markup=False)
            yield Button("Close", id="tips-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tips-close":
            self.dismiss()

    def action_dismiss_tips(self) -> None:
        self.dismiss()


__all__ = ["CATEGORIES", "TipsScreen", "classify_connection_failure", "tips_for"]
