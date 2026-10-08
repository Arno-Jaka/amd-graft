"""Blocking, informative overlay shown while a workspace operation runs."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import LoadingIndicator, Static


class LoadingOverlay(ModalScreen[None]):
    """Keep the app responsive while making background work unmistakable."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [("escape", "ignore_escape", "")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="loading-card"):
            yield Static("GRAFT / WORKING", id="loading-eyebrow", markup=False)
            yield LoadingIndicator(id="loading-spinner")
            yield Static(self.message, id="loading-message", markup=False)
            yield Static(
                "This can take a little while. Results will appear when the operation finishes.",
                id="loading-hint",
                markup=False,
            )

    def set_message(self, message: str) -> None:
        self.message = message
        if self.is_mounted:
            self.query_one("#loading-message", Static).update(message)

    def action_ignore_escape(self) -> None:
        """The active operation owns the overlay until its worker completes."""


__all__ = ["LoadingOverlay"]
