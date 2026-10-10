"""Popup offering to copy the text the user highlighted.

Textual already performs the highlight: dragging with the left button
selects text in any `Static` and `Screen.get_selected_text()` returns it.
What was missing is a way to act on that selection with the mouse, so a
right-click opens this popup with the selected text and one button that
puts it on the clipboard.

The popup copies on open-then-click, never on open: a right-click alone
must not touch the clipboard, because terminals reach the clipboard
through OSC 52 and an unexpected write is hostile on a machine whose
clipboard is a password manager.
"""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

# Long selections are common (a whole report gets highlighted); the popup
# previews a bounded prefix so it renders instantly, while the copy itself
# always carries the full text.
PREVIEW_CHARS = 1200
PREVIEW_TAIL = 120


def preview_text(text: str) -> str:
    """Clamp a selection for display, saying how much was left out."""
    if len(text) <= PREVIEW_CHARS:
        return text
    kept = text[:PREVIEW_CHARS]
    omitted = len(text) - PREVIEW_CHARS
    return f"{kept}\n… {omitted} more characters not shown"


class CopyPopup(ModalScreen[None]):
    """Show the highlighted text and copy it to the clipboard on request."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss_popup", "Close"),
        ("enter", "copy_selection", "Copy"),
    ]

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="copy-popup-card"):
            yield Static("COPY SELECTION", id="copy-popup-eyebrow", markup=False)
            yield Static(
                f"{len(self.text)} characters selected",
                id="copy-popup-count",
                markup=False,
            )
            with VerticalScroll(id="copy-popup-scroll"):
                yield Static(preview_text(self.text), id="copy-popup-text", markup=False)
            with Vertical(id="copy-popup-actions"):
                yield Button(
                    "Copy to clipboard", id="copy-popup-copy", variant="primary"
                )
                yield Button("Dismiss", id="copy-popup-dismiss")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-popup-copy":
            self.action_copy_selection()
        elif event.button.id == "copy-popup-dismiss":
            self.dismiss()

    def action_copy_selection(self) -> None:
        if not self.text:
            self.notify("Nothing to copy", severity="warning")
            self.dismiss()
            return
        self.app.copy_to_clipboard(self.text)
        self.notify(f"Copied {len(self.text)} characters to the clipboard")
        self.dismiss()

    def action_dismiss_popup(self) -> None:
        self.dismiss()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        # A right-click inside the popup closes it, matching how the popup
        # was opened, rather than layering a second popup on itself.
        if event.button == 3:
            self.dismiss()


__all__ = ["CopyPopup", "preview_text"]
