"""Popup overlay holding the original and converted source.

The report stays a summary: the two files are long, and inlining both pushed
the measured evidence off the screen. This modal carries the same numbered
panes the report used to render inline, one scroll for each side and a copy
button apiece, so the comparison is one click away instead of a scroll.
"""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from graft.tui.screens.results import numbered_source


class SourceComparePopup(ModalScreen[None]):
    """Original versus converted source in a scrollable overlay."""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss_popup", "Close"),
    ]

    def __init__(
        self,
        filename: str,
        original: str,
        converted: str,
        *,
        source_path: str = "",
    ) -> None:
        super().__init__()
        self.filename = filename
        self.original = original
        self.converted = converted
        self.source_path = source_path

    def compose(self) -> ComposeResult:
        with Vertical(id="compare-popup-card"):
            yield Static("SOURCE COMPARISON", id="compare-popup-eyebrow")
            yield Static(
                f"{self.filename} · original "
                f"{len(self.original.splitlines())} line(s) · converted "
                f"{len(self.converted.splitlines())} line(s)",
                id="compare-popup-title",
                markup=False,
            )
            yield Static(
                "The original is what you selected. The converted file is what "
                "the migration produced; this report, not the source, is the "
                "measured evidence.",
                id="compare-popup-note",
                markup=False,
            )
            with VerticalScroll(id="compare-popup-scroll"):
                yield Static("ORIGINAL", classes="compare-popup-section")
                yield Static(
                    self.source_path or self.filename,
                    classes="compare-popup-path",
                    markup=False,
                )
                yield Static(
                    numbered_source(self.original) if self.original else "(empty)",
                    classes="compare-popup-code",
                    markup=False,
                )
                yield Static("CONVERTED", classes="compare-popup-section")
                yield Static(
                    self.filename,
                    classes="compare-popup-path",
                    markup=False,
                )
                yield Static(
                    numbered_source(self.converted)
                    if self.converted
                    else "not available",
                    classes="compare-popup-code",
                    markup=False,
                )
            with Vertical(id="compare-popup-actions"):
                yield Button("Copy original", id="compare-popup-copy-original")
                yield Button("Copy converted", id="compare-popup-copy-converted")
                yield Button("Dismiss", id="compare-popup-dismiss")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "compare-popup-copy-original":
            if self.original:
                self.app.copy_to_clipboard(self.original)
                self.notify(f"Copied the original {self.filename} to the clipboard")
        elif event.button.id == "compare-popup-copy-converted":
            if self.converted:
                self.app.copy_to_clipboard(self.converted)
                self.notify("Copied the converted file to the clipboard")
        elif event.button.id == "compare-popup-dismiss":
            self.dismiss()

    def action_dismiss_popup(self) -> None:
        self.dismiss()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 3:
            self.dismiss()


__all__ = ["SourceComparePopup"]
