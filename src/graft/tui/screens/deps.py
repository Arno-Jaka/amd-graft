"""Report-only overlay shown when a GPU target's tools are incomplete.

Connecting to a target used to bury the dependency list in the results screen,
after the fact. This overlay inverts that: the moment a local or remote GPU is
checked, it opens with the checklist the pipeline needs (hipify-perl, hipcc,
PyTorch, ...), naming what is present and what is missing.

It is report-only: no install action. A target's dependency state is the
operator's business to fix by hand (e.g. via the cloud console or an SSH
session) — Graft only reports what it found so the operator knows why a later
step might fail.
"""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from graft.target.dependencies import Dependency, scan_dependencies
from graft.target.detect import TargetFacts

_PRESENT = "present"
_MANUAL = "manual"


class DependencyOverlay(ModalScreen[None]):
    """Report-only checklist of the tools a target is missing.

    Args:
        facts: a `TargetFacts`-shaped mapping for the connected target.
        fields: connection form values for a remote target, or ``None``
            when the target is this machine. Unused for now; kept for
            backward compatibility with callers that pass it.
        label: a short human label for the target, e.g. a host name.
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "dismiss_overlay", "Close"),
    ]

    def __init__(
        self,
        facts: dict[str, Any],
        *,
        fields: dict[str, Any] | None = None,
        label: str = "target",
        on_refreshed: Any = None,
    ) -> None:
        super().__init__()
        self._facts = dict(facts or {})
        self._label = label
        self._deps: list[Dependency] = []

    def compose(self) -> ComposeResult:
        try:
            parsed = TargetFacts(**self._facts)
        except TypeError:
            parsed = TargetFacts(ok=False)
        self._deps = scan_dependencies(parsed)
        kind = "this machine" if self._facts else self._label
        with Vertical(id="deps-card"):
            yield Static("TARGET DEPENDENCIES", id="deps-eyebrow", markup=False)
            yield Static(
                f"{parsed.summary_line()} · checked on {kind}",
                id="deps-title",
                markup=False,
            )
            with VerticalScroll(id="deps-list"):
                for dep in self._deps:
                    yield Static(
                        self._row_text(dep),
                        markup=False,
                        classes=f"dep-row {self._row_class(dep)}",
                        id=f"dep-{dep.key}",
                    )
            yield Static("", id="deps-status", markup=False)
            yield Button("Close", id="deps-close", variant="primary")

    def _row_text(self, dep: Dependency) -> str:
        state = _PRESENT if dep.present else _MANUAL if not dep.command else "missing"
        if state == _PRESENT:
            mark, detail = "ok", dep.detail or "present"
        elif state == _MANUAL:
            mark, detail = "!!", dep.detail or "install by hand"
        else:
            mark, detail = "missing", dep.detail or "not found"
        line = f"{mark:<8} {dep.label}  {detail}"
        if not dep.present and dep.command:
            line += f"\n         to install: {dep.command}"
        return line

    @staticmethod
    def _row_class(dep: Dependency) -> str:
        if dep.present:
            return "dep-present"
        return "dep-missing" if dep.command else "dep-manual"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "deps-close":
            self.dismiss()

    def action_dismiss_overlay(self) -> None:
        self.dismiss()


__all__ = ["DependencyOverlay"]
