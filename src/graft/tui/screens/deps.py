"""Overlay shown the moment a GPU target connects: what it is missing.

Connecting to a target used to bury the dependency list in the results
screen, after the fact. This overlay inverts that: the moment a local or
remote GPU is checked, it opens with the checklist the pipeline needs
(hipify-perl, hipcc, PyTorch, …), names what is missing, and offers to
install the pieces Graft can install, showing live progress per tool.

Installs are never automatic. They are system-wide and can be large, so
the button is only enabled when something is both missing and installable,
and the exact command runs only after it is pressed. A local target runs
the same fixed command table through a subprocess; a remote one goes over
the already-open transport, exactly like the results-screen flow it
complements.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, ProgressBar, Static

from graft.target.dependencies import (
    Dependency,
    install_dependencies,
    scan_dependencies,
)
from graft.target.detect import TargetFacts

_PENDING = "pending"
_INSTALLING = "installing"
_PRESENT = "present"
_MANUAL = "manual"


class DependencyOverlay(ModalScreen[None]):
    """Checklist of the tools a target is missing, with an install action.

    Args:
        facts: a `TargetFacts`-shaped mapping for the connected target.
        fields: connection form values for a remote target, or ``None``
            when the target is this machine.
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
        on_refreshed: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        self._facts = dict(facts or {})
        self._fields = dict(fields) if fields else None
        self._label = label
        self._on_refreshed = on_refreshed
        self._installing = False
        self._deps: list[Dependency] = []
        self._states: dict[str, str] = {}
        self._results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        try:
            parsed = TargetFacts(**self._facts)
        except TypeError:
            parsed = TargetFacts(ok=False)
        self._deps = scan_dependencies(parsed)
        self._states = {
            dep.key: (_PRESENT if dep.present else self._initial_state(dep))
            for dep in self._deps
        }
        kind = "this machine" if self._fields is None else self._label
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
            yield ProgressBar(id="deps-progress", show_eta=False)
            yield Static("", id="deps-status", markup=False)
            with Horizontal(id="deps-actions"):
                yield Button(
                    "Install missing dependencies",
                    id="deps-install",
                    variant="primary",
                    disabled=not self._installable(),
                )
                yield Button("Close", id="deps-close")

    @staticmethod
    def _initial_state(dep: Dependency) -> str:
        return _PENDING if dep.command else _MANUAL

    def _installable(self) -> bool:
        return any(dep.command and not dep.present for dep in self._deps)

    def _row_text(self, dep: Dependency) -> str:
        state = self._states.get(dep.key, _PENDING)
        if state == _PRESENT:
            mark, detail = "ok", dep.detail or "present"
        elif state == _INSTALLING:
            mark, detail = "···", "installing…"
        elif state == _MANUAL:
            mark, detail = "!!", dep.detail or "install by hand"
        else:
            mark, detail = "missing", dep.detail or "not found"
        line = f"{mark:<8} {dep.label}  {detail}"
        if state in {_PENDING, _MANUAL} and dep.command:
            line += f"\n         run: {dep.command}"
        return line

    @staticmethod
    def _row_class(dep: Dependency) -> str:
        if dep.present:
            return "dep-present"
        return "dep-missing" if dep.command else "dep-manual"

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "deps-install":
            self._start_install()
        elif event.button.id == "deps-close":
            self.dismiss()

    def action_dismiss_overlay(self) -> None:
        # While an install runs the escape key is inert: the commands are
        # already in flight on the target and the user needs the outcome.
        if not self._installing:
            self.dismiss()

    def _refresh_rows(self) -> None:
        for dep in self._deps:
            row = self.query_one(f"#dep-{dep.key}", Static)
            row.update(self._row_text(dep))
            row.set_class(True, self._row_class(dep))

    def _set_status(self, message: str) -> None:
        self.query_one("#deps-status", Static).update(message)

    # ------------------------------------------------------------------
    # install
    # ------------------------------------------------------------------
    def _start_install(self) -> None:
        if self._installing:
            return
        missing = [dep for dep in self._deps if dep.command and not dep.present]
        if not missing:
            return
        self._installing = True
        self.query_one("#deps-install", Button).disabled = True
        self.query_one("#deps-close", Button).disabled = True
        progress = self.query_one("#deps-progress", ProgressBar)
        progress.update(total=len(missing), progress=0)
        self._set_status(f"Installing {len(missing)} dependencies on {self._label}…")
        self._do_install()

    @work(exclusive=True, thread=True)
    def _do_install(self) -> None:
        """Run the installs off the UI thread, reporting each step live."""
        from graft.target import parse_form
        from graft.target.local import LocalTransport

        transport = None
        spec = None
        try:
            if self._fields is not None:
                spec = parse_form(self._fields)
                if spec.kind == "ssh":
                    from graft.target.ssh import SSHTransport

                    transport = SSHTransport(spec)
                else:
                    from graft.target.jupyter import JupyterTransport

                    transport = JupyterTransport(spec)
            else:
                transport = LocalTransport()
            facts = TargetFacts(**self._facts)
            with transport:
                results = install_dependencies(
                    transport,
                    spec,
                    facts,
                    on_progress=self._report_started,
                    on_result=self._report_finished,
                )
                # Re-detect afterwards so the checklist shows what the
                # target actually has now, not what it had before.
                refreshed: dict[str, Any] | None
                if self._fields is not None:
                    from graft.target.detect import detect

                    refreshed = detect(transport).as_dict()
                else:
                    from graft.target.local import local_target_facts

                    refreshed = local_target_facts()
        except Exception as exc:  # a broken transport must not kill the app
            self.app.call_from_thread(
                self._install_failed, f"{type(exc).__name__}: {exc}"
            )
            return
        self.app.call_from_thread(self._install_done, results, refreshed)

    def _report_started(self, key: str, message: str) -> None:
        self.app.call_from_thread(self._mark_installing, key, message)

    def _report_finished(self, outcome: dict[str, Any]) -> None:
        self.app.call_from_thread(self._mark_result, outcome)

    def _mark_installing(self, key: str, message: str) -> None:
        self._states[key] = _INSTALLING
        self._refresh_rows()
        self._set_status(message)

    def _mark_result(self, outcome: dict[str, Any]) -> None:
        key = str(outcome.get("key") or "")
        self._states[key] = _PRESENT if outcome.get("ok") else _MANUAL
        self._results.append(outcome)
        progress = self.query_one("#deps-progress", ProgressBar)
        progress.advance(1)
        label = outcome.get("label") or key
        if outcome.get("ok"):
            self._set_status(f"Installed {label}")
        else:
            self._set_status(f"Could not install {label} · see the row for why")
        self._refresh_rows()

    def _install_failed(self, message: str) -> None:
        self._installing = False
        # A tool whose row still says "installing…" never got an answer; it
        # goes back to its prior state rather than lying about progress.
        for dep in self._deps:
            if self._states.get(dep.key) == _INSTALLING:
                self._states[dep.key] = (
                    _PRESENT if dep.present else self._initial_state(dep)
                )
        self._refresh_rows()
        self._set_status(f"Dependency install failed · {message}")
        self.query_one("#deps-close", Button).disabled = False
        self.query_one("#deps-install", Button).disabled = not self._installable()

    def _install_done(
        self,
        results: list[dict[str, Any]],
        refreshed: dict[str, Any] | None,
    ) -> None:
        self._installing = False
        self._results = results
        failed = [item for item in results if not item.get("ok")]
        if refreshed:
            self._facts = refreshed
            try:
                parsed = TargetFacts(**refreshed)
                self._deps = scan_dependencies(parsed)
                self._states = {
                    dep.key: (_PRESENT if dep.present else self._initial_state(dep))
                    for dep in self._deps
                }
                self.query_one("#deps-title", Static).update(
                    f"{parsed.summary_line()} · re-checked after install"
                )
            except TypeError:
                pass
        self._refresh_rows()
        if failed:
            names = ", ".join(str(item.get("label") or item.get("key")) for item in failed)
            self._set_status(f"Finished · still missing: {names}")
        else:
            self._set_status("Finished · all installable dependencies are present")
        self.query_one("#deps-close", Button).disabled = False
        self.query_one("#deps-install", Button).disabled = not self._installable()
        # Let the workspace re-gate its buttons on the refreshed facts.
        if self._on_refreshed is not None and refreshed is not None:
            self._on_refreshed(refreshed)


__all__ = ["DependencyOverlay"]
