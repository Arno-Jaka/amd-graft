"""The TUI results screen renders the evidence the worker actually returned.

These run headless through Textual's test pilot. They exist mainly to keep the
optional model sections honest: the AI review is computed by the worker, so it
must be visible here, and it must never be presented as deterministic evidence.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, ProgressBar, Select, Static

from graft.target.dependencies import scan_dependencies
from graft.target.detect import TargetFacts
from graft.target.metrics import MetricsSnapshot
from graft.tui.app import GraftTUI
from graft.tui.screens.copy_popup import CopyPopup
from graft.tui.screens.deps import DependencyOverlay
from graft.tui.screens.results import ResultsScreen
from graft.tui.screens.target import TargetScreen
from graft.tui.screens.tips import TipsScreen, classify_connection_failure, tips_for
from graft.tui.widgets import GPUMetrics, GPUStatusPanel


class _Host(App):
    def compose(self) -> ComposeResult:
        yield ResultsScreen(id="results")


def _summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "conversion": {
            "verdict": "needs_review",
            "verdict_line": "converted, engineer review advised",
            "route": "pytorch",
            "backend": "hipify",
            "verdict_basis": "static analysis",
            "changes": [],
            "findings": [],
        },
        "parity": {"measured": False},
        "execution": {"measured": False},
        "environment": {},
        "benchmarks": [],
        "narrative": {},
        "converted_source": "print('converted')\n",
    }
    summary.update(overrides)
    return summary


def _render(summary: dict[str, Any]) -> str:
    async def main() -> str:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.query_one(ResultsScreen)
            screen.show_local(summary, "m.py", "/tmp/m.py", "print('original')\n")
            await pilot.pause()
            return "\n".join(str(widget.render()) for widget in screen.query(Static))

    return asyncio.run(main())


def test_local_results_render_an_available_ai_review():
    text = _render(
        _summary(
            ai_review={
                "available": True,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "recommendation": "engineer_review_required",
                "summary": "Looks plausible but unverified.",
                "comments": [
                    {
                        "file": "converted",
                        "line": 3,
                        "severity": "review",
                        "title": "Check the stream",
                        "comment": "Confirm the stream matches the original.",
                        "suggestion": "Use torch.cuda.current_stream().",
                    }
                ],
            }
        )
    )
    assert "AI REVIEW" in text
    assert "engineer_review_required" in text
    assert "Looks plausible but unverified." in text
    assert "Check the stream" in text
    assert "Suggested change" in text
    assert "not proof of compilation" in text


def test_local_results_render_an_unavailable_ai_review_reason():
    text = _render(
        _summary(ai_review={"available": False, "reason": "No API key was provided."})
    )
    assert "AI REVIEW" in text
    assert "No API key was provided." in text


def test_local_results_omit_the_ai_section_when_no_review_was_requested():
    text = _render(_summary(ai_review=None))
    assert "AI REVIEW" not in text


# --- connection tips --------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "kind", "expected"),
    [
        ("Could not reach root@host: Connection refused", "ssh", "banned"),
        ("Timed out connecting to host on port 22", "ssh", "timeout"),
        (
            "Authentication failed. Check the username, key path, and passphrase.",
            "ssh",
            "auth",
        ),
        ("Jupyter refused the request while listing files. Check the token.", "jupyter", "auth"),
        ("Could not reach host: Name or service not known", "jupyter", "unreachable"),
        ("something else entirely", "ssh", "other"),
    ],
)
def test_classify_connection_failure(error, kind, expected):
    assert classify_connection_failure(error, kind) == expected


def test_a_banned_failure_leads_with_unban_advice_not_a_retry():
    headline, tips = tips_for("banned", "ssh")
    joined = " ".join(tips)
    assert "fail2ban" in joined
    assert "unban" in joined
    # The last thing a banned user needs is to retry in a loop.
    assert any("Do not retry" in tip for tip in tips)
    assert headline


def test_jupyter_auth_and_ssh_auth_give_different_advice():
    assert tips_for("auth", "jupyter")[1] != tips_for("auth", "ssh")[1]
    assert "token" in " ".join(tips_for("auth", "jupyter")[1]).lower()


def test_the_tips_modal_renders_its_headline_and_tips():
    async def main() -> str:
        app = _Host()
        async with app.run_test() as pilot:
            app.push_screen(TipsScreen("banned", "ssh", "Connection refused"))
            await pilot.pause()
            screen = app.screen
            return "\n".join(str(widget.render()) for widget in screen.query(Static))

    text = asyncio.run(main())
    assert "CONNECTION HELP" in text
    assert "fail2ban" in text


# --- live metrics widget ----------------------------------------------------


class _MetricsHost(App):
    def compose(self) -> ComposeResult:
        yield GPUMetrics(id="metrics")


def _render_metrics(snapshot: MetricsSnapshot) -> str:
    async def main() -> str:
        app = _MetricsHost()
        async with app.run_test() as pilot:
            widget = app.query_one(GPUMetrics)
            widget.set_metrics(snapshot)
            await pilot.pause()
            return str(widget.render())

    return asyncio.run(main())


def test_metrics_widget_draws_bars_for_a_measured_target():
    text = _render_metrics(
        MetricsSnapshot(
            source="remote",
            gpu_name="AMD Instinct MI300X",
            vram_used_bytes=40_000_000_000,
            vram_total_bytes=100_000_000_000,
            gpu_util_pct=41,
            temp_c=52,
            power_w=143,
            disk_used_bytes=250_000_000_000,
            disk_total_bytes=500_000_000_000,
        )
    )
    assert "VRAM" in text
    assert "40%" in text
    assert "41%" in text
    assert "52°C" in text
    assert "TARGET" in text


def test_metrics_widget_says_not_measured_rather_than_zero():
    text = _render_metrics(
        MetricsSnapshot(source="local", error="No local GPU visible to torch")
    )
    assert "not measured" in text
    assert "No local GPU visible to torch" in text


# --- dependency section in results ------------------------------------------


def test_results_render_a_dependency_checklist():
    async def main() -> str:
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_dependency_install(
                [
                    {
                        "key": "torch",
                        "label": "PyTorch (ROCm)",
                        "ok": True,
                        "output": "",
                        "error": "",
                    },
                    {
                        "key": "hipcc",
                        "label": "hipcc (ROCm compiler)",
                        "ok": False,
                        "output": "",
                        "error": "not apt-based",
                    },
                ],
                {
                    "ok": True,
                    "os_name": "Ubuntu 24.04.5 LTS",
                    "tools": {"hipcc": "/opt/rocm/bin/hipcc"},
                    "python_path": "/usr/bin/python3",
                },
            )
            await pilot.pause()
            return "\n".join(str(widget.render()) for widget in screen.query(Static))

    text = asyncio.run(main())
    assert "Dependency check" in text
    assert "PyTorch (ROCm): installed" in text
    assert "hipcc (ROCm compiler): needs attention" in text


# --- copyable results -------------------------------------------------------


def test_copy_results_puts_the_displayed_text_on_the_clipboard():
    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_dependency_install(
                [
                    {
                        "key": "torch",
                        "label": "PyTorch (ROCm)",
                        "ok": True,
                        "output": "",
                        "error": "",
                    }
                ],
                {"ok": True, "python_path": "/usr/bin/python3"},
            )
            await pilot.pause()
            await pilot.click("#copy-results")
            await pilot.pause()
            return app._clipboard

    text = asyncio.run(main())
    assert text
    assert "PyTorch (ROCm): installed" in text


def test_copy_results_button_is_hidden_until_there_are_results():
    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            empty = app.query_one("#result-tools").display
            screen = app.query_one(ResultsScreen)
            screen.show_dependency_install([], {"ok": True})
            await pilot.pause()
            return empty, app.query_one("#result-tools").display

    empty, after = asyncio.run(main())
    assert empty is False
    assert after is True


def test_results_render_a_red_missing_dependency_error():
    dep = next(
        item
        for item in scan_dependencies(TargetFacts(ok=True, python_path="/usr/bin/python3"))
        if item.key == "torch"
    )

    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_missing_dependencies(
                "Running the converted file",
                [dep],
                "Running the converted file needs PyTorch (ROCm) on the target.",
            )
            await pilot.pause()
            heading = screen.query_one("#result-heading", Static)
            card = screen.query_one(".result-card", Static)
            return (
                str(heading.render()),
                "error" in heading.classes,
                "error" in card.classes,
                str(card.render()),
            )

    heading, heading_red, card_red, body = asyncio.run(main())
    assert heading == "Missing dependencies"
    assert heading_red and card_red
    assert "PyTorch (ROCm)" in body
    assert "install:" in body


# --- target gating and pre-flight checks ------------------------------------


def _target_facts(**overrides):
    base: dict = {
        "ok": True,
        "os_name": "Ubuntu 24.04.5 LTS",
        "rocm_version": "6.2.0",
        "gfx_arch": "gfx942",
        "gpu_count": 1,
        "gpu_name": "AMD Instinct MI300X",
        "tools": {
            "hipcc": "/opt/rocm/bin/hipcc",
            "hipify-perl": "/opt/rocm/bin/hipify-perl",
        },
        "python_path": "/usr/bin/python3",
    }
    base.update(overrides)
    return TargetFacts(**base).as_dict()


def test_install_button_only_shows_for_a_checked_and_lacking_target():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            button = screen.query_one("#install-deps", Button)
            before = (button.display, button.disabled)

            screen._connected_facts = _target_facts(torch_version="")
            screen._refresh_dependency_button()
            await pilot.pause()
            missing = (button.display, button.disabled)

            screen._connected_facts = _target_facts(torch_version="2.4.0+rocm6.2")
            screen._refresh_dependency_button()
            await pilot.pause()
            complete = (button.display, button.disabled)
            return before, missing, complete

    before, missing, complete = asyncio.run(main())
    assert before == (False, True)
    assert missing == (True, False)
    assert complete == (False, True)


def test_a_python_run_is_blocked_and_names_pytorch():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._connected_facts = _target_facts(torch_version="")
            blocking = screen._blocking_dependencies(".py")
            screen._show_dependency_block("Running the converted file", blocking)
            await pilot.pause()
            blob = "\n".join(
                str(widget.render())
                for widget in app.query_one("#results-view", ResultsScreen).query(Static)
            )
            return [dep.key for dep in blocking], app.query_one("#main-tabs").active, blob

    keys, active, blob = asyncio.run(main())
    assert keys == ["torch"]
    assert active == "results"
    assert "PyTorch" in blob
    assert "Install missing dependencies" in blob


def test_a_cu_run_and_verification_are_not_blocked_by_missing_torch():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._connected_facts = _target_facts(torch_version="")
            return (
                screen._blocking_dependencies(".cu"),
                screen._blocking_dependencies(None),
            )

    cu, verify = asyncio.run(main())
    assert cu == []
    assert verify == []


def test_a_finished_operation_reroutes_to_results():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            app.query_one("#main-tabs").active = "workspace"
            screen._local_probe_done({"gpu": False, "device": "CPU-only"}, "")
            await pilot.pause()
            return app.query_one("#main-tabs").active

    assert asyncio.run(main()) == "results"


def test_a_python_run_that_failed_on_missing_torch_shows_the_red_block():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._remote_done(
                {
                    "ok": False,
                    "stage": "pytorch_gpu_check",
                    "filename": "converted.py",
                    "facts": _target_facts(torch_version=""),
                    "error": "The target has no PyTorch installed.",
                },
                "file",
                "",
            )
            await pilot.pause()
            results = app.query_one("#results-view", ResultsScreen)
            return (
                app.query_one("#main-tabs").active,
                "\n".join(str(widget.render()) for widget in results.query(Static)),
            )

    active, blob = asyncio.run(main())
    assert active == "results"
    assert "Missing dependencies" in blob
    assert "PyTorch" in blob


def test_losing_the_live_stream_marks_the_target_disconnected():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._connected_facts = _target_facts(torch_version="")
            screen._connected_fields = {"kind": "ssh", "host": "h"}
            screen._refresh_dependency_button()
            await pilot.pause()
            shown_before = screen.query_one("#install-deps", Button).display
            screen._handle_metrics_loss(
                "The live connection to h was closed by the target."
            )
            await pilot.pause()
            panel = screen._gpu_panel()
            status = screen.query_one("#status", Static)
            return (
                shown_before,
                screen._connected_facts,
                screen._connected_fields,
                str(panel.render()),
                str(status.render()),
                screen.query_one("#install-deps", Button).display,
            )

    (
        shown_before,
        facts,
        fields,
        panel_text,
        status_text,
        shown_after,
    ) = asyncio.run(main())
    assert shown_before is True
    assert facts is None and fields is not None
    assert "CONNECTION LOST" in panel_text
    assert "connection lost" in status_text.lower()
    assert shown_after is False


def _cloud(app: GraftTUI) -> None:
    app.query_one("#location-select", Select).value = "cloud"


def _ssh_fields(**overrides):
    fields = {
        "kind": "ssh",
        "host": "gpu.example.com",
        "username": "root",
        "key_path": "/tmp/k",
        "key_passphrase": "",
        "port": "",
        "remote_dir": "/root/graft",
        "command_timeout_s": "900",
    }
    fields.update(overrides)
    return fields


def test_a_lost_stream_offers_a_reconnect_without_forgetting_the_target():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = _target_facts()
            screen._handle_metrics_loss("The live connection was closed.")
            await pilot.pause()
            return (
                screen._connected_fields,
                screen._connected_facts,
                screen.query_one("#reconnect-remote", Button).display,
                screen._reconnect_timer is not None,
                screen._reconnect_attempts,
            )

    fields, facts, button, scheduled, attempts = asyncio.run(main())
    assert fields is not None
    assert facts is None
    assert button is True
    assert scheduled is True
    assert attempts == 0


def test_a_failed_reconnect_keeps_the_settings_and_schedules_the_next_try():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._reconnect_attempts = 1
            screen._reconnect_done(
                _ssh_fields(),
                None,
                "Could not reach gpu.example.com: Connection timed out",
            )
            await pilot.pause()
            return (
                screen._connected_fields,
                screen._connected_facts,
                screen.query_one("#reconnect-remote", Button).display,
                screen._reconnect_timer is not None,
                screen.query_one("#status", Static).render(),
            )

    fields, facts, button, scheduled, status_text = asyncio.run(main())
    assert fields is not None
    assert facts is None
    assert button is True
    assert scheduled is True
    assert "retrying" in str(status_text).lower()


def test_an_auth_reconnect_failure_stops_retrying():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._reconnect_attempts = 1
            screen._reconnect_done(
                _ssh_fields(),
                None,
                "Authentication failed. Check the username, key path, and passphrase.",
            )
            await pilot.pause()
            return (
                screen._reconnect_timer,
                screen._reconnect_attempts,
                str(screen.query_one("#status", Static).render()),
            )

    timer, attempts, status_text = asyncio.run(main())
    assert timer is None
    assert attempts == 0
    assert "reconnect failed" in status_text.lower()


def test_reconnect_gives_up_after_the_budget():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._reconnect_attempts = len(screen.RECONNECT_BACKOFF_S)
            screen._schedule_reconnect("still unreachable")
            await pilot.pause()
            return (
                screen._reconnect_timer,
                str(screen.query_one("#status", Static).render()),
            )

    timer, status_text = asyncio.run(main())
    assert timer is None
    assert "gave up" in status_text.lower()


def test_a_successful_reconnect_restores_the_target(monkeypatch):
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            monkeypatch.setattr(screen, "_start_metrics", lambda *a, **k: None)
            screen._reconnect_attempts = 2
            screen._reconnect_done(_ssh_fields(), _target_facts(), "")
            await pilot.pause()
            panel = screen._gpu_panel()
            return (
                screen._connected_facts,
                screen._reconnect_attempts,
                screen.query_one("#reconnect-remote", Button).display,
                str(panel.render()),
                str(screen.query_one("#status", Static).render()),
            )

    facts, attempts, button, panel_text, status_text = asyncio.run(main())
    assert facts is not None
    assert attempts == 0
    assert button is False
    assert "CONNECTED" in panel_text
    assert "reconnected" in status_text.lower()


def test_reconnect_is_skipped_while_another_operation_runs():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._operation_active = True
            started = []
            screen.run_worker = lambda *a, **k: started.append(k)
            screen._reconnect_attempts = 0
            screen._start_reconnect(manual=True)
            return (
                started,
                screen._reconnect_attempts,
                screen._reconnect_timer is not None,
            )

    started, attempts, scheduled = asyncio.run(main())
    assert started == []
    assert attempts == 0
    assert scheduled is True


def test_manual_reconnect_resets_attempt_budget():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._reconnect_attempts = 3
            screen._reconnect_timer = screen.set_timer(5, lambda: None)
            started = []

            def fake_worker(*a, **k):
                started.append(k)

            screen.run_worker = fake_worker
            screen._start_reconnect(manual=True)
            await pilot.pause()
            return (
                started,
                screen._reconnect_attempts,
                screen._reconnect_timer,
            )

    started, attempts, timer = asyncio.run(main())
    assert started
    assert attempts == 1  # incremented
    assert timer is None


def test_reconnect_button_visibility_rules():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            btn = screen.query_one("#reconnect-remote", Button)
            # Local mode: never shown
            app.query_one("#location-select", Select).value = "local"
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._refresh_reconnect_button()
            await pilot.pause()
            local_shown = btn.display
            # Cloud but connected (facts present)
            app.query_one("#location-select", Select).value = "cloud"
            screen._connected_facts = _target_facts()
            screen._refresh_reconnect_button()
            await pilot.pause()
            cloud_connected = btn.display
            # Cloud but known but lost (facts None)
            screen._connected_facts = None
            screen._refresh_reconnect_button()
            await pilot.pause()
            cloud_lost = btn.display
            return local_shown, cloud_connected, cloud_lost

    local_shown, cloud_connected, cloud_lost = asyncio.run(main())
    assert local_shown is False
    assert cloud_connected is False
    assert cloud_lost is True


def test_switching_location_while_reconnecting_cancels_the_probe(monkeypatch):
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            _cloud(app)
            await pilot.pause()
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            # Start reconnect so there is a state
            screen._reconnect_attempts = 0
            screen._operation_active = False
            started = []
            screen.run_worker = lambda *a, **k: started.append("yes")
            screen._start_reconnect(manual=False)
            await pilot.pause()
            # Simulate in-flight result arriving after location changed
            app.query_one("#location-select", Select).value = "local"
            screen._reconnect_done(_ssh_fields(), _target_facts(), "")
            await pilot.pause()
            return screen._reconnect_attempts, screen._connected_facts

    attempts, facts = asyncio.run(main())
    assert attempts == 0
    assert facts is None





# --- app-level GPU dock ------------------------------------------------------
# The animation and live readings live in one dock fixed at the upper right of
# every screen, not inside the workspace tab, so switching tabs must not take
# the GPU state away.


def test_the_gpu_dock_is_app_level_and_survives_tab_switches():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            dock = app.query_one("#gpu-dock")
            panel = app.query_one(GPUStatusPanel)
            metrics = app.query_one(GPUMetrics)
            before = (dock.display, panel.is_mounted, metrics.is_mounted)
            # The panel is driven by the target screen through the app.
            screen = app.query_one(TargetScreen)
            screen._gpu_status("set_active", "AMD Instinct MI300X")
            app.query_one("#main-tabs").active = "results"
            await pilot.pause()
            after_switch = dock.display
            text = str(panel.render())
            return before, after_switch, "CONNECTED" in text, "MI300X" in text

    before, after_switch, connected, name = asyncio.run(main())
    assert before == (True, True, True)
    assert after_switch is True
    assert connected and name


def test_the_gpu_dock_shrinks_then_hides_as_the_terminal_narrows():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            dock = app.query_one("#gpu-dock")
            wide = set(dock.classes)
            await pilot.resize_terminal(100, 40)
            await pilot.pause()
            narrow = set(dock.classes)
            await pilot.resize_terminal(70, 30)
            await pilot.pause()
            tiny = set(dock.classes)
            await pilot.resize_terminal(50, 20)
            await pilot.pause()
            hidden = set(dock.classes)
            return wide, narrow, tiny, hidden

    wide, narrow, tiny, hidden = asyncio.run(main())
    assert "narrow" not in wide and "tiny" not in wide and "hidden" not in wide
    assert "narrow" in narrow
    assert "tiny" in tiny
    assert "hidden" in hidden


# --- right-click copy popup --------------------------------------------------
# Highlighting works through Textual's built-in selection; the popup turns a
# right-click into a copy that really reaches the clipboard.


def test_right_clicking_a_selection_opens_the_copy_popup_and_copies():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(ResultsScreen)
            screen.show_dependency_install(
                [{"key": "hipcc", "label": "hipcc (ROCm compiler)", "ok": True}],
                {"ok": True, "python_path": "/usr/bin/python3"},
            )
            app.query_one("#main-tabs").active = "results"
            await pilot.pause()
            card = next(iter(screen.query(".result-card")))
            await pilot.mouse_down(card, offset=(0, 0))
            await pilot.mouse_up(card, offset=(12, 0))
            await pilot.pause()
            selected = app.screen.get_selected_text()
            await pilot.click(card, button=3)
            await pilot.pause()
            popup = app.screen
            return selected, type(popup).__name__, app._clipboard

    selected, popup_name, clipboard_before = asyncio.run(main())
    # The drag selected the head of the card's text, and the right-click
    # opened the popup over it.
    assert selected and "hipcc" in selected
    assert popup_name == "CopyPopup"
    # A right-click alone must not write the clipboard; only the button does.
    assert clipboard_before in (None, "")


def test_the_copy_popup_button_puts_the_selection_on_the_clipboard():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(ResultsScreen)
            screen.show_error("Example", "hipcc missing on the target")
            app.query_one("#main-tabs").active = "results"
            await pilot.pause()
            card = next(iter(screen.query(".result-card")))
            await pilot.mouse_down(card, offset=(0, 0))
            await pilot.mouse_up(card, offset=(6, 0))
            await pilot.pause()
            selected = app.screen.get_selected_text()
            await pilot.click(card, button=3)
            await pilot.pause()
            await pilot.click("#copy-popup-copy")
            await pilot.pause()
            return selected, app._clipboard

    selected, clipboard = asyncio.run(main())
    assert selected
    # The button put exactly the highlighted text on the clipboard.
    assert clipboard == selected


def test_the_copy_popup_shows_the_selection_size_and_dismisses():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            app.push_screen(CopyPopup("alpha beta gamma"))
            await pilot.pause()
            popup = app.screen
            count = str(popup.query_one("#copy-popup-count").render())
            text = str(popup.query_one("#copy-popup-text").render())
            await pilot.press("escape")
            await pilot.pause()
            return count, text, type(app.screen).__name__

    count, text, after = asyncio.run(main())
    assert "16 characters" in count
    assert "alpha beta gamma" in text
    assert after != "CopyPopup"


def test_a_long_selection_is_clamped_in_the_preview_but_not_lost():
    from graft.tui.screens.copy_popup import preview_text

    long = "x" * 5000
    preview = preview_text(long)
    assert len(preview) < len(long)
    assert "3800 more characters" in preview
    assert preview_text("short") == "short"


# --- dependency overlay ------------------------------------------------------
# Connecting now checks the target immediately and offers the install with
# live progress, instead of hiding the gap until an operation fails.


def test_a_local_probe_opens_the_dependency_overlay():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(TargetScreen)
            screen._local_probe_done({"gpu": True, "device": "AMD"}, "")
            await pilot.pause()
            overlay = app.screen
            title = (
                str(overlay.query_one("#deps-title").render())
                if isinstance(overlay, DependencyOverlay)
                else ""
            )
            return type(overlay).__name__, title

    name, title = asyncio.run(main())
    assert name == "DependencyOverlay"
    assert title  # a real summary line, not an empty box


def test_a_remote_probe_opens_the_dependency_overlay_and_lists_missing_tools():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(TargetScreen)
            monkeypatched = _target_facts(
                torch_version="", tools={"hipcc": "/opt/rocm/bin/hipcc"}
            )
            screen._remote_done({"ok": True, "facts": monkeypatched}, "probe", "")
            await pilot.pause()
            overlay = app.screen
            if not isinstance(overlay, DependencyOverlay):
                return type(overlay).__name__, "", "", False
            rows = {
                dep.key: str(overlay.query_one(f"#dep-{dep.key}").render())
                for dep in overlay._deps
            }
            install = overlay.query_one("#deps-install", Button)
            return (
                type(overlay).__name__,
                rows.get("hipify-perl", ""),
                rows.get("hipcc", ""),
                install.disabled,
            )

    name, hipify_row, hipcc_row, disabled = asyncio.run(main())
    assert name == "DependencyOverlay"
    # hipify-perl is missing and installable; hipcc is present.
    assert "missing" in hipify_row and "rocm-hip-sdk" in hipify_row
    assert hipcc_row.startswith("ok")
    assert disabled is False


def test_a_probe_of_a_fully_equipped_target_does_not_open_the_overlay():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(TargetScreen)
            screen._remote_done(
                {"ok": True, "facts": _target_facts(torch_version="2.4.0+rocm6.2")},
                "probe",
                "",
            )
            await pilot.pause()
            return type(app.screen).__name__, app.query_one("#main-tabs").active

    name, active = asyncio.run(main())
    assert name != "DependencyOverlay"
    assert active == "results"


def test_verify_shows_its_results_instead_of_an_overlay():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(TargetScreen)
            # Even with a tool missing, verification is an explicit "show me
            # the result" action and must not be covered by the overlay.
            screen._remote_done(
                {
                    "ok": True,
                    "facts": _target_facts(torch_version=""),
                    "verdict": "verified",
                },
                "verify",
                "",
            )
            await pilot.pause()
            return type(app.screen).__name__, app.query_one("#main-tabs").active

    name, active = asyncio.run(main())
    assert name != "DependencyOverlay"
    assert active == "results"


def test_the_dependency_overlay_reports_live_install_progress():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            overlay = DependencyOverlay(
                _target_facts(tools={"hipcc": "/x"}, torch_version="")
            )
            app.push_screen(overlay)
            await pilot.pause()
            # Drive the same updates the worker posts, as the UI would see
            # them: a tool starts installing, then its outcome arrives.
            overlay._mark_installing("hipify-perl", "Installing hipify-perl…")
            await pilot.pause()
            during = str(overlay.query_one("#dep-hipify-perl").render())
            status_during = str(overlay.query_one("#deps-status").render())
            overlay._mark_result(
                {"key": "hipify-perl", "label": "hipify-perl", "ok": True}
            )
            await pilot.pause()
            after = str(overlay.query_one("#dep-hipify-perl").render())
            status_after = str(overlay.query_one("#deps-status").render())
            progress = overlay.query_one("#deps-progress", ProgressBar)
            return during, status_during, after, status_after, progress.progress

    during, status_during, after, status_after, progress = asyncio.run(main())
    assert "installing" in during
    assert "Installing hipify-perl" in status_during
    assert "ok" in after
    assert "Installed hipify-perl" in status_after
    assert progress >= 1


def test_a_failed_install_does_not_leave_rows_claiming_progress():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            overlay = DependencyOverlay(_target_facts(tools={}, torch_version=""))
            app.push_screen(overlay)
            await pilot.pause()
            overlay._mark_installing("hipify-perl", "Installing hipify-perl…")
            await pilot.pause()
            overlay._install_failed("the transport closed")
            await pilot.pause()
            row = str(overlay.query_one("#dep-hipify-perl").render())
            status = str(overlay.query_one("#deps-status").render())
            install = overlay.query_one("#deps-install", Button)
            return row, status, install.disabled

    row, status, disabled = asyncio.run(main())
    # The row falls back to "missing", never stuck on "installing…".
    assert "installing" not in row
    assert "missing" in row
    assert "install failed" in status
    # The install stays available: nothing was installed, so it can be retried.
    assert disabled is False


def test_the_dependency_overlay_does_not_install_without_the_button():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            overlay = DependencyOverlay(_target_facts(tools={}))
            app.push_screen(overlay)
            await pilot.pause()
            before = overlay._states
            # Escape closes an idle overlay without running anything.
            await pilot.press("escape")
            await pilot.pause()
            return before, type(app.screen).__name__

    before, after = asyncio.run(main())
    assert all(state != "installing" for state in before.values())
    assert after != "DependencyOverlay"


def test_an_install_refreshes_the_workspace_facts():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            screen = app.query_one(TargetScreen)
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = _target_facts(torch_version="")
            screen._show_dependency_overlay(
                screen._connected_facts,
                fields=screen._connected_fields,
                label="gpu.example.com",
            )
            await pilot.pause()
            overlay = app.screen
            refreshed = _target_facts(torch_version="2.4.0+rocm6.2")
            overlay._install_done([], refreshed)
            await pilot.pause()
            return (
                screen._connected_facts.get("torch_version"),
                screen.query_one("#install-deps", Button).display,
            )

    torch_version, button_shown = asyncio.run(main())
    assert torch_version == "2.4.0+rocm6.2"
    # With torch present there is nothing left to install, so the workspace
    # install button hides again.
    assert button_shown is False


# --- numbered source preview -------------------------------------------------


def test_the_source_preview_renders_numbered_lines_like_an_ide():
    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_source_preview(
                "m.py", "/tmp/m.py", "import torch\nx = 1\ny = 2\n"
            )
            await pilot.pause()
            blob = "\n".join(
                str(widget.render()) for widget in screen.query(Static)
            )
            return blob

    blob = asyncio.run(main())
    assert "1 │ import torch" in blob
    assert "2 │ x = 1" in blob
    assert "3 │ y = 2" in blob
    assert "3 line(s) · read-only preview" in blob


def test_the_numbered_gutter_grows_past_999_lines():
    from graft.tui.screens.results import numbered_source

    rendered = numbered_source("\n".join(f"line {i}" for i in range(1, 1201)))
    lines = rendered.splitlines()
    assert lines[0].startswith("   1 │ ")
    assert lines[-1].startswith("1200 │ ")


def test_the_source_comparison_panes_are_numbered_too():
    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_local(_summary(), "m.py", "/tmp/m.py", "print('original')\n")
            await pilot.pause()
            blob = "\n".join(
                str(widget.render()) for widget in screen.query(Static)
            )
            return blob

    blob = asyncio.run(main())
    assert "1 │ print('original')" in blob
    # The converted pane is numbered as well, not left as raw text.
    assert "1 │ print('converted')" in blob
