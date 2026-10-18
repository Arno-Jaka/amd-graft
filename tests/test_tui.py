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
from textual.widgets import Button, Input, Select, Static

from graft.target.dependencies import scan_dependencies
from graft.target.detect import TargetFacts
from graft.target.metrics import MetricsSnapshot
from graft.tui.app import GraftTUI
from graft.tui.screens.deps import DependencyOverlay
from graft.tui.screens.instructions import GUIDE_SECTIONS, InstructionsScreen
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
                "provider": "gemma",
                "model": "Gemma-4-31B-IT",
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




def test_remote_report_renders_the_target_the_report_and_the_source_diff():
    async def main() -> str:
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_remote_report(
                _summary(ai_review=None),
                "m.py",
                "/tmp/m.py",
                "print('original')\n",
                facts={
                    "gpu_name": "AMD Instinct MI300X VF",
                    "gpu_count": 1,
                    "rocm_version": "7.14.0",
                    "torch_hip": "7.14.60850",
                },
            )
            await pilot.pause()
            return "\n".join(str(w.render()) for w in screen.query(Static))

    text = asyncio.run(main())
    assert "Remote migration report" in text
    assert "REMOTE TARGET" in text
    assert "AMD Instinct MI300X VF" in text
    assert "SOURCE COMPARISON" in text
    assert "original · /tmp/m.py · 1 line(s)" in text
    assert "converted · 1 line(s)" in text
    assert "print('original')" not in text


def test_remote_report_includes_the_locally_generated_ai_review():
    async def main() -> str:
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_remote_report(
                _summary(
                    ai_review={
                        "available": True,
                        "provider": "gemma",
                        "model": "Gemma-4-31B-IT",
                        "recommendation": "engineer_review_required",
                        "summary": "Plausibly ported; verify at runtime.",
                        "comments": [],
                    }
                ),
                "m.py",
                "/tmp/m.py",
                "print('original')\n",
                facts={"gpu_name": "MI300X"},
            )
            await pilot.pause()
            return "\n".join(str(w.render()) for w in screen.query(Static))

    text = asyncio.run(main())
    assert "AI REVIEW" in text
    assert "Plausibly ported; verify at runtime." in text


def test_remote_verify_of_a_selected_file_shows_the_full_report():
    """End to end through the target screen: a verify result carrying a
    pipeline summary renders the report and source comparison, not the raw
    step transcript."""

    async def main() -> str:
        app = GraftTUI()
        async with app.run_test(size=(160, 60)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._verify_original_source = "print('original')\n"
            screen._verify_source_path = "/tmp/m.py"
            screen._remote_done(
                {
                    "ok": True,
                    "filename": "m.py",
                    "facts": _target_facts(),
                    "summary": _summary(ai_review=None),
                },
                "verify",
                "",
            )
            await pilot.pause()
            results = app.query_one("#results-view", ResultsScreen)
            return "\n".join(str(w.render()) for w in results.query(Static))

    blob = asyncio.run(main())
    assert "Remote migration report" in blob
    assert "SOURCE COMPARISON" in blob
    assert "original · /tmp/m.py · 1 line(s)" in blob


def test_the_source_comparison_opens_in_an_overlay_with_both_files():
    """The report keeps a summary; the two files live in a popup."""

    async def main() -> tuple[bool, str, str]:
        app = GraftTUI()
        async with app.run_test(size=(160, 60)) as pilot:
            results = app.query_one("#results-view", ResultsScreen)
            results.show_remote_report(
                _summary(ai_review=None),
                "m.py",
                "/tmp/m.py",
                "print('original')\n",
                facts=_target_facts(),
            )
            app.query_one("#main-tabs").active = "results"
            await pilot.pause()
            button = results.query_one("#compare-source", Button)
            before = button.disabled
            results.post_message(Button.Pressed(button))
            await pilot.pause()
            popup = app.screen
            blob = "\n".join(str(w.render()) for w in popup.query(Static))
            return before, type(popup).__name__, blob

    before, popup_name, blob = asyncio.run(main())
    assert before is False
    assert popup_name == "SourceComparePopup"
    assert "ORIGINAL" in blob
    assert "CONVERTED" in blob
    assert "1 │ print('original')" in blob
    assert "1 │ print('converted')" in blob


def test_a_passed_verification_arms_the_upload_button():
    async def main() -> tuple[bool, bool]:
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            button = screen.query_one("#run-converted", Button)
            before = button.disabled
            screen._remote_done(
                {
                    "ok": True,
                    "filename": "m.py",
                    "facts": _target_facts(),
                    "summary": _summary(ai_review=None),
                },
                "verify",
                "",
            )
            await pilot.pause()
            return before, button.disabled

    before, after = asyncio.run(main())
    assert before is True
    assert after is False


def test_a_verification_that_did_not_complete_leaves_the_upload_button_disabled():
    """Nothing unverified may be pushed to a GPU."""

    async def main() -> tuple[bool, str]:
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            button = screen.query_one("#run-converted", Button)
            screen._set_converted_file("print('converted')\n", "m.py", verified=True)
            await pilot.pause()
            armed = button.disabled
            screen._remote_done(
                {
                    "ok": False,
                    "filename": "m.py",
                    "facts": _target_facts(),
                    "error": "The migration did not complete on the target.",
                },
                "verify",
                "",
            )
            await pilot.pause()
            return armed, button.disabled

    armed, after = asyncio.run(main())
    assert armed is False
    assert after is True


def test_a_missing_torch_verification_shows_a_short_recommendation():
    """A target without PyTorch reads as a one-line fix, not a step dump."""

    async def main() -> str:
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._remote_done(
                {
                    "ok": False,
                    "filename": "m.py",
                    "facts": _target_facts(torch_version=""),
                    "missing_torch": True,
                    "error": (
                        "PyTorch was not found on the target host. Use the "
                        "Jupyter option, or install PyTorch on the host, to run "
                        "the migration."
                    ),
                    "steps": {
                        "verify_hip": {"returncode": 0},
                        "pipeline": {"returncode": 1},
                    },
                },
                "verify",
                "",
            )
            await pilot.pause()
            results = app.query_one("#results-view", ResultsScreen)
            return "\n".join(str(w.render()) for w in results.query(Static))

    blob = asyncio.run(main())
    assert "PyTorch not found on the target" in blob
    assert "Jupyter" in blob
    assert "steps:" not in blob
    assert "pipeline" not in blob


def test_verifying_the_bundled_example_does_not_arm_the_upload_button():
    """The fallback run never touched the operator's file."""

    async def main() -> bool:
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._remote_done(
                {
                    "ok": True,
                    "filename": "",
                    "facts": _target_facts(),
                    "summary": _summary(ai_review=None),
                },
                "verify",
                "",
            )
            await pilot.pause()
            return screen.query_one("#run-converted", Button).disabled

    assert asyncio.run(main()) is True


def test_a_local_migration_that_returns_a_converted_file_arms_the_upload_button():
    async def main() -> tuple[bool, bool]:
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            button = screen.query_one("#run-converted", Button)
            before = button.disabled
            screen._local_migration_done(
                _summary(ai_review=None), "m.py", "/tmp/m.py", "print('original')\n", ""
            )
            await pilot.pause()
            return before, button.disabled

    before, after = asyncio.run(main())
    assert before is True
    assert after is False





def test_results_offer_ai_conversion_when_manual_port_is_needed():
    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_local(
                _summary(
                    conversion={
                        "verdict": "manual_port",
                        "verdict_line": "manual port required",
                        "route": "cuda",
                        "backend": "hipify",
                        "verdict_basis": "blocker found",
                        "changes": [],
                        "findings": [
                            {"severity": "blocker", "symbol": "__shfl", "reason": "no mapping"}
                        ],
                    }
                ),
                "m.py",
                "/tmp/m.py",
                "print('original')\n",
            )
            await pilot.pause()
            text = "\n".join(str(w.render()) for w in screen.query(Static))
            return len(screen.query("#ai-convert")), text

    count, text = asyncio.run(main())
    assert count == 1
    assert "AI-ASSISTED PORT" in text


def test_results_do_not_offer_ai_conversion_on_a_clean_port():
    text = _render(
        _summary(
            conversion={
                "verdict": "production_ready",
                "verdict_line": "production ready",
                "route": "pytorch",
                "backend": "torch_rewriter",
                "verdict_basis": "clean",
                "changes": [],
                "findings": [],
            },
            parity={"measured": True, "passed": 4, "total": 4, "all_passed": True},
        )
    )
    assert "AI-ASSISTED PORT" not in text


def test_ai_conversion_result_mounts_preview_and_run_buttons():
    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            screen = app.query_one(ResultsScreen)
            screen.show_local(
                _summary(), "m.py", "/tmp/m.py", "print('original')\n",
                ai_options={"base_url": "http://h/v1", "model": "gemma-4", "api_key": ""},
            )
            await pilot.pause()
            screen._ai_conversion_done(
                {
                    "available": True,
                    "provider": "gemma",
                    "model": "gemma-4",
                    "filename": "m_gemma.py",
                    "summary": "Finished the port.",
                    "code": "print('converted')\n",
                    "comments": [
                        {
                            "line": 1,
                            "severity": "review",
                            "title": "Check stream",
                            "comment": "Confirm the stream.",
                        }
                    ],
                }
            )
            await pilot.pause()
            return (
                len(screen.query("#ai-preview")),
                len(screen.query("#ai-run")),
                "\n".join(str(w.render()) for w in screen.query(Static)),
            )

    preview, run, text = asyncio.run(main())
    assert preview == 1
    assert run == 1
    assert "AI version ready" in text


def test_ai_popup_renders_the_code_and_the_models_notes():
    from graft.tui.screens.ai_popup import AIConversionPopup

    async def main():
        app = _Host()
        async with app.run_test() as pilot:
            app.push_screen(
                AIConversionPopup(
                    "m_gemma.py",
                    "print('hello')\n",
                    [
                        {
                            "line": 1,
                            "severity": "review",
                            "title": "Check stream",
                            "comment": "Confirm the stream matches.",
                        }
                    ],
                )
            )
            await pilot.pause()
            popup = app.screen
            return "\n".join(str(w.render()) for w in popup.query(Static))

    text = asyncio.run(main())
    assert "AI-CONVERTED VERSION" in text
    assert "print('hello')" in text
    assert "Check stream" in text
    assert "generated by a language model" in text




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
    assert "Install it on the target" in blob


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


def test_a_successful_remote_verification_does_not_label_its_steps_failed():
    """The remote script reports exit codes, not an "ok" flag.

    Reading only an "ok" key made every step of a passed verification show as
    "failed", which is how a green run came back as "verify_hip: failed".
    """

    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._remote_done(
                {
                    "ok": True,
                    "facts": _target_facts(),
                    "steps": {
                        "verify_hip": {"returncode": 0, "output": "HIP VERIFICATION PASSED"},
                        "pipeline": {"returncode": 0, "output": "{}"},
                    },
                },
                "verify",
                "",
            )
            await pilot.pause()
            results = app.query_one("#results-view", ResultsScreen)
            return "\n".join(str(widget.render()) for widget in results.query(Static))

    blob = asyncio.run(main())
    assert "verify_hip: ok" in blob
    assert "pipeline: ok" in blob
    assert "failed" not in blob


def test_losing_the_live_stream_marks_the_target_disconnected():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen._connected_facts = _target_facts(torch_version="")
            screen._connected_fields = {"kind": "ssh", "host": "h"}
            await pilot.pause()
            screen._handle_metrics_loss(
                "The live connection to h was closed by the target."
            )
            await pilot.pause()
            panel = screen._gpu_panel()
            status = screen.query_one("#status", Static)
            return (
                screen._connected_facts,
                screen._connected_fields,
                str(panel.render()),
                str(status.render()),
            )

    facts, fields, panel_text, status_text = asyncio.run(main())
    assert facts is None and fields is not None
    assert "CONNECTION LOST" in panel_text
    assert "connection lost" in status_text.lower()


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
    assert attempts == 1
    assert timer is None


def test_reconnect_button_visibility_rules():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            btn = screen.query_one("#reconnect-remote", Button)
            app.query_one("#location-select", Select).value = "local"
            screen._connected_fields = _ssh_fields()
            screen._connected_facts = None
            screen._refresh_reconnect_button()
            await pilot.pause()
            local_shown = btn.display
            app.query_one("#location-select", Select).value = "cloud"
            screen._connected_facts = _target_facts()
            screen._refresh_reconnect_button()
            await pilot.pause()
            cloud_connected = btn.display
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
            screen._reconnect_attempts = 0
            screen._operation_active = False
            started = []
            screen.run_worker = lambda *a, **k: started.append("yes")
            screen._start_reconnect(manual=False)
            await pilot.pause()
            app.query_one("#location-select", Select).value = "local"
            screen._reconnect_done(_ssh_fields(), _target_facts(), "")
            await pilot.pause()
            return screen._reconnect_attempts, screen._connected_facts

    attempts, facts = asyncio.run(main())
    assert attempts == 0
    assert facts is None







def test_detect_token_is_disabled_until_a_valid_server_url_is_entered():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            field = screen.query_one("#jupyter-host", Input)
            button = screen.query_one("#detect-token", Button)

            field.value = ""
            screen._refresh_detect_token_button()
            await pilot.pause()
            empty = button.disabled

            field.value = "gpu.example.com"
            screen._refresh_detect_token_button()
            await pilot.pause()
            bare = button.disabled

            field.value = "https://gpu.example.com/lab"
            screen._refresh_detect_token_button()
            await pilot.pause()
            valid = button.disabled
            return empty, bare, valid

    empty, bare, valid = asyncio.run(main())
    assert empty is True
    assert bare is True
    assert valid is False


def test_detect_token_without_a_url_reports_the_missing_server():
    async def main() -> str:
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            screen = app.query_one(TargetScreen)
            await pilot.pause()
            screen.query_one("#jupyter-host", Input).value = ""
            screen._detect_jupyter_token()
            await pilot.pause()
            return str(screen.query_one("#status", Static).render())

    status = asyncio.run(main())
    assert "Jupyter Server URL" in status




def test_the_gpu_dock_is_app_level_and_survives_tab_switches():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            dock = app.query_one("#gpu-dock")
            panel = app.query_one(GPUStatusPanel)
            metrics = app.query_one(GPUMetrics)
            before = (dock.display, panel.is_mounted, metrics.is_mounted)
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
    assert title


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
                return type(overlay).__name__, "", ""
            rows = {
                dep.key: str(overlay.query_one(f"#dep-{dep.key}").render())
                for dep in overlay._deps
            }
            return (
                type(overlay).__name__,
                rows.get("hipify-perl", ""),
                rows.get("hipcc", ""),
            )

    name, hipify_row, hipcc_row = asyncio.run(main())
    assert name == "DependencyOverlay"
    assert "missing" in hipify_row and "rocm-hip-sdk" in hipify_row
    assert hipcc_row.startswith("ok")


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
            screen._show_source_comparison()
            await pilot.pause()
            popup = app.screen
            return type(popup).__name__, "\n".join(
                str(widget.render()) for widget in popup.query(Static)
            )

    popup_name, blob = asyncio.run(main())
    assert popup_name == "SourceComparePopup"
    assert "1 │ print('original')" in blob
    assert "1 │ print('converted')" in blob


def test_the_instructions_tab_shows_the_workflow_in_order():
    async def main():
        app = GraftTUI()
        async with app.run_test(size=(140, 48)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs")
            app.action_show_instructions()
            await pilot.pause()
            screen = app.query_one(InstructionsScreen)
            blob = "\n".join(str(widget.render()) for widget in screen.query(Static))
            return tabs.active, blob

    active, blob = asyncio.run(main())
    assert active == "instructions"
    assert "Convert & review" in blob
    assert "Verify on GPU" in blob
    assert "Upload & run converted file" in blob
    assert "Results tab" in blob


def test_the_guide_names_the_buttons_it_tells_you_to_press():
    """The wording must stay honest when the buttons change.

    A guide that points at a renamed control is worse than no guide, so the
    controls that carry the workflow are checked against what the workspace
    actually renders.
    """
    steps = "\n".join(step for _, items in GUIDE_SECTIONS for step in items)
    for label in (
        "Convert & review",
        "Detect target",
        "Detect token",
        "Verify on GPU",
        "Upload & run converted file",
        "Save converted file",
    ):
        assert label in steps, label
