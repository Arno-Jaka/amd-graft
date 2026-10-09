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
from textual.widgets import Static

from graft.target.metrics import MetricsSnapshot
from graft.tui.screens.results import ResultsScreen
from graft.tui.screens.tips import TipsScreen, classify_connection_failure, tips_for
from graft.tui.widgets import GPUMetrics


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

