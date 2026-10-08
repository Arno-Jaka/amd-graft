"""The TUI results screen renders the evidence the worker actually returned.

These run headless through Textual's test pilot. They exist mainly to keep the
optional model sections honest: the AI review is computed by the worker, so it
must be visible here, and it must never be presented as deterministic evidence.
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import Static

from graft.tui.screens.results import ResultsScreen


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
