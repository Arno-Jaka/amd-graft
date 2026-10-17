"""Flatten a PipelineResult into JSON-native data for the web UI.

Why this exists
---------------
`pipeline.run()` imports the file it is analysing, so it cannot be called
inside a long-lived web server process. The UI therefore runs it in a
disposable child (see `worker.py`) and needs the results back as something
that survives a process boundary.

That turns out to be easy: every value the UI actually renders is already a
scalar, a string, or a dict of strings. This module flattens a
PipelineResult into exactly those, and nothing else. There is deliberately
no attempt to round-trip dataclasses or enum members; a summary is a
projection, not a serialization format, and a field that the UI cannot
render is not worth transporting.

Being a pure function of a PipelineResult, `summarize` is testable without
a subprocess and without a browser.
"""

from __future__ import annotations

from typing import Any

from graft import pipeline
from graft.report import report


def _text(value: Any, fallback: str = report.NOT_MEASURED) -> str:
    """Render any value as display text without inventing a measurement."""
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _number(value: Any) -> float | None:
    """Pass a number through, or None so the UI can say "not measured"."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def summarize(result: pipeline.PipelineResult) -> dict[str, Any]:
    """Project a PipelineResult into JSON-native data.

    Every section is always present as a key, even when the pipeline skipped
    it, so the UI can tell "not measured" apart from "key missing".
    """
    inputs = result.inputs

    return {
        "report_text": result.report_text,
        "skipped": list(result.skipped),
        "environment": _environment(inputs.telemetry_snapshot),
        "conversion": _conversion(inputs),
        "execution": _execution(inputs.execution),
        "parity": _parity(inputs.parity),
        "benchmarks": _benchmarks(inputs.benchmarks),
        "narrative": _narrative(inputs.narratives),
        # The ported source itself, so a UI can show the original next to the
        # conversion. Empty (not missing) when the pipeline skipped migration.
        "converted_source": _converted_source(inputs),
    }


def _environment(snapshot: dict | None) -> dict[str, Any]:
    snapshot = snapshot or {}
    total_vram = _number(snapshot.get("total_vram_mb"))
    peak_vram = _number(snapshot.get("peak_vram_mb"))
    return {
        "backend": _text(snapshot.get("backend")),
        "gpu_name": _text(snapshot.get("gpu_name"), "none detected"),
        "gpu_available": bool(snapshot.get("gpu_available")),
        "total_vram_mb": total_vram,
        "peak_vram_mb": peak_vram,
    }


def _converted_source(inputs) -> str:
    migration = inputs.migration
    if migration is None:
        return ""
    return str(migration.ported_source or "")


def _conversion(inputs) -> dict[str, Any]:
    migration = inputs.migration
    if migration is None:
        return {"measured": False}

    verdict = migration.coverage.verdict
    parity = inputs.parity

    return {
        "measured": True,
        "verdict": str(verdict),
        # The basis strings are computed by the report module rather than
        # reworded here, so the page and the Markdown report cannot drift.
        "verdict_line": report._verdict_line(verdict),
        "verdict_basis": report._verdict_basis(migration, parity),
        "route": _text(getattr(migration.route, "lang", None), "unknown"),
        "backend": _text(migration.backend),
        "changes": [
            {
                "symbol": _text(change.get("symbol"), "?"),
                "reason": _text(change.get("reason"), ""),
                "line": change.get("line"),
            }
            for change in migration.changes
        ],
        # Read from the structured form (unmapped), not coverage.findings:
        # that field holds human-readable strings, not Finding objects.
        "findings": report.coverage_findings(migration),
    }


def _execution(execution) -> dict[str, Any]:
    if execution is None:
        return {"measured": False}

    return {
        "measured": True,
        "ok": bool(execution.ok),
        "case_count": execution.case_count or 0,
        "timed_out": bool(execution.timed_out),
        "exit_code": execution.exit_code,
        "error_type": execution.error_type,
        "error_message": execution.error_message,
    }


def _parity(summary: dict | None) -> dict[str, Any]:
    if not summary:
        return {"measured": False}

    return {
        "measured": True,
        "passed": int(summary.get("passed", 0)),
        "total": int(summary.get("total", 0)),
        "all_passed": bool(summary.get("all_passed")),
        "worst_case": _text(summary.get("worst_case"), "n/a"),
        "max_abs_error": _number(summary.get("max_abs_error")),
    }


def _benchmarks(rows: list[dict] | None) -> list[dict[str, Any]]:
    out = []
    for row in rows or []:
        out.append(
            {
                "label": _text(row.get("label"), "unnamed"),
                "ratio": _number(row.get("ratio")),
                "speedup": _number(row.get("speedup")),
                "verdict": _text(row.get("verdict"), "n/a"),
            }
        )
    return out


# Key order is the order the report renders them in.
NARRATIVE_TITLES = {
    "narration": "Summary",
    "parity_explainer": "Parity, explained",
    "benchmark_explainer": "Benchmarks, explained",
    "remediation": "Suggested remediation",
}


def _narrative(narratives: dict | None) -> dict[str, Any]:
    """Split narratives into present prose and reasons it is absent.

    An agent that is disabled or failed returns a reason string starting with
    UNAVAILABLE_MARKER. Treating that as prose would print "not available" as
    if it were an explanation, so it is filtered out here using the same
    rule the report itself uses.
    """
    prose: dict[str, str] = {}
    reasons: list[str] = []

    for key, value in (narratives or {}).items():
        if key not in NARRATIVE_TITLES or not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        if text.lower().startswith(report.UNAVAILABLE_MARKER):
            reasons.append(text)
        else:
            prose[key] = text

    return {"prose": prose, "reasons": reasons}
