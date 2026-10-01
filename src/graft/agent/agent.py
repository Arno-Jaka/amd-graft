"""Stage 8 — The Agno agent.

Handles the long tail that deterministic conversion misses: explanation,
narrative, suggested mappings, proposed extra test cases. Removing this
directory must leave a fully working product. That is the layering test,
and `is_available()` is what enforces it.

Four rules hold throughout this module.

1. The agent never decides correctness. It is handed measured results and
   narrates them. Every number it can emit comes from verify/, bench/, or
   coverage/, all of which it reaches through read-only tools.
2. It never runs user code. runner.py is not exposed as a tool, so the
   model cannot ask the harness to execute anything.
3. Absence is a normal state, not an error. No endpoint configured means
   the narrative sections render as "not available" and the report still
   completes. A missing optional layer must not fail the product.
4. Failure is contained. A timeout, a 500, or a malformed response yields
   a string saying so. It never propagates into the report as a traceback,
   because a report that fails to render is worse than one with no prose.

Every function here is synchronous and importable without agno or a
network, so tests can exercise the degradation paths directly.
"""

from __future__ import annotations

import os

from graft import config

from . import prompts
from .tools import build_tools

NOT_AVAILABLE = "not available"

# Applied to every prompt. The model proposes; the deterministic layers
# decide. Without this the model will happily invent a correctness claim.
GUARDRAIL = prompts.SYSTEM


def endpoint_configured() -> bool:
    """True when both a model id and a base URL are present.

    Checked separately from agent construction so the UI can show why the
    narrative sections are empty instead of just showing empty.
    """
    return bool(config.LLM_MODEL and config.LLM_BASE_URL)


def unavailable_reason() -> str:
    if not config.LLM_MODEL:
        return f"{NOT_AVAILABLE}: no LLM_MODEL configured"
    if not config.LLM_BASE_URL:
        return f"{NOT_AVAILABLE}: no LLM_BASE_URL configured"
    return ""


def build_model():
    """Construct the Agno model, or None if it cannot be built.

    vLLM is the target: an OpenAI-compatible server on the AMD instance,
    which is what makes a self-hosted demo possible without an API key.
    """
    if not endpoint_configured():
        return None

    try:
        from agno.models.vllm import VLLM
    except ImportError:
        return None

    api_key = (
        os.environ.get(config.LLM_API_KEY_ENV, "") or config.LLM_PLACEHOLDER_API_KEY
    )
    return VLLM(id=config.LLM_MODEL, base_url=config.LLM_BASE_URL, api_key=api_key)


def build_agent():
    """Construct the agent, or None when it cannot be constructed."""
    if not config.AGENT_ENABLED:
        return None

    model = build_model()
    if model is None:
        return None

    try:
        from agno.agent import Agent
    except ImportError:
        return None

    return Agent(
        model=model,
        name="graft-narrator",
        instructions=GUARDRAIL,
        tools=build_tools(),
        markdown=False,
        telemetry=False,
        retries=2,
    )


def is_available() -> bool:
    """Whether narrative generation can be attempted.

    The layering test: if this is False, the product still converts,
    verifies, benchmarks, and reports. Nothing in core/, verify/, bench/,
    or report/ imports this module.
    """
    if not endpoint_configured():
        return False
    try:
        import agno  # noqa: F401
    except ImportError:
        return False
    return True


def _run(agent, prompt: str, timeout: float | None = None) -> str:
    """One model call, fully contained.

    Returns prose on success and a plain explanation on any failure. The
    caller cannot distinguish "the model said something unhelpful" from
    "the model did not answer", which is intentional: both render as a
    sentence in a report rather than as an exception.
    """
    if agent is None:
        return unavailable_reason()

    try:
        response = agent.run(prompt)
    except Exception as exc:  # every failure becomes text, never propagates
        return f"{NOT_AVAILABLE}: {type(exc).__name__}: {exc}"

    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return f"{NOT_AVAILABLE}: model returned no content"


def narrate(summary: dict, agent=None) -> str:
    """Five-sentence summary of a migration.

    ``summary`` carries the measured values. The prompt states them
    verbatim so the model reports rather than infers.
    """
    prompt = prompts.NARRATE.format(
        backend=summary.get("backend", NOT_AVAILABLE),
        gpu_present=summary.get("gpu_present", "unknown"),
        conversion_verdict=summary.get("conversion_verdict", NOT_AVAILABLE),
        parity_summary=summary.get("parity_summary", NOT_AVAILABLE),
        execution_summary=summary.get("execution_summary", NOT_AVAILABLE),
        benchmark_summary=summary.get("benchmark_summary", NOT_AVAILABLE),
    )
    return _run(agent or build_agent(), prompt)


def explain_finding(finding: dict, gpu_name: str = "AMD GPU", agent=None) -> str:
    """Explain one coverage finding and what to do about it."""
    severity = finding.get("severity", "review")
    prompt = prompts.EXPLAIN_FINDING.format(
        symbol=finding.get("symbol", "?"),
        line=finding.get("line", "?"),
        severity=severity,
        severity_meaning={
            "blocker": "no mechanical translation exists",
            "review": "translation exists but verify the semantics",
            "info": "informational",
        }.get(severity, severity),
        reason=finding.get("reason", finding.get("note", "")),
        remediation=finding.get("remediation", "none suggested"),
        gpu_name=gpu_name,
    )
    return _run(agent or build_agent(), prompt)


def explain_change(change: dict, index: int, total: int, agent=None) -> str:
    """Explain one applied rewrite."""
    prompt = prompts.EXPLAIN_CHANGE.format(
        index=index,
        total=total,
        symbol=change.get("symbol", "?"),
        line=change.get("line", "?"),
        reason=change.get("reason", ""),
        before=change.get("before", "(not captured)"),
        after=change.get("after", "(not captured)"),
    )
    return _run(agent or build_agent(), prompt)


def summarize_parity(summary: dict, device: str = "the test device", agent=None) -> str:
    """Plain-language parity outcome.

    The numbers are interpolated into the prompt rather than left for the
    model to recall, so a hallucinated pass count is not reachable.
    """
    failures = summary.get("failures") or []
    rendered = (
        "\n".join(f"- {name}: {detail}" for name, detail in failures)
        if failures
        else "none"
    )
    prompt = prompts.SUMMARIZE_PARITY.format(
        device=device,
        passed=summary.get("passed", "?"),
        total=summary.get("total", "?"),
        all_passed=summary.get("all_passed", "unknown"),
        worst_case=summary.get("worst_case", "n/a"),
        max_abs_error=summary.get("max_abs_error", "n/a"),
        tolerance_note=summary.get("tolerance_note", "per-dtype rtol/atol"),
        failures=rendered,
    )
    return _run(agent or build_agent(), prompt)


def summarize_benchmark(rows: list[dict], noise_band: float, agent=None) -> str:
    """Explain benchmark numbers and what they do not show."""

    def _headline(row: dict) -> str:
        # Rows come straight from bench.compare_* and are keyed by ratio or
        # speedup, never "result". Reading a missing key would print
        # "not available" next to a real number already in the row.
        for key in ("ratio", "speedup"):
            if row.get(key) is not None:
                return f"{key} {row[key]}"
        return NOT_AVAILABLE

    rendered = (
        "\n".join(
            f"- {row.get('label', 'unnamed')}: {_headline(row)}"
            f" ({row.get('verdict', NOT_AVAILABLE)})"
            for row in rows
        )
        or "- no benchmarks recorded"
    )
    prompt = prompts.SUMMARIZE_BENCHMARK.format(
        rows=rendered, noise_pct=round((noise_band - 1) * 100)
    )
    return _run(agent or build_agent(), prompt)


def propose_remediation(unmapped: list[dict], agent=None) -> str:
    """Propose a specific fix per unmapped item."""
    rendered = (
        "\n".join(
            f"- {item.get('symbol', '?')} (line {item.get('line', '?')}): "
            f"{item.get('reason') or item.get('note', '')}"
            for item in unmapped
        )
        or "- none"
    )
    return _run(
        agent or build_agent(), prompts.PROPOSE_REMEDIATION.format(unmapped=rendered)
    )


def propose_test_cases(operation: str, agent=None) -> str:
    """Propose additional parity inputs for the given operation."""
    from .tools import existing_case_coverage, suggest_test_shapes

    current = existing_case_coverage()
    suggestions = suggest_test_shapes(operation)
    prompt = prompts.PROPOSE_TEST_CASE.format(
        case_count=current["case_count"],
        operation=operation,
        distributions=", ".join(current["distributions"]),
        shapes=", ".join(str(shape) for shape in current["shapes"]),
    )
    _run(agent or build_agent(), prompt)
    return {
        "deterministic_suggestions": suggestions,
        "note": (
            "Shape suggestions are generated deterministically, not by the "
            "model, so they exist even without an LLM endpoint."
        ),
    }


def narrative_sections(report_inputs) -> dict[str, str]:
    """Every narrative section for a report, or explicit unavailability.

    Built defensively: one failing section must not stop the others, and
    the whole set must not stop the report.
    """
    if not is_available():
        reason = unavailable_reason()
        return {
            "narration": reason,
            "parity_explainer": reason,
            "benchmark_explainer": reason,
            "remediation": reason,
        }

    sections: dict[str, str] = {}
    telemetry = report_inputs.telemetry_snapshot or {}
    parity_summary = report_inputs.parity or {}
    bench_rows = report_inputs.benchmarks or []

    try:
        sections["narration"] = narrate(
            {
                "backend": telemetry.get("backend", NOT_AVAILABLE),
                "gpu_present": telemetry.get("gpu_available", False),
                "conversion_verdict": str(
                    report_inputs.migration.coverage.verdict
                    if report_inputs.migration
                    else NOT_AVAILABLE
                ),
                "parity_summary": (
                    f"{parity_summary.get('passed', '?')}/{parity_summary.get('total', '?')} passed"
                    if parity_summary
                    else NOT_AVAILABLE
                ),
                "execution_summary": (
                    report_inputs.execution.summary()
                    if report_inputs.execution
                    else NOT_AVAILABLE
                ),
                "benchmark_summary": (
                    "; ".join(row.get("verdict", NOT_AVAILABLE) for row in bench_rows)
                    or NOT_AVAILABLE
                ),
            }
        )
    except Exception as exc:
        sections["narration"] = f"{NOT_AVAILABLE}: {type(exc).__name__}: {exc}"

    try:
        sections["parity_explainer"] = summarize_parity(
            parity_summary, telemetry.get("backend", "the test device")
        )
    except Exception as exc:
        sections["parity_explainer"] = f"{NOT_AVAILABLE}: {type(exc).__name__}: {exc}"

    try:
        sections["benchmark_explainer"] = summarize_benchmark(
            bench_rows, config.BENCH_NOISE_BAND
        )
    except Exception as exc:
        sections["benchmark_explainer"] = (
            f"{NOT_AVAILABLE}: {type(exc).__name__}: {exc}"
        )

    unmapped = (
        report_inputs.migration.coverage.unmapped if report_inputs.migration else []
    )
    try:
        sections["remediation"] = propose_remediation(unmapped)
    except Exception as exc:
        sections["remediation"] = f"{NOT_AVAILABLE}: {type(exc).__name__}: {exc}"

    return sections
