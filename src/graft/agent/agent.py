"""Stage 8 — The Gemma narration agent.

Handles the long tail that deterministic conversion misses: explanation,
narrative, suggested mappings, proposed extra test cases. Removing this
directory must leave a fully working product. That is the layering test,
and `is_available()` is what enforces it.

The model is Google Gemma, reached through AMD's OpenAI-compatible endpoint
(see ``gemma.py``). There is no agent framework in the loop: one configured
model, one system guardrail, one user prompt per call.

Four rules hold throughout this module.

1. The agent never decides correctness. It is handed measured results and
   narrates them. Every number it can emit comes from verify/, bench/, or
   coverage/; it is given those as text and cannot recompute them.
2. It never runs user code. runner.py is not reachable from here, so the
   model cannot ask the harness to execute anything.
3. Absence is a normal state, not an error. No endpoint configured means
   the narrative sections render as "not available" and the report still
   completes. A missing optional layer must not fail the product.
4. Failure is contained. A timeout, a 500, or a malformed response yields
   a string saying so. It never propagates into the report as a traceback,
   because a report that fails to render is worse than one with no prose.

Every function here is synchronous and importable without a network, so
tests can exercise the degradation paths directly.
"""

from __future__ import annotations

from graft import config

from . import prompts

NOT_AVAILABLE = "not available"

GUARDRAIL = prompts.SYSTEM


def endpoint_configured() -> bool:
    """True when both a model id and a base URL are present.

    Checked separately from agent construction so the UI can show why the
    narrative sections are empty instead of just showing empty.
    """
    return bool(config.LLM_MODEL and config.LLM_BASE_URL)


def unavailable_reason() -> str:
    """Why no model can be called, or "" when one can.

    Never returns an empty string while the model is unusable. An earlier
    version returned "" as soon as both endpoint variables were set, which
    meant a missing backend produced blank narrative sections under a
    heading claiming a language model wrote them.
    """
    if not config.LLM_MODEL:
        return f"{NOT_AVAILABLE}: no LLM_MODEL configured"
    if not config.LLM_BASE_URL:
        return f"{NOT_AVAILABLE}: no LLM_BASE_URL configured"
    if not config.AGENT_ENABLED:
        return f"{NOT_AVAILABLE}: agent layer disabled by AGENT_ENABLED"

    reason = _model_error
    if reason:
        return f"{NOT_AVAILABLE}: {reason}"
    return ""


_model_error: str | None = None


def build_model():
    """Construct the Gemma client, or None if it cannot be built.

    Gemma is served from an OpenAI-compatible endpoint, so a single client
    covers the default AMD token factory and any self-hosted replacement.
    The request timeout is set on the client, so a hung endpoint fails as a
    timeout rather than stalling the pipeline indefinitely.
    """
    global _model_error
    _model_error = None

    if not endpoint_configured():
        return None

    try:
        from .gemma import GemmaModel
    except Exception as exc:
        _model_error = f"gemma client unavailable: {exc}"
        return None

    return GemmaModel(
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        timeout=config.AGENT_TIMEOUT_S,
    )


class _Response:
    """The subset of a model response the narrative layer reads."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.text = content


class _Narrator:
    """One system guardrail plus one user prompt, sent to the model once."""

    def __init__(self, model) -> None:
        self.model = model

    def run(self, prompt: str) -> _Response:
        return _Response(self.model.complete(GUARDRAIL, prompt))


def build_agent():
    """Construct the narrator, or None when it cannot be constructed."""
    if not config.AGENT_ENABLED:
        return None

    model = build_model()
    if model is None:
        return None

    return _Narrator(model)


def is_available() -> bool:
    """Whether narrative generation can be attempted.

    The layering test: if this is False, the product still converts,
    verifies, benchmarks, and reports. Nothing in core/, verify/, bench/,
    or report/ imports this module.

    Requires a model that can actually be constructed, not merely an
    endpoint that is set, so the answer matches reality.
    """
    if not endpoint_configured():
        return False
    return build_model() is not None


def _run(agent, prompt: str, timeout: float | None = None) -> str:
    """One model call, fully contained.

    Returns prose on success and a plain explanation on any failure. The
    caller cannot distinguish "the model said something unhelpful" from
    "the model did not answer", which is intentional: both render as a
    sentence in a report rather than as an exception.

    ``timeout`` is a backstop for an agent object injected by a caller.
    The default path bounds the request on the model itself, which is the
    only place a hung server can actually be interrupted.
    """
    if agent is None:
        return unavailable_reason() or f"{NOT_AVAILABLE}: no model available"

    if timeout is not None:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FutureTimeout

        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                response = pool.submit(agent.run, prompt).result(timeout=timeout)
            except FutureTimeout:
                return f"{NOT_AVAILABLE}: timed out after {timeout:.0f}s"
            except Exception as exc:
                return f"{NOT_AVAILABLE}: {type(exc).__name__}: {exc}"
    else:
        try:
            response = agent.run(prompt)
        except Exception as exc:
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


def propose_test_cases(operation: str, agent=None) -> dict:
    """Propose additional parity inputs for the given operation.

    Returns both halves honestly labelled: deterministic shape candidates
    that always exist, and the model's own proposal when an endpoint is
    configured. Neither is mixed into the other, so a reader can tell
    which list a shape came from.
    """
    from .tools import existing_case_coverage, suggest_test_shapes

    current = existing_case_coverage()
    suggestions = suggest_test_shapes(operation)
    prompt = prompts.PROPOSE_TEST_CASE.format(
        case_count=current["case_count"],
        operation=operation,
        distributions=", ".join(current["distributions"]),
        shapes=", ".join(str(shape) for shape in current["shapes"]),
    )
    proposal = _run(agent or build_agent(), prompt)
    model_proposals = [] if proposal.startswith(NOT_AVAILABLE) else [proposal]
    return {
        "deterministic_suggestions": suggestions,
        "model_proposals": model_proposals,
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
