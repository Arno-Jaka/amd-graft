"""Stage 4 — Markdown report generation.

Assembles change log, unmapped items, parity results, benchmarks, and a
readiness verdict into one exportable document. Demoable with no AI in
the loop: every number below comes from a stage that was actually run.

Two rules keep the report from lying:

1. A missing measurement is printed as "not measured", never as a pass.
   Silence is not evidence.
2. The verdict is derived from results that exist. Anything unmeasured
   downgrades readiness rather than being quietly omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graft import config
from graft.bench import telemetry
from graft.core import runner
from graft.core.converter import MigrationResult, Verdict

NOT_MEASURED = "not measured"

# Leading marker the agent layer uses for anything it could not obtain.
# Matched as a literal because report/ must not import agent/.
UNAVAILABLE_MARKER = "not available"


@dataclass
class ReportInputs:
    """Everything the report may cite. Absent stages stay None.

    Deliberately loose typing: the report consumes dicts produced by
    several stages and must render a partial run without crashing.
    """

    migration: MigrationResult | None = None
    parity: dict | None = None
    parity_results: list | None = None
    benchmarks: list[dict] = field(default_factory=list)
    telemetry_snapshot: dict | None = None
    execution: runner.ExecutionResult | None = None
    narratives: dict[str, str] = field(default_factory=dict)
    skipped_notes: list[str] = field(default_factory=list)


def _fmt(value, spec="{:.3e}", missing=NOT_MEASURED) -> str:
    if value is None:
        return missing
    if isinstance(value, float):
        return spec.format(value)
    return str(value)


def _verdict_line(verdict) -> str:
    labels = {
        Verdict.PRODUCTION_READY: "PRODUCTION READY",
        Verdict.NEEDS_REVIEW: "NEEDS REVIEW",
        Verdict.MANUAL_PORT: "MANUAL PORT REQUIRED",
        Verdict.UNSUPPORTED: "UNSUPPORTED",
    }
    return labels.get(verdict, str(verdict))


def _verdict_basis(migration: MigrationResult, parity: dict | None) -> str:
    """What evidence the conversion verdict actually rests on.

    A verdict is only as strong as the thing behind it, and for the two
    routes those things are not comparable. The PyTorch route can be
    backed by parity cases that were compiled and executed. The CUDA C++
    route cannot: Graft rewrites the text and nobody has built it, so
    "production ready" there means "no unmapped constructs found", which
    is a scan rather than a proof.

    Printed next to the verdict so the word carries its evidence. Without
    this, a skim of the Conversion section suggests a stronger result than
    the run performed, even though the Readiness section downgrades it
    correctly further down.
    """
    route = migration.route.lang if migration.route else ""

    if route == "cuda_cpp":
        return "text analysis only - not compiled, not executed"

    if parity and parity.get("total"):
        passed, total = parity.get("passed", 0), parity["total"]
        if passed == total:
            return f"AST analysis, {passed}/{total} parity cases executed"
        return f"AST analysis, {passed}/{total} parity cases passed"

    return "AST analysis only - not executed"


def _conversion_section(
    migration: MigrationResult | None, parity: dict | None = None
) -> list[str]:
    if migration is None:
        return ["## Conversion", "", f"{NOT_MEASURED} (no conversion run).", ""]

    lines = [
        "## Conversion",
        "",
        f"- Verdict: {_verdict_line(migration.coverage.verdict)}"
        f" ({_verdict_basis(migration, parity)})",
        f"- Route: {migration.route.lang if migration.route else 'unknown'}",
        f"- Backend: {migration.backend}",
        f"- Changes applied: {len(migration.changes)}",
    ]
    for change in migration.changes:
        symbol = change.get("symbol", "?")
        reason = change.get("reason") or change.get("note") or ""
        line = change.get("line")
        where = f" (line {line})" if line else ""
        text = f"- `{symbol}`{where}"
        lines.append(f"{text}: {reason}" if reason else text)

    items = migration.coverage.unmapped
    if items:
        lines += ["", "### Coverage findings", ""]
        # Grouped by severity so the reader sees how much work is actually
        # left: a blocker means a rewrite, a review means verification.
        for level, heading in (
            ("blocker", "Blockers — no mechanical translation exists"),
            ("review", "Needs review — translation exists, verify semantics"),
            ("info", "Notes — worth knowing, changes no verdict"),
        ):
            group = [
                item
                for item in items
                if isinstance(item, dict)
                # An item with no severity predates the severity model.
                # Defaulted to review rather than dropped: silently hiding
                # a finding because it lacked metadata is the exact
                # failure mode this section exists to prevent.
                and item.get("severity", "review") == level
            ]
            if not group:
                continue
            lines += [f"#### {heading}", ""]
            for item in group:
                symbol = item.get("symbol", "?")
                line = item.get("line")
                where = f" (line {line})" if line else ""
                reason = item.get("reason") or item.get("note") or ""
                lines.append(f"- `{symbol}`{where}: {reason}")
                fix = item.get("remediation")
                if fix:
                    lines.append(f"  - fix: {fix}")
            lines.append("")
    else:
        lines += ["", "### Coverage findings", "", "None."]

    if migration.notes:
        lines += ["", "### Notes", ""]
        lines += [f"- {note}" for note in migration.notes]
    return [*lines, ""]


def _parity_section(inputs: ReportInputs) -> list[str]:
    summary = inputs.parity
    lines = ["## Parity", ""]
    if summary is None:
        lines += [f"{NOT_MEASURED} (parity harness not run).", ""]
    else:
        passed = summary.get("passed", 0)
        total = summary.get("total", 0)
        state = "PASS" if summary.get("all_passed") else "FAIL"
        lines += [
            f"**{state}** — {passed}/{total} cases within tolerance",
            "",
            f"- Worst case: {_fmt(summary.get('worst_case'), '{:.0f}', 'n/a')}"
            if summary.get("worst_case")
            else "- Worst case: n/a",
            f"- Max absolute error: {_fmt(summary.get('max_abs_error'))}",
        ]
        failures = [r for r in (inputs.parity_results or []) if not r.passed]
        if failures:
            lines += ["", "### Failures", ""]
            for result in failures:
                detail = result.error or "tolerance exceeded"
                lines.append(f"- `{result.name}` — {detail}")
        lines.append("")

    lines += [
        "Tolerances are per-dtype. Comparison is port vs an independent "
        "reference on the same device, not CUDA vs ROCm.",
        "",
    ]
    return lines


def _benchmark_section(inputs: ReportInputs) -> list[str]:
    lines = ["## Benchmarks", ""]
    if not inputs.benchmarks:
        lines += [f"{NOT_MEASURED} (no benchmark data).", ""]
    else:
        lines += [
            "| Comparison | Result | Verdict |",
            "| --- | --- | --- |",
        ]
        for entry in inputs.benchmarks:
            if "ratio" in entry:
                headline = f"ratio {entry['ratio']}"
            elif "speedup" in entry:
                headline = f"{entry['speedup']}x"
            else:
                headline = NOT_MEASURED
            lines.append(
                f"| {entry.get('label', 'unnamed')} | {headline} "
                f"| {entry.get('verdict', NOT_MEASURED)} |"
            )
        lines += [
            "",
            f"Wall-clock noise band: ±{(config.BENCH_NOISE_BAND - 1) * 100:.0f}%. "
            "Ratios inside it are reported as noise, not as passes.",
            "",
        ]
    return lines


def _entry_ok(entry) -> bool:
    """Read the ok flag from a dict or an attribute-bearing stub."""
    if isinstance(entry, dict):
        return bool(entry.get("ok"))
    return bool(getattr(entry, "ok", False))


def failed_entries(results) -> int:
    return sum(1 for entry in results or [] if not _entry_ok(entry))


def _execution_section(inputs: ReportInputs) -> list[str]:
    execution = inputs.execution
    lines = ["## Execution", ""]
    if execution is None:
        lines += [f"{NOT_MEASURED} (ported code was not executed).", ""]
    else:
        state = "OK" if execution.ok else "FAILED"
        lines += [
            f"**{state}** — {execution.summary()}",
            "",
            f"- Exit code: {execution.exit_code if execution.exit_code is not None else 'n/a'}",
            f"- Timed out: {execution.timed_out}",
        ]
        if execution.protocol_error:
            lines.append(f"- Protocol error: {execution.protocol_error}")
        if execution.error_message and not execution.ok:
            lines.append(f"- Error: {execution.error_message}")
        lines.append("")

    # Stated unconditionally. The isolation in force is process-level, and
    # a reader who assumes a real sandbox would over-trust this report.
    lines += ["### Isolation", ""]
    lines += [f"- {note}" for note in runner.sandbox_notes()]
    lines.append("")
    return lines


def _narrative_section(inputs: ReportInputs) -> list[str]:
    """LLM prose, clearly fenced off from the measured results above.

    Rendered after every number-bearing section and under an explicit
    heading, because the prose is the one part of this document a model
    wrote. A reader must never have to guess which is which.

    The agent marks anything it could not obtain with a leading
    "not available:", and those are reported as unavailability rather than
    under a heading claiming a model wrote them. The marker is matched as
    a literal because report/ must not import agent/.
    """
    titles = {
        "narration": "Summary",
        "parity_explainer": "Parity, explained",
        "benchmark_explainer": "Benchmarks, explained",
        "remediation": "Suggested remediation",
    }
    narratives = inputs.narratives or {}
    prose = {
        key: text.strip()
        for key, text in narratives.items()
        if key in titles
        and isinstance(text, str)
        and text.strip()
        and not text.strip().lower().startswith(UNAVAILABLE_MARKER)
    }

    if not prose:
        lines = ["## Narrative", "", "No narrative generated."]
        reasons = [
            text.strip()
            for key, text in narratives.items()
            if key in titles and isinstance(text, str) and text.strip()
        ]
        if reasons:
            lines += ["", "The agent layer could not produce prose:", ""]
            lines += [f"- {reason}" for reason in reasons]
        else:
            lines.append(
                "This section is optional and does not affect the verdict below."
            )
        lines += ["", "It is the only part of this report a model could write.", ""]
        return lines

    lines = ["## Narrative (LLM-generated)", ""]
    lines += [
        "> Generated by a language model from the measured results above. "
        "It explains and proposes; it does not decide correctness.",
        "",
    ]
    for key, title in titles.items():
        text = prose.get(key)
        if not text:
            continue
        lines += [f"### {title}", "", text, ""]
    return lines


def _readiness(inputs: ReportInputs) -> list[str]:
    """Downgrade rather than omit anything that was not measured."""
    blockers: list[str] = []
    migration = inputs.migration

    if migration is None:
        blockers.append("Conversion was not run.")
    elif not migration.converted:
        blockers.append("Conversion produced no ported source.")
    elif migration.coverage.verdict is not Verdict.PRODUCTION_READY:
        blockers.append(
            f"Conversion verdict: {_verdict_line(migration.coverage.verdict)}"
        )

    execution = inputs.execution
    if execution is None:
        blockers.append("Ported code was not executed.")
    elif not execution.ok:
        blockers.append(f"Execution failed: {execution.summary()}")
    elif failed_entries(execution.results):
        failed = failed_entries(execution.results)
        blockers.append(f"{failed} case(s) raised during execution.")

    if inputs.parity is None:
        blockers.append("Parity was not measured.")
    elif not inputs.parity.get("all_passed"):
        blockers.append("One or more parity cases exceeded tolerance.")

    if not inputs.benchmarks:
        blockers.append("No benchmarks were recorded.")

    snapshot = inputs.telemetry_snapshot
    if snapshot is None:
        blockers.append("Hardware telemetry was not captured.")
    elif not snapshot.get("gpu_available"):
        blockers.append(
            "No AMD GPU was detected; results are CPU-only and do not "
            "demonstrate ROCm performance."
        )

    blockers += inputs.skipped_notes

    verdict = (
        Verdict.PRODUCTION_READY
        if not blockers
        else (Verdict.NEEDS_REVIEW if migration else Verdict.UNSUPPORTED)
    )
    return (
        ["## Readiness", "", f"**{_verdict_line(verdict)}**", ""]
        + [f"- {item}" for item in blockers]
        + [""]
    )


def render(inputs: ReportInputs) -> str:
    snapshot = inputs.telemetry_snapshot or telemetry.device_snapshot()

    lines = [
        "# Graft migration report",
        "",
        f"Generated by {config.PROJECT_NAME} on the "
        f"{snapshot.get('backend', 'unknown backend')} backend.",
        "",
        "## Environment",
        "",
        f"- Backend: {snapshot.get('backend', NOT_MEASURED)}",
        f"- GPU present: {snapshot.get('gpu_available', False)}",
        f"- GPU: {snapshot.get('gpu_name', NOT_MEASURED)}",
        f"- Total VRAM: {_fmt(snapshot.get('total_vram_mb'), '{:.0f}', 'n/a')} MB",
        f"- Peak VRAM: {_fmt(snapshot.get('peak_vram_mb'), '{:.1f}', 'n/a')} MB",
        "",
    ]
    lines += _conversion_section(inputs.migration, inputs.parity)
    lines += _execution_section(inputs)
    lines += _parity_section(inputs)
    lines += _benchmark_section(inputs)
    lines += _narrative_section(inputs)
    lines += _readiness(inputs)
    return "\n".join(lines).rstrip() + "\n"
