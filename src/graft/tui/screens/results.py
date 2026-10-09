"""Results screen: the evidence ledger for migrations and GPU checks.

Every section renders a value that actually came back from the worker. When a
stage did not run, the screen says so instead of inventing a number. The LLM
narrative is the only prose that a language model can write and it never
changes a verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from graft.target.dependencies import scan_dependencies
from graft.target.detect import TargetFacts

NOT_MEASURED = "not measured"

NARRATIVE_TITLES = {
    "narration": "Summary",
    "parity_explainer": "Parity, explained",
    "benchmark_explainer": "Benchmarks, explained",
    "remediation": "Suggested remediation",
}

SEVERITY_ORDER = ("blocker", "warning", "info")


def _kv(label: str, value: Any) -> Static:
    return Static(f"{label}: {value}", markup=False, classes="result-kv")


def _finding_rows(findings: list[dict[str, Any]]) -> list[Static]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        buckets.setdefault(str(finding.get("severity") or "info"), []).append(finding)
    ordered = [level for level in SEVERITY_ORDER if level in buckets]
    ordered += [level for level in buckets if level not in SEVERITY_ORDER]
    rows: list[Static] = []
    for level in ordered:
        items = buckets[level]
        rows.append(
            Static(
                f"{level.upper()} · {len(items)} finding(s)",
                markup=False,
                classes="finding-group",
            )
        )
        for finding in items[:8]:
            literal = finding.get("matched_text") or finding.get("symbol") or "?"
            reason = finding.get("reason") or ""
            rows.append(
                Static(f"  {literal} — {reason}", markup=False, classes="finding")
            )
        if len(items) > 8:
            rows.append(
                Static(f"  +{len(items) - 8} more", markup=False, classes="finding")
            )
    return rows


def _ai_review_rows(review: dict[str, Any]) -> list[Static]:
    """Render the optional model review without letting it look like evidence."""
    if not review.get("available"):
        return [
            Static(
                str(review.get("reason") or "AI review was not completed."),
                markup=False,
                classes="result-card",
            )
        ]
    rows = [
        Static(
            "\n".join(
                [
                    f"provider/model : {review.get('provider', 'unknown')} / {review.get('model', 'unknown')}",
                    f"recommendation : {review.get('recommendation', 'engineer_review_required')}",
                ]
            ),
            markup=False,
            classes="result-card",
        )
    ]
    summary_text = str(review.get("summary") or "")
    if summary_text:
        rows.append(Static(summary_text, markup=False, classes="result-card"))
    for comment in review.get("comments") or []:
        line = comment.get("line")
        location = (
            f"{comment.get('file', 'converted')}:{line}"
            if line
            else str(comment.get("file", "converted"))
        )
        severity = str(comment.get("severity", "review")).upper()
        rows.append(
            Static(
                f"{severity} · {location} · {comment.get('title', 'Review note')}",
                markup=False,
                classes="result-section",
            )
        )
        if comment.get("comment"):
            rows.append(
                Static(str(comment["comment"]), markup=False, classes="result-card")
            )
        if comment.get("suggestion"):
            rows.append(
                Static(
                    f"Suggested change: {comment['suggestion']}",
                    markup=False,
                    classes="result-kv",
                )
            )
    if not review.get("comments"):
        rows.append(
            Static(
                "No line-specific comments were returned.",
                markup=False,
                classes="result-kv",
            )
        )
    rows.append(
        Static(
            "This is a model review of the source text, not proof of compilation, "
            "runtime behavior, parity, or performance.",
            markup=False,
            classes="result-kv",
        )
    )
    return rows


def _dependency_lines(facts: Any) -> list[str]:
    """A short checklist of the tools Graft needs, from the probe's facts."""
    if not isinstance(facts, dict):
        return []
    try:
        parsed = TargetFacts(**facts)
    except TypeError:
        return []
    lines = ["", "dependencies:"]
    for dep in scan_dependencies(parsed):
        if dep.present:
            mark = "ok"
        elif dep.required:
            mark = "MISSING"
        else:
            mark = "optional, missing"
        lines.append(f"  {dep.label}: {mark}")
    return lines


class ResultsScreen(VerticalScroll):
    """Display the evidence returned by local and remote operations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._converted_source = ""
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Static("Results", id="result-heading")
        yield Static("", id="result-summary")
        with Horizontal(id="result-tools"):
            yield Button("Copy results", id="copy-results")
        with Horizontal(id="save-row"):
            yield Input(placeholder="Save converted file to…", id="output-path")
            yield Button("Save converted file", id="save-output")
        yield Vertical(id="results-body")

    def on_mount(self) -> None:
        self.show_empty()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-output":
            self._save_converted()
        elif event.button.id == "copy-results":
            self._copy_results()

    def _copy_results(self) -> None:
        text = self._copyable_text()
        if not text.strip():
            return
        self.app.copy_to_clipboard(text)
        button = self.query_one("#copy-results", Button)
        button.label = "Copied to clipboard"
        self.set_timer(1.5, self._reset_copy_label)

    def _reset_copy_label(self) -> None:
        if self.is_mounted:
            self.query_one("#copy-results", Button).label = "Copy results"

    def _copyable_text(self) -> str:
        """The currently displayed results as one plain-text block."""
        chunks: list[str] = []
        for widget in self.query(Static):
            try:
                text = str(widget.render())
            except Exception:
                continue
            if text.strip():
                chunks.append(text)
        return "\n".join(chunks)

    def _save_converted(self) -> None:
        target = self.query_one("#output-path", Input).value.strip()
        status = self.query_one("#result-summary", Static)
        if not self._converted_source:
            status.update("There is no converted file to save.")
            status.set_classes("result-summary error")
            return
        if not target:
            status.update("Enter an output path for the converted file.")
            status.set_classes("result-summary warning")
            return
        path = Path(target).expanduser()
        try:
            if path.is_dir():
                path = path / "graft_converted.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._converted_source, encoding="utf-8")
        except OSError as exc:
            status.update(f"Could not save converted file: {exc}")
            status.set_classes("result-summary error")
            return
        status.update(f"Saved converted file · {path}")
        status.set_classes("result-summary success")

    # ------------------------------------------------------------------
    # internal rendering helpers
    # ------------------------------------------------------------------
    def _body(self) -> Vertical:
        return self.query_one("#results-body", Vertical)

    def _set_heading(self, text: str, tone: str = "info") -> None:
        heading = self.query_one("#result-heading", Static)
        heading.update(text)
        heading.set_classes(f"result-heading {tone}")

    def _set_summary(self, text: str | None, tone: str = "info") -> None:
        summary = self.query_one("#result-summary", Static)
        if text:
            summary.update(text)
            summary.set_classes(f"result-summary {tone}")
            summary.display = True
        else:
            summary.display = False

    def _reset(self) -> None:
        self._converted_source = ""
        self.query_one("#save-row").display = False
        self.query_one("#result-tools").display = False
        self.query_one("#output-path", Input).value = ""
        self._body().remove_children()

    def _mount_body(self, *widgets: Any, copyable: bool = True) -> None:
        self._body().mount(*widgets)
        self.query_one("#result-tools").display = copyable

    # ------------------------------------------------------------------
    # public views
    # ------------------------------------------------------------------
    def show_empty(self) -> None:
        self._set_heading("Results", "info")
        self._set_summary(None)
        self.query_one("#save-row").display = False
        self._body().remove_children()
        self._mount_body(
            Static(
                "Choose a source file and run a migration, or detect a GPU "
                "target. Results appear here.",
                markup=False,
                classes="result-card",
            ),
            copyable=False,
        )

    def show_source_preview(
        self,
        name: str,
        path: str,
        source: str | None = None,
        message: str | None = None,
    ) -> None:
        self._reset()
        if message:
            self._set_heading(f"Source not loaded · {name or 'source'}", "error")
            self._mount_body(Static(message, markup=False, classes="result-card"))
            return
        self._set_heading(f"Source preview · {name}", "info")
        self._mount_body(
            Static(path, markup=False, classes="result-kv"),
            Static(source or "", markup=False, classes="source-pane"),
        )

    def show_local(
        self,
        summary: dict[str, Any],
        filename: str,
        source_path: str,
        original_source: str,
    ) -> None:
        self._reset()
        conversion = summary.get("conversion") or {}
        parity = summary.get("parity") or {}
        execution = summary.get("execution") or {}
        environment = summary.get("environment") or {}
        benchmarks = summary.get("benchmarks") or []
        narrative = summary.get("narrative") or {}
        ai_review = summary.get("ai_review") or {}
        findings = conversion.get("findings") or []
        changes = conversion.get("changes") or []
        self._converted_source = str(summary.get("converted_source") or "")

        verdict = str(conversion.get("verdict") or "").lower()
        blockers = sum(
            1
            for finding in findings
            if str(finding.get("severity")) in {"blocker", "error"}
        )
        parity_measured = bool(parity.get("measured"))
        parity_failed = parity_measured and int(parity.get("passed", 0)) < int(
            parity.get("total", 0)
        )
        if verdict == "production_ready" and parity_measured and not parity_failed:
            tone = "success"
        elif verdict in {"unsupported"} or parity_failed or blockers:
            tone = "error"
        else:
            tone = "warning"

        self._set_heading(f"Migration report · {filename}", tone)
        self._set_summary(conversion.get("verdict_line") or "conversion complete", tone)
        self.query_one("#save-row").display = bool(self._converted_source)

        parity_text = (
            f"{parity.get('passed', 0)}/{parity.get('total', 0)}"
            if parity_measured
            else "not measured"
        )
        device = "GPU found" if environment.get("gpu_available") else "CPU-only"
        tiles = Static(
            "\n".join(
                [
                    f"rewrites applied : {len(changes)}",
                    f"parity cases     : {parity_text}",
                    f"blocking findings: {blockers}",
                    f"execution context: {device}",
                ]
            ),
            markup=False,
            classes="result-card",
        )

        details = [
            Static("CONVERSION", markup=False, classes="result-section"),
            Static(
                "\n".join(
                    [
                        f"route   : {conversion.get('route') or 'unknown'}",
                        f"backend : {conversion.get('backend') or 'not measured'}",
                        f"basis   : {conversion.get('verdict_basis') or ''}".rstrip(),
                    ]
                ),
                markup=False,
                classes="result-card",
            ),
        ]
        if changes:
            details.append(
                Static("REWRITE TRACE", markup=False, classes="result-section")
            )
            for change in changes[:20]:
                line = change.get("line")
                where = f":{line}" if isinstance(line, int) else ""
                details.append(
                    Static(
                        f"{change.get('symbol', '?')}{where} — {change.get('reason', '')}",
                        markup=False,
                        classes="result-kv",
                    )
                )
            if len(changes) > 20:
                details.append(
                    Static(
                        f"+{len(changes) - 20} additional rewrites",
                        markup=False,
                        classes="result-kv",
                    )
                )
        details.append(Static("COVERAGE SCAN", markup=False, classes="result-section"))
        if findings:
            details.extend(_finding_rows(findings))
        else:
            details.append(
                Static(
                    "No unmapped constructs found.", markup=False, classes="result-kv"
                )
            )

        environment_rows = Static(
            "\n".join(
                [
                    f"device      : {environment.get('gpu_name') or 'not measured'}",
                    f"gpu present : {'yes' if environment.get('gpu_available') else 'no'}",
                    f"backend     : {environment.get('backend') or 'not measured'}",
                    f"total vram  : {environment.get('total_vram_mb') or 'not measured'}",
                ]
            ),
            markup=False,
            classes="result-card",
        )

        if execution.get("measured"):
            execution_rows = Static(
                "\n".join(
                    [
                        f"result   : {'ok' if execution.get('ok') else 'failed'}",
                        f"cases    : {execution.get('case_count', 0)}",
                        f"timed out: {'yes' if execution.get('timed_out') else 'no'}",
                        f"exit code: {execution.get('exit_code')}",
                    ]
                ),
                markup=False,
                classes="result-card",
            )
        else:
            execution_rows = Static("not measured", markup=False, classes="result-card")

        if parity_measured:
            error = parity.get("max_abs_error")
            parity_rows = Static(
                "\n".join(
                    [
                        f"score     : {parity.get('passed', 0)}/{parity.get('total', 0)} "
                        f"({'passing' if parity.get('all_passed') else 'failing'})",
                        f"worst case: {parity.get('worst_case') or 'n/a'}",
                        f"max error : {error:.3e}"
                        if isinstance(error, (int, float))
                        else "max error : n/a",
                        "The port is compared with an independent reference on this same device.",
                    ]
                ),
                markup=False,
                classes="result-card",
            )
        else:
            parity_rows = Static("not measured", markup=False, classes="result-card")

        if benchmarks:
            bench_lines = []
            for row in benchmarks:
                ratio, speedup = row.get("ratio"), row.get("speedup")
                if isinstance(ratio, (int, float)):
                    headline = f"ratio {ratio:.2f}x"
                elif isinstance(speedup, (int, float)):
                    headline = f"speedup {speedup:.2f}x"
                else:
                    headline = "n/a"
                bench_lines.append(
                    f"{row.get('label', 'unnamed')}: {headline} · {row.get('verdict', 'n/a')}"
                )
            benchmark_rows = Static(
                "\n".join(bench_lines), markup=False, classes="result-card"
            )
        else:
            benchmark_rows = Static("not measured", markup=False, classes="result-card")

        prose = narrative.get("prose") or {}
        narrative_widgets: list[Any] = []
        if prose:
            for key, text in prose.items():
                narrative_widgets.append(
                    Static(
                        key.replace("_", " ").title(),
                        markup=False,
                        classes="result-section",
                    )
                )
                narrative_widgets.append(
                    Static(text, markup=False, classes="result-card")
                )
        else:
            reasons = narrative.get("reasons") or []
            narrative_widgets.append(
                Static("No narrative generated.", markup=False, classes="result-card")
            )
            for reason in reasons:
                narrative_widgets.append(
                    Static(reason, markup=False, classes="result-kv")
                )
            narrative_widgets.append(
                Static(
                    "This optional section is the only part a language model could have written, and it never affects a verdict.",
                    markup=False,
                    classes="result-kv",
                )
            )

        skipped = summary.get("skipped") or []
        if skipped:
            narrative_widgets.append(
                Static(
                    "SKIPPED / UNMEASURED STAGES",
                    markup=False,
                    classes="result-section",
                )
            )
            for item in skipped:
                narrative_widgets.append(
                    Static(item, markup=False, classes="result-kv")
                )

        ai_section: list[Any] = []
        if ai_review:
            ai_section.append(
                Static(
                    "AI REVIEW (model, optional)",
                    markup=False,
                    classes="result-section",
                )
            )
            ai_section.extend(_ai_review_rows(ai_review))

        self._mount_body(
            tiles,
            *details,
            Static("ENVIRONMENT", markup=False, classes="result-section"),
            environment_rows,
            Static("ISOLATED EXECUTION", markup=False, classes="result-section"),
            execution_rows,
            Static("PARITY SIGNAL", markup=False, classes="result-section"),
            parity_rows,
            Static("BENCHMARK SIGNAL", markup=False, classes="result-section"),
            benchmark_rows,
            Static(
                "NARRATIVE (model, optional)", markup=False, classes="result-section"
            ),
            *narrative_widgets,
            *ai_section,
            Static("SOURCE COMPARISON", markup=False, classes="result-section"),
            Static(f"original · {source_path}", markup=False, classes="editor-title"),
            Static(original_source, markup=False, classes="source-pane"),
            Static("converted", markup=False, classes="editor-title"),
            Static(
                self._converted_source or "not available",
                markup=False,
                classes="source-pane",
            ),
        )

    def show_local_probe(self, facts: dict[str, Any]) -> None:
        self._reset()
        gpu = bool(facts.get("gpu"))
        self._set_heading("Local capability check", "success" if gpu else "info")
        lines = [f"{key}: {value}" for key, value in sorted(facts.items())]
        self._mount_body(Static("\n".join(lines), markup=False, classes="result-card"))

    def show_error(self, title: str, message: str) -> None:
        self._reset()
        self._set_heading(title, "error")
        self._mount_body(Static(message, markup=False, classes="result-card error"))

    def show_missing_dependencies(
        self, action: str, missing: list[Any], message: str
    ) -> None:
        """A red block naming the target tool that stopped an operation."""
        self._reset()
        self._set_heading("Missing dependencies", "error")
        lines = [message, ""]
        for dep in missing:
            detail = getattr(dep, "detail", "") or "not found"
            lines.append(f"{getattr(dep, 'label', 'dependency')}: {detail}")
            command = getattr(dep, "command", "")
            if command:
                lines.append(f"  install: {command}")
        self._mount_body(
            Static("\n".join(lines), markup=False, classes="result-card error")
        )

    def show_migration_error(
        self, title: str, message: str, original_source: str
    ) -> None:
        self._reset()
        self._set_heading(title, "error")
        self._mount_body(
            Static(message, markup=False, classes="result-card error"),
            Static("original source", markup=False, classes="editor-title"),
            Static(original_source, markup=False, classes="source-pane"),
        )

    def show_remote_file(self, result: dict[str, Any]) -> None:
        self._reset()
        ok = bool(result.get("ok"))
        self._set_heading("Converted file run on GPU", "success" if ok else "error")
        self._mount_body(self._remote_body(result))

    def show_remote(self, result: dict[str, Any], mode: str) -> None:
        self._reset()
        ok = bool(result.get("ok"))
        self._set_heading(
            f"Remote {'detect' if mode == 'probe' else 'verification'}",
            "success" if ok else "error",
        )
        self._mount_body(self._remote_body(result))

    def show_dependency_install(
        self, results: list[dict[str, Any]], facts: dict[str, Any] | None
    ) -> None:
        self._reset()
        failed = [item for item in results if not item.get("ok")]
        tone = "warning" if failed else "success"
        self._set_heading("Dependency check", tone)
        lines: list[str] = []
        if not results:
            lines.append("All required dependencies are already present.")
        for item in results:
            label = item.get("label") or item.get("key") or "dependency"
            lines.append(f"{label}: {'installed' if item.get('ok') else 'needs attention'}")
            if item.get("output"):
                lines.append(f"  {item['output']}")
            if item.get("error"):
                lines.append(f"  {item['error']}")
        lines.extend(_dependency_lines(facts or {}))
        self._mount_body(Static("\n".join(lines), markup=False, classes="result-card"))

    def _remote_body(self, result: dict[str, Any]) -> Static:
        lines: list[str] = []
        facts = result.get("facts")
        if isinstance(facts, dict):
            for key, value in facts.items():
                lines.append(f"{key}: {value}")
            lines.extend(_dependency_lines(facts))
        steps = result.get("steps")
        if isinstance(steps, dict):
            lines.append("")
            lines.append("steps:")
            for name, step in steps.items():
                if isinstance(step, dict):
                    status = "ok" if step.get("ok") else "failed"
                    lines.append(f"  {name}: {status}")
                else:
                    lines.append(f"  {name}: {step}")
        error = result.get("error")
        if error:
            lines.append("")
            lines.append(f"error: {error}")
        return Static(
            "\n".join(lines) or "No details returned.",
            markup=False,
            classes="result-card",
        )
