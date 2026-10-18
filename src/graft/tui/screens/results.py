"""Results screen: the evidence ledger for migrations and GPU checks.

Every section renders a value that actually came back from the worker. When a
stage did not run, the screen says so instead of inventing a number. The LLM
narrative is the only prose that a language model can write and it never
changes a verdict.
"""

from __future__ import annotations

import contextlib
from functools import partial
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from graft.target.dependencies import scan_dependencies
from graft.target.detect import TargetFacts

NOT_MEASURED = "not measured"

NEEDS_MANUAL_PORT = {"needs_review", "manual_port", "unsupported"}


def numbered_source(source: str) -> str:
    """Render source with a right-aligned line-number gutter, like an IDE.

    The gutter is three columns wide for files up to 999 lines, which covers
    everything the UI will preview (the upload limit is 1 MB of source); past
    that it grows rather than misaligning.
    """
    lines = source.splitlines()
    width = max(3, len(str(len(lines) or 1)))
    return "\n".join(
        f"{index:>{width}} │ {line}" for index, line in enumerate(lines, start=1)
    )

NARRATIVE_TITLES = {
    "narration": "Summary",
    "parity_explainer": "Parity, explained",
    "benchmark_explainer": "Benchmarks, explained",
    "remediation": "Suggested remediation",
}

SEVERITY_ORDER = ("blocker", "review", "info")


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


def _step_status(step: Any) -> str:
    """Render a remote step as ok/failed.

    The remote scripts report a step with a ``returncode`` and its output; some
    paths also carry an explicit ``ok``. Reading only ``ok`` marked every
    successful remote step as failed, because the key was never there.
    """
    if not isinstance(step, dict):
        return "failed"
    if "ok" in step:
        return "ok" if step.get("ok") else "failed"
    if "returncode" in step:
        return "ok" if step.get("returncode") == 0 else "failed"
    return "failed"


class ResultsScreen(VerticalScroll):
    """Display the evidence returned by local and remote operations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._converted_source = ""
        self._original_source = ""
        self._source_filename = "converted.py"
        self._source_path = ""
        self._ai_options: dict[str, str] = {}
        self._ai_result: dict[str, Any] | None = None
        self._ai_converting = False
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Static("Results", id="result-heading")
        yield Static("", id="result-summary")
        with Horizontal(id="save-row"):
            yield Input(placeholder="Save converted file to…", id="output-path")
            yield Button("Save converted file", id="save-output")
        yield Vertical(id="results-body")

    def on_mount(self) -> None:
        self.show_empty()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-output":
            self._save_converted()
        elif event.button.id == "ai-convert":
            self._start_ai_conversion()
        elif event.button.id == "ai-preview":
            self._preview_ai()
        elif event.button.id == "ai-run":
            self._run_ai_on_gpu()
        elif event.button.id == "compare-source":
            self._show_source_comparison()

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

    def _show_source_comparison(self) -> None:
        """Open the original and converted files in an overlay.

        The report renders a summary of the comparison; both files live in the
        popup so a long port does not bury the measured evidence above it.
        """
        from graft.tui.screens.compare_popup import SourceComparePopup

        if not self._original_source.strip() and not self._converted_source.strip():
            self.notify("There is no source to compare.", severity="warning")
            return
        self.app.push_screen(
            SourceComparePopup(
                self._source_filename,
                self._original_source,
                self._converted_source,
                source_path=self._source_path,
            )
        )

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
        self._ai_result = None
        self._ai_converting = False
        self.query_one("#save-row").display = False
        self.query_one("#output-path", Input).value = ""
        self._body().remove_children()

    def _mount_body(self, *widgets: Any) -> None:
        self._body().mount(*widgets)

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
        body = source or ""
        line_count = len(body.splitlines())
        self._mount_body(
            Static(path, markup=False, classes="result-kv"),
            Static(
                f"{line_count} line(s) · read-only preview",
                markup=False,
                classes="result-kv",
            ),
            Static(numbered_source(body), markup=False, classes="source-pane"),
        )

    def show_local(
        self,
        summary: dict[str, Any],
        filename: str,
        source_path: str,
        original_source: str,
        ai_options: dict[str, str] | None = None,
    ) -> None:
        self._render_report(
            summary,
            filename,
            source_path,
            original_source,
            ai_options=ai_options,
            heading=f"Migration report · {filename}",
        )

    def show_remote_report(
        self,
        summary: dict[str, Any],
        filename: str,
        source_path: str,
        original_source: str,
        ai_options: dict[str, str] | None = None,
        facts: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        """The full report for a migration measured on a remote GPU.

        Same body as a local run — conversion, execution, parity, benchmarks,
        the AI review, and the original-vs-converted source comparison — with a
        short block naming the remote target the numbers came from.
        """
        self._render_report(
            summary,
            filename,
            source_path,
            original_source,
            ai_options=ai_options,
            heading=f"Remote migration report · {filename}",
            facts=facts,
            note=note,
        )

    def _render_report(
        self,
        summary: dict[str, Any],
        filename: str,
        source_path: str,
        original_source: str,
        ai_options: dict[str, str] | None = None,
        *,
        heading: str,
        facts: dict[str, Any] | None = None,
        note: str = "",
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
        self._original_source = original_source or ""
        self._source_filename = filename or "converted.py"
        self._source_path = source_path or ""
        self._ai_options = dict(ai_options or {})

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

        self._set_heading(heading, tone)
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

        remote_widgets: list[Any] = []
        if isinstance(facts, dict) and facts:
            remote_lines = []
            for key in ("gpu_name", "gpu_count", "rocm_version", "torch_hip"):
                if facts.get(key) not in (None, ""):
                    remote_lines.append(f"{key}: {facts[key]}")
            for key, value in (facts.get("tools") or {}).items():
                remote_lines.append(f"{key}: {'present' if value else 'missing'}")
            notes = facts.get("notes")
            if isinstance(notes, list) and notes:
                remote_lines.extend(f"note: {note}" for note in notes)
            elif notes:
                remote_lines.append(str(notes))
            if remote_lines:
                remote_widgets = [
                    Static("REMOTE TARGET", markup=False, classes="result-section"),
                    Static(
                        "\n".join(remote_lines), markup=False, classes="result-card"
                    ),
                ]
        if note:
            remote_widgets.append(
                Static(note, markup=False, classes="result-card")
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

        needs_manual = (
            blockers > 0
            or parity_failed
            or verdict in NEEDS_MANUAL_PORT
        )
        self._ai_result = None
        ai_assist: list[Any] = []
        if needs_manual:
            ai_assist = self._ai_assist_widgets()

        self._mount_body(
            tiles,
            *remote_widgets,
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
            *ai_assist,
            Static("SOURCE COMPARISON", markup=False, classes="result-section"),
            Static(
                f"original · {source_path or filename} · "
                f"{len(original_source.splitlines())} line(s)",
                markup=False,
                classes="result-kv",
            ),
            Static(
                (
                    f"converted · {len(self._converted_source.splitlines())} line(s)"
                    if self._converted_source
                    else "converted · not available"
                ),
                markup=False,
                classes="result-kv",
            ),
            Horizontal(
                Button("View original vs converted", id="compare-source"),
                id="compare-actions",
            ),
        )

    def _ai_assist_widgets(self) -> list[Any]:
        """The labelled block that offers to finish the port with Gemma."""
        model = self._ai_options.get("model") or "the configured Gemma model"
        widgets: list[Any] = [
            Static("AI-ASSISTED PORT (model, optional)", markup=False, classes="result-section"),
            Static(
                "The mechanical port left work for a human. You can ask Google "
                f"Gemma ({model}) to produce a complete file that finishes it. "
                "The AI file is kept separate from the measured results above — "
                "the verdict does not change, and the AI version is not evidence "
                "until you run it on a GPU and Graft verifies it.",
                markup=False,
                classes="result-card",
            ),
            Horizontal(
                Button("Convert with Google Gemma", id="ai-convert"),
                id="ai-assist-actions",
            ),
        ]
        return widgets

    def _start_ai_conversion(self) -> None:
        if self._ai_converting:
            return
        if not self._original_source.strip():
            self.notify("There is no source to convert.", severity="warning")
            return
        self._ai_converting = True
        with contextlib.suppress(Exception):
            button = self.query_one("#ai-convert", Button)
            button.label = "Asking Gemma…"
            button.disabled = True
        self.run_worker(
            partial(self._run_ai_conversion, self._original_source, self._source_filename),
            name="ai-conversion",
            group="graft-operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_ai_conversion(self, original_source: str, filename: str) -> None:
        from graft import pipeline

        evidence: dict[str, Any] = {
            "filename": filename,
            "options": self._ai_options,
        }
        try:
            result = pipeline.convert_with_ai(
                original_source,
                evidence,
                self._ai_options,
                filename=filename,
            )
        except Exception as exc:
            result = {
                "available": False,
                "reason": f"AI conversion failed ({type(exc).__name__}).",
            }
        self.app.call_from_thread(self._ai_conversion_done, result)

    def _ai_conversion_done(self, result: dict[str, Any]) -> None:
        self._ai_converting = False
        self._ai_result = result
        with contextlib.suppress(Exception):
            button = self.query_one("#ai-convert", Button)
            button.disabled = False
            button.label = "Convert with Google Gemma"
        if not result.get("available"):
            reason = result.get("reason") or "The AI conversion did not complete."
            self.notify(str(reason), severity="warning", timeout=8)
            return
        code = str(result.get("code") or "")
        self.notify(
            f"Gemma returned {result.get('filename') or 'a converted file'} "
            f"({len(code.splitlines())} lines). Review it before running.",
            timeout=8,
        )
        self._show_ai_result()

    def _show_ai_result(self) -> None:
        """Render the AI result controls beneath the offer block."""
        if not self._ai_result or not self._ai_result.get("available"):
            return
        result = self._ai_result
        code = str(result.get("code") or "")
        widgets: list[Any] = [
            Static(
                f"AI version ready · {result.get('filename')} · "
                f"{len(code.splitlines())} line(s)",
                markup=False,
                classes="result-card ai-flag",
            ),
            Static(str(result.get("summary") or ""), markup=False, classes="result-card"),
        ]
        comments = result.get("comments") or []
        if comments:
            widgets.append(
                Static("AI CHANGE NOTES", markup=False, classes="result-section")
            )
            for comment in comments[:12]:
                line = comment.get("line")
                where = f"line {line}" if line else "general"
                severity = str(comment.get("severity", "review")).upper()
                widgets.append(
                    Static(
                        f"{severity} · {where} · {comment.get('title', 'Change')}",
                        markup=False,
                        classes="result-kv",
                    )
                )
                if comment.get("comment"):
                    widgets.append(
                        Static(str(comment["comment"]), markup=False, classes="result-card")
                    )
        widgets.append(
            Static(
                "AI output can be wrong. Preview it, then upload and run it on a "
                "GPU: Graft re-runs the deterministic pipeline on the AI file and "
                "reports parity and benchmarks separately.",
                markup=False,
                classes="result-kv",
            )
        )
        widgets.append(
            Horizontal(
                Button("Preview AI version", id="ai-preview"),
                Button(
                    "Upload & run AI version on GPU",
                    id="ai-run",
                    variant="primary",
                ),
                id="ai-result-actions",
            )
        )
        body = self._body()
        with contextlib.suppress(Exception):
            body.mount(*widgets)

    def _preview_ai(self) -> None:
        if not self._ai_result or not self._ai_result.get("available"):
            return
        from graft.tui.screens.ai_popup import AIConversionPopup

        self.app.push_screen(
            AIConversionPopup(
                str(self._ai_result.get("filename") or "converted_gemma.py"),
                str(self._ai_result.get("code") or ""),
                list(self._ai_result.get("comments") or []),
            )
        )

    def _run_ai_on_gpu(self) -> None:
        if not self._ai_result or not self._ai_result.get("available"):
            return
        code = str(self._ai_result.get("code") or "")
        filename = str(self._ai_result.get("filename") or "converted_gemma.py")
        if not code.strip():
            self.notify("The AI version is empty.", severity="warning")
            return
        try:
            target = self.app.query_one("#target-screen")
        except Exception:
            self.notify("Open the Target tab to set the GPU connection first.", severity="warning")
            return
        start = getattr(target, "start_ai_remote_file", None)
        if start is None:
            self.notify("This build cannot run the AI version remotely.", severity="warning")
            return
        self.notify(f"Uploading {filename} to the GPU target…")
        start(filename, code)

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
            Static(
                numbered_source(original_source),
                markup=False,
                classes="source-pane",
            ),
        )

    def show_remote_file(self, result: dict[str, Any]) -> None:
        self._reset()
        ok = bool(result.get("ok"))
        self._set_heading("Converted file run on GPU", "success" if ok else "error")
        self._mount_body(self._remote_body(result))

    def show_remote_ai_file(self, result: dict[str, Any]) -> None:
        """Show the outcome of running + verifying an AI file on the GPU.

        This is the payoff of the AI path: not the model's word, but the
        deterministic pipeline run against the AI file on real AMD hardware.
        """
        self._reset()
        ok = bool(result.get("ok"))
        self._set_heading(
            "AI version run & verified on GPU", "success" if ok else "error"
        )
        lines: list[str] = []
        filename = result.get("filename")
        if filename:
            lines.append(f"file: {filename}")
        facts = result.get("facts")
        if isinstance(facts, dict):
            for key in ("gpu_name", "gpu_count", "rocm_version", "backend"):
                if facts.get(key) is not None:
                    lines.append(f"{key}: {facts[key]}")

        verification = result.get("verification")
        if isinstance(verification, dict) and verification:
            conversion = verification.get("conversion") or {}
            parity = verification.get("parity") or {}
            lines.append("")
            lines.append("deterministic verification of the AI file:")
            lines.append(f"  verdict : {conversion.get('verdict') or 'not measured'}")
            if parity.get("measured"):
                lines.append(
                    f"  parity  : {parity.get('passed', 0)}/{parity.get('total', 0)}"
                )
            else:
                lines.append("  parity  : not measured")
            for key in ("route", "backend"):
                if conversion.get(key):
                    lines.append(f"  {key}: {conversion[key]}")
        else:
            lines.append("")
            lines.append("deterministic verification did not return a result.")

        steps = result.get("steps")
        if isinstance(steps, dict):
            lines.append("")
            lines.append("steps:")
            for name, step in steps.items():
                if isinstance(step, dict):
                    status = _step_status(step)
                    lines.append(f"  {name}: {status}")
                else:
                    lines.append(f"  {name}: {step}")

        execution = result.get("execution_output")
        if execution:
            lines.append("")
            lines.append("run output:")
            lines.append(str(execution))

        error = result.get("error")
        if error:
            lines.append("")
            lines.append(f"error: {error}")

        report_text = result.get("report_text")
        if report_text:
            lines.append("")
            lines.append("report:")
            lines.append(str(report_text))

        lines.append("")
        lines.append(
            "The AI model wrote the candidate file; compilation, execution, "
            "parity, and benchmarks above were measured by Graft on the target."
        )
        self._mount_body(
            Static("\n".join(lines) or "No details returned.", markup=False, classes="result-card")
        )

    def show_remote(self, result: dict[str, Any], mode: str, note: str = "") -> None:
        self._reset()
        ok = bool(result.get("ok"))
        self._set_heading(
            f"Remote {'detect' if mode == 'probe' else 'verification'}",
            "success" if ok else "error",
        )
        self._mount_body(self._remote_body(result, note))

    def _remote_body(self, result: dict[str, Any], note: str = "") -> Static:
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
                    status = _step_status(step)
                    lines.append(f"  {name}: {status}")
                else:
                    lines.append(f"  {name}: {step}")
        error = result.get("error")
        if error:
            lines.append("")
            lines.append(f"error: {error}")
        if note:
            lines.append("")
            lines.append(note)
        return Static(
            "\n".join(lines) or "No details returned.",
            markup=False,
            classes="result-card",
        )
