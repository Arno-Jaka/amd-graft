"""The one call that runs everything.

Until this module existed, every stage was reachable but nothing connected
them: conversion, execution, parity, benchmarking, and the report were all
tested in isolation and assembled by hand. The agent had a narrative API
with no caller, so its output never reached a report.

`run()` performs the full sequence and returns the rendered document plus
the measured inputs it was built from, so a caller can show the numbers
and the prose without re-running anything.

Ordering is not arbitrary. Conversion must precede execution because the
ported source is what gets executed. Parity and benchmarking must follow
execution so the agent narrates measured results rather than intentions.
The narrative is attached last, immediately before rendering, so a model
failure can never affect a verdict: report.render() is called with the
narratives already resolved into text.

Every stage is skippable and every skip is recorded. A missing or broken
optional stage downgrades the report instead of failing the run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from graft import config
from graft.bench import bench, telemetry
from graft.core import converter, runner
from graft.report import report
from graft.verify import cases as case_module
from graft.verify import parity

__all__ = ["PipelineResult", "load_module", "run"]


@dataclass
class PipelineResult:
    """What a run produced. Report text plus everything behind it."""

    report_text: str
    inputs: report.ReportInputs
    skipped: list[str] = field(default_factory=list)
    ai_review: dict | None = None

    @property
    def verdict(self):
        return self.inputs.migration.coverage.verdict if self.inputs.migration else None

    @property
    def all_passed(self) -> bool | None:
        summary = self.inputs.parity or {}
        return summary.get("all_passed")


def load_module(path: Path) -> ModuleType:
    """Import a Python file by path, without putting its directory on sys.path.

    Graft migrates one file at a time and that file may import its
    neighbours, so the parent directory is added for the duration of the
    import and removed afterwards. Leaving it there would let one test's
    imports satisfy another test's module, which is how a green suite ends
    up testing the wrong file.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"graft_target_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(parent)
    return module


def _load_ported(
    migration, path: Path, skipped: list[str]
) -> tuple[ModuleType | None, bool]:
    """Import the ported source so parity and benchmarks test the port.

    The original file is deliberately not the thing loaded here. Parity and
    benchmarking exist to judge the conversion, so running them against the
    input would report a clean result while proving nothing about the output.

    The ported text is written next to the original so its sibling imports
    still resolve, imported, then removed. Only the AST (PyTorch) route
    produces importable Python; the CUDA C++ route yields HIP source that no
    Python interpreter can load.

    Returns ``(module, produced)``. ``produced`` says whether the route
    yielded a port at all, which is what tells the caller whether it may fall
    back to the original. A port that exists but fails to import returns
    ``(None, True)`` - a broken port must be reported, never masked by
    silently verifying the input instead.
    """
    if migration is None or not migration.ported_source:
        return None, False
    if migration.backend != "ast":
        return None, False

    parent = Path(path).parent
    try:
        fd, tmp = tempfile.mkstemp(prefix=".graft_ported_", suffix=".py", dir=parent)
    except OSError:
        fd, tmp = tempfile.mkstemp(prefix=".graft_ported_", suffix=".py")

    parent_str = str(parent)
    added = parent_str not in sys.path
    if added:
        sys.path.insert(0, parent_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(migration.ported_source)
        return load_module(Path(tmp)), True
    except Exception as exc:
        skipped.append(f"converted module not importable: {type(exc).__name__}: {exc}")
        return None, True
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent_str)
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def _benchmark(module: ModuleType, device, skipped: list[str]) -> list[dict]:
    """Time the port against its reference, and optionally against CPU.

    Two different questions, kept separate because conflating them is how
    a porting tool ends up claiming a 4.5x regression that is really a
    fused-versus-eager comparison:

    * port vs reference on the same device - implementation quality
    * port on GPU vs the same port on CPU - hardware headline

    Both need ``kernel``, ``reference``, and a module-level ``arg``
    tensor. A ported file that does not expose those is skipped with a
    note rather than benchmarked against something guessed.
    """
    rows: list[dict] = []
    kernel = getattr(module, "kernel", None)
    reference = getattr(module, "reference", None)
    arg = getattr(module, "arg", None)
    if kernel is None or reference is None or arg is None:
        missing = [
            name
            for name, value in (
                ("kernel", kernel),
                ("reference", reference),
                ("arg", arg),
            )
            if value is None
        ]
        skipped.append(f"benchmark: module does not export {', '.join(missing)}")
        return rows

    if device.type != "cuda":
        skipped.append(
            "benchmark: port vs reference requires a GPU device; CPU-only run"
        )
        return rows

    arg = arg.to(device)
    port = bench.measure("port", kernel, arg, device)
    ref = bench.measure("reference", reference, arg, device)
    quality = bench.compare_port_vs_reference(port, ref)
    rows.append(
        {
            "label": "port vs reference (same device, implementation quality)",
            "ratio": quality["ratio"],
            "verdict": quality["verdict"],
        }
    )

    cpu = torch_device("cpu")
    cpu_arg = arg.detach().to(cpu)
    cpu_result = bench.measure("port on cpu", kernel, cpu_arg, cpu)
    headline = bench.compare_against_cpu(port, cpu_result)
    rows.append(
        {
            "label": "gpu port vs cpu port (hardware headline)",
            "speedup": headline["speedup"],
            "verdict": headline["verdict"],
        }
    )
    return rows


def torch_device(kind: str):
    import torch

    return torch.device(kind)


def run(
    path: Path | str,
    *,
    device=None,
    run_parity: bool = True,
    run_bench: bool = True,
    with_narrative: bool = True,
    ai_review: dict | None = None,
) -> PipelineResult:
    """Convert, verify, benchmark, narrate, and render in one pass.

    Args:
        path: the source file to migrate.
        device: torch device; defaults to config.resolve_device().
        run_parity: generate the standard case set and compare.
        run_bench: time the kernel. Requires a GPU; skipped otherwise.
        with_narrative: ask the agent layer for prose. Ignored when no
            endpoint is configured, which is the normal case.
        ai_review: request one Gemma review of the final output. A dict with
            ``base_url``, ``model``, and ``api_key``. Routed through this
            composition root so no other layer imports the optional agent
            package.

    Returns a PipelineResult. Does not raise on a failed conversion, a
    failed parity check, or a missing agent: those are outcomes the report
    exists to describe.
    """
    path = Path(path)
    inputs = report.ReportInputs()
    skipped: list[str] = []

    original_source = path.read_text(encoding="utf-8")
    inputs.migration = converter.convert(original_source, path)
    inputs.telemetry_snapshot = telemetry.device_snapshot()
    device = device or config.resolve_device()

    try:
        original_module = load_module(path)
    except Exception as exc:
        skipped.append(f"module not importable: {type(exc).__name__}: {exc}")
        original_module = None

    ported_module, produced = _load_ported(inputs.migration, path, skipped)
    module = ported_module if produced else original_module

    case_list = case_module.generate() if run_parity else []
    if run_parity and module is not None:
        try:
            inputs.parity_results = parity.check(module, case_list)
            inputs.parity = parity.summarize(inputs.parity_results)
        except Exception as exc:
            skipped.append(f"parity: {type(exc).__name__}: {exc}")

    if original_module is not None:
        inputs.execution = runner.run_cases(inputs.migration.ported_source, case_list)
    else:
        skipped.append("execution: nothing to run")

    if run_bench and module is not None:
        try:
            inputs.benchmarks = _benchmark(module, device, skipped)
        except Exception as exc:
            skipped.append(f"benchmark: {type(exc).__name__}: {exc}")
    elif not run_bench:
        skipped.append("benchmark: disabled by caller")

    if with_narrative:
        inputs.narratives = _narrative(inputs)

    inputs.skipped_notes = skipped
    report_text = report.render(inputs)
    review = None
    if ai_review is not None:
        review = _ai_review(original_source, inputs, ai_review)
        report_text = _render_ai_review(report_text, review)
    return PipelineResult(report_text, inputs, skipped, ai_review=review)


def _ai_review(
    original_source: str, inputs: report.ReportInputs, options: dict
) -> dict:
    """Ask the optional agent layer for one provider-specific review.

    The import is lazy and scoped here because ``pipeline`` is the only
    module outside the agent package permitted to reach for it. A missing
    or broken agent package must not stop a deterministic run from
    completing, so failures are returned as an unavailable review.
    """
    try:
        from graft.agent.review import review_conversion
    except Exception as exc:
        return {
            "available": False,
            "reason": f"The AI review layer is not available ({type(exc).__name__}).",
        }

    converted_source = inputs.migration.ported_source if inputs.migration else ""
    return review_conversion(
        original_source,
        converted_source,
        _review_evidence(inputs),
        base_url=str(options.get("base_url") or config.LLM_BASE_URL),
        model_id=str(options.get("model") or config.LLM_MODEL),
        api_key=str(options.get("api_key") or config.LLM_API_KEY),
    )


def convert_with_ai(
    original_source: str,
    evidence: dict,
    options: dict,
    *,
    filename: str = "converted.py",
) -> dict:
    """Ask Gemma to finish a port the deterministic converter could not.

    This is the composition entry for the AI conversion path. Like
    ``_ai_review`` it lives in the one module permitted to import the optional
    agent package, and it imports lazily so a missing or broken agent layer
    degrades to ``{"available": False, ...}`` instead of breaking the caller.
    """
    try:
        from graft.agent.convert import convert_with_gemma
    except Exception as exc:
        return {
            "available": False,
            "reason": f"The AI conversion layer is not available ({type(exc).__name__}).",
        }

    return convert_with_gemma(
        original_source,
        evidence,
        base_url=str(options.get("base_url") or config.LLM_BASE_URL),
        model_id=str(options.get("model") or config.LLM_MODEL),
        api_key=str(options.get("api_key") or config.LLM_API_KEY),
        filename=filename,
    )


def review_with_ai(
    original_source: str,
    converted_source: str,
    summary: dict,
    options: dict,
) -> dict:
    """Ask Gemma to review an already-measured run.

    The composition entry for reusing a measured run's evidence when the run
    itself did not call the model — for example a migration executed on a
    remote GPU, whose evidence is summarised locally. The provider key stays on
    this machine; only the evidence projection is sent. Like the other agent
    entries this imports lazily and degrades to ``{"available": False, ...}``.
    """
    try:
        from graft.agent.review import review_conversion
    except Exception as exc:
        return {
            "available": False,
            "reason": f"The AI review layer is not available ({type(exc).__name__}).",
        }

    return review_conversion(
        original_source,
        converted_source,
        _review_evidence_from_summary(summary),
        base_url=str(options.get("base_url") or config.LLM_BASE_URL),
        model_id=str(options.get("model") or config.LLM_MODEL),
        api_key=str(options.get("api_key") or config.LLM_API_KEY),
    )


def _review_evidence_from_summary(summary: dict) -> dict:
    """Rebuild the model-facing evidence projection from a UI summary.

    Mirrors :func:`_review_evidence` so a run measured in another process (a
    remote GPU) yields the same projection the in-process path would have sent,
    and the model can still only cite numbers that were actually measured.
    """
    conversion = summary.get("conversion") or {}
    if not conversion.get("measured"):
        conversion_evidence: dict = {
            "verdict": report.NOT_MEASURED,
            "basis": report.NOT_MEASURED,
            "route": "unknown",
            "findings": [],
        }
    else:
        conversion_evidence = {
            "verdict": conversion.get("verdict_line") or report.NOT_MEASURED,
            "basis": conversion.get("verdict_basis") or report.NOT_MEASURED,
            "route": conversion.get("route") or "unknown",
            "findings": conversion.get("findings") or [],
        }

    execution = summary.get("execution") or {}
    execution_evidence = None
    if execution.get("measured"):
        execution_evidence = {
            "ok": bool(execution.get("ok")),
            "case_count": execution.get("case_count") or 0,
            "timed_out": bool(execution.get("timed_out")),
            "exit_code": execution.get("exit_code"),
            "error_type": execution.get("error_type"),
            "error_message": execution.get("error_message"),
        }

    parity = summary.get("parity") or {}
    return {
        "conversion": conversion_evidence,
        "execution": execution_evidence,
        "parity": parity if parity.get("measured") else None,
        "benchmarks": summary.get("benchmarks") or [],
        "environment": summary.get("environment") or {},
    }


def _render_ai_review(report_text: str, review: dict) -> str:
    try:
        from graft.agent.review import render_report_section
    except Exception:
        return report_text
    return render_report_section(report_text, review)


def _review_evidence(inputs: report.ReportInputs) -> dict:
    """Shape the measured results the same way the UI summary does.

    The model receives a projection of the run, never raw dataclasses, so
    it can only cite numbers that were actually measured.
    """
    migration = inputs.migration
    if migration is None:
        conversion: dict = {
            "verdict": report.NOT_MEASURED,
            "basis": report.NOT_MEASURED,
            "route": "unknown",
            "findings": [],
        }
    else:
        conversion = {
            "verdict": report._verdict_line(migration.coverage.verdict),
            "basis": report._verdict_basis(migration, inputs.parity),
            "route": migration.route.lang if migration.route else "unknown",
            "findings": report.coverage_findings(migration),
        }

    execution = inputs.execution
    execution_evidence = None
    if execution is not None:
        execution_evidence = {
            "ok": bool(execution.ok),
            "case_count": execution.case_count or 0,
            "timed_out": bool(execution.timed_out),
            "exit_code": execution.exit_code,
            "error_type": execution.error_type,
            "error_message": execution.error_message,
        }

    return {
        "conversion": conversion,
        "execution": execution_evidence,
        "parity": inputs.parity,
        "benchmarks": inputs.benchmarks,
        "environment": inputs.telemetry_snapshot,
    }


def _narrative(inputs: report.ReportInputs) -> dict[str, str]:
    """Narrative text, or the reason there is none.

    Imported lazily and defensively. If graft.agent is missing, broken, or
    simply not configured, the deterministic report must still render, so
    every failure mode here returns text rather than propagating.
    """
    try:
        from graft.agent import agent as agent_module
    except Exception as exc:
        reason = f"not available: agent layer could not be imported ({exc})"
        return {
            "narration": reason,
            "parity_explainer": reason,
            "benchmark_explainer": reason,
            "remediation": reason,
        }

    try:
        return agent_module.narrative_sections(inputs)
    except Exception as exc:
        reason = f"not available: {type(exc).__name__}: {exc}"
        return {
            "narration": reason,
            "parity_explainer": reason,
            "benchmark_explainer": reason,
            "remediation": reason,
        }
