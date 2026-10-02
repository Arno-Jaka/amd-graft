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

import importlib.util
import sys
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
) -> PipelineResult:
    """Convert, verify, benchmark, narrate, and render in one pass.

    Args:
        path: the source file to migrate.
        device: torch device; defaults to config.resolve_device().
        run_parity: generate the standard case set and compare.
        run_bench: time the kernel. Requires a GPU; skipped otherwise.
        with_narrative: ask the agent layer for prose. Ignored when no
            endpoint is configured, which is the normal case.

    Returns a PipelineResult. Does not raise on a failed conversion, a
    failed parity check, or a missing agent: those are outcomes the report
    exists to describe.
    """
    path = Path(path)
    inputs = report.ReportInputs()
    skipped: list[str] = []

    inputs.migration = converter.convert(path.read_text(encoding="utf-8"), path)
    inputs.telemetry_snapshot = telemetry.device_snapshot()
    device = device or config.resolve_device()

    try:
        module = load_module(path)
    except Exception as exc:
        skipped.append(f"module not importable: {type(exc).__name__}: {exc}")
        module = None

    case_list = case_module.generate() if run_parity else []
    if run_parity and module is not None:
        inputs.parity_results = parity.check(module, case_list)
        inputs.parity = parity.summarize(inputs.parity_results)

    if module is not None:
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
    return PipelineResult(report.render(inputs), inputs, skipped)


def _narrative(inputs: report.ReportInputs) -> dict[str, str]:
    """Narrative text, or the reason there is none.

    Imported lazily and defensively. If graft.agent is missing, broken, or
    simply not configured, the deterministic report must still render, so
    every failure mode here returns text rather than propagating.
    """
    try:
        from graft.agent import agent as agent_module
    except Exception as exc:  # agent layer absent or broken
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
