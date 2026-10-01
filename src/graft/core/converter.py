"""Stage 1 — Conversion router and result shape.

Picks a backend from the detected route, collects the change log, scores
coverage, and returns the MigrationResult that every downstream stage
consumes. This module defines the project's internal data contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from . import detector, hipify_backend, torch_rewriter
from .detector import Route


class Verdict(StrEnum):
    PRODUCTION_READY = "production_ready"
    NEEDS_REVIEW = "needs_review"
    MANUAL_PORT = "manual_port"
    UNSUPPORTED = "unsupported"


@dataclass
class CoverageReport:
    verdict: Verdict = Verdict.NEEDS_REVIEW
    findings: list[str] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)


@dataclass
class MigrationResult:
    source_path: Path | None = None
    original_source: str = ""
    ported_source: str = ""
    route: Route | None = None
    backend: str = "none"
    changes: list[dict] = field(default_factory=list)
    coverage: CoverageReport = field(default_factory=CoverageReport)
    notes: list[str] = field(default_factory=list)

    @property
    def converted(self) -> bool:
        return self.backend != "none"


def _apply_verdict(result: MigrationResult) -> None:
    route = result.route
    if route is None:
        return
    if route.has_ptx or route.has_cutlass or route.has_nvcc:
        result.coverage.verdict = Verdict.MANUAL_PORT
    elif result.coverage.unmapped:
        result.coverage.verdict = Verdict.NEEDS_REVIEW
    else:
        result.coverage.verdict = Verdict.PRODUCTION_READY


def _convert_cuda_cpp(result: MigrationResult, route: Route) -> MigrationResult:
    """CUDA C++ to HIP via the hipify backend.

    Inline PTX has no mechanical translation, so it short-circuits to
    MANUAL_PORT before any renaming happens. Renaming around PTX would
    produce a file that compiles and is still wrong.
    """
    if route.has_ptx:
        result.coverage.verdict = Verdict.MANUAL_PORT
        result.notes.append(
            "inline PTX detected: no mechanical translation to HIP exists"
        )
        return result

    hipified = hipify_backend.translate(result.original_source, result.source_path)
    result.ported_source = hipified.source
    result.backend = hipified.backend
    result.changes = hipified.changes
    result.coverage.unmapped = hipified.unmapped
    result.coverage.findings = route.notes
    result.notes = [*hipified.notes, *result.notes]
    result.notes.append(f"build with: {hipify_backend.build_command()}")

    _apply_verdict(result)
    return result


def convert(source: str, path: Path | None = None) -> MigrationResult:
    route = detector.detect(path, source)
    result = MigrationResult(source_path=path, original_source=source, route=route)

    if route.lang is detector.Lang.TRITON:
        result.coverage.verdict = Verdict.UNSUPPORTED
        result.notes.append(
            "Triton conversion is not supported: AMD needs the "
            "triton-windows or ROCm-native path, which is a separate port"
        )
        return result

    if route.lang is detector.Lang.CUDA_CPP:
        return _convert_cuda_cpp(result, route)

    if route.lang is detector.Lang.UNKNOWN:
        result.coverage.verdict = Verdict.UNSUPPORTED
        result.notes.append("input could not be classified as PyTorch source")
        return result

    ported, log = torch_rewriter.rewrite(source, route)
    result.ported_source = ported
    result.backend = "ast"
    result.changes = log.changes
    result.coverage.unmapped = log.unmapped
    result.coverage.findings = route.notes

    if "GPU_DEVICE_INDEX" in ported and "from graft.config import" not in ported:
        ported = "from graft.config import GPU_DEVICE_INDEX\n\n" + ported
        result.ported_source = ported

    _apply_verdict(result)
    return result
