"""Stage 1 — Conversion router and result shape.

Picks a backend from the detected route, collects the change log, scores
coverage, and returns the MigrationResult that every downstream stage
consumes. This module defines the project's internal data contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from . import coverage, detector, hipify_backend, torch_rewriter
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
    """Score the ported source and set the verdict from real findings.

    Scans the ported text rather than the route flags. The route records
    what was in the INPUT; the question here is what survived INTO the
    output, and those differ once a backend has rewritten things.
    """
    if not result.ported_source:
        return

    assessment = coverage.assess(result.ported_source)
    result.coverage.findings = [
        finding.reason for finding in assessment.findings
    ] + result.coverage.findings
    result.coverage.unmapped = [
        finding.to_dict() for finding in assessment.findings
    ] or result.coverage.unmapped
    result.coverage.verdict = Verdict(assessment.verdict)


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
    # Backend findings are merged, not replaced. The scanner re-reads the
    # ported source and finds most of the same things, but the backend knows
    # intent that a regex cannot recover; dropping its notes would lose the
    # reason a construct was flagged. _apply_verdict replaces unmapped with
    # the scanner output, which carries the same symbols at higher fidelity.
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
    result.coverage.findings = route.notes

    # The GPU_DEVICE_INDEX import is ensured by the rewriter, on the AST,
    # where it can see existing imports and respect __future__ placement.

    _apply_verdict(result)
    return result
