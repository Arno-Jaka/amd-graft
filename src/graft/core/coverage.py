"""Stage 7 — Coverage and readiness scoring.

The question this module answers is narrow and important: of everything
Graft did mechanically, what still needs a human? A conversion tool that
reports "done" on a file containing inline PTX is worse than useless,
because the failure surfaces later as wrong numbers.

Three severities, because they imply different amounts of work:

    BLOCKER  no mechanical translation exists. A human must rewrite it.
             Verdict cannot exceed MANUAL_PORT.
    REVIEW   a correct translation exists but the semantics are not
             provably equivalent on AMD. Verdict caps at NEEDS_REVIEW.
    INFO     worth knowing, changes nothing. Does not cap the verdict.

Findings carry a line number and a reason. A finding without both is a
finding nobody can act on, and a bare "PTX found" does not tell anyone
which of the seven occurrences is the problem.

The scanner reads the PORTED source, not the original. Checking the
original would re-report what the backend already handled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    BLOCKER = "blocker"
    REVIEW = "review"
    INFO = "info"


# Cap each verdict by the worst severity present. Lower wins.
_SEVERITY_CAP = {
    Severity.BLOCKER: "manual_port",
    Severity.REVIEW: "needs_review",
    Severity.INFO: "production_ready",
}


@dataclass
class Finding:
    """One thing that needs attention, with enough detail to act on."""

    severity: Severity
    category: str
    symbol: str
    line: int
    reason: str
    remediation: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": str(self.severity),
            "category": self.category,
            "symbol": self.symbol,
            "line": self.line,
            "reason": self.reason,
            "note": self.reason,
            "remediation": self.remediation,
        }

    def __repr__(self) -> str:
        return (
            f"Finding({self.severity}, {self.category}, {self.symbol!r}, "
            f"line={self.line})"
        )


# Ordered so that BLOCKER patterns are tested and reported first within a
# category; each entry is independent.
_BLOCKERS: tuple[tuple[str, str, str, str], ...] = (
    (
        "ptx",
        r"asm\s+(?:volatile\s*)?\(",
        "inline PTX",
        "PTX has no HIP equivalent; each instruction needs its AMD intrinsic",
        "rewrite the asm block with HIP intrinsics, or precompile for gfx",
    ),
    (
        "ptx",
        r"%laneid|%ctaid|%tid\b|mma\.sync|wgmma|cvta\.to\.global",
        "PTX special register or instruction",
        "PTX register or instruction syntax has no HIP equivalent",
        "rewrite using HIP intrinsics; mma.sync maps to __builtin_amdgcn_mfma",
    ),
    (
        "cutlass",
        r"#include\s*[<\"]cutlass|cute::",
        "CUTLASS / CuTe",
        "CUTLASS targets NVCC internals and does not build under hipcc",
        "port to rocWMMA or the rocPRIM library, or vendor an AMD CUTLASS fork",
    ),
    (
        "cooperative_groups",
        r"cooperative_groups|grid_group|grid\.sync\(\)",
        "cooperative groups grid sync",
        "grid-wide sync has no portable CDNA equivalent",
        "replace with a custom multi-block barrier or a rocPRIM cooperative kernel",
    ),
)

_REVIEWS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "warp_size",
        r"\bwarp_?size\b|warpSize|WARP_SIZE|warpSize32",
        "hardcoded warp size",
        "CUDA warps are 32 lanes; AMD wavefronts are 64",
        "confirm every lane-dependent index, shuffle mask, and reduction "
        "assumption, or query the wavefront width at runtime",
    ),
    (
        "shuffle",
        r"__shfl_(?:up|down|xor|up_sync|down_sync|xor_sync)\b",
        "warp shuffle intrinsic",
        "shuffle lane masks and widths differ across vendors",
        "swap for __shfl via hip, then verify lane indices for 64-wide waves",
    ),
    (
        "ballot",
        r"__ballot\b|__activemask\b|__syncwarp\b",
        "ballot / activemask / syncwarp",
        "these operate on warp or wavefront width, which differs",
        "verify against __builtin_amdgcn_wave_ballot and wave_barrier",
    ),
    (
        "atomics",
        r"atomic(Add|AddAndFetch|CAS|Exch)\(.*__shared__|__shared__.*atomic",
        "shared-memory atomic",
        "ordering guarantees differ between vendors",
        "confirm the required ordering for the reduction",
    ),
)

_INFOS: tuple[tuple[str, str, str, str], ...] = (
    (
        "texture",
        r"texture<|cudaTexture|cudaBindTexture",
        "texture object",
        "CUDA texture objects have no direct HIP texture equivalent",
        "use hipTextureObject or a linear-memory gather",
    ),
    (
        "nvcc_flags",
        r"-gencode|arch=compute_|--use_fast_math",
        "NVCC build flag",
        "NVCC flags do not carry over to hipcc",
        "translate to --offload-arch=gfx<arch>; check fast-math semantics",
    ),
    (
        "float16_math",
        r"__half|__nv_|float4|__builtin_ia32",
        "vendor-specific intrinsic type",
        "may not exist in the HIP headers",
        "confirm the type is available under hipcc, or add the equivalent",
    ),
)


def _line_of(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def _scan(source: str, rules, severity: Severity) -> list[Finding]:
    """One finding per rule per line, not per regex match.

    Several patterns match multiple times on the same line (a gencode flag
    string carries both -gencode and arch=compute_), and reporting each
    match separately produces several identical findings that a reader
    cannot act on differently. Distinct lines are kept separate, because
    those are distinct places to fix.
    """
    findings: list[Finding] = []
    seen: set[tuple[str, str, int]] = set()
    for category, pattern, symbol, reason, remediation in rules:
        for match in re.finditer(pattern, source):
            line = _line_of(source, match.start())
            key = (category, symbol, line)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                Finding(
                    severity=severity,
                    category=category,
                    symbol=symbol,
                    line=line,
                    reason=reason,
                    remediation=remediation,
                )
            )
    return findings


def scan(source: str) -> list[Finding]:
    """Find everything in ``source`` that a human still has to handle."""
    findings: list[Finding] = []
    findings += _scan(source, _BLOCKERS, Severity.BLOCKER)
    findings += _scan(source, _REVIEWS, Severity.REVIEW)
    findings += _scan(source, _INFOS, Severity.INFO)
    return sorted(findings, key=lambda f: (f.line, str(f.severity)))


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    counts = {str(level): 0 for level in Severity}
    for finding in findings:
        counts[str(finding.severity)] += 1
    return counts


def verdict_for(findings: list[Finding], fallback: str = "production_ready") -> str:
    """Worst severity present, capped. Never better than ``fallback``."""
    worst = fallback
    for finding in findings:
        cap = _SEVERITY_CAP[finding.severity]
        # Verdict ordering: manual_port > needs_review > production_ready.
        if cap == "manual_port" or (cap == "needs_review" and worst != "manual_port"):
            worst = cap
    return worst


@dataclass
class Assessment:
    """Everything the report and the UI need about coverage."""

    findings: list[Finding] = field(default_factory=list)
    verdict: str = "production_ready"
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKER]

    @property
    def reviews(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.REVIEW]

    @property
    def infos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.INFO]

    @property
    def manual_work_required(self) -> bool:
        return bool(self.blockers)

    def summary(self) -> str:
        if not self.findings:
            return "no findings; nothing needs human attention"
        parts = [
            f"{self.counts[str(level)]} {level}"
            for level in Severity
            if self.counts[str(level)]
        ]
        return ", ".join(parts)


def assess(source: str, fallback: str = "production_ready") -> Assessment:
    findings = scan(source)
    return Assessment(
        findings=findings,
        verdict=verdict_for(findings, fallback),
        counts=count_by_severity(findings),
    )
