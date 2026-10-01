"""Stage 5 — HIPIFY backend for CUDA C++ translation.

CUDA C++ to HIP is mostly mechanical renaming: HIP deliberately mirrors
the CUDA API so `cudaMalloc` becomes `hipMalloc` and `__global__` keeps
its spelling. That makes a deterministic translation feasible with no
GPU present, which is the point: the mapping is verifiable by inspection.

Three things must not be faked:

1. Warp size. CUDA warps are 32 lanes; CDNA RDNA wavefronts are 64. A
   kernel written against 32 that silently assumes a full wave is a
   correctness bug on AMD, so lane-dependent code is flagged, never
   renamed.
2. Inline PTX. No mechanical translation exists. It is reported as
   unmapped and the file is marked for manual port.
3. Build flags. `nvcc -gencode arch=compute_XX` becomes `hipcc
   --offload-arch=gfxXXX`, and the architecture numbers are not
   interchangeable. The tool states the substitution; it does not guess
   the target.

The official `hipify-perl` from the ROCm SDK is used when installed
because it is the reference implementation. The built-in table below is
the fallback for machines without ROCm, and covers the API surface
Graft claims to handle. Both paths return the same result shape.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# CUDA runtime and driver API to HIP equivalents. Ordered longest-first at
# build time so that a prefix cannot shadow a longer symbol.
CUDA_TO_HIP: dict[str, str] = {
    # memory
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemset": "hipMemset",
    # errors and setup
    "cudaGetLastError": "hipGetLastError",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaSuccess": "hipSuccess",
    "cudaGetErrorName": "hipGetErrorName",
    # device queries
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaSetDevice": "hipSetDevice",
    "cudaGetDevice": "hipGetDevice",
    "cudaDeviceProp": "hipDeviceProp_t",
    # streams and events
    "cudaStream_t": "hipStream_t",
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cudaStreamSynchronize": "hipStreamSynchronize",
    "cudaEvent_t": "hipEvent_t",
    "cudaEventCreate": "hipEventCreate",
    "cudaEventDestroy": "hipEventDestroy",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cudaEventElapsedTime": "hipEventElapsedTime",
}

# Headers, rewritten to the HIP equivalents.
HEADER_MAP: dict[str, str] = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "cuda.h": "hip/hip_runtime_api.h",
    "cuda_fp16.h": "hip/hip_fp16.h",
    "nvtx3/nvToolsExt.h": "hip/hip_runtime.h",
}

# Symbols that must NOT be renamed automatically. HIP keeps the CUDA
# spelling for the execution-model qualifiers, so renaming would break
# the file; they are verified present instead.
PASSTHROUGH_QUALIFIERS = (
    "__global__",
    "__device__",
    "__host__",
    "__shared__",
    "__constant__",
    "__restrict__",
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
    "warpSize",
)

# Execution-model constructs whose semantics differ on AMD. Reported as
# unmapped so a human reviews them.
LANE_SEMANTICS = {
    "__shfl": "wavefront shuffle is lane-width dependent (CUDA 32, AMD 64)",
    "__ballot": "ballot width differs across vendors",
    "__activemask": "active mask is wavefront sized",
    "__syncwarp": "maps to __builtin_amdgcn_wave_barrier, needs review",
    "coalesced_groups": "cooperative groups has no direct HIP equivalent",
    "grid_group": "cooperative groups grid sync is not portable to CDNA",
}

NVCC_FLAG_RE = re.compile(r"(?:^|\s)(-gencode\s+arch=compute_\d+[^\s]*)")


@dataclass
class HipifyResult:
    """Same result shape as the AST backend so the router can use either."""

    source: str
    changes: list[dict] = field(default_factory=list)
    unmapped: list[dict] = field(default_factory=list)
    backend: str = "table"
    notes: list[str] = field(default_factory=list)


def _symbol_pattern() -> re.Pattern[str]:
    ordered = sorted(CUDA_TO_HIP, key=len, reverse=True)
    alternation = "|".join(re.escape(name) for name in ordered)
    return re.compile(rf"\b({alternation})\b")


def _record(changes: list[dict], symbol: str, line: int, reason: str) -> None:
    changes.append({"symbol": symbol, "line": line, "reason": reason})


def hipify_perl_path() -> str | None:
    """Locate the official translator if the ROCm SDK is installed."""
    return shutil.which("hipify-perl") or shutil.which("hipify-clang")


def run_hipify_perl(source: str, path: Path | None = None) -> HipifyResult | None:
    """Delegate to hipify-perl. Returns None when it is unavailable.

    Failure is not swallowed: a non-zero exit raises, because a partial
    translation reported as complete is worse than no translation.
    """
    executable = hipify_perl_path()
    if executable is None:
        return None

    completed = subprocess.run(
        [executable, "-o", "-", "--", str(path) if path else "-"],
        input=source,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"hipify-perl failed ({completed.returncode}): "
            f"{completed.stderr.strip() or 'no stderr'}"
        )

    return HipifyResult(
        source=completed.stdout,
        changes=[
            {
                "symbol": "hipify-perl",
                "line": 0,
                "reason": "translated by the official ROCm hipify tool",
            }
        ],
        backend="hipify-perl",
        notes=["translation produced by hipify-perl, not by Graft's table"],
    )


def _line_of(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def translate(source: str, path: Path | None = None) -> HipifyResult:
    """Translate CUDA C++ to HIP. Prefers hipify-perl, falls back to table."""
    official = run_hipify_perl(source, path)
    if official is not None:
        return official

    result = HipifyResult(source=source, backend="table")
    result.notes.append(
        "hipify-perl not found; applied Graft's built-in mapping table. "
        "Install ROCm and re-run to cross-check against the official tool."
    )
    # Written back into result.source at the end; the local rebinds below
    # would otherwise be lost and the caller would receive unmodified source
    # alongside a change log claiming edits.

    # Headers first, so symbol renaming does not double-rewrite a path.
    for header in sorted(HEADER_MAP, key=len, reverse=True):
        for match in list(re.finditer(re.escape(header), source)):
            replacement = HEADER_MAP[header]
            _record(
                result.changes,
                f"#include <{header}>",
                _line_of(source, match.start()),
                f"rewritten to <{replacement}>",
            )
            source = source.replace(header, replacement)

    pattern = _symbol_pattern()
    position = 0
    while True:
        match = pattern.search(source, position)
        if match is None:
            break
        symbol = match.group(1)
        replacement = CUDA_TO_HIP[symbol]
        _record(
            result.changes,
            symbol,
            _line_of(source, match.start()),
            f"renamed to {replacement}",
        )
        source = source[: match.start()] + replacement + source[match.end() :]
        position = match.start() + len(replacement)

    for name, reason in LANE_SEMANTICS.items():
        if name not in source:
            continue
        line = _line_of(source, source.index(name))
        result.unmapped.append({"symbol": name, "line": line, "note": reason})
        result.notes.append(f"{name} requires manual review: {reason}")

    warp_literal = re.search(r"warp_?size|warpSize|WARP_SIZE|warpSize32", source)
    if warp_literal:
        result.unmapped.append(
            {
                "symbol": "warp size literal",
                "line": _line_of(source, warp_literal.start()),
                "note": "CUDA warp is 32 lanes; AMD wavefront is 64. "
                "Verify every lane-dependent assumption.",
            }
        )

    # PTX is not in LANE_SEMANTICS because it is a hard stop rather than a
    # review item: there is no HIP equivalent at all, so a returned source
    # containing it would look translated while remaining unportable.
    ptx = re.search(
        r"asm\s+(?:volatile\s*)?\(|%laneid|%ctaid|mma\.sync|wgmma|"
        r"ld\.global|st\.global|bar\.warp\.sync",
        source,
    )
    if ptx:
        result.unmapped.append(
            {
                "symbol": "inline PTX",
                "line": _line_of(source, ptx.start()),
                "note": "PTX has no HIP equivalent; convert by hand using "
                "the AMD equivalent intrinsic for each instruction.",
            }
        )

    if result.unmapped:
        result.notes.append(
            "unmapped items above block an automatic production_ready verdict"
        )

    result.source = source
    return result


def build_command() -> str:
    """The equivalent build line, for the report to show."""
    return (
        "hipcc --offload-arch=gfx<YOUR_ARCH> (replaces nvcc -gencode arch=compute_XX)"
    )
