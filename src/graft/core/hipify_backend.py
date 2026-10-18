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

import difflib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CUDA_TO_HIP: dict[str, str] = {
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemset": "hipMemset",
    "cudaGetLastError": "hipGetLastError",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaSuccess": "hipSuccess",
    "cudaGetErrorName": "hipGetErrorName",
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaSetDevice": "hipSetDevice",
    "cudaGetDevice": "hipGetDevice",
    "cudaDeviceProp": "hipDeviceProp_t",
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

HEADER_MAP: dict[str, str] = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "cuda.h": "hip/hip_runtime_api.h",
    "cuda_fp16.h": "hip/hip_fp16.h",
    "nvtx3/nvToolsExt.h": "hip/hip_runtime.h",
}

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

    Three things this gets right that are easy to get wrong, all of them
    found by running the real tool on a ROCm machine rather than by reading
    its documentation:

    1. hipify-perl takes a *file*, not stdin, and decides how to translate
       from the extension. The source is staged in a temporary file that
       keeps the original suffix, because a `.cu` staged as `.txt` is not
       translated at all.
    2. Translated source goes to stdout by *default*. `-o` names an output
       file, so the intuitive `-o -` makes hipify-perl write to a file
       literally called `-` and return empty stdout with exit status 0.
       That failure is invisible unless the output is checked.
    3. Empty output is an error, never a successful translation. An empty
       "translation" would otherwise be reported as a completed port, with
       the downstream stages declining to scan it.

    Raises RuntimeError on a non-zero exit or on empty output: a partial or
    absent translation reported as complete is worse than no translation.
    """
    executable = hipify_perl_path()
    if executable is None:
        return None

    suffix = path.suffix if path is not None and path.suffix else ".cu"
    with tempfile.TemporaryDirectory(prefix="graft-hipify-") as tmp:
        staged = Path(tmp) / f"input{suffix}"
        staged.write_text(source, encoding="utf-8")
        completed = subprocess.run(
            [executable, str(staged)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"hipify-perl failed ({completed.returncode}): "
            f"{completed.stderr.strip() or 'no stderr'}"
        )

    translated = completed.stdout
    if not translated.strip():
        raise RuntimeError(
            "hipify-perl exited successfully but produced no output. "
            f"stderr: {completed.stderr.strip() or '(empty)'}. "
            "Refusing to report an empty translation as a successful port."
        )

    return HipifyResult(
        source=translated,
        changes=_diff_changes(source, translated),
        backend="hipify-perl",
        notes=[
            "translation produced by hipify-perl, not by Graft's table",
            f"hipify-perl warnings: {completed.stderr.strip() or 'none'}",
        ],
    )


MAX_REPORTED_CHANGES = 60


def _identifier_positions(source: str) -> list[tuple[str, int]]:
    """Every C/C++ identifier with its offset, for line attribution."""
    return [
        (match.group(0), match.start())
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", source)
    ]


def _diff_changes(before: str, after: str) -> list[dict]:
    """Describe what hipify-perl actually renamed.

    hipify-perl reports no per-symbol detail, so the change list is recovered
    by diffing identifiers. Reporting the tool's existence as a single opaque
    change would hide the one thing a reviewer needs: which CUDA symbol
    became which HIP symbol, and where.
    """
    old = _identifier_positions(before)
    new = _identifier_positions(after)
    old_names = [name for name, _ in old]
    new_names = [name for name, _ in new]

    matcher = difflib.SequenceMatcher(None, old_names, new_names, autojunk=False)
    new_name_set = set(new_names)
    old_name_set = set(old_names)

    changes: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        left = old[i1:i2]
        right = new[j1:j2]

        if len(left) == len(right):
            for (name, offset), (replacement, _) in zip(left, right, strict=True):
                if name != replacement:
                    changes.append(
                        {
                            "symbol": name,
                            "line": _line_of(before, offset),
                            "reason": f"renamed to {replacement}",
                        }
                    )
            continue

        for (name, offset), (replacement, _) in zip(left, right, strict=False):
            if name != replacement:
                changes.append(
                    {
                        "symbol": name,
                        "line": _line_of(before, offset),
                        "reason": f"renamed to {replacement}",
                    }
                )
        for name, offset in left[len(right) :]:
            if name not in new_name_set:
                changes.append(
                    {
                        "symbol": name,
                        "line": _line_of(before, offset),
                        "reason": "removed by hipify-perl",
                    }
                )
        for name, offset in right[len(left) :]:
            if name not in old_name_set:
                changes.append(
                    {
                        "symbol": name,
                        "line": _line_of(before, offset),
                        "reason": f"added by hipify-perl (was {name[len('hip') :]})"
                        if name.startswith("hip")
                        else "added by hipify-perl",
                    }
                )

    if len(changes) > MAX_REPORTED_CHANGES:
        hidden = len(changes) - MAX_REPORTED_CHANGES
        changes = changes[:MAX_REPORTED_CHANGES]
        changes.append(
            {"symbol": "…", "line": 0, "reason": f"{hidden} further changes not listed"}
        )
    return changes


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
