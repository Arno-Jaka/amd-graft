"""Stage 1 — Input classification.

Given a dropped file, decide the route: CUDA C++, PyTorch Python, or
Triton. Records the detected language plus an inventory of CUDA-specific
surface that downstream coverage scoring consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Lang(StrEnum):
    PYTORCH_PYTHON = "pytorch_python"
    TRITON = "triton"
    CUDA_CPP = "cuda_cpp"
    UNKNOWN = "unknown"


@dataclass
class Route:
    lang: Lang
    path: Path | None
    uses_cuda_api: bool = False
    cuda_symbols: list[str] = field(default_factory=list)
    has_ptx: bool = False
    has_cutlass: bool = False
    has_nvcc: bool = False
    warp_literals: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


CUDA_API_RE = re.compile(r"torch\.cuda\.(\w+)")
# PTX is identified by instruction mnemonics and special registers, not by
# the asm( syntax: real CUDA emits asm volatile("..."), and PyTorch kernels
# carry PTX in plain strings. Both must be caught.
PTX_RE = re.compile(
    r"asm\s+(?:volatile\s*)?\(|"
    r"%laneid|%ctaid|%tid|"
    r"mma\.sync|wgmma|"
    r"ld\.global|st\.global|bar\.warp\.sync|cvta\.to\.global"
)
CUTLASS_RE = re.compile(r"cutlass|cute::", re.IGNORECASE)
NVCC_RE = re.compile(r"nvcc|-gencode|arch=compute", re.IGNORECASE)
TRITON_RE = re.compile(r"@triton\.jit|import\s+triton")
WARP_LITERAL_RE = re.compile(r"\b(32|64)\b")

CPP_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".hpp", ".h"}


CUDA_CPP_MARKERS = (
    "__global__",
    "__device__",
    "<<<",
    "cuda_runtime.h",
    "cudaMalloc",
    "blockIdx",
)


def detect(path: Path | None, source: str) -> Route:
    route = Route(lang=Lang.UNKNOWN, path=path)

    suffix_is_cpp = path is not None and path.suffix.lower() in CPP_SUFFIXES
    content_is_cpp = any(marker in source for marker in CUDA_CPP_MARKERS)

    # Extension is not sufficient on its own: pasted snippets arrive with no
    # path, and a .py wrapper can still contain raw CUDA. Content wins when
    # the two disagree, because compiling the wrong route wastes more time
    # than a misclassification here.
    if (suffix_is_cpp and not TRITON_RE.search(source)) or content_is_cpp:
        route.lang = Lang.CUDA_CPP
    elif TRITON_RE.search(source):
        route.lang = Lang.TRITON
    elif "torch" in source:
        route.lang = Lang.PYTORCH_PYTHON
    elif suffix_is_cpp:
        route.lang = Lang.CUDA_CPP

    route.uses_cuda_api = "torch.cuda" in source
    route.cuda_symbols = sorted(set(CUDA_API_RE.findall(source)))
    route.has_ptx = bool(PTX_RE.search(source))
    route.has_cutlass = bool(CUTLASS_RE.search(source))
    route.has_nvcc = bool(NVCC_RE.search(source))

    if route.lang in (Lang.CUDA_CPP, Lang.TRITON):
        route.warp_literals = sorted({int(m) for m in WARP_LITERAL_RE.findall(source)})

    if route.has_ptx:
        route.notes.append("inline PTX detected: no mechanical translation exists")
    if route.has_cutlass:
        route.notes.append("CUTLASS include: requires manual port")
    if route.has_nvcc:
        route.notes.append("NVCC build flags: replace with hipcc")

    return route
