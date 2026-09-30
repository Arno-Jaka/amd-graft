"""Stage 1 — Input classification.

Given a dropped file, decide the route: CUDA C++, PyTorch Python, or
Triton. Records the detected language plus an inventory of CUDA-specific
surface that downstream coverage scoring consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Lang(str, Enum):
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
PTX_RE = re.compile(r"\.ptx\b|asm\s*\(\s*\"")
CUTLASS_RE = re.compile(r"cutlass|cute::", re.IGNORECASE)
NVCC_RE = re.compile(r"nvcc|-gencode|arch=compute", re.IGNORECASE)
TRITON_RE = re.compile(r"@triton\.jit|import\s+triton")
WARP_LITERAL_RE = re.compile(r"\b(32|64)\b")

CPP_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".hpp", ".h"}


def detect(path: Path | None, source: str) -> Route:
    route = Route(lang=Lang.UNKNOWN, path=path)

    if path is not None and path.suffix.lower() in CPP_SUFFIXES:
        route.lang = Lang.CUDA_CPP
    elif TRITON_RE.search(source):
        route.lang = Lang.TRITON
    elif "torch" in source:
        route.lang = Lang.PYTORCH_PYTHON

    route.uses_cuda_api = "torch.cuda" in source
    route.cuda_symbols = sorted(set(CUDA_API_RE.findall(source)))
    route.has_ptx = bool(PTX_RE.search(source))
    route.has_cutlass = bool(CUTLASS_RE.search(source))
    route.has_nvcc = bool(NVCC_RE.search(source))

    if route.lang in (Lang.CUDA_CPP, Lang.TRITON):
        route.warp_literals = sorted(set(WARP_LITERAL_RE.findall(source)))

    if route.has_ptx:
        route.notes.append("inline PTX detected: no mechanical translation exists")
    if route.has_cutlass:
        route.notes.append("CUTLASS include: requires manual port")
    if route.has_nvcc:
        route.notes.append("NVCC build flags: replace with hipcc")

    return route