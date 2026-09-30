"""Stage 1 — detector classification and CUDA surface inventory."""

from pathlib import Path

import pytest

from graft.core.detector import Lang, detect

PYTORCH_SRC = """
import torch

def f(x):
    return x.cuda()
"""

TRITON_SRC = """
import torch
import triton

@triton.jit
def k(x):
    pass
"""

CUDA_C_SRC = """
__global__ void k(float* x) { }
"""

CUTLASS_SRC = "#include <cutlass/gemm/device/gemm.h>\n"


def test_detects_pytorch_python():
    route = detect(Path("m.py"), PYTORCH_SRC)
    assert route.lang is Lang.PYTORCH_PYTHON


def test_detects_triton_before_pytorch():
    route = detect(Path("m.py"), TRITON_SRC)
    assert route.lang is Lang.TRITON


def test_detects_cuda_cpp_by_extension():
    route = detect(Path("m.cu"), CUDA_C_SRC)
    assert route.lang is Lang.CUDA_CPP


def test_unknown_when_nothing_matches():
    route = detect(Path("notes.txt"), "just prose, no imports here")
    assert route.lang is Lang.UNKNOWN


def test_triton_file_importing_torch_is_not_pytorch():
    """A Triton kernel imports torch too; order of checks matters."""
    src = "import torch\nimport triton\n"
    assert detect(Path("m.py"), src).lang is Lang.TRITON


def test_cuda_symbols_deduplicated_and_sorted():
    src = "torch.cuda.synchronize()\ntorch.cuda.is_available()\ntorch.cuda.synchronize()\n"
    route = detect(Path("m.py"), src)
    assert route.cuda_symbols == ["is_available", "synchronize"]


def test_ptx_detected():
    route = detect(Path("m.cu"), 'asm volatile("mov.u32 %0, %%laneid;")')
    assert route.has_ptx is True
    assert route.notes


def test_ptx_detected_without_asm_syntax():
    """Regression: PTX is often carried in a plain string, not asm(...)."""
    route = detect(Path("k.py"), 'kernels = "mma.sync.aligned.m16n8k16"')
    assert route.has_ptx is True


def test_ptx_detected_from_special_registers():
    route = detect(Path("m.cu"), "int tid = %ctaid.x;")
    assert route.has_ptx is True


def test_cutlass_detected():
    route = detect(Path("m.cu"), CUTLASS_SRC)
    assert route.has_cutlass is True


def test_nvcc_flags_detected():
    route = detect(Path("m.py"), 'build = "nvcc -gencode arch=compute_90"')
    assert route.has_nvcc is True


def test_warp_literals_only_scanned_for_native_kernels():
    """Scanning Python for 32 would flag every array size: pure noise."""
    py = detect(Path("m.py"), "x = torch.zeros(32)\n")
    cpp = detect(Path("m.cu"), "const int TILE = 64;\n")
    assert py.warp_literals == []
    assert cpp.warp_literals == [64]


def test_clean_file_has_no_notes():
    route = detect(Path("m.py"), PYTORCH_SRC)
    assert route.notes == []


@pytest.mark.parametrize("bad", [None, Path("m.ptx")])
def test_pathless_input_still_classifies(bad):
    assert detect(bad, PYTORCH_SRC).lang is Lang.PYTORCH_PYTHON
