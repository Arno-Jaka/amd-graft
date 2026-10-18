"""HIPIFY backend tests.

The backend's job is to be trustworthy about what it does and does not
translate. These tests check the mechanical renaming, and more importantly
check that the risky constructs are reported rather than silently passed
through as if they were handled.
"""

from __future__ import annotations

import pytest

from graft import config
from graft.core import converter, detector
from graft.core import hipify_backend as hipify
from graft.core.converter import Verdict

CUDA_SNIPPET = """
#include <cuda_runtime.h>

__global__ void k(const float* in, float* out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = in[i] * 2.0f;
}

int main() {
    float* p = nullptr;
    cudaMalloc(&p, 1024);
    cudaMemset(p, 0, 1024);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { return 1; }
    cudaFree(p);
    return 0;
}
"""




def test_runtime_symbols_are_renamed():
    out = hipify.translate(CUDA_SNIPPET).source
    for cuda in (
        "cudaMalloc",
        "cudaMemset",
        "cudaDeviceSynchronize",
        "cudaGetLastError",
        "cudaSuccess",
        "cudaFree",
    ):
        assert cuda not in out, f"{cuda} survived translation"
    for hip in (
        "hipMalloc",
        "hipMemset",
        "hipDeviceSynchronize",
        "hipGetLastError",
        "hipSuccess",
        "hipFree",
    ):
        assert hip in out


def test_header_is_rewritten():
    out = hipify.translate(CUDA_SNIPPET).source
    assert "hip/hip_runtime.h" in out
    assert "cuda_runtime.h" not in out


def test_execution_model_qualifiers_are_left_alone():
    """HIP keeps CUDA spellings here; renaming would break compilation."""
    out = hipify.translate(CUDA_SNIPPET).source
    assert "__global__" in out
    for builtin in ("threadIdx", "blockIdx", "blockDim"):
        assert builtin in out


def test_kernel_launch_syntax_is_untouched():
    """The <<<>>> launch form is identical in HIP."""
    source = (
        "#include <cuda_runtime.h>\n"
        "__global__ void k(float* p) {}\n"
        "int main(){ k<<<1,1>>>(nullptr); return 0; }\n"
    )
    assert "k<<<1,1>>>" in hipify.translate(source).source


def test_longest_symbol_wins():
    """cudaMemcpy must not be rewritten before cudaMemcpyDeviceToHost."""
    source = "void f(){ cudaMemcpy(a, b, n, cudaMemcpyDeviceToHost); }"
    out = hipify.translate(source).source
    assert "hipMemcpyDeviceToHost" in out
    assert "hipMemcpyHostToDevice" not in out


def test_every_change_is_recorded_with_a_line_number():
    result = hipify.translate(CUDA_SNIPPET)
    assert result.changes
    for change in result.changes:
        assert change["line"] >= 1
        assert change["symbol"]
        assert change["reason"]


def test_translated_source_matches_the_change_log():
    """The returned source must be the edited text, not the original."""
    result = hipify.translate(CUDA_SNIPPET)
    assert result.source != CUDA_SNIPPET
    assert "hipMalloc" in result.source




def test_inline_ptx_is_reported_not_renamed():
    source = (
        "#include <cuda_runtime.h>\n"
        "__global__ void k(float* p) {\n"
        '  asm volatile("mma.sync.aligned.m16n8k8.f32.f32.f32.f32");\n'
        "}\n"
    )
    result = hipify.translate(source)
    assert any(
        "PTX" in u["note"] or "ptx" in u["note"].lower() for u in result.unmapped
    )


def test_warp_size_difference_is_flagged():
    """CUDA warp is 32 lanes, AMD wavefront is 64. Never auto-assumed."""
    source = (
        "#include <cuda_runtime.h>\n"
        "#define WARP_SIZE 32\n"
        "int lane = threadIdx.x % WARP_SIZE;\n"
    )
    result = hipify.translate(source)
    assert any("wavefront" in u["note"] or "64" in u["note"] for u in result.unmapped)


def test_shuffle_is_flagged_for_review():
    result = hipify.translate(
        "__global__ void k(){ float v = __shfl_xor_sync(0xffffffff, 1, 0); }"
    )
    assert result.unmapped
    assert any("shfl" in u["symbol"] or "shuffle" in u["note"] for u in result.unmapped)


def test_unmapped_items_block_production_ready():
    source = "#include <cuda_runtime.h>\n__global__ void k(){__shfl_xor_sync(1,2,0);}\n"
    result = converter.convert(source)
    assert result.coverage.verdict is Verdict.NEEDS_REVIEW
    assert result.coverage.unmapped


def test_ptx_short_circuits_before_any_renaming():
    """Renaming around PTX yields a file that compiles and is wrong."""
    source = (
        "#include <cuda_runtime.h>\n"
        "__global__ void k(float* p){\n"
        "  cudaMalloc(&p, 4);\n"
        '  asm volatile("ld.global.f32 %0,[%1];"::"l"(p):"f");\n'
        "}\n"
    )
    result = converter.convert(source)
    assert result.coverage.verdict is Verdict.MANUAL_PORT
    assert "cudaMalloc" in result.ported_source or not result.ported_source
    assert result.changes == []




def test_cu_file_routes_to_hipify():
    from pathlib import Path

    route = detector.detect(Path("x.cu"), CUDA_SNIPPET)
    assert route.lang is detector.Lang.CUDA_CPP


def test_clean_cuda_file_converts():
    result = converter.convert(CUDA_SNIPPET)
    assert result.backend in ("table", "hipify-perl")
    assert result.coverage.verdict is Verdict.PRODUCTION_READY
    assert result.converted


def test_build_command_is_stated():
    result = converter.convert(CUDA_SNIPPET)
    assert any("hipcc" in note for note in result.notes)


def test_triton_remains_unsupported():
    result = converter.convert("@triton.jit\ndef k(x): pass")
    assert result.coverage.verdict is Verdict.UNSUPPORTED


def test_unknown_source_remains_unsupported():
    result = converter.convert("x = 1 + 2")
    assert result.coverage.verdict is Verdict.UNSUPPORTED




def test_table_is_used_when_hipify_absent(monkeypatch):
    monkeypatch.setattr(hipify, "hipify_perl_path", lambda: None)
    result = hipify.translate(CUDA_SNIPPET)
    assert result.backend == "table"
    assert any("hipify-perl not found" in note for note in result.notes)


def test_missing_hipify_is_never_a_silent_pass(monkeypatch):
    """Falling back must announce itself, not look like a clean run."""
    monkeypatch.setattr(hipify, "hipify_perl_path", lambda: None)
    result = converter.convert(CUDA_SNIPPET)
    assert any("built-in mapping table" in note for note in result.notes)


def test_official_tool_failure_raises(monkeypatch):
    class Failed:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(hipify, "hipify_perl_path", lambda: "/usr/bin/hipify-perl")
    monkeypatch.setattr(hipify.subprocess, "run", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="hipify-perl failed"):
        hipify.translate(CUDA_SNIPPET)


def test_delegation_prefers_official_tool(monkeypatch):
    official = hipify.HipifyResult(
        source="// translated by ROCm hipify",
        backend="hipify-perl",
        notes=["translation produced by hipify-perl, not by Graft's table"],
    )
    monkeypatch.setattr(hipify, "run_hipify_perl", lambda *a, **k: official)
    result = hipify.translate(CUDA_SNIPPET)
    assert result.backend == "hipify-perl"
    assert result.source == official.source
    assert "cudaMalloc" not in result.source


def test_mapping_table_has_no_cuda_leftovers():
    for cuda, hip in hipify.CUDA_TO_HIP.items():
        assert not cuda.startswith("hip"), cuda
        assert hip.startswith("hip"), (cuda, hip)


def test_shipped_example_converts_cleanly():
    """The .cu example in the repo must survive the same path as a user file."""
    example = config.EXAMPLES_DIR / "02_cuda_add_kernel.cu"
    result = converter.convert(example.read_text(), example)
    assert result.coverage.verdict is Verdict.PRODUCTION_READY
    assert "hipMalloc" in result.ported_source
    assert "cudaMalloc" not in result.ported_source
    assert "add_kernel<<<blocks, threads>>>" in result.ported_source
    assert "__shared__" in result.ported_source
    assert "__syncthreads" in result.ported_source
