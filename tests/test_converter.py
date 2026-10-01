"""Stage 1 — converter routing, verdict logic, and result contract."""

from pathlib import Path

import pytest

from graft.core.converter import Verdict, convert

EXAMPLE = Path("examples/01_torch_cuda_softmax.py")


@pytest.fixture
def example_source() -> str:
    return EXAMPLE.read_text()


def test_example_converts(example_source):
    result = convert(example_source, EXAMPLE)
    assert result.route.lang.value == "pytorch_python"
    assert result.backend == "ast"
    assert result.converted is True


def test_ported_source_is_valid_python(example_source):
    import ast

    ast.parse(convert(example_source, EXAMPLE).ported_source)


def test_example_reaches_production_ready(example_source):
    """The demo example must stay clean, or the demo looks broken."""
    result = convert(example_source, EXAMPLE)
    assert result.coverage.verdict is Verdict.PRODUCTION_READY


def test_changes_are_line_attributed(example_source):
    result = convert(example_source, EXAMPLE)
    assert result.changes
    for change in result.changes:
        assert change["line"] > 0
        assert change["reason"]


def test_trivial_cuda_cpp_converts_without_renames():
    """A .cu file with no CUDA API surface still routes to the HIP backend.

    Superseded the old 'CUDA is unsupported' expectation: that stage is
    implemented, so asserting otherwise would lock in a false limitation.
    """
    result = convert("__global__ void k() {}", Path("m.cu"))
    assert result.backend in ("table", "hipify-perl")
    assert result.converted is True
    assert result.changes == []


def test_triton_routes_to_unsupported():
    src = "import torch\nimport triton\n\n@triton.jit\ndef k(x):\n    pass\n"
    result = convert(src, Path("m.py"))
    assert result.coverage.verdict is Verdict.UNSUPPORTED


def test_unclassifiable_input_is_rejected():
    result = convert("just some prose with no imports", Path("notes.txt"))
    assert result.coverage.verdict is Verdict.UNSUPPORTED
    assert result.converted is False


def test_ptx_downgrades_to_manual_port():
    """Worst case wins: PTX can never be production-ready."""
    src = 'import torch\nkernels = "mma.sync.aligned.m16n8k16"\n'
    assert convert(src, Path("m.py")).coverage.verdict is Verdict.MANUAL_PORT


def test_nvcc_flags_are_reported_but_not_a_blocker():
    """Build flags need translation, not a hand rewrite of the kernel.

    Was asserted as MANUAL_PORT while coverage scoring was still a flag
    check. Under the current severity model an -gencode flag is INFO: the
    source translates cleanly and only the build command changes, so
    capping the verdict at manual_port overstated the work required.
    """
    result = convert(
        'import torch\ncmd = "nvcc -gencode arch=compute_90"\n', Path("m.py")
    )
    assert any("NVCC" in f or "nvcc" in f for f in result.coverage.findings)
    assert result.coverage.verdict is Verdict.PRODUCTION_READY


def test_config_import_prepended_once(example_source):
    ported = convert(example_source, EXAMPLE).ported_source
    assert "from graft.config import" in ported
    assert ported.count("from graft.config import") == 1


def test_convert_is_deterministic(example_source):
    a = convert(example_source, EXAMPLE).ported_source
    b = convert(example_source, EXAMPLE).ported_source
    assert a == b
