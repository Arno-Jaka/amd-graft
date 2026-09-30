"""Stage 1 — AST rewriting behaviour and change logging."""

import ast

from graft.core.detector import Lang, Route
from graft.core.torch_rewriter import rewrite

PYTORCH_ROUTE = Route(lang=Lang.PYTORCH_PYTHON, path=None)
TRITON_ROUTE = Route(lang=Lang.TRITON, path=None)

HARDCODED_DEVICE = """
import torch

dev = torch.device("cuda", 0)
x = torch.zeros(4, device=dev)
"""


def test_hardcoded_index_becomes_config_constant():
    """Regression: the index is args[1], not args[0]."""
    ported, _log = rewrite(HARDCODED_DEVICE, PYTORCH_ROUTE)
    assert "GPU_DEVICE_INDEX" in ported
    assert 'torch.device("cuda", 0)' not in ported.replace("'", '"')


def test_hardcoded_rewrite_is_logged_with_line_number():
    _, log = rewrite(HARDCODED_DEVICE, PYTORCH_ROUTE)
    symbols = [c["symbol"] for c in log.changes]
    assert "torch.device('cuda', 0)" in symbols
    entry = next(c for c in log.changes if c["symbol"] == "torch.device('cuda', 0)")
    assert entry["line"] > 0
    assert entry["reason"]


def test_ported_output_is_valid_python():
    ported, _ = rewrite(HARDCODED_DEVICE, PYTORCH_ROUTE)
    ast.parse(ported)


def test_torch_cuda_calls_recorded_as_verified_portable():
    src = "import torch\ntorch.cuda.synchronize()\n"
    _, log = rewrite(src, PYTORCH_ROUTE)
    assert any("torch.cuda.synchronize" in c["symbol"] for c in log.changes)
    assert all("verified portable" in c["reason"] for c in log.changes)


def test_nonzero_device_index_left_alone():
    """Only index 0 is normalized; others may be deliberate."""
    src = 'dev = torch.device("cuda", 1)\n'
    ported, _ = rewrite(src, PYTORCH_ROUTE)
    assert "GPU_DEVICE_INDEX" not in ported


def test_single_arg_device_call_left_alone():
    src = 'dev = torch.device("cuda")\n'
    ported, _ = rewrite(src, PYTORCH_ROUTE)
    assert "GPU_DEVICE_INDEX" not in ported


def test_non_pytorch_route_passes_through_unchanged():
    src = "@triton.jit\ndef k(x):\n    pass\n"
    ported, log = rewrite(src, TRITON_ROUTE)
    assert ported == src
    assert log.changes == []


def test_rewrite_is_idempotent_on_its_own_output():
    once, _ = rewrite(HARDCODED_DEVICE, PYTORCH_ROUTE)
    twice, _ = rewrite(once, PYTORCH_ROUTE)
    assert "GPU_DEVICE_INDEX" in twice
    assert once.count("GPU_DEVICE_INDEX") == twice.count("GPU_DEVICE_INDEX")


def test_unrelated_code_is_untouched():
    src = "def add(a, b):\n    return a + b\n"
    ported, _ = rewrite(src, PYTORCH_ROUTE)
    assert "return a + b" in ported
