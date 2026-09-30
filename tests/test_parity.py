"""Stage 2 — parity harness behaviour.

The first test is the important one: a harness that has never failed is
indistinguishable from a harness that cannot fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from graft.core.reference import get_reference
from graft.verify import cases, parity
from graft.verify.tolerances import tolerance_for

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "01_torch_cuda_softmax.py"


def _module(solve, operation: str = "softmax"):
    return SimpleNamespace(solve=solve, OPERATION=operation)


@pytest.fixture(scope="module")
def example():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("example_01", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the harness must be capable of failing -------------------------------


def test_harness_catches_a_deliberately_broken_kernel():
    """A kernel that inflates output by 1% must not pass."""
    broken = _module(lambda x: torch.softmax(x, dim=-1) * 1.01)
    results = parity.check(broken, cases.generate())
    assert results, "no cases ran"
    assert not parity.summarize(results)["all_passed"]
    assert all(not r.passed for r in results)


def test_harness_catches_a_sign_flipped_kernel():
    broken = _module(lambda x: -torch.softmax(x, dim=-1))
    results = parity.check(broken, cases.generate())
    assert not parity.summarize(results)["all_passed"]


def test_harness_catches_a_dropped_reduction():
    """Returns only the first row: shape-stable, numerically wrong."""
    broken = _module(lambda x: torch.softmax(x[..., :1], dim=-1))
    results = parity.check(broken, cases.generate())
    assert not parity.summarize(results)["all_passed"]


# --- correct code must pass -----------------------------------------------


def test_exact_match_passes():
    reference = get_reference("softmax")
    results = parity.check(_module(reference), cases.generate()[:5])
    assert parity.summarize(results)["all_passed"]


def test_real_example_passes_every_generated_case(example):
    """The gate CI relies on."""
    results = parity.check(example, cases.generate())
    summary = parity.summarize(results)
    assert summary["all_passed"], [r for r in results if not r.passed]
    assert summary["total"] == len(cases.SHAPES) * len(cases.DISTRIBUTIONS)


def test_real_example_keeps_error_small(example):
    results = parity.check(example, cases.generate())
    summary = parity.summarize(results)
    assert summary["max_abs_error"] < 1e-5


# --- failure modes must surface as failures, not crashes ------------------


def test_shape_mismatch_is_reported_not_raised():
    broken = _module(lambda x: torch.zeros(x.shape[0]))
    results = parity.check(broken, cases.generate()[:1])
    assert len(results) == 1
    assert results[0].passed is False
    assert "shape" in results[0].error


def test_compare_returns_consistent_arity():
    """compare() must return the same tuple length on every path."""
    ok = parity.compare(torch.zeros(4), torch.zeros(4), torch.float32)
    bad = parity.compare(torch.zeros(4), torch.zeros(8), torch.float32)
    assert len(ok) == len(bad)


def test_throwing_module_yields_failed_case_not_exception():
    def boom(x):
        raise RuntimeError("kernel exploded")

    results = parity.check(_module(boom), cases.generate()[:3])
    assert len(results) == 3
    assert all(r.passed is False for r in results)
    assert "kernel exploded" in results[0].error


def test_one_bad_case_does_not_abort_the_run():
    """A failure must not discard the cases that passed."""
    state = {"n": 0}

    def flaky(x):
        state["n"] += 1
        if state["n"] == 2:
            raise ValueError("transient")
        return torch.softmax(x, dim=-1)

    results = parity.check(_module(flaky), cases.generate()[:5])
    assert len(results) == 5
    assert sum(1 for r in results if r.passed) == 4


# --- NaN semantics --------------------------------------------------------


def test_matching_nan_positions_count_as_match():
    """0/0 on both sides is agreement, not divergence."""
    expected = torch.tensor([float("nan"), 1.0])
    actual = torch.tensor([float("nan"), 1.0])
    assert parity.compare(expected, actual, torch.float32)[0] is True


def test_extra_nan_on_the_port_side_fails():
    """NaN only in the output means the port introduced divergence."""
    expected = torch.tensor([1.0, 2.0])
    actual = torch.tensor([1.0, float("nan")])
    assert parity.compare(expected, actual, torch.float32)[0] is False


# --- summarise ------------------------------------------------------------


def test_summarize_reports_counts_and_worst_case(example):
    results = parity.check(example, cases.generate())
    summary = parity.summarize(results)
    assert summary["passed"] == summary["total"] == len(results)
    assert summary["worst_case"] in {c["name"] for c in cases.generate()}
    assert summary["max_abs_error"] is not None


def test_summarize_handles_empty_input():
    summary = parity.summarize([])
    assert summary == {
        "passed": 0,
        "total": 0,
        "all_passed": True,
        "worst_case": None,
        "max_abs_error": None,
    }


# --- case generation ------------------------------------------------------


def test_cases_are_reproducible_across_calls():
    a = cases.generate()
    b = cases.generate()
    assert [c["name"] for c in a] == [c["name"] for c in b]
    assert all(torch.equal(x["input"], y["input"]) for x, y in zip(a, b, strict=False))


def test_cases_cover_degenerate_shapes():
    shapes = {c["shape"] for c in cases.generate()}
    assert (1, 1) in shapes
    assert (3, 1) in shapes
    assert any(not all(d & (d - 1) == 0 for d in s) for s in shapes)


def test_case_dtype_is_honoured():
    for case in cases.generate(dtype=torch.float64):
        assert case["input"].dtype is torch.float64


# --- reference registry and tolerances -----------------------------------


def test_reference_lookup_rejects_unknown_operation():
    with pytest.raises(KeyError, match="no reference registered"):
        get_reference("definitely_not_an_operation")


def test_reference_is_independent_of_the_example(example):
    """Guard against someone 'simplifying' the reference into a copy."""
    reference = get_reference("softmax")
    source = reference.__code__.co_filename
    assert "reference.py" in source
    assert example.solve is not reference


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
def test_tolerances_exist_for_every_supported_dtype(dtype):
    rtol, atol = tolerance_for(dtype)
    assert 0 < atol <= rtol < 1


def test_tolerance_lookup_rejects_unknown_dtype():
    with pytest.raises(KeyError, match="no tolerance registered"):
        tolerance_for(torch.complex64)
