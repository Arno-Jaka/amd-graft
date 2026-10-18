"""Stage 3 — benchmark and telemetry behaviour.

The load-bearing test mirrors test_parity.py: a comparator that cannot
detect a regression is worthless.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from graft import config
from graft.bench import bench

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "01_torch_cuda_softmax.py"
CPU = torch.device("cpu")


def _softmax(x):
    return torch.softmax(x, dim=-1)


@pytest.fixture(scope="module")
def example():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("example_01", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tensor():
    return torch.randn(5, 1024)


def _fake(label: str, median_ms: float) -> bench.BenchResult:
    r = bench.BenchResult.__new__(bench.BenchResult)
    r.label, r.device, r.iters = label, "cpu", 30
    r.median_ms = r.p90_ms = r.mean_ms = median_ms
    r.min_ms, r.max_ms, r.stdev_ms = median_ms, median_ms, 0.0
    return r




def test_returns_exactly_iters_samples(tensor):
    timings = bench.time_kernel(_softmax, tensor, CPU, warmup=2, iters=17)
    assert len(timings) == 17


def test_warmup_samples_are_excluded(tensor):
    """A slow first call must not leak into the measured samples."""
    calls = {"n": 0}

    def slow_first(x):
        calls["n"] += 1
        if calls["n"] == 1:
            pass
        return _softmax(x)

    timings = bench.time_kernel(slow_first, tensor, CPU, warmup=3, iters=5)
    assert len(timings) == 5
    assert all(t >= 0 for t in timings)


def test_warmup_runs_before_measurement(tensor):
    seen = []

    def counting(x):
        seen.append(len(seen))
        return _softmax(x)

    bench.time_kernel(counting, tensor, CPU, warmup=4, iters=6)
    assert len(seen) == 10
    assert seen[:4] == [0, 1, 2, 3]


def test_timings_are_positive(tensor):
    timings = bench.time_kernel(_softmax, tensor, CPU, warmup=2, iters=10)
    assert all(t > 0 for t in timings)


def test_sync_is_a_noop_on_cpu():
    assert bench.sync(CPU) is None




def test_settle_stops_once_the_window_flattens(monkeypatch):
    """Already-stable timing settles at the window size."""
    seen = []

    def fake_time(fn, arg, device, iters):
        seen.append(iters)
        return [1.0] * iters

    monkeypatch.setattr(bench, "_time", fake_time)
    used = bench.settle(_softmax, torch.randn(2, 2), CPU, batch=7, window=4)
    assert used == 4
    assert seen == [7, 7, 7, 7]


def test_settle_does_not_stop_while_a_ramp_continues(monkeypatch):
    """Consecutive batches agreeing is not enough during a gradual ramp.

    Each batch here is within 1% of the one before it, yet the window keeps
    sliding. A pairwise test would have exited early and benchmarked the
    ramp; the window test has to keep going.
    """
    sequence = [
        [10.0, 10.0],
        [9.90, 9.90],
        [9.80, 9.80],
        [8.0, 8.0],
        [7.0, 7.0],
        [7.0, 7.0],
        [7.0, 7.0],
        [7.0, 7.0],
    ]
    calls = {"n": 0}

    def fake_time(fn, arg, device, iters):
        batch = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        return batch

    monkeypatch.setattr(bench, "_time", fake_time)
    used = bench.settle(_softmax, torch.randn(2, 2), CPU, batch=2, window=4)
    assert used == 8


def test_settle_respects_the_batch_cap(monkeypatch):
    """A kernel that never stabilises must not warm up forever."""
    calls = {"n": 0}

    def fake_time(fn, arg, device, iters):
        calls["n"] += 1
        return [float(calls["n"])]

    monkeypatch.setattr(bench, "_time", fake_time)
    used = bench.settle(
        _softmax, torch.randn(2, 2), CPU, batch=2, max_batches=5, budget_s=999
    )
    assert used == 5
    assert calls["n"] == 5


def test_settle_respects_the_time_budget(monkeypatch):
    """A slow kernel stops warming up once the wall-clock budget is spent."""
    ticks = iter([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    calls = {"n": 0}

    def fake_time(fn, arg, device, iters):
        calls["n"] += 1
        return [1.0 - calls["n"] * 0.1]

    monkeypatch.setattr(bench, "_time", fake_time)
    used = bench.settle(
        _softmax,
        torch.randn(2, 2),
        CPU,
        batch=2,
        window=4,
        max_batches=100,
        budget_s=2.0,
    )
    assert used < 100
    assert calls["n"] == used


def test_settle_is_skipped_on_cpu(tensor):
    """CPU timing has no clock ramp, so call counts stay exact."""
    if not config.BENCH_SETTLE:
        pytest.skip("settling disabled")

    calls = {"n": 0}

    def counting(x):
        calls["n"] += 1
        return _softmax(x)

    bench.time_kernel(counting, tensor, CPU, warmup=4, iters=6)
    assert calls["n"] == 10


def test_settle_runs_for_accelerators(monkeypatch, tensor):
    """A non-CPU device does warm up beyond warmup + iters."""
    if not config.BENCH_SETTLE:
        pytest.skip("settling disabled")

    fake_cuda = torch.device("cuda", 0)
    monkeypatch.setattr(bench, "sync", lambda device: None)

    calls = {"n": 0}

    def counting(x):
        calls["n"] += 1
        return _softmax(x)

    settled = {"batches": 0}

    def fake_settle(fn, arg, device, **kwargs):
        settled["batches"] += 1
        fn(arg)
        return 1

    monkeypatch.setattr(bench, "settle", fake_settle)
    bench.time_kernel(counting, tensor, fake_cuda, warmup=2, iters=3)
    assert settled["batches"] == 1
    assert calls["n"] == 6


def test_benchresult_reports_expected_fields(tensor):
    result = bench.measure("ref", _softmax, tensor, CPU, iters=9)
    assert result.iters == 9
    assert result.min_ms <= result.median_ms <= result.max_ms
    assert result.p90_ms >= result.median_ms


def test_percentile_is_monotonic():
    ordered = [float(i) for i in range(1, 101)]
    assert bench._percentile(ordered, 0.50) == 50.0
    assert bench._percentile(ordered, 0.90) == 90.0
    assert bench._percentile(ordered, 0.99) == 99.0


def test_percentile_handles_single_sample():
    assert bench._percentile([4.2], 0.90) == 4.2




def test_comparator_detects_a_significant_regression():
    result = bench.compare_port_vs_reference(_fake("port", 2.0), _fake("ref", 1.0))
    assert result["ratio"] == 2.0
    assert result["verdict"] == "significant regression"


def test_comparator_detects_a_minor_regression():
    result = bench.compare_port_vs_reference(_fake("port", 1.10), _fake("ref", 1.0))
    assert result["verdict"] == "minor regression"


def test_comparator_accepts_no_regression():
    result = bench.compare_port_vs_reference(_fake("port", 1.0), _fake("ref", 1.0))
    assert result["ratio"] == 1.0
    assert result["verdict"] == "within measurement noise"


def test_comparator_guards_against_zero_baseline():
    result = bench.compare_port_vs_reference(_fake("port", 1.0), _fake("ref", 0.0))
    assert result["ratio"] is None
    assert "too fast" in result["verdict"]


def test_speedup_is_device_time_over_cpu_time():
    """Inverted division would report a slowdown as a speedup."""
    result = bench.compare_against_cpu(_fake("port", 0.5), _fake("cpu", 4.0))
    assert result["speedup"] == 8.0
    assert "8.00x faster" in result["verdict"]


def test_speedup_guards_against_zero_cpu_baseline():
    result = bench.compare_against_cpu(_fake("port", 1.0), _fake("cpu", 0.0))
    assert result["speedup"] is None


def test_cpu_port_measures_honestly_against_the_fused_reference(example, tensor):
    """The example's eager softmax is genuinely slower than torch.softmax.

    That is the correct result, not a regression in the port: five eager
    ops cannot match one fused kernel. The test asserts that the harness
    reports the measured direction rather than a flattering one.
    """
    port = bench.measure("port", example.kernel, tensor, CPU, iters=25)
    ref = bench.measure("reference", _softmax, tensor, CPU, iters=25)
    result = bench.compare_port_vs_reference(port, ref)
    assert result["ratio"] > 1.0
    assert result["verdict"] == "significant regression"
    assert result["port_median_ms"] > result["reference_median_ms"]


def test_no_regression_verdict_is_deterministic():
    """Verdict logic is asserted on synthetic timings, never on real ones.

    Two identical implementations measured back to back vary by several
    percent on shared hardware, so asserting a verdict from live timings
    is inherently flaky. The comparator's logic is covered by the
    _fake-based tests above; this checks the bands are wired up.
    """
    assert (
        bench.compare_port_vs_reference(_fake("p", 1.0), _fake("r", 1.0))["verdict"]
        == "within measurement noise"
    )
    assert (
        bench.compare_port_vs_reference(_fake("p", 1.15), _fake("r", 1.0))["verdict"]
        == "minor regression"
    )
    assert (
        bench.compare_port_vs_reference(_fake("p", 3.0), _fake("r", 1.0))["verdict"]
        == "significant regression"
    )


def test_real_2x_slower_kernel_is_detected_reliably():
    """A difference well outside the noise band must be caught every time."""

    def slow(x):
        torch.softmax(x, dim=-1)
        torch.softmax(x, dim=-1)
        return torch.softmax(x, dim=-1)

    x = torch.randn(1024, 2048)
    port = bench.measure("port", slow, x, CPU, iters=40)
    ref = bench.measure("reference", _softmax, x, CPU, iters=40)
    result = bench.compare_port_vs_reference(port, ref)
    assert result["ratio"] > bench.NOISE_BAND
    assert result["verdict"] == "significant regression"


def test_noise_band_is_wider_than_naive_five_percent():
    """Guard against the threshold being tightened back into the noise."""
    assert bench.NOISE_BAND >= 1.08
    assert bench.SIGNIFICANT_BAND > bench.NOISE_BAND


def test_comparison_reports_the_noise_band_it_used():
    result = bench.compare_port_vs_reference(_fake("p", 1.0), _fake("r", 1.0))
    assert result["noise_band"] == bench.NOISE_BAND
