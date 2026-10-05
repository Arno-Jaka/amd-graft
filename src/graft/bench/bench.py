"""Stage 3 — Benchmark harness.

Wall-clock timing of device-resident kernels. Two disciplines matter:

1. Warmup iterations are discarded. The first calls trigger lazy kernel
   compilation and allocator growth; timing them measures one-time setup.
2. The device is synchronized inside the timed region. GPU kernel
   launches are asynchronous, so without a sync the timer stops while the
   GPU is still working and every measurement reads as launch overhead.
"""

from __future__ import annotations

import math
import statistics
import time

import torch

from graft import config

# Verdict thresholds live in config so the report, UI, and this module
# all quote the same numbers. They are calibrated against measured noise:
# identical kernels vary by several percent run to run on shared hardware,
# so a naive 5% band would report regressions that do not exist.
NOISE_BAND = config.BENCH_NOISE_BAND
SIGNIFICANT_BAND = config.BENCH_SIGNIFICANT_BAND


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, arg, device, iters: int) -> list[float]:
    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(arg)
        sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def settle(
    fn,
    arg: torch.Tensor,
    device: torch.device,
    batch: int = config.BENCH_SETTLE_BATCH,
    window: int = config.BENCH_SETTLE_WINDOW,
    tolerance: float = config.BENCH_SETTLE_TOLERANCE,
    max_batches: int = config.BENCH_SETTLE_MAX_BATCHES,
    budget_s: float = config.BENCH_SETTLE_BUDGET_S,
) -> int:
    """Warm the device until timing stops drifting. Returns batches used.

    On an AMD Instinct MI300X VF the card runs roughly 12-14% slower for the
    first couple of thousand consecutive kernel calls while boost clocks
    ramp. Measuring across that ramp makes an unchanged kernel look like a
    double-digit regression.

    Convergence is tested over a sliding *window* of batch medians rather
    than between consecutive pairs, because the ramp is gradual: each
    individual batch looks stable next to the previous one while the
    cumulative drift is large. A window only flattens once the ramp is
    genuinely over, which on this device takes roughly 40 batches.

    Two limits keep this affordable. `max_batches` caps the batch count and
    `budget_s` caps wall-clock time, so a slow kernel cannot spend minutes
    warming up. Hitting either limit means the device never settled, which
    is noted rather than hidden: the remaining drift shows up as variance.
    """
    deadline = time.perf_counter() + budget_s
    recent: list[float] = []
    for index in range(1, max_batches + 1):
        current = statistics.median(_time(fn, arg, device, batch))
        recent.append(current)
        if len(recent) > window:
            recent.pop(0)
        if len(recent) == window and recent[0] > 0:
            spread = (max(recent) - min(recent)) / min(recent)
            if spread <= tolerance:
                return index
        if time.perf_counter() >= deadline:
            break
    return index


def time_kernel(
    fn,
    arg: torch.Tensor,
    device: torch.device,
    warmup: int = config.BENCH_WARMUP_ITERS,
    iters: int = config.BENCH_ITERS,
) -> list[float]:
    for _ in range(warmup):
        fn(arg)
    sync(device)

    # Only for accelerators, and only when the kernel is long enough that
    # a few hundred extra calls are affordable. CPU timing has no clock
    # ramp, so it keeps its exact fixed-warmup behaviour.
    if config.BENCH_SETTLE and device.type != "cpu" and iters:
        settle(fn, arg, device)

    return _time(fn, arg, device, iters)


def _percentile(ordered: list[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


class BenchResult:
    __slots__ = (
        "device",
        "iters",
        "label",
        "max_ms",
        "mean_ms",
        "median_ms",
        "min_ms",
        "p90_ms",
        "stdev_ms",
    )

    def __init__(self, label, device, timings):
        ordered = sorted(timings)
        self.label = label
        self.device = str(device)
        self.iters = len(ordered)
        self.median_ms = statistics.median(ordered)
        self.p90_ms = _percentile(ordered, 0.90)
        self.mean_ms = statistics.fmean(ordered)
        self.min_ms = ordered[0]
        self.max_ms = ordered[-1]
        self.stdev_ms = statistics.stdev(ordered) if len(ordered) > 1 else 0.0

    def __repr__(self) -> str:
        return (
            f"BenchResult({self.label}, {self.device}, "
            f"median={self.median_ms:.4f}ms, p90={self.p90_ms:.4f}ms)"
        )


def measure(
    label: str,
    fn,
    arg: torch.Tensor,
    device: torch.device,
    warmup: int = config.BENCH_WARMUP_ITERS,
    iters: int = config.BENCH_ITERS,
) -> BenchResult:
    timings = time_kernel(fn, arg, device, warmup, iters)
    return BenchResult(label, device, timings)


def compare_port_vs_reference(port: BenchResult, reference: BenchResult) -> dict:
    """Ratio near 1.0 means no regression.

    Scope note: the parity reference (e.g. torch.softmax) is a FUSED
    kernel, while a migrated kernel is usually an eager op sequence.
    Comparing them measures implementation quality, not conversion
    fidelity. It is a true number, but label it accordingly.

    For conversion fidelity specifically, compare port@device against
    the same port running on CPU. For a hardware headline, use
    compare_against_cpu.
    """
    if reference.median_ms <= 0:
        return {"ratio": None, "verdict": "reference too fast to compare"}

    ratio = port.median_ms / reference.median_ms
    if ratio < NOISE_BAND:
        verdict = "within measurement noise"
    elif ratio < SIGNIFICANT_BAND:
        verdict = "minor regression"
    else:
        verdict = "significant regression"
    return {
        "ratio": round(ratio, 3),
        "port_median_ms": port.median_ms,
        "reference_median_ms": reference.median_ms,
        "noise_band": NOISE_BAND,
        "verdict": verdict,
    }


def compare_against_cpu(port: BenchResult, cpu_reference: BenchResult) -> dict:
    """Hardware headline. Not a porting result, and labelled as such."""
    if cpu_reference.median_ms <= 0:
        return {"speedup": None, "verdict": "cpu baseline too fast to compare"}

    speedup = cpu_reference.median_ms / port.median_ms
    return {
        "speedup": round(speedup, 2),
        "device_median_ms": port.median_ms,
        "cpu_median_ms": cpu_reference.median_ms,
        "verdict": f"{speedup:.2f}x faster than the CPU reference",
    }
