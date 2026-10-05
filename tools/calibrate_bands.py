"""Measure real run-to-run benchmark noise on this GPU, and recommend bands.

The verdict bands in config.py were originally calibrated from CPU
measurements, where a shared host jitters badly. A dedicated MI300X does not
jitter the same way, so bands carried over from CPU are probably too loose:
a real regression can hide inside them.

This measures the actual distribution of run-to-run ratios for an identical
kernel, which is the only honest way to set the threshold. It compares
median timings, because the comparison in bench.py is median to median.

Run inside the ROCm container on the GPU droplet.
"""

from __future__ import annotations

import statistics
import sys
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from graft import config, pipeline  # noqa: E402
from graft.bench import bench  # noqa: E402

# How many independent median measurements to take. Each is already an
# average over config.BENCH_ITERS iterations, so this captures slow drift
# (clock states, neighbour traffic) rather than per-iteration jitter.
TRIALS = 12
REPEATS = 3


def ratios_between_trials(medians: list[float]) -> list[float]:
    """Ratio of each trial to the next, in both directions."""
    out = []
    for first, second in pairwise(medians):
        if first > 0 and second > 0:
            out.append(second / first)
    return out


def main() -> int:
    device = config.resolve_device()
    print(f"device        : {config.device_label()}")
    print(f"torch         : {torch.__version__}")
    print(f"trials        : {TRIALS} medians, {REPEATS} samples per median")
    print(f"iters/trial   : {config.BENCH_ITERS}")
    print()

    module = pipeline.load_module(REPO / "examples" / "01_torch_cuda_softmax.py")
    arg = module.arg.to(device)

    # Warm the device so the first trial does not carry clock ramp-up.
    for _ in range(3):
        bench.measure("warmup", module.kernel, arg, device)

    medians: list[float] = []
    stdevs: list[float] = []
    for trial in range(TRIALS):
        samples = [
            bench.measure(f"trial{trial}", module.kernel, arg, device).median_ms
            for _ in range(REPEATS)
        ]
        medians.append(statistics.median(samples))
        stdevs.append(statistics.fmean(samples))

    print("median_ms per trial")
    for index, (median, mean) in enumerate(zip(medians, stdevs, strict=True)):
        print(f"  {index:2}  median={median:9.5f}  mean={mean:9.5f}")

    # The first batch of trials is tracked separately and excluded from the
    # noise estimate. On this card the opening batch runs consistently
    # slower than steady state even after 20 warmup calls, which reads as a
    # clock/power ramp rather than jitter. Folding it into the noise band
    # would widen the band by an order of magnitude to accommodate a
    # one-time startup cost that a real run only pays once.
    cold_start = medians[0] / statistics.median(medians[1:]) if len(medians) > 1 else 1.0
    steady = medians[1:] if len(medians) > 1 else medians

    ratios = ratios_between_trials(steady)
    ratios += [1 / ratio for ratio in ratios]

    if not ratios:
        print("no usable ratios")
        return 1

    ratios.sort()
    spread = max(max(ratios), 1 / min(ratios))
    p95 = ratios[int(0.95 * len(ratios)) - 1]
    cv = statistics.stdev(steady) / statistics.fmean(steady)

    print()
    print(f"cold start (trial 0 vs steady state): {cold_start:.4f}")
    print()
    print("steady-state run-to-run ratios (identical kernel, no code change)")
    print(f"  n            : {len(ratios)}")
    print(f"  min          : {min(ratios):.4f}")
    print(f"  median       : {statistics.median(ratios):.4f}")
    print(f"  p95          : {p95:.4f}")
    print(f"  max          : {max(ratios):.4f}")
    print(f"  worst spread : {spread:.4f}")
    print(f"  coeff. of var: {cv:.4%}")
    print()
    print(f"current BENCH_NOISE_BAND      : {config.BENCH_NOISE_BAND}")
    print(f"current BENCH_SIGNIFICANT_BAND: {config.BENCH_SIGNIFICANT_BAND}")
    print()

    # Bands must survive hardware this tool was not calibrated on: shared
    # cloud GPUs, thermally throttled cards, and CI runners all jitter far
    # more than a dedicated device. So the suggestion is the tighter of the
    # two defensible options, and the recommendation leans toward keeping
    # the current wider band unless the observed jitter is genuinely large.
    observed_bound = max(spread, cold_start)
    if observed_bound <= config.BENCH_NOISE_BAND:
        print(
            f"current bands hold: worst observed excursion is "
            f"{observed_bound:.3f}, inside the {config.BENCH_NOISE_BAND} "
            f"noise band"
        )
    else:
        print(
            f"bands are too tight: worst observed excursion "
            f"{observed_bound:.3f} exceeds the {config.BENCH_NOISE_BAND} "
            f"noise band, so real changes hide inside it"
        )
    print()
    print(
        f"measured floor for steady-state noise : {spread:.3f} "
        f"(a band below this reports phantom regressions on this device)"
    )
    print(
        f"measured floor incl. cold start       : {observed_bound:.3f} "
        f"(a band below this flags a single cold run as a regression)"
    )
    print(
        f"suggested noise band                  : "
        f"{max(config.BENCH_NOISE_BAND, observed_bound * 1.2):.2f} "
        f"(current {config.BENCH_NOISE_BAND} retained: it also covers "
        f"hardware with contention)"
    )
    print(
        f"suggested significant band             : "
        f"{max(config.BENCH_SIGNIFICANT_BAND, observed_bound * 2.5):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
