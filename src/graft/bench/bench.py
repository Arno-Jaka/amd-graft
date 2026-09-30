"""Stage 3 — Benchmark harness.

Warmup iterations followed by N timed runs, reporting median and p90 so
tail spikes stay visible. Comparison is port versus reference on the same
device; never framed against CUDA, which is unavailable here.
"""