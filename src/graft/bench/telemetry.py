"""Stage 3 — Hardware telemetry.

GPU name, peak VRAM, and live utilization via rocm-smi. Isolated so the
UI can poll it independently of the migration pipeline.
"""