"""Stage 2 — Test input generation.

Fixed seed plus known shapes, plus the cases that expose real bugs:
zeros, negatives, extreme magnitudes, non-power-of-2 dimensions, batch
size 1, and single-element inputs.
"""