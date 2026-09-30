"""Stage 2 — Tolerance table.

Dtype-to-tolerance mapping, fp32 tight / fp16 and bf16 loose, with the
rationale documented: reduction reordering legitimately perturbs trailing
digits. No other module may hardcode tolerances.
"""
