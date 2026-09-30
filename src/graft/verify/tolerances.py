"""Stage 2 — Per-dtype tolerance table.

Reduction reordering and fused-kernel differences legitimately perturb
trailing bits, so zero tolerance would fail correct code and teach you
nothing. bf16 carries8 mantissa bits, hence the very loose bound.

No other module may hardcode tolerances.
"""

from __future__ import annotations

import torch

TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float64: (1e-9, 1e-12),
    torch.float32: (1e-5, 1e-6),
    torch.float16: (1e-2, 1e-3),
    torch.bfloat16: (5e-2, 1e-2),
}


def tolerance_for(dtype: torch.dtype) -> tuple[float, float]:
    try:
        return TOLERANCES[dtype]
    except KeyError:
        raise KeyError(
            f"no tolerance registered for {dtype};known dtypes: {list(TOLERANCES)}"
        ) from None
