"""Stage 2 — Ground-truth reference implementations.

Each entry is deliberately an INDEPENDENT implementation of the same math
from the example it checks. If both sides shared code, parity would be
tautological. torch.softmax is a separate fused kernel from the example's
hand-rolled max/exp/div sequence, so agreement between them is evidence.
"""

from __future__ import annotations

import torch


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _layernorm(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + 1e-5)


REFERENCES: dict[str, object] = {
    "softmax": _softmax,
    "layernorm": _layernorm,
}


def get_reference(operation: str):
    try:
        return REFERENCES[operation]
    except KeyError:
        raise KeyError(
            f"no reference registered for {operation!r};"
            f"known operations: {sorted(REFERENCES)}"
        ) from None


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
