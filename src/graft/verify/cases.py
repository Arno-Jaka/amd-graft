"""Stage 2 — Test input generation.

Shapes cover power-of-two, non-power-of-two, degenerate and large.
Distributions cover the numerical edges: zeros, constants, extreme
magnitudes, all-negative.
"""

from __future__ import annotations

import torch

SEED = 0

SHAPES = [
    (1, 1),
    (4, 8),
    (7, 33),
    (16, 128),
    (1, 1024),
    (2, 3, 65),
    (3, 1),
    (5, 4096),
]


def _make_normal(shape, gen):
    return torch.randn(*shape, generator=gen)


def _make_zeros(shape, gen):
    return torch.zeros(*shape)


def _make_constant(shape, gen):
    return torch.full(shape, 3.0)


def _make_large(shape, gen):
    return torch.randn(*shape, generator=gen) * 1e4


def _make_negative(shape, gen):
    return -torch.rand(*shape, generator=gen) * 1e3


DISTRIBUTIONS = {
    "normal": _make_normal,
    "zeros": _make_zeros,
    "constant": _make_constant,
    "large": _make_large,
    "negative": _make_negative,
}


def generate(dtype: torch.dtype = torch.float32, seed: int = SEED) -> list[dict]:
    cases = []
    for shape in SHAPES:
        for dist_name, make in DISTRIBUTIONS.items():
            gen = torch.Generator().manual_seed(seed)
            tensor = make(shape, gen).to(dtype)
            cases.append(
                {
                    "name": f"{dist_name}/{shape}",
                    "shape": shape,
                    "dtype": dtype,
                    "input": tensor,
                }
            )
    return cases
