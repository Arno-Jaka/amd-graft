"""Idiomatic PyTorch using the torch.cuda.* call surface.

Exercises device-aware tensor creation, an explicit index, synchronization,
and a hand-rolled softmax reduction. The device resolves through
graft.config so this file runs unchanged on a CPU laptop and a ROCm GPU.

Contract:
    solve(x_cpu) -> tensor   entry point, shape-generic, CPU tensor in/out
    kernel(x) -> tensor      device-resident, no transfers, benchmarkable
    OPERATION                key into the reference registry
"""

import torch

from graft.config import resolve_device

OPERATION = "softmax"


def softmax(x):
    """Numerically stable softmax via max subtraction."""
    x_max = x.max(dim=-1, keepdim=True)[0]
    shifted = x - x_max
    e = torch.exp(shifted)
    return e / e.sum(dim=-1, keepdim=True)


def kernel(x):
    """Device-resident entry point. No transfers: safe to benchmark."""
    return softmax(x)


def solve(x_cpu):
    x = x_cpu.to(resolve_device())
    out = kernel(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out.cpu()


def reference(x_cpu):
    return torch.softmax(x_cpu, dim=-1)