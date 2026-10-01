"""Central configuration.

Single source of truth for paths, model settings, and device resolution.
Every other module reads from here instead of hardcoding literals.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
PROJECT_NAME = "Graft"

EXAMPLES_DIR = PROJECT_ROOT / "examples"

# LLM endpoint, typically vLLM on the AMD instance. Blank values mean the
# agent layer is disabled, which must leave the product fully working; see
# agent/agent.py for why that is the required behaviour rather than a
# convenience.
LLM_MODEL = ""
LLM_BASE_URL = ""
LLM_API_KEY_ENV = "VLLM_API_KEY"

# Placeholder key for a local vLLM server, which usually runs without auth.
# Agno raises rather than defaulting, so one is supplied here. This is not
# a credential and grants nothing.
LLM_PLACEHOLDER_API_KEY = "local"

AGENT_ENABLED = True
AGENT_MAX_TOOL_CALLS = 8

SUBPROCESS_TIMEOUT_S = 120
BENCH_WARMUP_ITERS = 5
BENCH_ITERS = 50

# Verdict bands for benchmark comparison. Calibrated against measured
# noise, not intuition: identical kernels vary by several percent run to
# run on shared hardware. Values under BENCH_NOISE_BAND are reported as
# noise rather than as a pass. Tighten once benchmarking on dedicated AMD
# hardware, where jitter is much lower.
BENCH_NOISE_BAND = 1.10
BENCH_SIGNIFICANT_BAND = 1.25

GPU_NAME_FALLBACK = "CPU"
GPU_DEVICE_INDEX = 0
GPU_DEVICE_NAME = "amd"

PREFER_GPU = True


def resolve_device():
    """Return the best available torch device.

    Prefers the AMD GPU when one is present and falls back to CPU, so the
    same converted code runs unchanged on a laptop and on a ROCm instance.
    ROCm builds expose themselves through torch.cuda, so one check covers
    both vendors.
    """
    import torch

    if PREFER_GPU and torch.cuda.is_available():
        return torch.device("cuda", GPU_DEVICE_INDEX)
    return torch.device("cpu")


def device_label() -> str:
    """Human-readable device name for telemetry and reports."""
    import torch

    if not torch.cuda.is_available():
        return GPU_NAME_FALLBACK
    props = torch.cuda.get_device_properties(GPU_DEVICE_INDEX)
    return getattr(props, "name", None) or GPU_DEVICE_NAME
