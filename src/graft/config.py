"""Central configuration.

Single source of truth for paths, model settings, and device resolution.
Every other module reads from here instead of hardcoding literals.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
PROJECT_NAME = "Graft"

EXAMPLES_DIR = PROJECT_ROOT / "examples"

DOTENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path = DOTENV_PATH) -> None:
    """Populate os.environ from a .env file, without overriding the shell.

    A dependency-free loader rather than python-dotenv: the file holds at
    most a handful of settings, and the agent layer must stay optional.
    Real environment variables win, so exporting a value in the shell still
    overrides the file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv()


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# LLM endpoint, typically vLLM on the AMD instance. Blank values mean the
# agent layer is disabled, which must leave the product fully working; see
# agent/agent.py for why that is the required behaviour rather than a
# convenience. Read from the environment so a deployed instance can be
# configured without editing code.
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()

# Name of the variable holding the API key, not the key itself. A local
# vLLM server usually runs without auth, so this is often unset entirely.
# Set LLM_API_KEY_ENV=VLLM_API_KEY for that convention.
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "LLM_API_KEY").strip()

# Placeholder key for a local vLLM server, which usually runs without auth.
# Agno raises rather than defaulting, so one is supplied here. This is not
# a credential and grants nothing.
LLM_PLACEHOLDER_API_KEY = "local"

AGENT_ENABLED = _flag("AGENT_ENABLED", True)

# Agno has no tool-call cap of its own, so the limit is enforced at the
# tool boundary instead. See agent/tools.py.
AGENT_MAX_TOOL_CALLS = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "8"))

# Seconds allowed for one model call. Enforced on the model object rather
# than around agent.run(), so a hung server cannot stall the pipeline.
AGENT_TIMEOUT_S = float(os.environ.get("AGENT_TIMEOUT_S", "60"))

SUBPROCESS_TIMEOUT_S = 120
BENCH_WARMUP_ITERS = 5
BENCH_ITERS = 50

# Clock settling, measured on an AMD Instinct MI300X VF (ROCm 7.14).
# Boost clocks there take roughly two thousand consecutive kernel calls to
# reach steady state, and the card runs 12-14% slower until they do. A fixed
# warmup count cannot clear a ramp that long, so a benchmark taken during it
# reports an unchanged kernel as a double-digit regression. The harness
# therefore re-times in batches until a sliding window of them flattens.
# Only applied to CUDA/ROCm devices: CPU timing has no equivalent ramp, and
# leaving it out keeps CPU behaviour deterministic.
#
# BENCH_SETTLE_BUDGET_S is the reason this is not free but is still cheap:
# a slow kernel would otherwise spend minutes warming up. Exceeding the
# budget is not an error, it just means the device had not settled and the
# residual drift shows up as benchmark variance.
BENCH_SETTLE = True
BENCH_SETTLE_BATCH = 50
BENCH_SETTLE_WINDOW = 4
BENCH_SETTLE_TOLERANCE = 0.005
BENCH_SETTLE_MAX_BATCHES = 60
BENCH_SETTLE_BUDGET_S = 2.0

# The web UI runs the whole pipeline in a disposable child process, because
# the pipeline imports the file it is analysing. This bounds the child, not
# the upload: a run that exceeds it is killed and reported as a timeout.
UI_WORKER_TIMEOUT_S = float(os.environ.get("UI_WORKER_TIMEOUT_S", "900"))

# Largest upload the UI will accept, in bytes. A migration source is a
# source file, not an archive; anything larger is a mistake or an attack.
UI_MAX_UPLOAD_BYTES = int(os.environ.get("UI_MAX_UPLOAD_BYTES", str(1024 * 1024)))

# Verdict bands for benchmark comparison. Calibrated against measured
# noise, not intuition: identical kernels vary run to run on real hardware.
# Values under BENCH_NOISE_BAND are reported as noise rather than as a pass.
#
# Measured on a dedicated MI300X VF (ROCm 7.14, torch 2.12.0+rocm7.14.0) by
# tools/calibrate_bands.py: steady-state run-to-run spread across 12 trials
# is under 0.5%, but the clock ramp above pushes a cold measurement about 7%
# high. Both sit inside the 1.10 band, so it holds on this device with a
# little headroom. It is deliberately not tightened further: these numbers
# come from an uncontended device, and shared or throttled cloud GPUs are
# noisier. Re-measure before changing them.
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
