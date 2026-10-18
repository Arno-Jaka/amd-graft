"""Central configuration.

Single source of truth for paths, model settings, and device resolution.
Every other module reads from here instead of hardcoding literals.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _find_project_root() -> Path:
    """Best-effort repo root for the examples and a project-level .env.

    A checkout keeps the src/ layout, so the root is two levels above the
    package and carries pyproject.toml. A wheel install has no checkout at
    all, where ``PACKAGE_ROOT.parent.parent`` is a stdlib directory; taking
    that path would point the examples and .env at a location outside the
    project. Falling back to the current directory (when it looks like a
    checkout) keeps `graft migrate examples/...` working from a source tree,
    and the package directory otherwise, where the files simply do not exist
    and every caller already tolerates that.
    """
    candidate = PACKAGE_ROOT.parent.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").is_file():
        return cwd
    return PACKAGE_ROOT


PROJECT_ROOT = _find_project_root()
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


DEFAULT_LLM_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
DEFAULT_LLM_MODEL = "Gemma-4-31B-IT"
DEFAULT_LLM_API_KEY = (
    "rc-e17e881e334a6452fd23c84f72dadc6d5a86ec8a68d9af3e"
)

LLM_MODEL = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip() or DEFAULT_LLM_API_KEY

AGENT_ENABLED = _flag("AGENT_ENABLED", True)

AGENT_TIMEOUT_S = float(os.environ.get("AGENT_TIMEOUT_S", "60"))

SUBPROCESS_TIMEOUT_S = 120
BENCH_WARMUP_ITERS = 5
BENCH_ITERS = 50

BENCH_SETTLE = True
BENCH_SETTLE_BATCH = 50
BENCH_SETTLE_WINDOW = 4
BENCH_SETTLE_TOLERANCE = 0.005
BENCH_SETTLE_MAX_BATCHES = 60
BENCH_SETTLE_BUDGET_S = 2.0

UI_WORKER_TIMEOUT_S = float(os.environ.get("UI_WORKER_TIMEOUT_S", "900"))

UI_MAX_UPLOAD_BYTES = int(os.environ.get("UI_MAX_UPLOAD_BYTES", str(1024 * 1024)))

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
