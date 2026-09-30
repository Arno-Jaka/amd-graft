"""Stage 3 — Hardware telemetry.

Reports the device actually in use. torch.version.hip distinguishes a
genuine ROCm build from a CPU or CUDA one, which is the difference between
"verified on AMD" and "hoped it works".
"""

from __future__ import annotations

import torch

from graft import config


def backend_name() -> str:
    if torch.version.hip is not None:
        return f"ROCm {torch.version.hip}"
    if torch.version.cuda is not None:
        return f"CUDA {torch.version.cuda}"
    return "CPU"


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def device_label() -> str:
    if not is_gpu_available():
        return config.GPU_NAME_FALLBACK
    return torch.cuda.get_device_name(config.GPU_DEVICE_INDEX)


def device_snapshot() -> dict:
    """One flat snapshot dict. Keys are fixed so consumers never guess.

    Note: previously this initialised 'gpu_name' and then assigned
    'device', leaving a stale 'CPU' value alongside the real name. Silent
    contradiction is worse than a crash for a telemetry surface.
    """
    available = is_gpu_available()
    snapshot = {
        "backend": backend_name(),
        "gpu_available": available,
        "gpu_name": config.GPU_NAME_FALLBACK,
        "total_vram_mb": None,
        "peak_vram_mb": None,
    }
    if not available:
        return snapshot

    props = torch.cuda.get_device_properties(config.GPU_DEVICE_INDEX)
    snapshot["gpu_name"] = props.name
    snapshot["total_vram_mb"] = round(props.total_memory / 1024**2, 1)
    return snapshot


def reset_peak_memory() -> None:
    if is_gpu_available():
        torch.cuda.reset_peak_memory_stats(config.GPU_DEVICE_INDEX)


def peak_vram_mb() -> float | None:
    if not is_gpu_available():
        return None
    used = torch.cuda.max_memory_allocated(config.GPU_DEVICE_INDEX)
    return round(used / 1024**2, 1)
