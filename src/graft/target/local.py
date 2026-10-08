"""Describe the machine Graft is running on, without touching credentials.

This is the local counterpart to `graft.target.detect`, which probes a remote
host. It only reports what the local interpreter can already see; a missing
GPU or a broken torch import yields an empty mapping rather than an error, so
a UI can render "not detected" instead of failing.
"""

from __future__ import annotations

from typing import Any

from graft import config


def local_capability() -> dict[str, Any]:
    """Summarize this machine without collecting or handling credentials."""
    try:
        import torch

        from graft.bench import telemetry

        snapshot = telemetry.device_snapshot()
        gpu = bool(torch.cuda.is_available())
        return {
            "device": config.device_label(),
            "backend": snapshot.get("backend", "unknown"),
            "gpu": gpu,
            "vram_mb": snapshot.get("total_vram_mb"),
            "torch": torch.__version__,
            "bands_calibrated": gpu
            and bool(getattr(config, "BANDS_CALIBRATED", False)),
        }
    except Exception:
        return {}
