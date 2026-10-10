"""Describe the machine Graft is running on, without touching credentials.

This is the local counterpart to `graft.target.detect`, which probes a remote
host. It only reports what the local interpreter can already see; a missing
GPU or a broken torch import yields an empty mapping rather than an error, so
a UI can render "not detected" instead of failing.
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from graft import config
from graft.target.transport import CommandResult, TransportError


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


# Tools Graft's pipeline shells out to. The remote probe searches the ROCm
# install layout as well as PATH because a non-login shell routinely misses a
# versioned bin directory; the same is true locally, so the same layout is
# searched here rather than trusting shutil.which alone.
_TOOLS = ("hipcc", "hipify-perl", "hipify-clang", "nvcc", "rocminfo", "rocm-smi")
_TOOL_DIRS = (
    "/opt/rocm/bin",
    "/opt/rocm/llvm/bin",
    "/opt/rocm/core-*/bin",
    "/opt/rocm-*/bin",
    "/opt/venv/bin",
)


def _find_tool(name: str) -> str:
    """Locate an executable on PATH or in the ROCm install layout."""
    found = shutil.which(name)
    if found:
        return found
    for pattern in _TOOL_DIRS:
        for candidate in sorted(glob.glob(os.path.join(pattern, name))):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return ""


def _local_rocm_version() -> str:
    """Read the ROCm version file, mirroring the remote probe."""
    for pattern in ("/opt/rocm/.info/version", "/opt/rocm/core-*/.info/version"):
        for version_file in sorted(glob.glob(pattern)):
            try:
                version = Path(version_file).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if version:
                return version
    return ""


def local_target_facts() -> dict[str, Any]:
    """A `TargetFacts`-shaped mapping for this machine.

    The remote probe and this function produce the same shape on purpose: the
    dependency checklist, the install flow, and the GPU panel all consume
    ``TargetFacts``, so a local target is described by the same code path
    rather than a second, divergent one. Anything unreadable is left empty,
    which the consumers render as "not detected" instead of failing.
    """
    tools = {name: path for name in _TOOLS if (path := _find_tool(name))}
    torch_version = ""
    torch_gpu = ""
    gpu_count = 0
    gpu_name = ""
    vram_mb = 0
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = str(torch.cuda.get_device_name(0))
            torch_gpu = gpu_name
            try:
                properties = torch.cuda.get_device_properties(0)
                vram_mb = int(getattr(properties, "total_memory", 0) / (1024 * 1024))
            except Exception:
                pass
    except Exception:
        pass
    return {
        "ok": True,
        "os_name": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "rocm_version": _local_rocm_version(),
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "vram_mb": vram_mb,
        "tools": tools,
        "torch_version": torch_version,
        "torch_gpu": torch_gpu,
        "python_path": sys.executable or "",
    }


class LocalTransport:
    """Run the pipeline's shell commands on this machine.

    Implements the parts of the `Transport` protocol the dependency install
    flow needs, so installing on a local target goes through exactly the same
    code (and the same fixed command table) as a remote one. `stream` and
    `push` are deliberately absent: they exist for remote orchestration, and
    pretending to support them here would invite use in the wrong place.
    """

    def __enter__(self) -> LocalTransport:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def probe(self) -> None:
        """This machine is by definition reachable."""

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
    ) -> CommandResult:
        timeout = timeout_s or config.SUBPROCESS_TIMEOUT_S
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(command, 124, "", f"Timed out after {timeout}s.")
        except OSError as exc:
            return CommandResult(command, 127, "", str(exc))
        return CommandResult(command, completed.returncode, completed.stdout, completed.stderr)

    def stream(self, command, on_line, *, stop) -> None:
        raise TransportError(
            "Streaming is not available for the local machine; live readings "
            "come from torch and sysfs directly."
        )

    def push(self, local: bytes, remote_path: str) -> None:
        raise TransportError(
            "Uploading is not needed for the local machine; write the file "
            "to its path directly."
        )
