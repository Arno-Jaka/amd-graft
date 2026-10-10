"""Live GPU and disk readings for the workspace panel.

One snapshot is a flat set of ``key=value`` lines, which is the cheapest
thing to produce on a remote host and the easiest to parse here. VRAM, GPU
busy, temperature, and power come from the amdgpu sysfs nodes rather than
``rocm-smi``: the sysfs layout is stable, while rocm-smi's human and CSV
output shifts between releases. Disk comes from ``df``.

The remote side can either print one snapshot (for a poll) or loop forever
(for a streamed update over a single connection, which is what the UI uses).
"""

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass, replace
from typing import Any

SENTINEL = "__GRAFT_METRICS_END__"

# One snapshot as a shell script. Kept free of `set -e` so a missing sysfs
# node skips one field instead of aborting the whole reading. Sensor names
# differ by card (temp1/temp2, power1_input/power1_average), so each family is
# globbed and the highest reading is kept.
_SAMPLE = r"""
emit() { printf '%s=%s\n' "$1" "$2"; }
total=0
used=0
busy=-1
temp=-1
power=-1
gpu_name=""
for dev in /sys/class/drm/card*/device; do
  if [ -r "$dev/mem_info_vram_total" ]; then
    total=$(( total + $(cat "$dev/mem_info_vram_total" 2>/dev/null) ))
    used=$(( used + $(cat "$dev/mem_info_vram_used" 2>/dev/null) ))
    b="$(cat "$dev/gpu_busy_percent" 2>/dev/null)"
    [ -n "$b" ] && busy="$b"
    for hw in "$dev"/hwmon/hwmon*; do
      for tf in "$hw"/temp*_input; do
        [ -r "$tf" ] || continue
        v=$(( $(cat "$tf" 2>/dev/null) / 1000 ))
        [ "$v" -gt "$temp" ] && temp="$v"
      done
      for pf in "$hw"/power1_average "$hw"/power1_input; do
        [ -r "$pf" ] || continue
        v=$(( $(cat "$pf" 2>/dev/null) / 1000000 ))
        [ "$v" -gt "$power" ] && power="$v"
      done
    done
  fi
done
emit vram_total_bytes "$total"
emit vram_used_bytes "$used"
emit gpu_util_pct "$busy"
emit temp_c "$temp"
emit power_w "$power"
set -- $(df -P -B1 / 2>/dev/null | awk 'NR==2 {print $2, $3}')
emit disk_total_bytes "${1:-0}"
emit disk_used_bytes "${2:-0}"
"""


def sample_command() -> str:
    """A one-shot shell command that prints a single metrics snapshot."""
    return "set +e\n" + _SAMPLE


def stream_command(interval_s: float = 1.0) -> str:
    """A shell command that streams a snapshot every ``interval_s`` seconds.

    Each snapshot ends with the sentinel so the reader knows when a complete
    reading has arrived; without it, partial lines would render as bars that
    jump around.
    """
    interval = max(0.2, float(interval_s))
    return (
        "set +e\nwhile :; do\n"
        + _SAMPLE
        + f"\nprintf '%s\\n' {SENTINEL}\nsleep {interval}\ndone\n"
    )


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """One reading. Any ``-1`` means "not measured", never a real value."""

    source: str = "local"
    gpu_name: str = ""
    gpu_count: int = 0
    vram_used_bytes: int = -1
    vram_total_bytes: int = -1
    gpu_util_pct: float = -1
    temp_c: float = -1
    power_w: float = -1
    disk_used_bytes: int = -1
    disk_total_bytes: int = -1
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "vram_used_bytes": self.vram_used_bytes,
            "vram_total_bytes": self.vram_total_bytes,
            "gpu_util_pct": self.gpu_util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "disk_used_bytes": self.disk_used_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "error": self.error,
        }


def _number(raw: Any, default: float = -1.0) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def snapshot_from_fields(fields: dict[str, str], *, source: str) -> MetricsSnapshot:
    """Build a snapshot from parsed ``key=value`` lines."""

    def as_int(key: str) -> int:
        value = _number(fields.get(key))
        return int(value) if value >= 0 else -1

    return MetricsSnapshot(
        source=source,
        gpu_name=fields.get("gpu_name", ""),
        gpu_count=int(_number(fields.get("gpu_count"), 0)),
        vram_used_bytes=as_int("vram_used_bytes"),
        vram_total_bytes=as_int("vram_total_bytes"),
        gpu_util_pct=_number(fields.get("gpu_util_pct")),
        temp_c=_number(fields.get("temp_c")),
        power_w=_number(fields.get("power_w")),
        disk_used_bytes=as_int("disk_used_bytes"),
        disk_total_bytes=as_int("disk_total_bytes"),
    )


def parse_metrics(text: str, *, source: str = "remote") -> MetricsSnapshot:
    """Parse a `key=value` block; unknown keys are ignored."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return snapshot_from_fields(fields, source=source)


def _read_int(path: str) -> int:
    """Read an integer from a sysfs node, or -1 when unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return -1


def _local_sysfs_fields() -> dict[str, str]:
    """Read local amdgpu sysfs nodes, the same nodes the remote script reads.

    The remote target gets utilization, temperature, and power; a local AMD
    workstation exposes the identical nodes under /sys, so leaving the local
    panel at VRAM-only was a gap rather than a limit. Anything missing (an
    NVIDIA card, a VM, macOS) yields nothing and the field stays "not
    measured".
    """
    fields: dict[str, str] = {}
    total = 0
    used = 0
    busy = -1
    temp = -1
    power = -1
    found = False
    for device in sorted(glob.glob("/sys/class/drm/card*/device")):
        vram_total = _read_int(os.path.join(device, "mem_info_vram_total"))
        if vram_total <= 0:
            continue
        found = True
        total += vram_total
        used += max(0, _read_int(os.path.join(device, "mem_info_vram_used")))
        busy = max(busy, _read_int(os.path.join(device, "gpu_busy_percent")))
        for hwmon in sorted(glob.glob(os.path.join(device, "hwmon", "hwmon*"))):
            for temp_input in sorted(glob.glob(os.path.join(hwmon, "temp*_input"))):
                temp = max(temp, _read_int(temp_input) // 1000)
            for power_input in (
                *sorted(glob.glob(os.path.join(hwmon, "power1_average"))),
                *sorted(glob.glob(os.path.join(hwmon, "power1_input"))),
            ):
                power = max(power, _read_int(power_input) // 1_000_000)
    if not found:
        return fields
    fields["vram_total_bytes"] = str(total)
    fields["vram_used_bytes"] = str(used)
    if busy >= 0:
        fields["gpu_util_pct"] = str(busy)
    if temp >= 0:
        fields["temp_c"] = str(temp)
    if power >= 0:
        fields["power_w"] = str(power)
    return fields


def local_snapshot() -> MetricsSnapshot:
    """Read this machine's GPU sensors (torch + sysfs) and root disk usage."""
    fields: dict[str, str] = {}
    try:
        import torch

        if torch.cuda.is_available():
            # torch is the authority for the GPU name, count, and the VRAM a
            # ROCm process may actually use (reservations, other processes);
            # sysfs below is the authority for utilization and sensors.
            fields["gpu_name"] = str(torch.cuda.get_device_name(0))
            fields["gpu_count"] = str(torch.cuda.device_count())
            free, total = torch.cuda.mem_get_info(0)
            fields["vram_total_bytes"] = str(int(total))
            fields["vram_used_bytes"] = str(int(total - free))
    except Exception:
        # A missing or unusable torch is expected on a CPU-only workstation;
        # disk and sysfs still work, so report what can be measured.
        pass
    sysfs_fields = _local_sysfs_fields()
    # sysfs reports the card's own totals. They are only used when torch could
    # not provide a live reading, so a measured value is never replaced by a
    # less precise one.
    if "vram_total_bytes" not in fields:
        fields.update({k: v for k, v in sysfs_fields.items() if k.startswith("vram_")})
    for key, value in sysfs_fields.items():
        if not key.startswith("vram_"):
            fields[key] = value
    try:
        usage = shutil.disk_usage("/")
        fields["disk_total_bytes"] = str(usage.total)
        fields["disk_used_bytes"] = str(usage.used)
    except OSError:
        pass
    snapshot = snapshot_from_fields(fields, source="local")
    if snapshot.vram_total_bytes < 0:
        snapshot = replace(
            snapshot,
            error="No local AMD GPU visible to torch or /sys/class/drm",
        )
    return snapshot


__all__ = [
    "SENTINEL",
    "MetricsSnapshot",
    "local_snapshot",
    "parse_metrics",
    "sample_command",
    "snapshot_from_fields",
    "stream_command",
]
