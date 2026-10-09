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


def local_snapshot() -> MetricsSnapshot:
    """Read this machine's VRAM (via torch) and root disk usage."""
    fields: dict[str, str] = {}
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            fields["vram_total_bytes"] = str(int(total))
            fields["vram_used_bytes"] = str(int(total - free))
            fields["gpu_name"] = str(torch.cuda.get_device_name(0))
            fields["gpu_count"] = str(torch.cuda.device_count())
    except Exception:
        # A missing or unusable torch is expected on a CPU-only workstation;
        # disk still works, so report what can be measured.
        pass
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
            error="No local GPU visible to torch",
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
