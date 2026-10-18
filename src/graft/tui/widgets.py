"""Terminal-cell geometry, live target state, and the fixed GPU dock."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from graft.target.metrics import MetricsSnapshot


class GPUStatusPanel(Static):
    """A dual-fan GPU silhouette rasterized with the terminal's Braille dots."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._state = "waiting"
        self._device_name = "GPU NOT DETECTED YET"
        self._message = "WAITING FOR GPU DETECTION"
        self._phase = 0
        self._pulse = False
        self._coast = 0
        self._columns = 44
        self._rows = 11
        kwargs.setdefault("markup", False)
        super().__init__("", *args, **kwargs)
        self._render_panel()

    def on_mount(self) -> None:
        self.set_interval(0.12, self._advance_fans)

    def on_resize(self, event: events.Resize) -> None:
        columns, rows = self._dimensions(event.size.width)
        if (columns, rows) != (self._columns, self._rows):
            self._columns, self._rows = columns, rows
            self._render_panel()

    @staticmethod
    def _dimensions(width: int) -> tuple[int, int]:
        if width < 33:
            return 24, 7
        if width < 49:
            return 34, 9
        return 44, 11

    def set_waiting(self, message: str = "WAITING FOR GPU DETECTION") -> None:
        self._state = "waiting"
        self._device_name = "GPU NOT DETECTED YET"
        self._message = message.upper()
        self._coast = 0
        self._set_state_class()
        self._render_panel()

    def set_local_facts(self, facts: dict[str, Any]) -> None:
        if facts.get("gpu"):
            self.set_active(str(facts.get("device") or "AMD GPU"))
        else:
            self.set_not_detected("AMD GPU NOT DETECTED ON THIS MACHINE")

    def set_target_facts(self, facts: dict[str, Any] | None) -> None:
        facts = facts or {}
        if int(facts.get("gpu_count") or 0) > 0:
            name = str(facts.get("gpu_name") or facts.get("torch_gpu") or "AMD GPU")
            self.set_active(name)
        else:
            self.set_not_detected("AMD GPU NOT DETECTED ON TARGET")

    def set_active(self, device_name: str) -> None:
        self._state = "active"
        self._device_name = device_name.strip() or "GPU"
        self._message = ""
        self._phase = 0
        self._coast = 0
        self._set_state_class()
        self._render_panel()

    def set_not_detected(self, message: str = "AMD GPU NOT DETECTED") -> None:
        self._state = "offline"
        self._device_name = "GPU NOT DETECTED"
        self._message = message.upper()
        self._coast = 0
        self._set_state_class()
        self._render_panel()

    def set_disconnected(self, message: str = "TARGET DISCONNECTED") -> None:
        self._state = "disconnected"
        self._message = message.upper()
        self._coast = 8
        self._set_state_class()
        self._render_panel()

    def _set_state_class(self) -> None:
        if not self.is_mounted:
            return
        for state in ("active", "offline", "disconnected", "waiting"):
            self.set_class(self._state == state, state)

    def _advance_fans(self) -> None:
        if self._state == "active":
            self._phase = (self._phase + 3) % 36
        elif self._state == "disconnected" and self._coast:
            self._phase = (self._phase + 2) % 36
            self._coast -= 1
        else:
            self._pulse = not self._pulse
        self._render_panel()

    @staticmethod
    def _line(pixels: set[tuple[int, int]], x0: int, y0: int, x1: int, y1: int) -> None:
        """Draw a Bresenham line into the Braille-dot canvas."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            pixels.add((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                error += dy
                x0 += step_x
            if doubled_error <= dx:
                error += dx
                y0 += step_y

    @staticmethod
    def _ring(pixels: set[tuple[int, int]], cx: int, cy: int, radius: int) -> None:
        for step in range(144):
            angle = step * math.tau / 144
            pixels.add(
                (
                    round(cx + math.cos(angle) * radius),
                    round(cy + math.sin(angle) * radius),
                )
            )

    @staticmethod
    def _fan(
        pixels: set[tuple[int, int]],
        cx: int,
        cy: int,
        radius: int,
        phase: float,
        *,
        broken: bool,
        pulse: bool,
    ) -> None:
        GPUStatusPanel._ring(pixels, cx, cy, radius)
        GPUStatusPanel._ring(pixels, cx, cy, max(2, radius - 2))

        if not broken:
            for blade in range(3):
                base = phase + blade * math.tau / 3
                for y in range(cy - radius + 2, cy + radius - 1):
                    for x in range(cx - radius + 2, cx + radius - 1):
                        dx, dy = x - cx, y - cy
                        distance = math.hypot(dx, dy)
                        if 2.5 < distance < radius - 2:
                            angle = math.atan2(dy, dx)
                            swept_angle = base + 0.65 * (distance / radius)
                            delta = (angle - swept_angle + math.pi) % math.tau - math.pi
                            if abs(delta) < 0.19:
                                pixels.add((x, y))
        else:
            pixels.add((cx, cy))
            if pulse:
                pixels.update(((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)))

        GPUStatusPanel._ring(pixels, cx, cy, 2)
        pixels.add((cx, cy))

    @staticmethod
    def _braille(pixels: set[tuple[int, int]], width: int, height: int) -> list[str]:
        dot_bits = {
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (1, 0): 3,
            (1, 1): 4,
            (1, 2): 5,
            (0, 3): 6,
            (1, 3): 7,
        }
        rows: list[str] = []
        for cell_y in range(height):
            chars: list[str] = []
            for cell_x in range(width):
                mask = 0
                for (dot_x, dot_y), bit in dot_bits.items():
                    if (cell_x * 2 + dot_x, cell_y * 4 + dot_y) in pixels:
                        mask |= 1 << bit
                chars.append(chr(0x2800 + mask) if mask else " ")
            rows.append("".join(chars).rstrip())
        return rows

    def _geometry(self) -> list[str]:
        width, height = self._columns * 2, self._rows * 4
        pixels: set[tuple[int, int]] = set()

        outline = [
            (8, 7),
            (14, 2),
            (width - 13, 2),
            (width - 5, 8),
            (width - 5, height - 10),
            (width - 12, height - 3),
            (15, height - 3),
            (8, height - 8),
            (8, 7),
        ]
        for start, end in pairwise(outline):
            self._line(pixels, *start, *end)

        for y in range(9, height - 9):
            pixels.update(((4, y), (7, y)))
        for x in (4, 5, 6, 7):
            pixels.update(((x, 9), (x, height - 10)))
        pixels.difference_update(
            {(4, height // 2 - 1), (4, height // 2), (7, height // 2)}
        )
        for x in range(18, width - 18, 4):
            self._line(pixels, x, 1, x + 2, 1)

        radius = min(12, (height - 8) // 2, (width // 2 - 9) // 2)
        center_y = height // 2
        fan_offset = radius + 3
        centers = (width // 2 - fan_offset, width // 2 + fan_offset)
        for index, center_x in enumerate(centers):
            self._fan(
                pixels,
                center_x,
                center_y,
                radius,
                self._phase * math.tau / 36 + index * math.tau / 6,
                broken=self._state == "disconnected" and self._coast == 0,
                pulse=self._pulse,
            )

        contact_y = height - 2
        for x in range(18, width - 18, 3):
            pixels.update(((x, contact_y), (x + 1, contact_y)))
        return self._braille(pixels, self._columns, self._rows)

    def _render_panel(self) -> None:
        if self._state == "active":
            state_line = "● CONNECTED · online"
        elif self._state == "disconnected":
            state_line = (
                "! CONNECTION LOST · GPU OFFLINE"
                if self._pulse
                else "◌ CONNECTION LOST · GPU OFFLINE"
            )
        elif self._state == "offline":
            state_line = "! GPU NOT DETECTED"
        else:
            state_line = "◌ WAITING FOR GPU DETECTION"

        lines = [
            *self._geometry(),
            "AMD  ·  " + self._device_name,
            self._message,
            state_line,
        ]
        self.update("\n".join(lines))


class GPUMetrics(Static):
    """Live VRAM, disk, utilization, temperature, and power under the GPU.

    Every reading is optional. A field that could not be measured shows as
    "not measured" rather than a zero, so an unreadable sensor is never
    mistaken for an idle GPU.

    The bar width follows the widget, because the dock narrows on smaller
    terminals: a fixed 18-cell bar turns every line into a wrap, which reads
    as corruption rather than as a gauge.
    """

    BAR_WIDTH = 18
    MIN_BAR_WIDTH = 6
    HORIZONTAL_PADDING = 2
    GAUGE_FIXED = 28
    COMPACT_FIXED = 13

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("markup", False)
        self._snapshot: MetricsSnapshot | None = None
        self._message = "AWAITING A TARGET FOR LIVE READINGS"
        self._bar_width = self.BAR_WIDTH
        self._width = 0
        super().__init__("", *args, **kwargs)
        self._render_panel()

    @property
    def _compact(self) -> bool:
        """Too narrow for gauges: abbreviate rather than wrap every line."""
        return 0 < self._width < self.GAUGE_FIXED + self.MIN_BAR_WIDTH + self.HORIZONTAL_PADDING

    def on_mount(self) -> None:
        self._render_panel()

    def on_resize(self, event: events.Resize) -> None:
        width = event.size.width
        available = max(0, width - self.HORIZONTAL_PADDING)
        bar_width = max(
            self.MIN_BAR_WIDTH, min(self.BAR_WIDTH, available - self.GAUGE_FIXED)
        )
        if bar_width != self._bar_width or width != self._width:
            self._bar_width = bar_width
            self._width = width
            self._render_panel()

    def set_metrics(self, snapshot: MetricsSnapshot) -> None:
        self._snapshot = snapshot
        self._message = snapshot.error
        self._render_panel()

    def clear(self, message: str = "AWAITING A TARGET FOR LIVE READINGS") -> None:
        self._snapshot = None
        self._message = message
        self._render_panel()

    @staticmethod
    def _bar(fraction: float, width: int = BAR_WIDTH) -> str:
        fraction = max(0.0, min(1.0, fraction))
        filled = round(fraction * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _gb(value: int) -> str:
        return f"{value / 1_000_000_000:.1f}"

    def _usage(self, label: str, used: int, total: int) -> str:
        if used < 0 or total <= 0:
            if self._compact:
                return f"{label}  not measured"
            return f"{label}  {'not measured':<{self._bar_width + 12}}"
        fraction = used / total
        if self._compact:
            return (
                f"{label}  {fraction * 100:3.0f}%  "
                f"{used / 1_000_000_000:.0f}/{total / 1_000_000_000:.0f}G"
            )
        return (
            f"{label}  {self._bar(fraction, self._bar_width)}  "
            f"{fraction * 100:3.0f}%  "
            f"{self._gb(used)}/{self._gb(total)} GB"
        )

    def _utilization(self, pct: float) -> str:
        if pct < 0:
            if self._compact:
                return "UTIL  not measured"
            return f"UTIL  {'not measured':<{self._bar_width + 12}}"
        if self._compact:
            return f"UTIL  {pct:3.0f}%"
        return (
            f"UTIL  {self._bar(pct / 100, self._bar_width)}  {pct:3.0f}%"
        )

    def _render_panel(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            self.update(self._message)
            return
        gauges: list[str] = []
        if snapshot.temp_c >= 0:
            gauges.append(f"{snapshot.temp_c:.0f}°C")
        if snapshot.power_w >= 0:
            gauges.append(f"{snapshot.power_w:.0f} W")
        if snapshot.gpu_name:
            gauges.append(snapshot.gpu_name)
        source = "TARGET" if snapshot.source == "remote" else "LOCAL"
        footer = " · ".join(gauges) if gauges else (snapshot.error or "no gauges")
        if self._width:
            budget = self._width - self.HORIZONTAL_PADDING - len(source) - 4
            if len(footer) > budget:
                footer = footer[: max(1, budget - 1)] + "…"
        lines = [
            self._usage("VRAM", snapshot.vram_used_bytes, snapshot.vram_total_bytes),
            self._usage("DISK", snapshot.disk_used_bytes, snapshot.disk_total_bytes),
            self._utilization(snapshot.gpu_util_pct),
            f"[{source}] {footer}",
        ]
        self.update("\n".join(lines))


class GPUDock(Vertical):
    """The fixed upper-right panel: animated GPU plus live readings.

    Mounted once by the app rather than per screen, so the same panel stays
    visible on every tab. Everything that drives it (the workspace target
    screen, the metrics thread) reaches it through the app.
    """

    def compose(self) -> ComposeResult:
        yield Static("GPU / LIVE", id="gpu-dock-title")
        yield GPUStatusPanel(id="gpu-visual")
        yield GPUMetrics(id="gpu-metrics")


__all__ = ["GPUDock", "GPUMetrics", "GPUStatusPanel"]
