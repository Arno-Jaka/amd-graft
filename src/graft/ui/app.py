"""Stage 9 — The web UI.

FastHTML plus MonsterUI. One screen, no build step, and no framework beyond
the two of them.

Four rules keep the presentation honest:

1. A verdict is never shown without the evidence behind it. The basis text
   is computed once, in `graft.report`, and transported verbatim, so the
   page cannot word a verdict more confidently than the Markdown report does.
2. Unmeasured renders as unmeasured. Absent parity, absent benchmarks, and
   absent narrative each say so rather than collapsing to zero or to a gap
   in the layout.
3. Findings are grouped by the severity the converter assigned, never
   re-judged here.
4. This module never calls `pipeline.run()` and never imports `graft.agent`.
   Uploaded code runs in a disposable child (see `worker.py`); the agent is
   an optional layer inside the pipeline, and the UI has no business knowing
   which prose was model-written.

Because rendering is a pure function of the summary dict, the whole page can
be tested without a browser and without a subprocess.

Visual language
---------------
An AMD-adjacent dark theme: near-black surfaces, one red accent, 2px radii,
thin translucent borders, an uppercase tracked eyebrow above every heading,
and their signature 326deg diagonal gradient. These are general stylistic
tendencies only. No AMD logo, wordmark, font file, or other brand asset is
included or imitated.

Motion is CSS-first. Keyframes live in `_STYLE` and are selected through
Tailwind's arbitrary `animate-[...]` syntax, so nearly everything animated
here needs no JavaScript. The only script counts stat tiles and forwards
pointer position into CSS variables, because those two genuinely need
per-frame state. Everything degrades under prefers-reduced-motion.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastcore.xml import Div, NotStr, PlainText
from monsterui.all import (
    H1,
    H2,
    Alert,
    Button,
    Card,
    CardBody,
    CardHeader,
    DivVStacked,
    Form,
    Input,
    Label,
    Small,
    Strong,
    Subtitle,
    fast_app,
)
from starlette.datastructures import UploadFile
from starlette.responses import PlainTextResponse

from graft import config
from graft.core import runner
from graft.ui import worker
from graft.ui.summary import NARRATIVE_TITLES

# --- palette ---------------------------------------------------------------
# Near-black page, two elevated greys, one accent hue with light/base/dark
# tints, and a translucent white border instead of a filled grey. The
# gradient's angle is the detail that reads as "AMD" without copying
# anything: it is a direction, not an asset.

BG = "#000000"
SURFACE = "#1a1a1b"
SURFACE_2 = "#27282b"
BORDER = "hsla(0,0,100%,0.15)"
TEXT = "#ffffff"
TEXT_DIM = "hsla(0,0,100%,0.7)"
TEXT_FAINT = "hsla(0,0,100%,0.5)"
RED = "#ed1c24"
RED_LIGHT = "#f14b52"
RED_DARK = "#c61017"
CYAN = "#00c2de"
GREEN = "#4aa564"
GOLD = "#c1a968"

BRAND_GRADIENT = f"linear-gradient(326deg,{RED_LIGHT},{RED} 58%,{RED_DARK})"

RADIUS = "[border-radius:2px]"

VERDICT_COLOR = {
    "production_ready": GREEN,
    "needs_review": GOLD,
    "manual_port": RED,
    "unsupported": RED,
}
SEVERITY_COLOR = {"blocker": RED, "review": GOLD, "info": CYAN}

EYEBROW = (
    "block text-[0.6875rem] uppercase tracking-[0.14em] font-semibold "
    f"text-[{TEXT_FAINT}]"
)
MONO = "font-mono text-xs"
BODY = f"text-sm text-[{TEXT_DIM}] leading-relaxed"
SURFACE_CLS = f"bg-[{SURFACE}] border border-[{BORDER}] {RADIUS}"

NOT_MEASURED = "not measured"

# Shown in the Execution panel. Read from the runner rather than restated
# here, so the page cannot drift into describing isolation the runner does
# not actually provide.
ISOLATION_NOTES = list(runner.sandbox_notes())


# --- motion ----------------------------------------------------------------

_STYLE = NotStr(f"""<style>
  .brand-grad {{ background-image: {BRAND_GRADIENT}; }}

  /* DotGrid: a telemetry texture from a tiled radial-gradient. Animating
     background-position stays on the compositor, so it is cheap. */
  .dotgrid {{
    background-image: radial-gradient(hsla(0,0,100%,0.11) 1px, transparent 1px);
    background-size: 26px 26px;
    animation: dotgrid-drift 22s linear infinite;
  }}
  @keyframes dotgrid-drift {{
    from {{ background-position: 0 0; }}
    to   {{ background-position: 26px 26px; }}
  }}

  /* ShinyText: a specular sweep clipped to the glyphs. */
  .shine {{
    background: linear-gradient(100deg,{TEXT} 0%,{TEXT} 38%,{TEXT_FAINT} 50%,{TEXT} 62%,{TEXT} 100%);
    background-size: 220% 100%;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shine-sweep 7s ease-in-out infinite;
  }}
  @keyframes shine-sweep {{
    0%,100% {{ background-position: 180% 0; }}
    50%     {{ background-position: -20% 0; }}
  }}

  @property --cursor-angle {{ syntax: "<angle>"; inherits: false; initial-value: 0deg; }}

  /* BorderGlow: a cursor-tracked conic gradient revealed through a mask. */
  .border-glow {{ position: relative; isolation: isolate; }}
  .border-glow::before {{
    content: ""; position: absolute; inset: 0; border-radius: inherit;
    padding: 1px; z-index: 2; pointer-events: none; opacity: 0;
    transition: opacity .3s ease;
    background: conic-gradient(from var(--cursor-angle) at 50% 50%,
      transparent 0deg, {RED} 70deg, {RED_LIGHT} 120deg, transparent 190deg);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
  }}
  .border-glow:hover::before {{ opacity: 1; animation: glow-spin 3.5s linear infinite; }}
  @keyframes glow-spin {{ to {{ --cursor-angle: 360deg; }} }}

  /* SpotlightCard: cursor-tracked radial highlight. */
  .spotlight {{ position: relative; }}
  .spotlight::after {{
    content: ""; position: absolute; inset: 0; border-radius: inherit;
    pointer-events: none; opacity: 0; transition: opacity .3s ease;
    background: radial-gradient(240px circle at var(--mouse-x, 50%) var(--mouse-y, 50%),
      hsla(353,86%,54%,0.16), transparent 70%);
  }}
  .spotlight:hover::after {{ opacity: 1; }}

  /* GlareHover: one diagonal sheen crossing on hover. */
  .glare {{ position: relative; overflow: hidden; }}
  .glare::after {{
    content: ""; position: absolute; inset: -50%;
    background: linear-gradient(105deg, transparent 42%, hsla(0,0,100%,0.07) 50%, transparent 58%);
    transform: translateX(-120%); transition: transform .7s ease;
  }}
  .glare:hover::after {{ transform: translateX(120%); }}

  /* Staggered reveal; each panel sets its own --i. */
  .rise {{ opacity: 0; animation: rise-in .55s cubic-bezier(.16,1,.3,1) forwards;
           animation-delay: calc(var(--i, 0) * 70ms); }}
  @keyframes rise-in {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}

  /* Scanline so an idle screen still breathes. */
  .scanline {{ position: relative; overflow: hidden; }}
  .scanline::before {{
    content: ""; position: absolute; left: 0; right: 0; height: 34%;
    background: linear-gradient(180deg, transparent, {BRAND_GRADIENT}, transparent);
    opacity: .45; animation: scan 4.5s cubic-bezier(.6,0,.4,1) infinite;
  }}
  @keyframes scan {{
    0%   {{ transform: translateY(-120%); }}
    100% {{ transform: translateY(340%); }}
  }}

  @keyframes pulse-ring {{
    0%   {{ box-shadow: 0 0 0 0 hsla(353,86%,54%,0.55); }}
    70%  {{ box-shadow: 0 0 0 9px hsla(353,86%,54%,0); }}
    100% {{ box-shadow: 0 0 0 0 hsla(353,86%,54%,0); }}
  }}
  .pulse {{ animation: pulse-ring 2.2s cubic-bezier(.4,0,.6,1) infinite; }}

  /* Severity meters fill in rather than appearing at full width. */
  .meter {{ width: 0; animation: meter-fill .9s cubic-bezier(.16,1,.3,1) forwards;
            animation-delay: calc(var(--i, 0) * 70ms + 200ms); }}
  @keyframes meter-fill {{ to {{ width: var(--w, 0%); }} }}

  /* Slow ambient wash, echoing their radial-from-above move. */
  .ambient {{
    background-image: radial-gradient(circle at 50% -10%,
      hsla(353,86%,54%,0.20), transparent 58%);
    animation: ambient-breathe 11s ease-in-out infinite;
  }}
  @keyframes ambient-breathe {{
    0%,100% {{ opacity: .75; }}
    50%     {{ opacity: 1; }}
  }}

  /* Scroll-linked reveal, feature-detected so a browser without it renders
     the content fully visible instead of hiding it. */
  @supports (animation-timeline: view()) {{
    .reveal {{
      animation: rise-in linear both;
      animation-timeline: view();
      animation-range: entry 5% cover 22%;
    }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    .shine, .dotgrid, .scanline::before, .ambient, .pulse, .rise,
    .glare::after, .border-glow::before, .meter, .reveal {{
      animation: none !important;
    }}
    .rise, .reveal {{ opacity: 1 !important; }}
    .meter {{ opacity: 1 !important; width: var(--w, 0%) !important; }}
  }}
</style>""")

# Two things here genuinely need JavaScript: counting a number up, and
# telling CSS where the pointer is. Everything else is keyframes above.
_SCRIPT = NotStr("""<script>
  document.querySelectorAll('[data-count]').forEach(function (el) {
    var target = parseFloat(el.dataset.count);
    if (isNaN(target)) return;
    var decimals = (String(target).split('.')[1] || '').length;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.textContent = target.toFixed(decimals);
      return;
    }
    var start = performance.now(), dur = 900;
    function tick(now) {
      var p = Math.min(1, (now - start) / dur);
      el.textContent = (target * (1 - Math.pow(1 - p, 3))).toFixed(decimals);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });

  document.addEventListener('pointermove', function (e) {
    var card = e.target.closest('.spotlight, .border-glow');
    if (!card) return;
    var r = card.getBoundingClientRect();
    card.style.setProperty('--mouse-x', ((e.clientX - r.left) / r.width * 100) + '%');
    card.style.setProperty('--mouse-y', ((e.clientY - r.top) / r.height * 100) + '%');
    var x = e.clientX - r.left - r.width / 2;
    var y = e.clientY - r.top - r.height / 2;
    card.style.setProperty('--cursor-angle',
      (Math.atan2(y, x) * 180 / Math.PI + 360) % 360 + 'deg');
  }, { passive: true });
</script>""")


# --- small builders --------------------------------------------------------


def eyebrow(text: str):
    return Small(text, cls=EYEBROW)


def stat_tile(label: str, value: Any, index: int = 0, tone: str = TEXT):
    """One KPI tile. Numbers count up; anything else renders as-is."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        content = Small(
            "0",
            **{
                "data-count": str(value),
                "class": "text-3xl font-black tabular-nums tracking-tight",
                "style": f"color:{tone}",
            },
        )
    else:
        content = Small(
            str(value),
            **{
                "class": "text-lg font-semibold",
                "style": f"color:{tone}",
            },
        )
    return Card(
        DivVStacked(eyebrow(label), content, cls="space-y-1"),
        cls=f"{SURFACE_CLS} spotlight glare p-5 relative overflow-hidden",
        style=f"--i:{index}",
    )


def badge(text: str, color: str):
    return Small(
        text.upper(),
        cls="inline-flex items-center px-2 py-0.5 text-[0.625rem] font-bold "
        "uppercase tracking-[0.12em]",
        style=f"color:{color};background:hsla(0,0,100%,0.06);"
        f"border:1px solid hsla(0,0,100%,0.12);border-radius:2px",
    )


def kv(label: str, value: str):
    return DivVStacked(
        eyebrow(label),
        Strong(value, cls="text-sm font-semibold break-words"),
        cls="space-y-0.5",
    )


def meter(share: float, color: str, index: int = 0):
    return Div(
        Div(cls="h-[3px] w-full", style="background:hsla(0,0,100%,0.07)"),
        Div(
            cls="meter h-[3px]",
            style=f"--w:{share:.0f}%;background:{color};--i:{index}",
        ),
        cls="space-y-1",
    )


def panel(title: str, *children, kicker: str = "measured", index: int = 0):
    return Card(
        CardHeader(
            DivVStacked(
                eyebrow(kicker),
                H2(title, cls="text-xl font-bold tracking-tight mt-1"),
                cls="space-y-0",
            ),
            cls="pb-3",
        ),
        CardBody(DivVStacked(*children, cls="space-y-3"), cls="pt-0"),
        cls=f"{SURFACE_CLS} border-glow rise",
        style=f"--i:{index}",
    )


def note(text: str, color: str, kicker: str):
    return Alert(
        DivVStacked(
            eyebrow(kicker),
            Strong(text, cls="text-sm font-normal"),
            cls="space-y-1",
        ),
        cls="text-sm",
        style=f"border:1px solid {color};background:hsla(0,0,100%,0.03);border-radius:2px",
    )


def bullet_list(items: list[str], cls: str = BODY):
    return DivVStacked(*[Small(item, cls=cls) for item in items], cls="space-y-1.5")


# --- panels ----------------------------------------------------------------


def environment_panel(env: dict, index: int):
    vram = env.get("total_vram_mb")
    children = [
        DivVStacked(
            kv("backend", env.get("backend", NOT_MEASURED)),
            kv(
                "gpu",
                env.get("gpu_name", NOT_MEASURED) if env.get("gpu_available") else "none detected",
            ),
            kv("gpu present", "yes" if env.get("gpu_available") else "no"),
            kv("total vram", f"{vram} MB" if isinstance(vram, (int, float)) else NOT_MEASURED),
            cls="grid grid-cols-2 md:grid-cols-4 gap-4",
        )
    ]
    if env.get("gpu_available"):
        children.append(
            note(
                "A GPU was detected, so the numbers below are device-local. "
                "They still say nothing about CUDA behaviour.",
                CYAN,
                "gpu present",
            )
        )
    else:
        children.append(
            note(
                "No AMD GPU was detected. Every number on this page is "
                "CPU-only, and none of it demonstrates ROCm behaviour.",
                GOLD,
                "cpu only",
            )
        )
    return panel("Environment", *children, kicker="where this ran", index=index)


def conversion_panel(conv: dict, index: int):
    if not conv.get("measured"):
        return panel("Conversion", Small(NOT_MEASURED, cls="text-sm"), index=index)

    colour = VERDICT_COLOR.get(conv.get("verdict", ""), TEXT_DIM)
    changes = conv.get("changes") or []
    return panel(
        "Conversion",
        DivVStacked(
            DivVStacked(
                Strong(
                    conv.get("verdict_line", NOT_MEASURED),
                    cls="text-xl font-black tracking-tight",
                    style=f"color:{colour}",
                ),
                badge(conv.get("verdict", "unknown"), colour),
                cls="flex items-center gap-3 flex-wrap",
            ),
            Small(conv.get("verdict_basis", ""), cls=BODY),
            DivVStacked(
                kv("route", conv.get("route", "unknown")),
                kv("backend", conv.get("backend", NOT_MEASURED)),
                kv("changes", str(len(changes))),
                cls="grid grid-cols-3 gap-4",
            ),
            _change_rows(changes),
            _findings(conv.get("findings") or []),
            cls="space-y-5",
        ),
        kicker="what changed",
        index=index,
    )


def _change_rows(changes: list[dict]):
    if not changes:
        return DivVStacked(
            eyebrow("changes"),
            Small("No rewrites applied.", cls=BODY),
            cls="space-y-1",
        )
    rows = []
    for change in changes[:14]:
        line = change.get("line")
        where = f":{line}" if isinstance(line, int) else ""
        rows.append(
            DivVStacked(
                Strong(f"{change.get('symbol', '?')}{where}", cls=f"{MONO} font-semibold"),
                Small(change.get("reason", "") or "", cls="text-xs"),
                cls="space-y-0",
            )
        )
    if len(changes) > 14:
        rows.append(Small(f"+{len(changes) - 14} more", cls="text-xs"))
    return DivVStacked(
        DivVStacked(
            eyebrow(f"{len(changes)} change(s) applied"),
            DivVStacked(
                *rows, cls="space-y-2 border-l pl-3", style=f"border-color:{BORDER}"
            ),
            cls="space-y-2",
        )
    )


def _findings(findings: list[dict]):
    """Severity buckets with proportional meters.

    The share is of the findings actually present, so three blockers read as
    100% blockers while three blockers and four notes show both in
    proportion rather than pretending the totals are comparable.
    """
    if not findings:
        return DivVStacked(
            eyebrow("coverage"),
            Small("No unmapped constructs found.", cls=BODY),
            cls="space-y-1",
        )

    buckets: dict[str, list[dict]] = {}
    for finding in findings:
        buckets.setdefault(str(finding.get("severity", "info")), []).append(finding)

    total = len(findings)
    blocks = []
    for order, (level, items) in enumerate(buckets.items()):
        colour = SEVERITY_COLOR.get(level, TEXT_DIM)
        share = len(items) / total * 100
        blocks.append(
            DivVStacked(
                DivVStacked(
                    badge(f"{level} x{len(items)}", colour),
                    Small(f"{share:.0f}% of findings", cls="text-xs"),
                    cls="flex items-center justify-between gap-3",
                ),
                meter(share, colour, order),
                _finding_items(items, colour),
                cls="space-y-2",
            )
        )
    return DivVStacked(
        DivVStacked(eyebrow("coverage"), *blocks, cls="space-y-4")
    )


def _finding_items(items: list[dict], colour: str):
    rows = []
    for finding in items[:8]:
        literal = finding.get("matched_text") or finding.get("symbol") or "?"
        rows.append(
            DivVStacked(
                Strong(str(literal), cls=f"{MONO} font-semibold", style=f"color:{colour}"),
                Small(finding.get("reason") or "", cls="text-xs"),
                cls="space-y-0",
            )
        )
    if len(items) > 8:
        rows.append(Small(f"+{len(items) - 8} more", cls="text-xs"))
    return DivVStacked(
        *rows, cls="space-y-1.5 pl-3 border-l", style=f"border-color:{colour}"
    )


def execution_panel(exec_data: dict, index: int):
    if not exec_data.get("measured"):
        return panel(
            "Execution",
            DivVStacked(
                Small(NOT_MEASURED, cls="text-sm"),
                Small(
                    "The Python runner cannot execute a .cu file. Compiling it "
                    "needs a ROCm toolchain, which this machine does not have.",
                    cls=BODY,
                ),
                cls="space-y-1.5",
            ),
            kicker="isolated run",
            index=index,
        )

    ok = bool(exec_data.get("ok"))
    children = [
        DivVStacked(
            kv("result", "ok" if ok else "failed"),
            kv("cases", str(exec_data.get("case_count", 0))),
            kv("timed out", "yes" if exec_data.get("timed_out") else "no"),
            kv("exit code", str(exec_data.get("exit_code"))),
            cls="grid grid-cols-2 md:grid-cols-4 gap-4",
        )
    ]
    if not ok:
        failure = ": ".join(
            part for part in (exec_data.get("error_type"), exec_data.get("error_message")) if part
        )
        children.append(note(failure or "the run failed", RED, "failed"))
    children.append(
        DivVStacked(
            eyebrow("isolation"),
            bullet_list(ISOLATION_NOTES, "text-xs"),
            cls="space-y-1",
        )
    )
    return panel("Execution", *children, kicker="isolated run", index=index)


def parity_panel(parity: dict, index: int):
    if not parity.get("measured"):
        return panel(
            "Parity",
            DivVStacked(
                Small(NOT_MEASURED, cls="text-sm"),
                Small(
                    "Parity needs an importable module with a callable to "
                    "compare against its reference.",
                    cls=BODY,
                ),
                cls="space-y-1.5",
            ),
            kicker="does it still compute",
            index=index,
        )

    passed, total = parity.get("passed", 0), parity.get("total", 0)
    share = (passed / total * 100) if total else 0
    all_passed = parity.get("all_passed")
    colour = GREEN if all_passed else (GOLD if passed else RED)
    error = parity.get("max_abs_error")

    return panel(
        "Parity",
        DivVStacked(
            DivVStacked(
                DivVStacked(
                    Small(
                        f"{passed}/{total}",
                        cls="text-3xl font-black tabular-nums",
                        style=f"color:{colour}",
                    ),
                    badge("passing" if all_passed else "failing", colour),
                    cls="flex items-center gap-3",
                ),
                meter(share, colour, 1),
                cls="space-y-2",
            ),
            DivVStacked(
                kv("worst case", parity.get("worst_case", "n/a")),
                kv(
                    "max abs error",
                    f"{error:.3e}" if isinstance(error, (int, float)) else "n/a",
                ),
                cls="grid grid-cols-2 gap-4",
            ),
            Small(
                "The port is compared against an independent reference on the "
                "same device. That is not CUDA compared against ROCm.",
                cls=BODY,
            ),
            cls="space-y-4",
        ),
        kicker="does it still compute",
        index=index,
    )


def benchmark_panel(rows: list[dict], index: int):
    if not rows:
        return panel(
            "Benchmarks",
            DivVStacked(
                Small(NOT_MEASURED, cls="text-sm"),
                Small(
                    "Timing needs a GPU. The verdict bands were calibrated from "
                    "CPU noise and are probably too loose on dedicated AMD "
                    "hardware.",
                    cls=BODY,
                ),
                cls="space-y-1.5",
            ),
            kicker="measured cost",
            index=index,
        )

    blocks = []
    for row in rows:
        ratio, speedup = row.get("ratio"), row.get("speedup")
        if isinstance(ratio, (int, float)):
            headline = f"ratio {ratio:.2f}x"
        elif isinstance(speedup, (int, float)):
            headline = f"speedup {speedup:.2f}x"
        else:
            headline = "n/a"
        blocks.append(
            DivVStacked(
                DivVStacked(
                    Strong(row.get("label", "unnamed"), cls="text-sm font-semibold"),
                    Strong(headline, cls=f"{MONO} font-bold"),
                    cls="flex items-center justify-between gap-3 gap-4",
                ),
                Small(row.get("verdict", "n/a"), cls="text-xs"),
                cls="space-y-1 border-l pl-3",
                style=f"border-color:{BORDER}",
            )
        )
    return panel("Benchmarks", *blocks, kicker="measured cost", index=index)


def narrative_panel(narr: dict, index: int):
    prose = narr.get("prose") or {}
    if not prose:
        children = [Small("No narrative generated.", cls="text-sm font-semibold")]
        reasons = narr.get("reasons") or []
        if reasons:
            children += [Small(reason, cls="text-xs") for reason in reasons]
        children.append(
            Small(
                "This section is optional. It is the only part of the page a "
                "language model could have written, and it never affects a "
                "verdict.",
                cls=BODY,
            )
        )
        return panel("Narrative", *children, kicker="model, optional", index=index)

    blocks = [
        DivVStacked(
            eyebrow(NARRATIVE_TITLES[key]),
            Small(text, cls=BODY),
            cls="space-y-1",
        )
        for key, text in prose.items()
        if key in NARRATIVE_TITLES
    ]
    return panel("Narrative", *blocks, kicker="model, optional", index=index)


def skipped_panel(skipped: list[str], index: int):
    if not skipped:
        return None
    return panel(
        "Skipped",
        DivVStacked(*[Strong(item, cls=MONO) for item in skipped], cls="space-y-1.5"),
        kicker="not measured, not hidden",
        index=index,
    )


# --- result --------------------------------------------------------------

def summary_tiles(summary: dict):
    conv = summary.get("conversion") or {}
    parity = summary.get("parity") or {}
    blockers = sum(
        1 for f in (conv.get("findings") or []) if str(f.get("severity")) == "blocker"
    )
    if parity.get("measured"):
        parity_text = f"{parity.get('passed', 0)}/{parity.get('total', 0)}"
    else:
        parity_text = NOT_MEASURED

    return DivVStacked(
        DivVStacked(
            stat_tile("changes applied", len(conv.get("changes") or []), 0),
            stat_tile("parity cases passed", parity_text, 1),
            stat_tile("blockers", blockers, 2, RED if blockers else GREEN),
            cls="grid grid-cols-1 sm:grid-cols-3 gap-4",
        ),
        cls="space-y-4",
    )


def export_button():
    """A real download, not an htmx swap.

    An htmx GET would render the Markdown into the page. A plain anchor with
    `download` gets the browser to save it, and the response carries a
    content-disposition header for the filename.
    """
    return Div(
        PlainText("Download Markdown report"),
        href="/export.md",
        download="graft-report.md",
        cls="inline-flex items-center px-4 py-2 text-sm font-semibold cursor-pointer",
        style=f"border:1px solid {BORDER};background:{SURFACE_2};"
        f"border-radius:2px;color:{TEXT}",
    )


def render_result(summary: dict):
    """Everything Graft measured, as one fragment.

    A pure function of the summary dict, so the page is testable without a
    server and cannot display a value the pipeline did not produce.
    """
    panels = [
        environment_panel(summary.get("environment") or {}, 1),
        conversion_panel(summary.get("conversion") or {}, 2),
        execution_panel(summary.get("execution") or {}, 3),
        parity_panel(summary.get("parity") or {}, 4),
        benchmark_panel(summary.get("benchmarks") or [], 5),
        narrative_panel(summary.get("narrative") or {}, 6),
        skipped_panel(summary.get("skipped") or [], 7),
    ]
    return DivVStacked(
        DivVStacked(
            DivVStacked(
                eyebrow("result"),
                H2("Migration report", cls="text-2xl font-bold tracking-tight"),
                cls="space-y-0",
            ),
            summary_tiles(summary),
            DivVStacked(export_button(), cls="flex items-center"),
            cls="space-y-4",
        ),
        *[item for item in panels if item is not None],
        cls="space-y-6",
    )


# --- page ------------------------------------------------------------------


def hero():
    return DivVStacked(
        DivVStacked(
            eyebrow("CUDA to ROCm, verified"),
            H1("Graft", cls="shine text-6xl sm:text-7xl font-black tracking-tight mt-2"),
            Subtitle(
                "Port a CUDA file, prove it still computes the right answer, "
                "and see exactly what a human still has to do.",
                cls="text-base max-w-xl mt-3",
            ),
            cls="max-w-3xl space-y-0",
        ),
        cls="ambient py-10",
    )


def drop_zone():
    return Form(
        DivVStacked(
            DivVStacked(
                eyebrow("input"),
                Strong("Choose a .py or .cu file", cls="text-lg font-semibold"),
                Small(
                    "Converted, executed, parity-checked, benchmarked, and "
                    "reported. Nothing on this page is simulated.",
                    cls="text-xs",
                ),
                cls="space-y-1.5 py-7 text-center pointer-events-none",
            ),
            DivVStacked(
                Label(
                    "Select file",
                    for_="graft-file",
                    cls="inline-flex items-center gap-2 px-4 py-2 text-sm "
                    f"font-semibold cursor-pointer brand-grad {RADIUS}",
                ),
                Button(
                    "Run migration",
                    type="submit",
                    cls="px-5 py-2 font-semibold",
                    style=f"background-image:{BRAND_GRADIENT};border:0;"
                    f"border-radius:2px;color:{TEXT}",
                ),
                cls="flex items-center justify-center gap-3 relative z-10",
            ),
            cls="space-y-3",
        ),
        Input(type="file", name="source", id="graft-file", cls="hidden"),
        method="post",
        action="/analyze",
        enctype="multipart/form-data",
        hx_post="/analyze",
        hx_encoding="multipart/form-data",
        hx_target="#result",
        hx_swap="innerHTML",
        hx_indicator="#working",
        cls=f"{SURFACE_CLS} scanline p-6 relative",
    )


def working():
    """htmx toggles .htmx-indicator for the duration of a request."""
    return Div(
        DivVStacked(
            DivVStacked(
                Div(cls="w-2 h-2 rounded-full pulse", style=f"background:{RED}"),
                Strong("Converting and verifying", cls="text-sm font-semibold"),
                cls="flex items-center gap-3 justify-center",
            ),
            Small("The port runs in a separate process.", cls="text-xs"),
            cls="space-y-1.5 py-5",
        ),
        id="working",
        cls="htmx-indicator",
        style="opacity:0;transition:opacity .2s ease",
    )


def page():
    return DivVStacked(
        _STYLE,
        _SCRIPT,
        Div(
            Div(cls="absolute inset-0 dotgrid opacity-70 pointer-events-none"),
            DivVStacked(
                hero(),
                drop_zone(),
                working(),
                Div(id="result", cls="mt-2"),
                Div(cls="h-20"),
                cls="relative max-w-5xl mx-auto px-5 py-10 space-y-8",
            ),
            cls="relative min-h-screen",
            style=f"background:{BG}",
        ),
        cls="min-h-screen",
    )


def error_panel(message: str):
    return note(message, RED, "could not read that")


# --- worker plumbing -------------------------------------------------------


class WorkerFailure(Exception):
    """The child could not produce a summary."""


def run_worker(source: str, filename: str, options: dict) -> dict:
    """Run the pipeline in a child process and return its summary.

    The one place in the UI that spawns anything. Everything about uploaded
    code happens inside that child: importing it, executing it, timing it.

    Failures are raised as WorkerFailure with a message fit for the page.
    The child is killed on timeout, which is the only way to stop code that
    does not cooperate.
    """
    request = json.dumps({"source": source, "filename": filename, "options": options})
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "graft.ui.worker"],
            input=request,
            capture_output=True,
            text=True,
            timeout=config.UI_WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerFailure(
            f"The migration was killed after {config.UI_WORKER_TIMEOUT_S:.0f}s. "
            "It was probably looping."
        ) from exc

    if not completed.stdout.strip():
        raise WorkerFailure(
            f"The worker produced no output (exit {completed.returncode}). "
            f"{(completed.stderr or '').strip()[-300:]}"
        )

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WorkerFailure("The worker returned something that was not JSON.") from exc

    if not payload.get("ok"):
        raise WorkerFailure(str(payload.get("error") or "the worker failed"))

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise WorkerFailure("The worker returned no summary.")

    return summary


# --- app -------------------------------------------------------------------

# This FastHTML version returns (app, app.route); the route helper is unused
# because every route here is registered explicitly below.
app = fast_app(title="Graft")[0]

# The most recent report, for the download link. Per-process and
# deliberately simple: a single-user local tool, not a multi-tenant service.
_LAST_REPORT = "Run a migration to generate a report."


def _validate(name: str, size: int) -> str | None:
    """Return an error message, or None if the upload is acceptable."""
    suffix = Path(name).suffix.lower()
    if suffix not in worker.ALLOWED_SUFFIXES:
        return f"Expected a {' or '.join(sorted(worker.ALLOWED_SUFFIXES))} file, got {suffix or 'no extension'}."
    if size > config.UI_MAX_UPLOAD_BYTES:
        limit = config.UI_MAX_UPLOAD_BYTES // 1024
        return f"That file is larger than the {limit} KB limit."
    return None


@app.get("/")
def index():
    return page()


@app.post("/analyze")
def analyze(source: UploadFile):
    global _LAST_REPORT

    name = Path(getattr(source, "filename", "") or "upload").name
    raw = source.file.read()

    problem = _validate(name, len(raw))
    if problem:
        return error_panel(problem)
    if not raw.strip():
        return error_panel("That file was empty.")

    try:
        summary = run_worker(
            raw.decode("utf-8", errors="replace"),
            name,
            {"run_parity": True, "run_bench": True, "with_narrative": True},
        )
    except WorkerFailure as exc:
        return error_panel(str(exc))

    _LAST_REPORT = str(summary.get("report_text") or _LAST_REPORT)
    return render_result(summary)


@app.get("/export.md")
def export():
    # A raw response, because returning a component would wrap the Markdown
    # in the page layout and the browser would render HTML instead of saving
    # a file.
    return PlainTextResponse(
        _LAST_REPORT,
        media_type="text/markdown; charset=utf-8",
        headers={"content-disposition": 'attachment; filename="graft-report.md"'},
    )


def serve(host: str = "127.0.0.1", port: int = 8000):  # pragma: no cover
    """Run the UI.

    Binds to localhost by default. The child process that executes uploaded
    code shares this user's filesystem and network, so exposing this on a
    public interface hands that to anyone who can reach the port.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)
