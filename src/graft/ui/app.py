



"""Animated web interface for Graft.

The server is only a view and an upload boundary.  Uploaded source is sent to
``worker.py`` for conversion and execution in a disposable child process; this
module never invokes the migration pipeline in-process.  The visible UI is a
motion-led evidence cockpit built from the worker's JSON summary.
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
    Button,
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

BG = "#070a11"
PANEL = "rgba(15, 20, 32, .76)"
LINE = "rgba(211, 232, 255, .13)"
TEXT = "#edf5ff"
DIM = "#9aa9bd"
FAINT = "#647185"
CYAN = "#7cefff"
MINT = "#a6ff73"
VIOLET = "#9d8cff"
ORANGE = "#ffb36a"
PINK = "#ff638b"
RED = "#ff5d70"

VERDICT_COLOR = {
    "production_ready": MINT,
    "needs_review": ORANGE,
    "manual_port": PINK,
    "unsupported": RED,
}
SEVERITY_COLOR = {"blocker": PINK, "review": ORANGE, "info": CYAN}
NOT_MEASURED = "not measured"
BODY = "text-sm leading-relaxed text-[#aab7c9]"

# These are the runner's real guarantees, rather than UI-authored claims.
ISOLATION_NOTES = list(runner.sandbox_notes())


# CSS UI ELEMENT MAP — search these class names below to change each surface:
# graft-shell / edge-line: page canvas and moving background field
# mast / brand-mark / live-pill: top navigation bar
# hero-* / stage-chip: headline and workflow rail
# radar-* : right-hand animated illustration
# upload-* / drop-field / file-slot: source upload console
# working-* : in-flight request indicator
# result-* / metric-* : report heading and summary statistics
# ui-card / data-grid / notice / meter-* : reusable evidence-card internals
_STYLE = NotStr(f"""<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing:border-box; }}
  body {{ background:{BG}; color:{TEXT}; }}
  ::selection {{ background:rgba(124,239,255,.24); color:{TEXT}; }}
  .graft-shell {{ min-height:100vh; overflow:hidden; position:relative; background:radial-gradient(900px 620px at 88% -10%,rgba(91,68,168,.24),transparent 68%),radial-gradient(720px 620px at -8% 26%,rgba(0,179,203,.15),transparent 66%),linear-gradient(155deg,#070a11 0%,#0a0e18 45%,#06080e 100%); }}
  .graft-shell:before {{ content:""; position:absolute; inset:0; pointer-events:none; opacity:.42; background-image:linear-gradient(rgba(151,180,220,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(151,180,220,.045) 1px,transparent 1px); background-size:42px 42px; mask-image:linear-gradient(to bottom,#000,transparent 73%); animation:field-drift 26s linear infinite; }}
  .graft-shell:after {{ content:""; position:absolute; inset:0; pointer-events:none; opacity:.25; background-image:radial-gradient(rgba(255,255,255,.8) .65px,transparent .8px); background-size:23px 23px; animation:star-shift 18s linear infinite; }}
  @keyframes field-drift {{ to {{ background-position:42px 42px; }} }} @keyframes star-shift {{ to {{ background-position:-23px 23px; }} }}
  .edge-line {{ height:1px; opacity:.72; background:linear-gradient(90deg,transparent,{CYAN},{VIOLET},transparent); background-size:200% 100%; animation:line-travel 5s linear infinite; }} @keyframes line-travel {{ to {{ background-position:-200% 0; }} }}
  .mast {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; }} .brand-lockup {{ display:flex; align-items:center; gap:.72rem; letter-spacing:.13em; }}
  .brand-mark {{ width:31px; height:31px; position:relative; display:grid; place-items:center; border:1px solid rgba(124,239,255,.6); border-radius:10px; transform:rotate(45deg); box-shadow:0 0 25px rgba(124,239,255,.18); animation:mark-breathe 3.4s ease-in-out infinite; }} .brand-mark:before {{ content:""; width:10px; height:10px; border-radius:3px; background:linear-gradient(135deg,{CYAN},{VIOLET}); box-shadow:0 0 16px {CYAN}; }} .brand-word {{ font-size:.72rem; font-weight:800; color:{TEXT}; }} @keyframes mark-breathe {{ 50% {{ transform:rotate(45deg) scale(1.08); box-shadow:0 0 35px rgba(124,239,255,.42); }} }}
  .live-pill {{ display:inline-flex; align-items:center; gap:.48rem; padding:.4rem .63rem; color:{DIM}; border:1px solid {LINE}; background:rgba(8,12,21,.44); border-radius:999px; font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; backdrop-filter:blur(12px); }} .live-dot {{ width:.44rem; height:.44rem; border-radius:999px; background:{MINT}; box-shadow:0 0 0 0 rgba(166,255,115,.55); animation:live-pulse 2.2s infinite; }} @keyframes live-pulse {{ 70% {{ box-shadow:0 0 0 8px rgba(166,255,115,0); }} 100% {{ box-shadow:0 0 0 0 rgba(166,255,115,0); }} }}
  .eyebrow {{ color:{CYAN}; font-size:.65rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
  .hero-grid {{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(290px,.85fr); align-items:center; gap:2.5rem; }} .hero-title {{ max-width:700px; font-size:clamp(3.3rem,9vw,7rem); font-weight:900; line-height:.89; letter-spacing:-.075em; }} .hero-title em {{ font-style:normal; color:transparent; background:linear-gradient(100deg,{CYAN},#c8f7ff 35%,{VIOLET} 70%,{PINK}); background-size:250% 100%; -webkit-background-clip:text; background-clip:text; animation:text-flux 7s ease-in-out infinite; }} @keyframes text-flux {{ 0%,100% {{ background-position:100% 0; }} 50% {{ background-position:0 0; }} }} .hero-copy {{ max-width:570px; color:{DIM}; font-size:1.02rem; line-height:1.72; }} .stage-rail {{ display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.5rem; }} .stage-chip {{ display:inline-flex; align-items:center; padding:.47rem .66rem; color:#c8d7e9; border:1px solid {LINE}; border-radius:8px; background:rgba(12,17,29,.48); font-size:.7rem; }} .stage-chip:before {{ content:""; display:block; width:5px; height:5px; margin-right:.45rem; border-radius:50%; background:{CYAN}; box-shadow:0 0 9px {CYAN}; }}
  .radar-wrap {{ min-height:410px; display:grid; place-items:center; position:relative; }} .radar {{ width:min(82vw,360px); aspect-ratio:1; position:relative; border-radius:50%; display:grid; place-items:center; }} .radar:before {{ content:""; position:absolute; inset:0; border-radius:50%; background:repeating-radial-gradient(circle,transparent 0 16%,rgba(124,239,255,.14) 16.35% 16.8%,transparent 17.1% 25%),conic-gradient(from 0deg,transparent 0 75%,rgba(124,239,255,.42) 90%,transparent 112%); border:1px solid rgba(124,239,255,.26); animation:radar-turn 8s linear infinite; box-shadow:inset 0 0 54px rgba(124,239,255,.06),0 0 70px rgba(96,82,210,.14); }} .radar:after {{ content:""; position:absolute; inset:12%; border:1px dashed rgba(157,140,255,.32); border-radius:50%; animation:orbit-turn 14s linear infinite reverse; }} @keyframes radar-turn {{ to {{ transform:rotate(360deg); }} }} @keyframes orbit-turn {{ to {{ transform:rotate(360deg); }} }} .radar-core {{ z-index:1; width:34%; aspect-ratio:1; display:grid; place-items:center; border-radius:50%; background:radial-gradient(circle at 35% 30%,#ecffff,{CYAN} 22%,#2d75aa 58%,#141a3a 73%); box-shadow:0 0 0 12px rgba(124,239,255,.06),0 0 55px rgba(124,239,255,.65); animation:core-hover 4s ease-in-out infinite; }} @keyframes core-hover {{ 50% {{ transform:scale(1.11); box-shadow:0 0 0 20px rgba(124,239,255,.035),0 0 85px rgba(124,239,255,.85); }} }} .radar-node {{ position:absolute; z-index:2; width:.68rem; height:.68rem; border-radius:50%; background:{MINT}; box-shadow:0 0 16px {MINT}; animation:node-blink 2.6s ease-in-out infinite; }} .radar-node.one {{ top:16%; left:23%; }} .radar-node.two {{ right:14%; bottom:27%; animation-delay:.9s; background:{PINK}; box-shadow:0 0 16px {PINK}; }} .radar-node.three {{ right:28%; top:22%; animation-delay:1.5s; background:{ORANGE}; box-shadow:0 0 16px {ORANGE}; }} @keyframes node-blink {{ 50% {{ opacity:.35; transform:scale(.5); }} }} .radar-readout {{ position:absolute; bottom:6%; right:0; z-index:4; padding:.75rem .85rem; min-width:158px; border:1px solid {LINE}; border-radius:10px; background:rgba(10,15,27,.76); box-shadow:0 16px 45px rgba(0,0,0,.24); backdrop-filter:blur(15px); animation:readout-float 5s ease-in-out infinite; }} @keyframes readout-float {{ 50% {{ transform:translateY(-10px); }} }} .radar-readout b {{ display:block; color:{TEXT}; font-size:.75rem; letter-spacing:.04em; }} .radar-readout span {{ display:block; margin-top:.25rem; color:{CYAN}; font:700 .65rem ui-monospace,SFMono-Regular,monospace; letter-spacing:.08em; }}
  .upload-console {{ position:relative; overflow:hidden; padding:1px; border-radius:17px; background:linear-gradient(125deg,rgba(124,239,255,.72),rgba(157,140,255,.34) 32%,rgba(255,99,139,.44) 70%,rgba(124,239,255,.48)); background-size:220% 220%; animation:border-flow 9s linear infinite; box-shadow:0 26px 90px rgba(0,0,0,.24); }} @keyframes border-flow {{ to {{ background-position:220% 220%; }} }} .upload-inner {{ padding:clamp(1.15rem,3vw,2rem); background:linear-gradient(135deg,rgba(15,22,37,.96),rgba(10,14,25,.91)); border-radius:16px; }} .drop-field {{ position:relative; min-height:192px; display:flex; align-items:center; justify-content:center; padding:1.5rem; text-align:center; border:1px dashed rgba(163,221,255,.24); border-radius:12px; overflow:hidden; transition:border-color .25s,background .25s,transform .25s; }} .drop-field:before {{ content:""; position:absolute; width:320px; height:320px; border-radius:50%; background:radial-gradient(circle,rgba(124,239,255,.12),transparent 63%); animation:field-breathe 4.6s ease-in-out infinite; }} .drop-field:after {{ content:""; position:absolute; inset:-50%; background:linear-gradient(105deg,transparent 43%,rgba(255,255,255,.07) 50%,transparent 57%); transform:translateX(-70%) rotate(4deg); animation:sheen 7s ease-in-out infinite; }} @keyframes field-breathe {{ 50% {{ transform:scale(1.2); opacity:.62; }} }} @keyframes sheen {{ 0%,28% {{ transform:translateX(-70%) rotate(4deg); }} 58%,100% {{ transform:translateX(70%) rotate(4deg); }} }} .drop-field.is-dragging {{ border-color:{CYAN}; background:rgba(124,239,255,.07); transform:scale(1.008); }} .upload-contents {{ position:relative; z-index:1; max-width:650px; }} .upload-emblem {{ width:40px; height:40px; display:grid; place-items:center; margin:0 auto .72rem; border:1px solid rgba(124,239,255,.58); border-radius:13px; color:{CYAN}; font:900 1.1rem ui-monospace,monospace; box-shadow:0 0 24px rgba(124,239,255,.16); animation:emblem-bob 3.8s ease-in-out infinite; }} @keyframes emblem-bob {{ 50% {{ transform:translateY(-6px) rotate(5deg); }} }} .file-slot {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-top:1rem; padding:.65rem .75rem; text-align:left; border:1px solid rgba(255,255,255,.08); border-radius:8px; background:rgba(2,5,11,.32); }} .file-slot span {{ overflow:hidden; color:{DIM}; font:.71rem ui-monospace,SFMono-Regular,monospace; text-overflow:ellipsis; white-space:nowrap; }} .file-slot small {{ flex:none; color:{FAINT}; font-size:.61rem; letter-spacing:.09em; text-transform:uppercase; }} .upload-actions {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; padding-top:1rem; }} .choose-button,.run-button,.export-button {{ transition:transform .2s ease,box-shadow .2s ease,background .2s ease; }} .choose-button:hover,.run-button:hover,.export-button:hover {{ transform:translateY(-2px); }} .choose-button {{ padding:.62rem .82rem; border:1px solid rgba(124,239,255,.34); border-radius:8px; color:{CYAN}; background:rgba(124,239,255,.06); font-size:.76rem; font-weight:750; cursor:pointer; }} .run-button {{ padding:.67rem 1rem; border:0; border-radius:8px; color:#071019; background:linear-gradient(105deg,{CYAN},{MINT}); font-size:.76rem; font-weight:900; box-shadow:0 8px 25px rgba(124,239,255,.16); }} .run-button:hover {{ box-shadow:0 12px 35px rgba(124,239,255,.3); }} .guard-copy {{ color:{FAINT}; font-size:.7rem; line-height:1.45; }}
  .working-card {{ padding:1rem 1.15rem; border:1px solid rgba(124,239,255,.25); border-radius:12px; background:rgba(11,17,28,.78); }} .working-stream {{ display:flex; gap:.45rem; align-items:center; }} .working-stream i {{ width:5px; height:5px; border-radius:50%; background:{CYAN}; animation:stream .9s ease-in-out infinite alternate; }} .working-stream i:nth-child(2) {{ animation-delay:.18s; }} .working-stream i:nth-child(3) {{ animation-delay:.36s; }} @keyframes stream {{ to {{ transform:translateY(-5px); opacity:.35; }} }}
  .result-horizon {{ display:flex; align-items:end; justify-content:space-between; gap:1rem; padding-bottom:1.45rem; border-bottom:1px solid {LINE}; }} .result-title {{ font-size:clamp(2rem,4vw,3.25rem); font-weight:850; letter-spacing:-.055em; }} .export-button {{ display:inline-flex; align-items:center; gap:.5rem; padding:.67rem .85rem; color:{TEXT}; border:1px solid {LINE}; border-radius:8px; background:rgba(255,255,255,.045); font-size:.75rem; font-weight:750; }} .export-button:before {{ content:"↓"; color:{CYAN}; font-size:1rem; line-height:0; }}
  .metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.78rem; }} .metric {{ position:relative; min-height:126px; padding:1rem; overflow:hidden; border:1px solid {LINE}; border-radius:12px; background:linear-gradient(145deg,rgba(22,31,47,.74),rgba(12,17,29,.67)); }} .metric:after {{ content:""; position:absolute; width:130px; height:130px; right:-52px; bottom:-70px; border:1px solid rgba(124,239,255,.18); border-radius:50%; box-shadow:0 0 0 15px rgba(124,239,255,.025),0 0 0 30px rgba(124,239,255,.018); transition:transform .35s ease; }} .metric:hover:after {{ transform:scale(1.25); }} .metric-value {{ position:relative; z-index:1; margin-top:1rem; font-size:clamp(1.5rem,3vw,2.15rem); font-weight:900; letter-spacing:-.055em; }} .metric-label {{ color:{FAINT}; font-size:.62rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }} .metric-kicker {{ position:absolute; right:.9rem; top:.9rem; width:.44rem; height:.44rem; border-radius:50%; background:var(--metric-tone,{CYAN}); box-shadow:0 0 14px var(--metric-tone,{CYAN}); }}
  .panel-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }} .span-full {{ grid-column:1 / -1; }} .ui-card {{ position:relative; overflow:hidden; padding:1.28rem; border:1px solid {LINE}; border-radius:14px; background:{PANEL}; box-shadow:0 18px 54px rgba(0,0,0,.14); transform-style:preserve-3d; transition:border-color .28s ease,box-shadow .28s ease,transform .28s ease; }} .ui-card:before {{ content:""; position:absolute; inset:0; pointer-events:none; opacity:0; background:radial-gradient(430px circle at var(--mx,50%) var(--my,50%),rgba(124,239,255,.12),transparent 48%); transition:opacity .28s; }} .ui-card:hover {{ border-color:rgba(124,239,255,.42); box-shadow:0 23px 65px rgba(0,0,0,.3); transform:perspective(900px) rotateX(var(--rx,0deg)) rotateY(var(--ry,0deg)) translateY(-3px); }} .ui-card:hover:before {{ opacity:1; }} .card-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; padding-bottom:1rem; border-bottom:1px solid rgba(211,232,255,.09); }} .card-title {{ margin-top:.22rem; color:{TEXT}; font-size:1.12rem; font-weight:820; letter-spacing:-.035em; }} .card-index {{ color:rgba(124,239,255,.56); font:700 .66rem ui-monospace,SFMono-Regular,monospace; }} .card-body {{ padding-top:1rem; }} .data-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.85rem; }} .data-grid.compact {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .data-grid.pairs {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .kv-label {{ color:{FAINT}; font-size:.61rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }} .kv-value {{ margin-top:.26rem; color:#dae7f6; font-size:.82rem; font-weight:700; overflow-wrap:anywhere; }}
  .badge {{ display:inline-flex; align-items:center; width:fit-content; padding:.3rem .47rem; border:1px solid color-mix(in srgb,var(--badge-color) 42%,transparent); border-radius:999px; color:var(--badge-color); background:color-mix(in srgb,var(--badge-color) 8%,transparent); font-size:.6rem; font-weight:900; letter-spacing:.1em; text-transform:uppercase; }} .verdict-line {{ font-size:clamp(1.5rem,3vw,2.3rem); font-weight:900; letter-spacing:-.06em; }} .basis {{ max-width:750px; color:{DIM}; font-size:.85rem; line-height:1.7; }} .notice {{ padding:.82rem .9rem; border:1px solid var(--notice-color); border-radius:9px; background:color-mix(in srgb,var(--notice-color) 5%,transparent); }} .notice strong {{ display:block; color:#dce9f6; font-size:.8rem; font-weight:650; line-height:1.5; }} .notice .eyebrow {{ color:var(--notice-color); margin-bottom:.28rem; }} .stream-list {{ border-left:1px solid rgba(124,239,255,.23); padding-left:.78rem; }} .stream-row {{ position:relative; }} .stream-row:before {{ content:""; position:absolute; left:-1.02rem; top:.36rem; width:.42rem; height:.42rem; border-radius:50%; background:{CYAN}; box-shadow:0 0 12px rgba(124,239,255,.55); }} .stream-row strong {{ color:#dfeafa; font-size:.73rem; }} .stream-row small {{ display:block; margin-top:.12rem; color:{DIM}; font-size:.72rem; line-height:1.45; }} .meter-track {{ height:5px; overflow:hidden; border-radius:999px; background:rgba(255,255,255,.07); }} .meter-fill {{ height:100%; width:100%; transform:scaleX(0); transform-origin:left; border-radius:inherit; background:var(--meter-color); box-shadow:0 0 12px var(--meter-color); }} html.motion-ready .meter-fill {{ animation:fill-meter .95s cubic-bezier(.16,1,.3,1) forwards; animation-delay:calc(var(--i,0) * 100ms + 200ms); }} @keyframes fill-meter {{ to {{ transform:scaleX(var(--meter-size)); }} }} .finding-group {{ padding:.82rem; border:1px solid rgba(255,255,255,.07); border-radius:9px; background:rgba(0,0,0,.12); }} .finding-row {{ padding-left:.75rem; border-left:1px solid var(--finding-color); }} .finding-row strong {{ color:var(--finding-color); font-size:.72rem; }} .finding-row small {{ display:block; margin-top:.16rem; color:{DIM}; font-size:.71rem; line-height:1.42; }} .parity-score {{ font-size:clamp(2rem,4vw,3.4rem); font-weight:900; letter-spacing:-.075em; line-height:.9; }} .bench-row {{ padding:.75rem .8rem; border-left:2px solid {VIOLET}; border-radius:0 7px 7px 0; background:rgba(157,140,255,.045); }} .narrative-block {{ padding:.74rem .8rem; border-left:2px solid {CYAN}; border-radius:0 8px 8px 0; background:rgba(124,239,255,.035); }}
  html.motion-ready .reveal {{ opacity:0; transform:translateY(18px) scale(.985); }} html.motion-ready .reveal.is-live {{ animation:reveal-up .68s cubic-bezier(.16,1,.3,1) forwards; animation-delay:calc(var(--i,0) * 76ms); }} @keyframes reveal-up {{ to {{ opacity:1; transform:translateY(0) scale(1); }} }}
  @media (max-width:820px) {{ .hero-grid {{ grid-template-columns:1fr; gap:.4rem; }} .radar-wrap {{ min-height:315px; }} .radar {{ width:min(80vw,300px); }} .metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .panel-grid {{ grid-template-columns:1fr; }} .span-full {{ grid-column:auto; }} .data-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .upload-actions {{ align-items:flex-start; flex-direction:column; }} }}
  @media (max-width:500px) {{ .mast .live-pill {{ display:none; }} .hero-title {{ font-size:3.25rem; }} .data-grid,.data-grid.compact,.data-grid.pairs {{ grid-template-columns:1fr 1fr; }} .radar-readout {{ right:-.5rem; }} .result-horizon {{ align-items:flex-start; flex-direction:column; }} }}
  @media (prefers-reduced-motion:reduce) {{ *,*:before,*:after {{ animation-duration:.01ms!important; animation-iteration-count:1!important; scroll-behavior:auto!important; transition-duration:.01ms!important; }} html.motion-ready .reveal {{ opacity:1; transform:none; }} .meter-fill {{ transform:scaleX(var(--meter-size)); }} }}
</style>""")

_SCRIPT = NotStr("""<script>
  (function () {
    function reduced() { return window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches; }
    function count(node) {
      if (node.dataset.countReady || reduced()) return;
      var target = Number(node.dataset.count); if (!Number.isFinite(target)) return;
      node.dataset.countReady = '1'; var decimals = (String(node.dataset.count).split('.')[1] || '').length;
      var began = performance.now();
      function tick(now) { var p = Math.min((now - began) / 820, 1); node.textContent = (target * (1 - Math.pow(1 - p, 4))).toFixed(decimals); if (p < 1) requestAnimationFrame(tick); }
      requestAnimationFrame(tick);
    }
    function tilt(card) {
      if (card.dataset.tiltReady || reduced() || matchMedia('(pointer: coarse)').matches) return;
      card.dataset.tiltReady = '1';
      card.addEventListener('pointermove', function (event) { var box = card.getBoundingClientRect(), x = (event.clientX - box.left) / box.width, y = (event.clientY - box.top) / box.height; card.style.setProperty('--mx', (x * 100) + '%'); card.style.setProperty('--my', (y * 100) + '%'); card.style.setProperty('--rx', ((.5 - y) * 3.1) + 'deg'); card.style.setProperty('--ry', ((x - .5) * 3.1) + 'deg'); });
      card.addEventListener('pointerleave', function () { card.style.removeProperty('--rx'); card.style.removeProperty('--ry'); });
    }
    function intake(root) {
      root.querySelectorAll('[data-drop-field]').forEach(function (field) {
        if (field.dataset.dropReady) return; field.dataset.dropReady = '1';
        var input = field.closest('form').querySelector('input[type=file]'), name = field.querySelector('[data-file-name]');
        function selected(file) { if (file) name.textContent = file.name + ' · ' + Math.max(1, Math.ceil(file.size / 1024)) + ' KB'; }
        input.addEventListener('change', function () { selected(input.files[0]); });
        ['dragenter','dragover'].forEach(function (kind) { field.addEventListener(kind, function (event) { event.preventDefault(); field.classList.add('is-dragging'); }); });
        ['dragleave','drop'].forEach(function (kind) { field.addEventListener(kind, function (event) { event.preventDefault(); field.classList.remove('is-dragging'); }); });
        field.addEventListener('drop', function (event) { if (!event.dataTransfer.files.length) return; try { input.files = event.dataTransfer.files; } catch (_) {} selected(event.dataTransfer.files[0]); });
      });
    }
    function reveal(root) {
      var cards = root.querySelectorAll('.reveal:not([data-reveal-ready])'); if (!cards.length) return;
      cards.forEach(function (card) { card.dataset.revealReady = '1'; });
      if (reduced() || !('IntersectionObserver' in window)) { cards.forEach(function (card) { card.classList.add('is-live'); }); return; }
      var observer = new IntersectionObserver(function (entries) { entries.forEach(function (entry) { if (entry.isIntersecting) { entry.target.classList.add('is-live'); observer.unobserve(entry.target); } }); }, { threshold:.08 });
      cards.forEach(function (card) { observer.observe(card); });
    }
    function hydrate(root) { root.querySelectorAll('[data-count]').forEach(count); root.querySelectorAll('[data-tilt]').forEach(tilt); intake(root); reveal(root); }
    function boot() { document.documentElement.classList.add('motion-ready'); hydrate(document); document.body.addEventListener('htmx:afterSwap', function (event) { hydrate(event.target); }); }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
  }());
</script>""")


# UI PRIMITIVE: vertical stack used to group a visible region's children.
def stack(*children, cls: str = "", style: str | None = None):
    return Div(*children, cls=f"flex flex-col {cls}".strip(), style=style)


# UI ELEMENT: small uppercase section label above titles and readouts.
def eyebrow(text: str):
    return Small(text, cls="eyebrow")


# UI ELEMENT: coloured severity/status pill.
def badge(text: str, color: str):
    return Small(text.upper(), cls="badge", style=f"--badge-color:{color}")


# UI ELEMENT: labelled value used in data grids across the report.
def kv(label: str, value: Any):
    return stack(Small(label, cls="kv-label"), Strong(str(value), cls="kv-value"), cls="min-w-0")


# UI ELEMENT: coloured evidence notice for warnings, context, and failures.
def note(text: str, color: str, kicker: str):
    return Div(eyebrow(kicker), Strong(text, cls="break-words"), cls="notice", style=f"--notice-color:{color}")


# UI ELEMENT: animated horizontal percentage meter.
def meter(share: float, color: str, index: int = 0):
    share = max(0.0, min(100.0, share))
    return Div(Div(cls="meter-fill", style=f"--meter-size:{share / 100:.4f};--meter-color:{color};--i:{index}"), cls="meter-track")


# UI ELEMENT: reusable report card with a section number, title, body, tilt,
# and staged reveal. Change this builder to change every evidence card.
def panel(title: str, *children, kicker: str = "evidence", index: int = 0, full: bool = False):
    return Div(
        Div(stack(eyebrow(kicker), H2(title, cls="card-title"), cls="gap-0"), Small(f"0{index + 1}", cls="card-index"), cls="card-head"),
        Div(*children, cls="card-body"),
        cls=f"ui-card reveal {'span-full' if full else ''}",
        style=f"--i:{index}",
        **{"data-tilt": "true"},
    )


# UI ELEMENT: one top-level evidence statistic. Numeric values count up.
def stat_tile(label: str, value: Any, index: int, tone: str = CYAN):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value_node = Strong("0", cls="metric-value tabular-nums", **{"data-count": str(value)})
    else:
        value_node = Strong(str(value), cls="metric-value")
    return Div(Small(label, cls="metric-label"), value_node, Div(cls="metric-kicker", style=f"--metric-tone:{tone}"), cls="metric reveal", style=f"--i:{index}")


# UI ELEMENT: compact plain-text list, used for process-isolation notes.
def bullet_list(items: list[str], cls: str = BODY):
    return stack(*[Small(item, cls=cls) for item in items], cls="gap-1.5")


# UI PANEL: hardware and backend context for a completed migration.
# The four ``kv`` calls below are respectively the backend, GPU name,
# availability flag, and total VRAM tiles.
def environment_panel(env: dict, index: int):
    vram, gpu = env.get("total_vram_mb"), bool(env.get("gpu_available"))
    details = Div(
        kv("backend", env.get("backend", NOT_MEASURED)),
        kv("GPU", env.get("gpu_name", NOT_MEASURED) if gpu else "none detected"),
        kv("GPU present", "yes" if gpu else "no"),
        kv("total VRAM", f"{vram} MB" if isinstance(vram, (int, float)) else NOT_MEASURED),
        cls="data-grid",
    )
    context = note(
        "A GPU was detected. Measurements are device-local, but they still do not establish CUDA behaviour.", CYAN, "device context"
    ) if gpu else note(
        "No AMD GPU was detected. Every number here is CPU-only; none demonstrates ROCm behaviour.", ORANGE, "CPU-only"
    )
    return panel("Execution environment", stack(details, context, cls="gap-5"), kicker="where this ran", index=index)


# UI ELEMENT: vertical rewrite timeline inside the conversion panel.
def _change_rows(changes: list[dict]):
    if not changes:
        return stack(eyebrow("rewrite trace"), Small("No mechanical rewrites were applied.", cls=BODY), cls="gap-1")
    rows = []
    for change in changes[:14]:
        line = change.get("line")
        where = f":{line}" if isinstance(line, int) else ""
        rows.append(Div(Strong(f"{change.get('symbol', '?')}{where}", cls="font-mono text-xs"), Small(change.get("reason", "") or "", cls="text-xs text-[#9aa9bd]"), cls="stream-row"))
    if len(changes) > 14:
        rows.append(Small(f"+{len(changes) - 14} additional rewrites", cls="text-xs text-[#9aa9bd]"))
    return stack(eyebrow(f"rewrite trace · {len(changes)} event(s)"), stack(*rows, cls="stream-list gap-3"), cls="gap-3")


# UI ELEMENT: individual construct row inside a severity finding group.
def _finding_items(items: list[dict], color: str):
    rows = []
    for finding in items[:8]:
        literal = finding.get("matched_text") or finding.get("symbol") or "?"
        rows.append(Div(Strong(str(literal), cls="font-mono"), Small(finding.get("reason") or ""), cls="finding-row", style=f"--finding-color:{color}"))
    if len(items) > 8:
        rows.append(Small(f"+{len(items) - 8} more", cls="text-xs text-[#9aa9bd]"))
    return stack(*rows, cls="gap-2")


# UI ELEMENT: severity groups, including their badge, percentage, meter, and
# construct rows. The label format is kept stable for report/test consistency.
def _findings(findings: list[dict]):
    if not findings:
        return stack(eyebrow("coverage scan"), Small("No unmapped constructs found.", cls=BODY), cls="gap-1")
    buckets: dict[str, list[dict]] = {}
    for finding in findings:
        buckets.setdefault(str(finding.get("severity", "info")), []).append(finding)
    groups, total = [], len(findings)
    for order, (level, items) in enumerate(buckets.items()):
        color, share = SEVERITY_COLOR.get(level, DIM), len(items) / total * 100
        groups.append(Div(Div(badge(f"{level} x{len(items)}", color), Small(f"{share:.0f}% of findings", cls="text-xs text-[#9aa9bd]"), cls="flex items-center justify-between gap-3"), meter(share, color, order), _finding_items(items, color), cls="finding-group flex flex-col gap-3"))
    return stack(eyebrow("coverage scan"), *groups, cls="gap-3")


# UI PANEL: headline translation verdict, route metadata, rewrite trace, and
# coverage findings. It intentionally takes a full grid row.
def conversion_panel(conv: dict, index: int):
    if not conv.get("measured"):
        return panel("Conversion verdict", Small(NOT_MEASURED, cls="text-sm text-[#aab7c9]"), kicker="translation evidence", index=index, full=True)
    color, changes = VERDICT_COLOR.get(conv.get("verdict", ""), DIM), conv.get("changes") or []
    header = Div(
        stack(Strong(conv.get("verdict_line", NOT_MEASURED), cls="verdict-line", style=f"color:{color}"), badge(conv.get("verdict", "unknown"), color), cls="gap-2"),
        Div(kv("route", conv.get("route", "unknown")), kv("backend", conv.get("backend", NOT_MEASURED)), kv("rewrites", len(changes)), cls="data-grid compact min-w-[280px]"),
        cls="flex flex-col lg:flex-row lg:items-end justify-between gap-6",
    )
    body = Div(Small(conv.get("verdict_basis", ""), cls="basis"), Div(_change_rows(changes), _findings(conv.get("findings") or []), cls="grid grid-cols-1 lg:grid-cols-2 gap-6"), cls="flex flex-col gap-6")
    return panel("Conversion verdict", header, body, kicker="translation evidence", index=index, full=True)


# UI PANEL: child-process outcome and the runner's actual isolation limits.
def execution_panel(exec_data: dict, index: int):
    if not exec_data.get("measured"):
        return panel("Isolated execution", stack(Small(NOT_MEASURED, cls="text-sm text-[#aab7c9]"), Small("The Python runner cannot execute a .cu file. Compilation needs a ROCm toolchain, which this machine does not have.", cls=BODY), cls="gap-2"), kicker="contained run", index=index)
    ok = bool(exec_data.get("ok"))
    details = Div(kv("result", "ok" if ok else "failed"), kv("cases", exec_data.get("case_count", 0)), kv("timed out", "yes" if exec_data.get("timed_out") else "no"), kv("exit code", exec_data.get("exit_code")), cls="data-grid")
    children: list[Any] = [details]
    if not ok:
        failure = ": ".join(part for part in (exec_data.get("error_type"), exec_data.get("error_message")) if part)
        children.append(note(failure or "the run failed", RED, "execution failed"))
    children.append(stack(eyebrow("process boundary"), bullet_list(ISOLATION_NOTES, "text-xs leading-relaxed text-[#9aa9bd]"), cls="gap-2"))
    return panel("Isolated execution", stack(*children, cls="gap-5"), kicker="contained run", index=index)


# UI PANEL: numerical-correctness score, pass/fail badge, error details, and
# animated completion meter.
def parity_panel(parity: dict, index: int):
    if not parity.get("measured"):
        return panel("Parity signal", stack(Small(NOT_MEASURED, cls="text-sm text-[#aab7c9]"), Small("Parity needs an importable module with a callable to compare against its reference.", cls=BODY), cls="gap-2"), kicker="correctness check", index=index)
    passed, total = parity.get("passed", 0), parity.get("total", 0)
    share, complete = (passed / total * 100) if total else 0, bool(parity.get("all_passed"))
    color, error = MINT if complete else (ORANGE if passed else PINK), parity.get("max_abs_error")
    content = stack(
        Div(stack(Strong(f"{passed}/{total}", cls="parity-score", style=f"color:{color}"), badge("passing" if complete else "failing", color), cls="gap-2"), Div(kv("worst case", parity.get("worst_case", "n/a")), kv("max abs error", f"{error:.3e}" if isinstance(error, (int, float)) else "n/a"), cls="data-grid pairs min-w-[220px]"), cls="flex items-end justify-between gap-5"),
        meter(share, color, 1),
        Small("The port is compared with an independent reference on this same device. This is not CUDA compared against ROCm.", cls=BODY),
        cls="gap-5",
    )
    return panel("Parity signal", content, kicker="correctness check", index=index)


# UI PANEL: device timing rows. It deliberately says "not measured" without
# GPU benchmarks rather than displaying invented performance values.
def benchmark_panel(rows: list[dict], index: int):
    if not rows:
        return panel("Benchmark signal", stack(Small(NOT_MEASURED, cls="text-sm text-[#aab7c9]"), Small("Timing needs a GPU. The verdict bands were calibrated from CPU noise and are probably too loose on dedicated AMD hardware.", cls=BODY), cls="gap-2"), kicker="measured cost", index=index)
    blocks = []
    for row in rows:
        ratio, speedup = row.get("ratio"), row.get("speedup")
        headline = f"ratio {ratio:.2f}x" if isinstance(ratio, (int, float)) else f"speedup {speedup:.2f}x" if isinstance(speedup, (int, float)) else "n/a"
        blocks.append(Div(Div(Strong(row.get("label", "unnamed"), cls="text-sm font-bold"), Strong(headline, cls="font-mono text-xs text-[#d9d0ff]"), cls="flex items-center justify-between gap-3"), Small(row.get("verdict", "n/a"), cls="mt-2 block text-xs text-[#9aa9bd]"), cls="bench-row"))
    return panel("Benchmark signal", stack(*blocks, cls="gap-3"), kicker="measured cost", index=index)


# UI PANEL: optional LLM prose. This remains visually separate from measured
# evidence because it never determines a verdict.
def narrative_panel(narr: dict, index: int):
    prose = narr.get("prose") or {}
    if not prose:
        children: list[Any] = [Strong("No narrative generated.", cls="text-sm")]
        children += [Small(reason, cls="text-xs text-[#9aa9bd]") for reason in narr.get("reasons") or []]
        children.append(Small("This optional section is the only part a language model could have written, and it never affects a verdict.", cls=BODY))
        return panel("Narrative layer", stack(*children, cls="gap-3"), kicker="model, optional", index=index)
    blocks = [Div(eyebrow(NARRATIVE_TITLES[key]), Small(text, cls=BODY), cls="narrative-block flex flex-col gap-2") for key, text in prose.items() if key in NARRATIVE_TITLES]
    return panel("Narrative layer", stack(*blocks, cls="gap-4"), kicker="model, optional", index=index)


# UI PANEL: explicit list of stages that were skipped or unavailable.
def skipped_panel(skipped: list[str], index: int):
    if not skipped:
        return None
    return panel("Skipped / unmeasured stages", stack(*[Strong(item, cls="font-mono text-xs text-[#b9c7d9]") for item in skipped], cls="gap-2"), kicker="not hidden", index=index, full=True)


# UI REGION: four animated summary tiles at the start of a migration report:
# rewrites, parity, blocking findings, and execution environment.
def summary_tiles(summary: dict):
    conv, parity, environment = summary.get("conversion") or {}, summary.get("parity") or {}, summary.get("environment") or {}
    blockers = sum(1 for finding in conv.get("findings") or [] if str(finding.get("severity")) == "blocker")
    parity_text = f"{parity.get('passed', 0)}/{parity.get('total', 0)}" if parity.get("measured") else NOT_MEASURED
    device = "GPU found" if environment.get("gpu_available") else "CPU-only"
    return Div(
        # UI ELEMENT: number of mechanically applied rewrites.
        stat_tile("rewrites applied", len(conv.get("changes") or []), 0),
        # UI ELEMENT: passed/total numerical parity cases.
        stat_tile("parity cases", parity_text, 1, MINT),
        # UI ELEMENT: blocker count from the coverage scan.
        stat_tile("blocking findings", blockers, 2, PINK if blockers else MINT),
        # UI ELEMENT: GPU or CPU-only execution context.
        stat_tile("execution context", device, 3, CYAN if environment.get("gpu_available") else ORANGE),
        cls="metric-grid",
    )


# UI CONTROL: browser-native Markdown report download.
def export_button():
    """Use a real download anchor, never an htmx content swap."""
    return Div(PlainText("Download report"), href="/export.md", download="graft-report.md", cls="export-button cursor-pointer")


# UI REGION: complete post-analysis evidence ledger. Change the ``panels``
# order to rearrange report cards; each panel remains an honest worker output.
def render_result(summary: dict):
    """Render an honest evidence ledger from the worker's summary."""
    panels = [
        conversion_panel(summary.get("conversion") or {}, 0),
        environment_panel(summary.get("environment") or {}, 1),
        execution_panel(summary.get("execution") or {}, 2),
        parity_panel(summary.get("parity") or {}, 3),
        benchmark_panel(summary.get("benchmarks") or [], 4),
        narrative_panel(summary.get("narrative") or {}, 5),
        skipped_panel(summary.get("skipped") or [], 6),
    ]
    return Div(
        Div(stack(eyebrow("evidence packet / complete"), H2("Migration report", cls="result-title"), Small("A visual ledger of what ran, what passed, and what still needs human attention.", cls="text-sm text-[#9aa9bd]"), cls="gap-2"), export_button(), cls="result-horizon reveal", style="--i:0"),
        summary_tiles(summary),
        Div(*[item for item in panels if item is not None], cls="panel-grid"),
        cls="mt-10 flex flex-col gap-5",
    )


# UI ELEMENT: persistent top bar with the Graft mark and active-state badge.
def masthead():
    return Div(
        # UI ELEMENT: product mark and wordmark at the top left.
        Div(Div(cls="brand-mark"), Strong("GRAFT", cls="brand-word"), cls="brand-lockup"),
        # UI ELEMENT: live/status pill at the top right.
        Div(Div(cls="live-dot"), Small("Verification cockpit", cls="font-semibold"), cls="live-pill"),
        cls="mast",
    )


# UI ELEMENT: decorative animated radar. It is presentation only and makes no
# claim about measured hardware state; the actual environment panel does that.
def radar():
    return Div(
        # UI ELEMENT: three orbiting decorative radar nodes and centre core.
        Div(
            Div(cls="radar-node one"),
            Div(cls="radar-node two"),
            Div(cls="radar-node three"),
            Div("↗", cls="radar-core text-lg font-black text-[#061017]"),
            cls="radar",
        ),
        # UI ELEMENT: decorative radar caption (not a measured status).
        Div(Strong("EVIDENCE ONLINE"), Small("translate · execute · verify", cls="block"), cls="radar-readout"),
        cls="radar-wrap",
    )


# UI REGION: landing headline, explainer text, workflow chips, and radar.
def hero():
    return Div(
        stack(
            # UI ELEMENT: hero eyebrow.
            eyebrow("CUDA → ROCm / verified migration"),
            # UI ELEMENT: primary marketing headline.
            H1(NotStr("Port the code.<br><em>Keep the proof.</em>"), cls="hero-title"),
            # UI ELEMENT: product explainer paragraph.
            Subtitle("Graft turns a CUDA source file into a transparent migration record: mechanical changes, execution evidence, parity checks, and the gaps that still need an engineer.", cls="hero-copy mt-2"),
            # UI ELEMENT: four-stage workflow chips.
            Div(Small("Inspect", cls="stage-chip"), Small("Translate", cls="stage-chip"), Small("Execute", cls="stage-chip"), Small("Verify", cls="stage-chip"), cls="stage-rail"),
            cls="gap-4",
        ),
        # UI ELEMENT: right-hand animated illustration.
        radar(),
        cls="hero-grid pt-14 pb-8",
    )


# UI REGION: animated upload console. Its elements are, in order: drop field,
# hidden file picker, filename readout, choose-source control, and submit
# control. The data attributes are used by the small enhancement script.
def drop_zone():
    return Div(
        Form(
            # UI ELEMENT: drag-and-drop target and its instructional content.
            Div(
                Div(
                    # UI ELEMENT: upload arrow emblem.
                    Div("↥", cls="upload-emblem"),
                    # UI ELEMENT: upload section label.
                    eyebrow("source intake"),
                    # UI ELEMENT: upload call to action.
                    Strong("Drop a CUDA workload into the field", cls="mt-2 block text-xl font-extrabold tracking-tight"),
                    # UI ELEMENT: supported-file explanation.
                    Small("Python and CUDA C++ are supported. Graft reports only the checks it actually ran.", cls="mt-2 block text-sm text-[#9aa9bd]"),
                    # UI ELEMENT: selected filename and upload-size limit.
                    Div(Small("Awaiting a .py or .cu file", **{"data-file-name": "true"}), Small("max 1 MB"), cls="file-slot"),
                    cls="upload-contents",
                ),
                cls="drop-field",
                **{"data-drop-field": "true"},
            ),
            # UI ELEMENT: hidden native file picker opened by the label below.
            Input(type="file", name="source", id="graft-file", accept=".py,.cu", cls="hidden"),
            # UI ELEMENT: upload action row.
            Div(
                # UI ELEMENT: visible trigger for the hidden file picker.
                Label("Choose source", for_="graft-file", cls="choose-button"),
                Div(
                    # UI ELEMENT: process-isolation assurance copy.
                    Small("Runs in a disposable child process.", cls="guard-copy"),
                    # UI ELEMENT: form submit button that starts the migration.
                    Button("Run migration", type="submit", cls="run-button"),
                    cls="flex items-center gap-4",
                ),
                cls="upload-actions",
            ),
            cls="upload-inner",
            method="post",
            action="/analyze",
            enctype="multipart/form-data",
            hx_post="/analyze",
            hx_encoding="multipart/form-data",
            hx_target="#result",
            hx_swap="innerHTML",
            hx_indicator="#working",
        ),
        cls="upload-console",
    )


# UI ELEMENT: htmx request indicator shown while the disposable worker runs.
def working():
    return Div(
        # UI ELEMENT: in-flight status card shown only while htmx awaits /analyze.
        Div(
            Div(
                # UI ELEMENT: three-dot animated activity indicator.
                Div(Div(), Div(), Div(), cls="working-stream"),
                # UI ELEMENT: in-flight status headline.
                Strong("Building the evidence packet", cls="text-sm font-bold"),
                cls="flex items-center gap-3",
            ),
            # UI ELEMENT: in-flight isolation description.
            Small("The uploaded code is running in its own process.", cls="mt-1 block text-xs text-[#9aa9bd]"),
            cls="working-card",
        ),
        id="working",
        cls="htmx-indicator mt-4",
    )


# UI PAGE: assembles the visual shell in top-to-bottom screen order.
# Keep _STYLE and _SCRIPT first so all marked elements are ready to animate.
def page():
    return Div(
        # UI ELEMENT: page-scoped visual rules and animation definitions.
        _STYLE,
        # UI ELEMENT: progressive enhancement for counters, drag/drop, tilt, and reveal.
        _SCRIPT,
        # UI ELEMENT: animated gradient line at the top of the viewport.
        Div(cls="edge-line"),
        # UI ELEMENT: centred application content column.
        Div(
            masthead(),
            hero(),
            drop_zone(),
            working(),
            # UI ELEMENT: htmx target where the migration evidence ledger appears.
            Div(id="result"),
            # UI ELEMENT: bottom breathing room.
            Div(cls="h-20"),
            cls="relative mx-auto max-w-6xl px-5 py-6 sm:px-8",
        ),
        cls="graft-shell",
    )


# UI ELEMENT: inline upload/worker error notice, rendered in the result slot.
def error_panel(message: str):
    return Div(note(message, RED, "could not read that"), cls="mt-6")


class WorkerFailure(Exception):
    """The disposable child could not return a useful summary."""


def run_worker(source: str, filename: str, options: dict) -> dict:
    """Run uploaded source only in the disposable worker process."""
    request = json.dumps({"source": source, "filename": filename, "options": options})
    try:
        completed = subprocess.run([sys.executable, "-m", "graft.ui.worker"], input=request, capture_output=True, text=True, timeout=config.UI_WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise WorkerFailure(f"The migration was killed after {config.UI_WORKER_TIMEOUT_S:.0f}s. It was probably looping.") from exc
    if not completed.stdout.strip():
        raise WorkerFailure(f"The worker produced no output (exit {completed.returncode}). {(completed.stderr or '').strip()[-300:]}")
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


app = fast_app(title="Graft")[0]
_LAST_REPORT = "Run a migration to generate a report."


def _validate(name: str, size: int) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix not in worker.ALLOWED_SUFFIXES:
        return f"Expected a {' or '.join(sorted(worker.ALLOWED_SUFFIXES))} file, got {suffix or 'no extension'}."
    if size > config.UI_MAX_UPLOAD_BYTES:
        return f"That file is larger than the {config.UI_MAX_UPLOAD_BYTES // 1024} KB limit."
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
        summary = run_worker(raw.decode("utf-8", errors="replace"), name, {"run_parity": True, "run_bench": True, "with_narrative": True})
    except WorkerFailure as exc:
        return error_panel(str(exc))
    _LAST_REPORT = str(summary.get("report_text") or _LAST_REPORT)
    return render_result(summary)


@app.get("/export.md")
def export():
    return PlainTextResponse(_LAST_REPORT, media_type="text/markdown; charset=utf-8", headers={"content-disposition": 'attachment; filename="graft-report.md"'})


def serve(host: str = "127.0.0.1", port: int = 8000):  # pragma: no cover
    """Run the local UI server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)
