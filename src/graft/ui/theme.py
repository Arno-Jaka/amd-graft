"""The visual design system for Graft.

The look is AMD-inspired without using any AMD trademark: near-black canvas,
their signature red, hard angular geometry built from clipped corners and
chevrons, and motion that snaps rather than drifts. Nothing here is loaded
from a CDN or an external font, so the interface renders identically offline
and ships no third-party assets.

Two decisions drive most of the sheet:

* Angles instead of curves. Cards use ``clip-path`` to cut a corner rather than
  rounding it, and chevrons carry the eye down the page. This is the single
  biggest difference from a typical glassmorphic dashboard.
* Short, decisive motion. Durations sit between 120ms and 900ms with
  ``cubic-bezier(.16,1,.3,1)``, which leaves fast. Long soft fades read as
  consumer-app; sharp cuts read as instrumentation.

``prefers-reduced-motion`` disables every animation and transition, and the
reveal and meter animations have explicit "already finished" end states so
content is never hidden behind an animation that will not run.
"""

from __future__ import annotations

from fastcore.xml import NotStr

# --- palette ---------------------------------------------------------------

VOID = "#000000"
INK = "#070708"
PANEL = "#0d0d0f"
PANEL_RAISED = "#131316"
EDGE = "rgba(255,255,255,.11)"
EDGE_BRIGHT = "rgba(255,255,255,.24)"
TEXT = "#ffffff"
MUTED = "#a3a3a6"
FAINT = "#6b6b70"

# The accent. Red on black is the whole identity, so it is used sparingly and
# only where something is actionable or a verdict is at stake.
RED = "#ed1c24"
RED_DEEP = "#b3121a"
RED_GLOW = "rgba(237,28,36,.55)"

# Secondary tones for verdict states. Kept desaturated so red stays dominant.
PASS = "#4ade80"
WARN = "#fbbf24"
FAIL = "#f87171"
INFO = "#7dd3fc"

VERDICT_COLOR = {
    "production_ready": PASS,
    "needs_review": WARN,
    "manual_port": "#fb923c",
    "unsupported": FAIL,
}
SEVERITY_COLOR = {"blocker": FAIL, "review": WARN, "info": INFO}

# Corner cuts. Two values so nested elements do not all notch identically.
NOTCH = "14px"
NOTCH_SM = "8px"

# Angled clip paths, expressed as a polygon so the cut scales with the box.
CLIP_PANEL = (
    "polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px))"
)
CLIP_SMALL = (
    "polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px))"
)

MOTION = "cubic-bezier(.16,1,.3,1)"

MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = (
    "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
)

# The full sheet. Kept in one place so a colour or duration can be changed in
# one edit, and so the class names below have a single source of truth.
STYLE = NotStr(f"""<style>
  :root {{
    color-scheme: dark;
    --void: {VOID};
    --ink: {INK};
    --panel: {PANEL};
    --raised: {PANEL_RAISED};
    --edge: {EDGE};
    --edge-bright: {EDGE_BRIGHT};
    --text: {TEXT};
    --muted: {MUTED};
    --faint: {FAINT};
    --red: {RED};
    --red-deep: {RED_DEEP};
    --notch: {NOTCH};
    --notch-sm: {NOTCH_SM};
    --clip-panel: {CLIP_PANEL};
    --clip-small: {CLIP_SMALL};
    --motion: {MOTION};
    --mono: {MONO};
    --sans: {SANS};
  }}
  * {{ box-sizing:border-box; }}
  html {{ scrollbar-color: {RED_DEEP} {INK}; }}
  body {{ margin:0; background:var(--void); color:var(--text); font-family:var(--sans); -webkit-font-smoothing:antialiased; }}
  ::selection {{ background:{RED}; color:{TEXT}; }}

  /* --- canvas ---------------------------------------------------------
     A blueprint grid, a slow diagonal sweep, and a red bloom in one corner.
     The sweep is the only ambient motion on the page; everything else moves
     in response to the reader. */
  .shell {{ position:relative; min-height:100vh; overflow:hidden; background:
      radial-gradient(1100px 700px at 100% 0%, rgba(237,28,36,.16), transparent 62%),
      linear-gradient(168deg,#000 0%,#08080a 46%,#000 100%); }}
  .shell:before {{ content:""; position:absolute; inset:0; pointer-events:none; z-index:0;
      background-image:
        linear-gradient(rgba(255,255,255,.038) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.038) 1px, transparent 1px);
      background-size:56px 56px;
      mask-image:linear-gradient(to bottom,#000 0%,transparent 62%);
      animation:grid-slide 22s linear infinite; }}
  .shell:after {{ content:""; position:absolute; inset:-40%; pointer-events:none; z-index:0;
      background:linear-gradient(104deg,transparent 44%,rgba(237,28,36,.10) 50%,transparent 56%);
      animation:bar-sweep 11s var(--motion) infinite; }}
  @keyframes grid-slide {{ to {{ background-position:56px 56px; }} }}
  @keyframes bar-sweep {{ 0% {{ transform:translateX(-62%) rotate(3deg); }} 55%,100% {{ transform:translateX(62%) rotate(3deg); }} }}
  .layer {{ position:relative; z-index:1; }}

  /* --- masthead ------------------------------------------------------- */
  .mast {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; }}
  .lockup {{ display:flex; align-items:center; gap:.7rem; }}
  /* A chevron mark, cut rather than drawn: two stacked parallelograms. */
  .mark {{ position:relative; width:34px; height:34px; clip-path:var(--clip-small);
      background:linear-gradient(135deg,var(--red),var(--red-deep));
      box-shadow:0 0 26px {RED_GLOW}; animation:mark-throb 3.6s ease-in-out infinite; }}
  .mark:before,.mark:after {{ content:""; position:absolute; left:7px; right:7px; height:4px;
      background:{TEXT}; clip-path:polygon(0 0,78% 0,100% 100%,22% 100%); }}
  .mark:before {{ top:9px; }} .mark:after {{ bottom:9px; opacity:.55; }}
  @keyframes mark-throb {{ 50% {{ box-shadow:0 0 44px {RED_GLOW}; transform:scale(1.06); }} }}
  .wordmark {{ font-size:.7rem; font-weight:800; letter-spacing:.2em; }}
  .live {{ display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .6rem;
      clip-path:var(--clip-small); border:1px solid var(--edge); background:rgba(255,255,255,.03);
      color:var(--muted); font-size:.62rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
  .live i {{ width:.4rem; height:.4rem; background:var(--red); box-shadow:0 0 12px var(--red);
      clip-path:polygon(0 0,100% 50%,0 100%); animation:live-pulse 1.9s var(--motion) infinite; }}
  @keyframes live-pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.25; }} }}

  /* --- rules and type ------------------------------------------------- */
  .rule {{ height:2px; border:0; margin:0;
      background:linear-gradient(90deg,var(--red) 0%,var(--red-deep) 22%,transparent 55%);
      clip-path:polygon(0 0,100% 0,calc(100% - 10px) 100%,0 100%); }}
  .eyebrow {{ color:var(--red); font-size:.62rem; font-weight:800; letter-spacing:.2em;
      text-transform:uppercase; }}
  .display {{ font-size:clamp(3rem,8.4vw,6.6rem); font-weight:900; line-height:.86;
      letter-spacing:-.055em; text-transform:uppercase; }}
  /* The one gradient in the interface, kept to a single word so it reads as
     an accent rather than a theme. */
  .display em {{ font-style:normal; color:transparent;
      background:linear-gradient(96deg,var(--red) 0%,#ff6b6b 42%,var(--text) 100%);
      -webkit-background-clip:text; background-clip:text; }}
  .lede {{ max-width:60ch; color:var(--muted); font-size:1rem; line-height:1.75; }}
  .mono {{ font-family:var(--mono); }}
  .rail {{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.6rem; }}
  .rail span {{ display:inline-flex; align-items:center; gap:.45rem; padding:.4rem .65rem;
      clip-path:var(--clip-small); border:1px solid var(--edge); color:var(--muted);
      font-size:.68rem; letter-spacing:.05em; }}
  .rail span:before {{ content:""; width:6px; height:6px; background:var(--red);
      clip-path:polygon(0 0,100% 50%,0 100%); }}

  /* --- plates: the clipped card primitive ----------------------------- */
  .plate {{ position:relative; padding:1.3rem; clip-path:var(--clip-panel);
      background:linear-gradient(150deg,var(--panel),#08080a); border:0;
      box-shadow:inset 0 0 0 1px var(--edge); transition:box-shadow .22s var(--motion),transform .22s var(--motion); }}
  /* A hairline down the leading edge picks up the accent on hover. */
  .plate:before {{ content:""; position:absolute; left:0; top:14px; bottom:14px; width:2px;
      background:var(--red); transform:scaleY(0); transform-origin:top;
      transition:transform .3s var(--motion); }}
  .plate:hover {{ box-shadow:inset 0 0 0 1px var(--edge-bright),0 18px 48px rgba(0,0,0,.55);
      transform:translateY(-2px); }}
  .plate:hover:before {{ transform:scaleY(1); }}
  .plate-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;
      padding-bottom:.9rem; border-bottom:1px solid var(--edge); }}
  .plate-title {{ margin-top:.2rem; font-size:1.08rem; font-weight:800; letter-spacing:-.02em; }}
  .plate-idx {{ color:var(--red); font:700 .64rem var(--mono); }}
  .plate-body {{ padding-top:1rem; }}
  .grid {{ display:grid; gap:.9rem; }}
  .g2 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .g3 {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
  .g4 {{ grid-template-columns:repeat(4,minmax(0,1fr)); }}
  .wide {{ grid-column:1 / -1; }}

  /* --- figures -------------------------------------------------------- */
  .figure {{ position:relative; padding:1rem; overflow:hidden;
      clip-path:var(--clip-small); background:var(--raised);
      box-shadow:inset 0 0 0 1px var(--edge); }}
  .figure:after {{ content:""; position:absolute; right:0; top:0; width:14px; height:14px;
      background:var(--tone,var(--red)); clip-path:polygon(100% 0,100% 100%,0 0); }}
  .figure-label {{ color:var(--faint); font-size:.6rem; font-weight:800; letter-spacing:.16em;
      text-transform:uppercase; }}
  .figure-value {{ margin-top:.85rem; font-size:clamp(1.5rem,2.9vw,2.1rem); font-weight:900;
      letter-spacing:-.05em; }}

  /* --- badges and verdicts -------------------------------------------- */
  .tag {{ display:inline-flex; align-items:center; padding:.28rem .5rem; width:fit-content;
      clip-path:var(--clip-small); color:var(--tone,var(--muted));
      background:color-mix(in srgb,var(--tone,var(--muted)) 12%,transparent);
      box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--tone,var(--muted)) 45%,transparent);
      font-size:.58rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }}
  .verdict {{ font-size:clamp(1.4rem,2.8vw,2.2rem); font-weight:900; letter-spacing:-.05em;
      text-transform:uppercase; }}

  /* --- data rows ------------------------------------------------------ */
  .kv {{ display:flex; flex-direction:column; gap:.22rem; }}
  .kv-label {{ color:var(--faint); font-size:.58rem; font-weight:800; letter-spacing:.14em;
      text-transform:uppercase; }}
  .kv-value {{ font-size:.82rem; font-weight:700; overflow-wrap:anywhere; }}
  .feed {{ border-left:1px solid var(--edge); padding-left:.85rem; }}
  .feed-row {{ position:relative; }}
  /* Chevrons instead of dots, matching the mark. */
  .feed-row:before {{ content:""; position:absolute; left:-1.09rem; top:.34rem; width:8px; height:8px;
      background:var(--red); clip-path:polygon(0 0,100% 50%,0 100%); }}
  .feed-row strong {{ display:block; font-size:.73rem; }}
  .feed-row small {{ display:block; margin-top:.1rem; color:var(--muted); font-size:.72rem;
      line-height:1.5; }}
  .notice {{ padding:.8rem .9rem; clip-path:var(--clip-small);
      background:color-mix(in srgb,var(--tone,var(--muted)) 7%,transparent);
      box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--tone,var(--muted)) 40%,transparent); }}
  .notice strong {{ display:block; font-size:.79rem; font-weight:650; line-height:1.55; }}

  /* --- meters --------------------------------------------------------- */
  .meter {{ height:6px; background:rgba(255,255,255,.07); overflow:hidden;
      clip-path:polygon(0 0,100% 0,calc(100% - 3px) 100%,0 100%); }}
  .meter i {{ display:block; height:100%; background:var(--tone,var(--red));
      box-shadow:0 0 12px var(--tone,var(--red)); transform:scaleX(0); transform-origin:left; }}
  html.motion .meter i {{ animation:meter-fill .8s var(--motion) forwards;
      animation-delay:calc(var(--i,0) * 90ms + 180ms); }}
  @keyframes meter-fill {{ to {{ transform:scaleX(var(--size,.5)); }} }}

  /* --- forms ----------------------------------------------------------
     Inputs are angled to match the plates, with a red caret accent rather
     than a rounded focus ring. */
  .field {{ display:flex; flex-direction:column; gap:.4rem; }}
  .field label {{ color:var(--faint); font-size:.6rem; font-weight:800; letter-spacing:.14em;
      text-transform:uppercase; }}
  .field input,.field select {{ width:100%; padding:.62rem .7rem; color:var(--text);
      background:#000; border:0; clip-path:var(--clip-small);
      box-shadow:inset 0 0 0 1px var(--edge); font-family:var(--mono); font-size:.82rem;
      transition:box-shadow .18s var(--motion); }}
  .field input:focus,.field select:focus {{ outline:none; box-shadow:inset 0 0 0 1px var(--red); }}
  .field input::placeholder {{ color:#45454a; }}
  .hint {{ color:var(--faint); font-size:.66rem; line-height:1.5; }}
  .btn {{ display:inline-flex; align-items:center; gap:.5rem; padding:.68rem .95rem; cursor:pointer;
      clip-path:var(--clip-small); border:0; color:var(--text);
      background:linear-gradient(135deg,var(--red),var(--red-deep));
      box-shadow:0 10px 30px rgba(237,28,36,.24);
      font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
      transition:transform .18s var(--motion),box-shadow .18s var(--motion),filter .18s var(--motion); }}
  .btn:hover {{ transform:translateY(-2px); box-shadow:0 16px 40px rgba(237,28,36,.36); filter:brightness(1.08); }}
  .btn:active {{ transform:translateY(0); }}
  .btn.ghost {{ background:transparent; box-shadow:inset 0 0 0 1px var(--edge-bright); color:var(--muted); }}
  .btn.ghost:hover {{ color:var(--text); box-shadow:inset 0 0 0 1px var(--red); }}

  /* --- intake --------------------------------------------------------- */
  .intake {{ position:relative; padding:clamp(1.1rem,3vw,1.9rem); clip-path:var(--clip-panel);
      background:linear-gradient(140deg,#101013,#08080a); box-shadow:inset 0 0 0 1px var(--edge); }}
  .drop {{ position:relative; min-height:180px; display:flex; align-items:center; justify-content:center;
      padding:1.5rem; text-align:center; overflow:hidden; cursor:pointer;
      border:1px dashed rgba(237,28,36,.34); transition:background .2s var(--motion),border-color .2s var(--motion); }}
  .drop:before {{ content:""; position:absolute; inset:0;
      background:repeating-linear-gradient(-45deg,rgba(237,28,36,.05) 0 10px,transparent 10px 20px);
      animation:drop-drift 6s linear infinite; }}
  @keyframes drop-drift {{ to {{ background-position:28px 0; }} }}
  .drop.over {{ border-color:var(--red); background:rgba(237,28,36,.07); }}
  .drop-inner {{ position:relative; z-index:1; max-width:620px; }}
  /* Corner ticks, like a viewfinder. */
  .drop-mark {{ width:52px; height:52px; margin:0 auto .85rem; background:var(--red);
      clip-path:polygon(0 0,38% 0,62% 38%,62% 0,100% 0,100% 100%,0 100%);
      animation:mark-throb 3.6s ease-in-out infinite; }}
  .file-name {{ margin-top:.5rem; color:var(--muted); font:600 .74rem var(--mono); }}

  /* --- working indicator ---------------------------------------------- */
  .working {{ display:flex; align-items:center; gap:.75rem; padding:.9rem 1.05rem;
      clip-path:var(--clip-small); background:var(--raised); box-shadow:inset 0 0 0 1px var(--edge); }}
  /* A bar that fills, rather than dots that bounce. Reads as progress. */
  .working i {{ flex:1; height:4px; background:rgba(255,255,255,.07); overflow:hidden; position:relative; }}
  .working i:after {{ content:""; position:absolute; inset:0; width:38%;
      background:linear-gradient(90deg,transparent,var(--red),transparent);
      animation:run-bar 1.25s var(--motion) infinite; }}
  @keyframes run-bar {{ from {{ transform:translateX(-100%); }} to {{ transform:translateX(320%); }} }}

  /* --- reveal --------------------------------------------------------- */
  /* Hidden only once JS has confirmed motion is acceptable, so a failed
     script can never leave the page blank. */
  html.motion .reveal {{ opacity:0; transform:translateY(16px); }}
  html.motion .reveal.live {{ animation:rise .6s var(--motion) forwards;
      animation-delay:calc(var(--i,0) * 65ms); }}
  @keyframes rise {{ to {{ opacity:1; transform:none; }} }}

  /* --- responsive ----------------------------------------------------- */
  @media (max-width:900px) {{
    .g4 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .g2,.g3 {{ grid-template-columns:1fr; }}
    .wide {{ grid-column:auto; }}
  }}
  @media (max-width:560px) {{
    .mast .live {{ display:none; }}
    .display {{ font-size:2.7rem; }}
    .g4 {{ grid-template-columns:1fr; }}
  }}
  @media (prefers-reduced-motion:reduce) {{
    *,*:before,*:after {{ animation-duration:.01ms!important; animation-iteration-count:1!important;
      transition-duration:.01ms!important; scroll-behavior:auto!important; }}
    html.motion .reveal {{ opacity:1; transform:none; }}
    .meter i {{ transform:scaleX(var(--size,.5)); }}
  }}
</style>""")

# Motion and interaction. Kept small and defensive: every entry point checks
# for reduced motion first, and every listener is attached at most once.
SCRIPT = NotStr("""<script>
  (function () {
    var reduced = function () {
      return window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    };

    function count(node) {
      if (node.dataset.done || reduced()) return;
      var target = Number(node.dataset.count);
      if (!isFinite(target)) return;
      node.dataset.done = '1';
      var decimals = (String(node.dataset.count).split('.')[1] || '').length;
      var t0 = performance.now();
      function tick(now) {
        var p = Math.min((now - t0) / 700, 1);
        var eased = 1 - Math.pow(1 - p, 3);
        node.textContent = (target * eased).toFixed(decimals);
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    function intake(root) {
      root.querySelectorAll('[data-drop]').forEach(function (field) {
        if (field.dataset.ready) return;
        field.dataset.ready = '1';
        var form = field.closest('form');
        if (!form) return;
        var input = form.querySelector('input[type=file]');
        var label = field.querySelector('[data-file-name]');
        function show(file) {
          if (label && file) {
            label.textContent = file.name + '  ' + Math.max(1, Math.ceil(file.size / 1024)) + ' KB';
          }
        }
        if (input) {
          input.addEventListener('change', function () { show(input.files[0]); });
        }
        ['dragenter', 'dragover'].forEach(function (kind) {
          field.addEventListener(kind, function (e) {
            e.preventDefault(); field.classList.add('over');
          });
        });
        ['dragleave', 'drop'].forEach(function (kind) {
          field.addEventListener(kind, function (e) {
            e.preventDefault(); field.classList.remove('over');
          });
        });
        field.addEventListener('drop', function (e) {
          if (!e.dataTransfer.files.length || !input) return;
          try { input.files = e.dataTransfer.files; } catch (_) {}
          show(e.dataTransfer.files[0]);
        });
      });
    }

    function reveal(root) {
      var nodes = root.querySelectorAll('.reveal:not([data-seen])');
      if (!nodes.length) return;
      Array.prototype.forEach.call(nodes, function (n) { n.dataset.seen = '1'; });
      if (reduced() || !('IntersectionObserver' in window)) {
        Array.prototype.forEach.call(nodes, function (n) { n.classList.add('live'); });
        return;
      }
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('live');
            io.unobserve(entry.target);
          }
        });
      }, { threshold: 0.06 });
      Array.prototype.forEach.call(nodes, function (n) { io.observe(n); });
    }

    function hydrate(root) {
      root.querySelectorAll('[data-count]').forEach(count);
      intake(root);
      reveal(root);
    }

    function boot() {
      // Added only after the checks pass, so .reveal is never left hidden.
      document.documentElement.classList.add('motion');
      hydrate(document);
      document.body.addEventListener('htmx:afterSwap', function (e) { hydrate(e.target); });
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', boot);
    } else {
      boot();
    }
  }());
</script>""")
