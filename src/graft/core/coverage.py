
"""Stage 2 — Coverage / readiness scoring.

Detects what genuinely requires human attention: inline PTX, CUTLASS
includes, hardcoded warp-size assumptions, custom inline assembly.
Emits a per-function verdict: production-ready / needs review / manual
port required.
"""


