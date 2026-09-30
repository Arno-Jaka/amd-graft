"""Stage 1 — Deterministic CUDA C++ conversion backend.

Thin wrapper around hipify-perl / hipify-clang. Captures stdout, stderr,
and warnings. Isolated so the backend can be swapped without touching
callers. Must degrade gracefully when the tool is not installed.
"""