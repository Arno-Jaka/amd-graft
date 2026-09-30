"""Stage 2 — Sandboxed execution.

Runs converted code in an isolated temp directory under a hard timeout,
capturing stdout, stderr, and exit status. User-supplied code is executed
here, so isolation is not optional.
"""
