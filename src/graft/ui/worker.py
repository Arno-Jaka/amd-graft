"""Disposable child process that runs the pipeline for the web UI.

Threat model
------------
`pipeline.run()` imports the file it analyses, calls its `kernel`, and times
it. For a CLI that is fine: the user is running their own code. For an HTTP
endpoint it is remote code execution, because `import` alone is enough and the
whole module body runs at import time.

So the server never calls `pipeline.run()`. It spawns this module, which:

1. reads a JSON request from stdin: source text, filename, and options,
2. writes the source into a throwaway directory,
3. runs the full pipeline against it,
4. writes a JSON response to stdout: the UI summary plus the Markdown report.

The child inherits the server's OS user, filesystem, and network access. That
is stated plainly rather than papered over: this is process isolation, not a
sandbox, and it is the same guarantee `core/runner.py` gives for a single
kernel run. A hostile upload can still read files or open sockets as the
serving user. What it cannot do is compromise the server process itself,
which is why the endpoint stays safe enough to bind to a real interface only
when the operator has accepted that limit.

Killing the child mid-run is the timeout mechanism. That is a clean way to
stop code, and also a blunt one: a killed child may leave GPU state behind
and cannot be asked to tidy up after itself.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path

PROTOCOL_VERSION = 1

# Files the pipeline can actually route. Anything else is rejected before a
# child is spawned rather than failing confusingly inside it.
ALLOWED_SUFFIXES = {".py", ".cu"}

# How much stdout the analysed file may print before its output is
# truncated. Some is kept because it is often the reason a run failed, but
# it is bounded so a looping print cannot exhaust memory.
CODE_STDOUT_LIMIT = 8192


def _respond(payload: dict) -> None:
    """Write one JSON document to stdout, which is the whole channel."""
    json.dump(payload, sys.stdout)
    sys.stdout.flush()


def _fail(message: str, detail: str = "") -> None:
    _respond({"ok": False, "error": message, "detail": detail})
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail("worker received malformed request", str(exc))
        return 1

    source = request.get("source")
    if not isinstance(source, str) or not source.strip():
        _fail("worker received no source")

    filename = Path(str(request.get("filename") or "upload.py")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        _fail(f"unsupported file type: {suffix or 'none'}")

    options = request.get("options") or {}
    run_parity = bool(options.get("run_parity", True))
    run_bench = bool(options.get("run_bench", True))
    with_narrative = bool(options.get("with_narrative", True))

    from graft import pipeline
    from graft.ui import summary as summary_module

    with tempfile.TemporaryDirectory(prefix="graft-worker-") as tmp:
        target = Path(tmp) / filename
        target.write_text(source, encoding="utf-8")

        # The analysed file is imported by the pipeline, so it runs in this
        # process and inherits both stdout and stderr. A single print() would
        # corrupt the JSON response, and a print loop to stderr would grow
        # the pipe the parent buffers without limit. Both streams are
        # therefore captured and clipped, then forwarded on stderr where the
        # noise is still visible for debugging.
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                result = pipeline.run(
                    target,
                    run_parity=run_parity,
                    run_bench=run_bench,
                    with_narrative=with_narrative,
                )
        except Exception as exc:
            _fail(
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(limit=6),
            )
            return 1
        finally:
            noise = captured.getvalue()
            if noise:
                clipped = noise[:CODE_STDOUT_LIMIT]
                if len(noise) > CODE_STDOUT_LIMIT:
                    clipped += f"\n... {len(noise) - CODE_STDOUT_LIMIT} more characters"
                print(clipped, file=sys.stderr)

        # Flattened inside the child, while the dataclasses are still live.
        # The parent only ever sees JSON.
        payload = {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "summary": summary_module.summarize(result),
        }

    _respond(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
