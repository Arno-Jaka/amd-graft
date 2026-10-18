"""Disposable child process that runs the pipeline for a UI.

Threat model
------------
`pipeline.run()` imports the file it analyses, calls its `kernel`, and times
it. For a CLI that is fine: the user is running their own code. For an HTTP
endpoint it is remote code execution, because `import` alone is enough and the
whole module body runs at import time.

So neither UI calls `pipeline.run()` in-process. Both spawn this module, which:

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
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from graft import config

PROTOCOL_VERSION = 1

ALLOWED_SUFFIXES = {".py", ".cu"}

CODE_STDOUT_LIMIT = 8192


class WorkerFailure(Exception):
    """The disposable worker could not be run or did not answer."""


def run_worker(source: str, filename: str, options: dict) -> dict:
    """Run source only in the disposable worker process.

    This is the launcher half of the module: both the web UI and the TUI call
    it, and it is the only place the child is spawned. The analyzed source
    never imports into the caller's long-lived process.
    """
    request = json.dumps({"source": source, "filename": filename, "options": options})
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "graft.worker"],
            input=request,
            capture_output=True,
            text=True,
            timeout=config.UI_WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerFailure(
            f"The migration was killed after {config.UI_WORKER_TIMEOUT_S:.0f}s. It was probably looping."
        ) from exc
    if not completed.stdout.strip():
        raise WorkerFailure(
            f"The worker produced no output (exit {completed.returncode}). {(completed.stderr or '').strip()[-300:]}"
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
    review_options = options.get("ai_review")
    if not isinstance(review_options, dict):
        review_options = None

    from graft import pipeline
    from graft import summary as summary_module

    with tempfile.TemporaryDirectory(prefix="graft-worker-") as tmp:
        target = Path(tmp) / filename
        target.write_text(source, encoding="utf-8")

        captured = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(captured),
                contextlib.redirect_stderr(captured),
            ):
                result = pipeline.run(
                    target,
                    run_parity=run_parity,
                    run_bench=run_bench,
                    with_narrative=with_narrative and review_options is None,
                    ai_review=review_options,
                )
                summary = summary_module.summarize(result)
                summary["ai_review"] = result.ai_review
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

        payload = {
            "ok": True,
            "version": PROTOCOL_VERSION,
            "summary": summary,
        }

    _respond(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
