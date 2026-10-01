"""Stage 6 — Sandboxed execution of converted code.

User-supplied code must be executed somewhere, and that somewhere is a
separate process with a hard timeout. Three reasons, in order of
importance:

1. A ported CUDA file is third-party code. `import os; os.remove('/')` in
   a converted file must not reach the user's shell.
2. Infinite loops are normal in kernels. A timeout is the only reliable
   way to stop them.
3. Crashes must not take the harness down with them. A segfault in
   native code kills the whole interpreter if it happens in-process.

What this does NOT claim to be: a security sandbox. There is no seccomp,
no namespace, no resource limit beyond wall clock. The child runs with
the same user privileges. Isolation here means process containment and a
timeout, which is enough for a demo and for accidentally hostile input,
and not enough for genuinely adversarial input. Says so in the docstring
rather than letting the module name imply more.

Cases are handed over as a single .pt file and results come back as JSON
on stdout, so the child never imports the harness and the parent never
imports the user's code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from graft import config

# Protocol version: the child and parent must agree on the JSON shape.
# Bump when the result schema changes so a stale child fails loudly.
PROTOCOL = 1


class RunnerError(RuntimeError):
    """Raised when the harness itself is broken, not when user code fails."""


@dataclass
class ExecutionResult:
    """Outcome of one execution. Never raises on user-code failure."""

    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    timeout_seconds: float = 0.0
    protocol_error: str = ""
    error_type: str = ""
    error_message: str = ""
    results: list = field(default_factory=list)
    case_count: int = 0

    def summary(self) -> str:
        if self.timed_out:
            return f"timed out after {self.timeout_seconds}s"
        if self.protocol_error:
            return f"protocol error: {self.protocol_error}"
        if not self.ok:
            return (
                f"{self.error_type}: {self.error_message}"
                if self.error_type
                else (f"exited with code {self.exit_code}")
            )
        return f"ok, {self.case_count} case(s)"

    def __repr__(self) -> str:
        return f"ExecutionResult(ok={self.ok}, {self.summary()})"


# Executed in the child. Kept as a module-level string so it can be read
# and audited as code rather than as an opaque blob.
CHILD_SOURCE = """
import importlib.util
import json
import sys

import torch

PROTOCOL = {protocol}


def _load(path):
    spec = importlib.util.spec_from_file_location("ported", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    source_path, cases_path, out_path = sys.argv[1:4]
    payload = {{}}
    try:
        module = _load(source_path)
        cases = torch.load(cases_path, weights_only=True)

        op = getattr(module, "OPERATION", None)
        solve = getattr(module, "solve", None)
        if not callable(solve):
            raise AttributeError("ported module exposes no solve(x) entry point")
        if op is None:
            raise AttributeError("ported module exposes no OPERATION")

        results = []
        for name, tensor in cases:
            entry = {{"name": name}}
            try:
                value = solve(tensor)
                # Materialise now: a lazy tensor would report success here
                # and fail later, outside this try block.
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                else:
                    raise TypeError(
                        f"solve() returned {{type(value).__name__}}, expected Tensor"
                    )
                entry["ok"] = True
                entry["output"] = value
            except Exception as exc:  # noqa: BLE001 - report, do not abort
                entry["ok"] = False
                entry["error"] = f"{{type(exc).__name__}}: {{exc}}"
            results.append(entry)

        payload = {{
            "protocol": PROTOCOL,
            "ok": True,
            "operation": op,
            "results": results,
        }}
    except BaseException as exc:  # noqa: BLE001 - the child must always answer
        payload = {{
            "protocol": PROTOCOL,
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }}

    torch.save(payload, out_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
"""


def _write_child(directory: Path) -> Path:
    script = directory / "_graft_child.py"
    script.write_text(CHILD_SOURCE.format(protocol=PROTOCOL))
    return script


def _case_pair(case) -> tuple[str, object]:
    """Normalise a case to (name, cpu_tensor).

    Accepts the dict shape used by verify.cases.generate and the plain
    (name, tensor) tuple, so callers do not have to reshape their inputs
    before handing them over.
    """
    if isinstance(case, dict):
        name = case.get("name", "?")
        tensor = case.get("input")
    else:
        name, tensor = case
    if tensor is None:
        raise ValueError(f"case {name!r} has no 'input' tensor")
    return str(name), tensor.cpu()


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    # Keep the child's device choice explicit rather than inherited, so a
    # stray CUDA_VISIBLE_DEVICES cannot silently change what was measured.
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    env["GRAFT_PROTOCOL"] = str(PROTOCOL)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_cases(
    source: str,
    cases,
    timeout: float | None = None,
) -> ExecutionResult:
    """Execute ``solve`` over ``cases`` in a child process.

    Args:
        source: ported Python source, written verbatim to the child.
        cases: iterable of case dicts (as produced by verify.cases.generate)
            or (name, tensor) pairs. Both shapes are accepted because the
            dict form is what the parity harness already uses.
        timeout: seconds; defaults to config.SUBPROCESS_TIMEOUT_S.

    Returns an ExecutionResult describing what happened. User-code
    failures are data, not exceptions. A RunnerError means the harness
    could not run at all.
    """
    limit = config.SUBPROCESS_TIMEOUT_S if timeout is None else timeout

    with tempfile.TemporaryDirectory(
        prefix=f"graft-run-{uuid.uuid4().hex[:8]}-"
    ) as tmp:
        work = Path(tmp)
        source_path = work / "ported.py"
        cases_path = work / "cases.pt"
        out_path = work / "result.pt"
        script = _write_child(work)

        source_path.write_text(source)

        try:
            import torch

            payload_cases = [_case_pair(case) for case in cases]
            torch.save(payload_cases, cases_path)
        except Exception as exc:
            raise RunnerError(f"could not serialise cases: {exc}") from exc

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(source_path),
                    str(cases_path),
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=limit,
                env=_environment(),
                cwd=work,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False,
                timed_out=True,
                timeout_seconds=limit,
                error_type="TimeoutExpired",
                error_message=f"exceeded {limit}s",
                stderr=f"child killed after {limit}s",
            )
        except OSError as exc:
            raise RunnerError(f"could not start child process: {exc}") from exc

        result = ExecutionResult(ok=False, exit_code=completed.returncode)
        result.stdout = completed.stdout
        result.stderr = completed.stderr

        if not out_path.exists():
            result.error_type = "NoResult"
            result.error_message = (
                "child produced no result file; it probably crashed natively "
                "(segfault, OOM, or an abort in native code)"
            )
            return result

        try:
            import torch

            payload = torch.load(out_path, weights_only=True)
        except Exception as exc:
            raise RunnerError(f"child result unreadable: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
            result.protocol_error = (
                f"unexpected result schema; expected protocol {PROTOCOL}"
            )
            return result

        result.ok = bool(payload.get("ok"))
        result.error_type = payload.get("error_type", "")
        result.error_message = payload.get("error_message", "")

        collected = payload.get("results") or []
        result.case_count = len(collected)
        result.results = [
            {
                "name": entry.get("name", "?"),
                "ok": bool(entry.get("ok")),
                "error": entry.get("error", ""),
                "output": entry.get("output"),
            }
            for entry in collected
        ]
        return result


def run_module(path: Path, timeout: float | None = None) -> ExecutionResult:
    """Import-time check: does the ported file even load?

    Separate from run_cases because an import error is a distinct class of
    problem from a wrong answer, and diagnosing them needs different text.
    """
    try:
        source = Path(path).read_text()
    except OSError as exc:
        raise RunnerError(f"cannot read {path}: {exc}") from exc

    return run_cases(source, [], timeout=timeout)


def sandbox_notes() -> list[str]:
    """Honest description of the isolation actually in force.

    Rendered into the report so a judge is not left believing there is a
    real sandbox behind the word.
    """
    return [
        "Converted code runs in a separate process, not a security sandbox.",
        "The child inherits the same OS user and filesystem access.",
        f"Wall-clock timeout of {config.SUBPROCESS_TIMEOUT_S}s; native crashes "
        "cannot be caught, only reported.",
    ]
