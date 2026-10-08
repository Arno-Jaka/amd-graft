"""Run Graft on a remote AMD target and bring the evidence back.

Three steps, each reported separately so a failure says which one broke:

1. Push a tarball of the current working tree to the target.
2. Run one script remotely that executes the verification and prints a single
   JSON object between markers.
3. Parse that object into the same summary shape the local UI already renders.

Results come back as JSON on stdout rather than as a file download, so a
failure part-way through still yields whatever the remote managed to produce.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graft.target.detect import TargetFacts, detect
from graft.target.spec import TargetSpec
from graft.target.transport import TransportError, clip, quote

ARCHIVE_NAME = "graft-remote.tar.gz"
RESULT_MARKER = "__GRAFT_JSON__"
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024

# What is left out of the uploaded tree. .git matters most: it is large, and
# shipping a repository's history to a third-party host is not something to do
# by default.
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".ruff_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}

# The remote script. It runs Graft's own verification tools rather than
# reimplementing anything, so the numbers it reports come from the same code
# path the local run uses.
# Single braces throughout: this string is only .replace()d, never
# .format()d, so doubled braces would print literally and the JSON
# below would not parse.
REMOTE_SCRIPT = r"""
set -o pipefail
cd __REMOTE_DIR__ || exit 2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

status=0
printf '%s\n' __MARKER__
if [ ! -f src/graft/__init__.py ]; then
  printf '{"error":"project did not arrive intact"}\n'
  exit 3
fi

# tools/verify_hip.py needs the ROCm pip wheel's libs on the loader path; the
# wheel ships them versioned only, without the unversioned symlink the linker
# looks for.
sdk_lib="$(python3 -c 'import pathlib,site;print(next(iter(pathlib.Path(site.getsitepackages()[0]).glob("_rocm_sdk_core/lib")),""))' 2>/dev/null)"
if [ -n "$sdk_lib" ]; then
  for so in "$sdk_lib"/*.so.*; do
    [ -e "$so" ] || continue
    base="$(basename "$so")"
    ln -sf "$so" "$sdk_lib/${base%%.so.*}.so" 2>/dev/null
  done
  LD_LIBRARY_PATH="$sdk_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH
fi

printf '{"steps":{'
first=1
run_step() {
  name="$1"; shift
  if [ $first -eq 0 ]; then printf ','; fi
  first=0
  out="$("$@" 2>&1)"
  rc=$?
  printf '"%s":{"returncode":%d,"output":%s}' \
    "$name" "$rc" "$(printf '%s' "$out" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')"
}

run_step verify_hip python3 tools/verify_hip.py
run_step pipeline python3 -m graft.main migrate examples/01_torch_cuda_softmax.py --json --no-narrative
printf '}}\n'
exit $status
"""


@dataclass(frozen=True, slots=True)
class RemoteRun:
    """Everything one remote attempt produced."""

    spec_label: str
    facts: TargetFacts
    started_at: float
    duration_s: float
    pushed_bytes: int = 0
    steps: dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_label": self.spec_label,
            "facts": self.facts.as_dict(),
            "duration_s": round(self.duration_s, 2),
            "pushed_bytes": self.pushed_bytes,
            "steps": self.steps,
            "ok": self.ok,
            "error": self.error,
        }


def build_archive(root: Path) -> bytes:
    """Tar the working tree, excluding VCS and build noise."""
    root = Path(root).resolve()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED_DIRS for part in relative.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            archive.add(path, arcname=str(relative))
    payload = buffer.getvalue()
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise TransportError(
            f"The project archive is {len(payload) // 1024} KB, over the "
            f"{MAX_ARCHIVE_BYTES // 1024} KB limit for an upload."
        )
    return payload


def _unpack_hint(result: Any, spec: TargetSpec) -> str:
    """Explain a failed unpack in terms the operator can act on.

    The usual cause is not a corrupt archive: the upload landed somewhere other
    than where the shell looked. A Jupyter contents path resolves against the
    server's root directory, so a remote directory outside that root silently
    puts the archive one level deeper than the command expects.
    """
    summary = result.summary()
    lowered = summary.lower()
    # "cannot open" also covers a permissions failure, which is a different
    # problem with a different fix, so check that case first.
    denied = "permission denied" in lowered
    missing = not denied and any(
        needle in lowered for needle in ("cannot open", "no such file")
    )
    if missing and spec.kind == "jupyter":
        return (
            f"{summary} The archive was not found where the shell looked. A "
            "Jupyter upload is written relative to the server's root directory, "
            "so the remote directory has to sit inside that root."
        )
    return summary


def _script_failure_hint(result: Any) -> str:
    """Explain a remote script that produced no result block.

    The two streams are reported separately: which one the script wrote to is
    the difference between "it died early" and "it ran and wrote its JSON
    somewhere the parser does not look".
    """
    if result.returncode != 0:
        lead = f"The verification script exited {result.returncode}."
    else:
        lead = "The verification script produced no result block."
    out = clip((result.stdout or "").strip(), 500)
    err = clip((result.stderr or "").strip(), 500)
    return f"{lead} stdout: {out or '(empty)'} stderr: {err or '(empty)'}"


def _extract_json(stdout: str) -> dict[str, Any] | None:
    """Pull the JSON object the remote script printed between its markers."""
    start = stdout.find(RESULT_MARKER)
    if start == -1:
        return None
    body = stdout[start + len(RESULT_MARKER) :].lstrip()
    # The object may be preceded by unrelated shell noise, so decode from the
    # first brace rather than trusting the line to start cleanly.
    depth = 0
    start_brace = -1
    in_string = False
    escaped = False
    for index, character in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start_brace = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start_brace != -1:
                candidate = body[start_brace : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


_VERIFY_PASS = re.compile(r"HIP VERIFICATION PASSED")


def run_remote(
    transport,
    spec: TargetSpec,
    *,
    source: Path | None = None,
    probe_only: bool = False,
) -> RemoteRun:
    """Detect the target, push Graft to it, and run the verification suite.

    ``source`` is the *local* tree to upload, which defaults to the installed
    package root. It is deliberately not ``spec.remote_dir``: that path only
    exists on the far side, so naming one parameter after the other invites
    packing whatever happens to sit at that path locally.

    ``probe_only`` stops after detection, which is what the UI's connect
    button uses so the operator can see what was found before anything is
    uploaded or executed.
    """
    started = time.monotonic()
    label = spec.safe_label()

    facts = detect(transport)
    if probe_only:
        return RemoteRun(
            spec_label=label,
            facts=facts,
            started_at=started,
            duration_s=time.monotonic() - started,
            ok=facts.ok,
            error="" if facts.ok else "; ".join(facts.problems),
        )

    archive = build_archive(
        source or Path(__file__).resolve().parent.parent.parent.parent
    )
    remote_archive = f"{spec.remote_dir.rsplit('/', 1)[0] or '/'}/{ARCHIVE_NAME}"
    transport.push(archive, remote_archive)

    unpack = transport.run(
        f"rm -rf {quote(spec.remote_dir)} && mkdir -p {quote(spec.remote_dir)} "
        f"&& tar xzf {quote(remote_archive)} -C {quote(spec.remote_dir)}",
        timeout_s=spec.command_timeout_s,
    )
    if not unpack.ok:
        return RemoteRun(
            spec_label=label,
            facts=facts,
            started_at=started,
            duration_s=time.monotonic() - started,
            pushed_bytes=len(archive),
            ok=False,
            error=f"The project uploaded but would not unpack. {_unpack_hint(unpack, spec)}",
        )

    script = REMOTE_SCRIPT.replace("__REMOTE_DIR__", quote(spec.remote_dir)).replace(
        "__MARKER__", RESULT_MARKER
    )
    result = transport.run(script, timeout_s=spec.command_timeout_s)

    payload = _extract_json(result.stdout) or {}
    if not payload:
        # The script speaks JSON between markers. If that is missing, the script
        # died before it got there, and "did not verify" on its own tells the
        # operator nothing about why. Show what actually came back.
        return RemoteRun(
            spec_label=label,
            facts=facts,
            started_at=started,
            duration_s=time.monotonic() - started,
            pushed_bytes=len(archive),
            steps={
                "remote_script": {
                    "returncode": result.returncode,
                    "output": result.summary(),
                }
            },
            ok=False,
            error=_script_failure_hint(result),
        )
    if payload.get("error"):
        return RemoteRun(
            spec_label=label,
            facts=facts,
            started_at=started,
            duration_s=time.monotonic() - started,
            pushed_bytes=len(archive),
            ok=False,
            error=str(payload["error"]),
        )

    steps = payload.get("steps", {})
    verify_output = (steps.get("verify_hip") or {}).get("output", "")
    verified = bool(_VERIFY_PASS.search(verify_output))

    problems = list(facts.problems)
    if not verified:
        problems.append(
            "The ported HIP did not verify on the target. See the verify step "
            "output for the failing check."
        )

    return RemoteRun(
        spec_label=label,
        facts=facts,
        started_at=started,
        duration_s=time.monotonic() - started,
        pushed_bytes=len(archive),
        steps=steps,
        ok=verified and not problems,
        error="; ".join(problems),
    )


def run_remote_file(
    transport,
    spec: TargetSpec,
    filename: str,
    source: str,
) -> dict[str, Any]:
    """Upload and execute one converted file on a detected AMD GPU.

    Each run uses a unique subdirectory under the selected remote workspace;
    existing files are never removed. CUDA C++ output is compiled with
    ``hipcc`` and then executed. Python output is run with the target's
    PyTorch interpreter after confirming that PyTorch can see the GPU.
    """
    started = time.monotonic()
    label = spec.safe_label()
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in {".py", ".cu"}:
        raise ValueError("Only .py and .cu converted files can run on the target.")
    if not source.strip():
        raise ValueError("The converted file is empty; there is nothing to upload.")

    facts = detect(transport)
    common = {
        "spec_label": label,
        "facts": facts.as_dict(),
        "filename": name,
    }
    if not facts.gpu_count:
        return {
            **common,
            "ok": False,
            "stage": "gpu_detection",
            "error": "No AMD GPU is visible on the target. The converted file was not uploaded or run.",
            "duration_s": round(time.monotonic() - started, 2),
        }

    run_dir = f"{spec.remote_dir}/graft-runs/{uuid.uuid4().hex}"
    mkdir = transport.run(
        f"mkdir -p {quote(run_dir)}", timeout_s=spec.command_timeout_s
    )
    if not mkdir.ok:
        return {
            **common,
            "ok": False,
            "stage": "prepare",
            "error": f"Could not prepare a run directory on the target. {mkdir.summary()}",
            "duration_s": round(time.monotonic() - started, 2),
        }

    remote_file = f"{run_dir}/{name}"
    transport.push(source.encode("utf-8"), remote_file)
    uploaded_bytes = len(source.encode("utf-8"))

    if suffix == ".cu":
        if not facts.tools.get("hipcc"):
            return {
                **common,
                "ok": False,
                "stage": "compile",
                "uploaded_bytes": uploaded_bytes,
                "remote_path": remote_file,
                "error": "The file was uploaded, but hipcc is not available on this target.",
                "duration_s": round(time.monotonic() - started, 2),
            }
        binary = f"{run_dir}/graft-converted"
        compile_command = (
            f"cd {quote(run_dir)} && {quote(facts.tools['hipcc'])} -O2 "
            f"{quote(name)} -o {quote(binary)}"
        )
        compile_result = transport.run(
            compile_command, timeout_s=spec.command_timeout_s
        )
        if not compile_result.ok:
            return {
                **common,
                "ok": False,
                "stage": "compile",
                "uploaded_bytes": uploaded_bytes,
                "remote_path": remote_file,
                "compile_output": compile_result.output,
                "error": f"hipcc could not compile the converted file (exit {compile_result.returncode}).",
                "duration_s": round(time.monotonic() - started, 2),
            }
        run_result = transport.run(
            f"cd {quote(run_dir)} && {quote(binary)}",
            timeout_s=spec.command_timeout_s,
        )
        return {
            **common,
            "ok": run_result.ok,
            "stage": "complete" if run_result.ok else "execute",
            "uploaded_bytes": uploaded_bytes,
            "remote_path": remote_file,
            "compile_output": compile_result.output,
            "execution_output": run_result.output,
            "exit_code": run_result.returncode,
            "error": ""
            if run_result.ok
            else f"The compiled HIP program exited {run_result.returncode}.",
            "duration_s": round(time.monotonic() - started, 2),
        }

    python_path = facts.python_path or "python3"
    gpu_probe_code = (
        "import torch,sys; "
        "ok=torch.cuda.is_available(); "
        "print('GPU=' + (torch.cuda.get_device_name(0) if ok else 'unavailable')); "
        "sys.exit(0 if ok else 2)"
    )
    gpu_probe = transport.run(
        f"{quote(python_path)} -c {quote(gpu_probe_code)}",
        timeout_s=spec.command_timeout_s,
    )
    if not gpu_probe.ok:
        return {
            **common,
            "ok": False,
            "stage": "pytorch_gpu_check",
            "uploaded_bytes": uploaded_bytes,
            "remote_path": remote_file,
            "gpu_probe": gpu_probe.output,
            "error": "The target has a GPU, but its PyTorch interpreter cannot use it. The converted script was not run.",
            "duration_s": round(time.monotonic() - started, 2),
        }

    run_result = transport.run(
        f"cd {quote(run_dir)} && {quote(python_path)} {quote(name)}",
        timeout_s=spec.command_timeout_s,
    )
    return {
        **common,
        "ok": run_result.ok,
        "stage": "complete" if run_result.ok else "execute",
        "uploaded_bytes": uploaded_bytes,
        "remote_path": remote_file,
        "gpu_probe": gpu_probe.output,
        "execution_output": run_result.output,
        "exit_code": run_result.returncode,
        "error": ""
        if run_result.ok
        else f"The converted Python script exited {run_result.returncode}.",
        "duration_s": round(time.monotonic() - started, 2),
    }
