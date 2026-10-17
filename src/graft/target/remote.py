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

from graft import config
from graft.target.detect import TargetFacts, detect
from graft.target.spec import TargetSpec
from graft.target.transport import TransportError, clip, quote

ARCHIVE_NAME = "graft-remote.tar.gz"
RESULT_MARKER = "__GRAFT_JSON__"
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024

# What is left out of the uploaded tree. .git matters most: it is large, and
# shipping a repository's history to a third-party host is not something to do
# by default. Credential files matter more: the tree is uploaded to a machine
# the operator may not own, so a root .env holding LLM_API_KEY must never ride
# along. Matching by name (not only by directory) is what stops that.
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", ".ruff_cache", ".mypy_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILES = {".env", ".envrc", "id_rsa", "id_ed25519", "credentials"}


def _is_secret_file(name: str) -> bool:
    return name in EXCLUDED_FILES or name.startswith(".env.")

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
run_step pipeline python3 -m graft.main migrate __PIPELINE_FILE__ --json --no-narrative
printf '}}\n'
exit $status
"""


# Remote script for running and verifying a Gemma-produced file. Graft is
# uploaded first, so the verification runs against the same deterministic
# pipeline a local run uses. Execution and verification are reported as
# separate steps so a failure names which one broke.
AI_REMOTE_SCRIPT = r"""
set -o pipefail
cd __REMOTE_DIR__ || exit 2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

status=0
printf '%s\n' __MARKER__
if [ ! -f src/graft/__init__.py ]; then
  printf '{"error":"project did not arrive intact"}\n'
  exit 3
fi

sdk_lib="$(__PYTHON__ -c 'import pathlib,site;print(next(iter(pathlib.Path(site.getsitepackages()[0]).glob("_rocm_sdk_core/lib")),""))' 2>/dev/null)"
if [ -n "$sdk_lib" ]; then
  for so in "$sdk_lib"/*.so.*; do
    [ -e "$so" ] || continue
    base="$(basename "$so")"
    ln -sf "$so" "$sdk_lib/${base%%.so.*}.so" 2>/dev/null
  done
  LD_LIBRARY_PATH="$sdk_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH
fi

AI_FILE=__AI_FILE__
AI_SUFFIX=__AI_SUFFIX__
AI_DIR=__AI_DIR__

printf '{"steps":{'
first=1
run_step() {
  name="$1"; shift
  if [ $first -eq 0 ]; then printf ','; fi
  first=0
  out="$("$@" 2>&1)"
  rc=$?
  printf '"%s":{"returncode":%d,"output":%s}' \
    "$name" "$rc" "$(printf '%s' "$out" | __PYTHON__ -c 'import json,sys;print(json.dumps(sys.stdin.read()))')"
}

if [ "$AI_SUFFIX" = "cu" ]; then
  if command -v hipcc >/dev/null 2>&1; then
    run_step compile hipcc -O2 "$AI_FILE" -o "$AI_DIR/graft-ai-bin"
    run_step run "$AI_DIR/graft-ai-bin"
  else
    run_step compile sh -c "echo 'hipcc is not available on this target'; exit 127"
  fi
else
  run_step run __PYTHON__ "$AI_FILE"
fi
run_step verification __PYTHON__ -m graft.main migrate "$AI_FILE" --json --no-narrative
run_step report __PYTHON__ -m graft.main migrate "$AI_FILE" --no-narrative
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
    filename: str = ""
    # Whether the HIP toolchain self-check passed. For a selected-file verify
    # this is informational, not a gate: a valid Python/ROCm target need not
    # have a C++ compiler.
    verified: bool = False
    # The pipeline's own JSON summary, when the verify run migrated a source
    # file. This is the same projection the local UI renders, so the remote
    # report can reuse it (source comparison included) without a second run.
    summary: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec_label": self.spec_label,
            "facts": self.facts.as_dict(),
            "duration_s": round(self.duration_s, 2),
            "pushed_bytes": self.pushed_bytes,
            "steps": self.steps,
            "ok": self.ok,
            "error": self.error,
            "filename": self.filename,
            "verified": self.verified,
            "summary": self.summary,
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
            if _is_secret_file(path.name):
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


def _resolve_dir(transport, remote_dir: str) -> str:
    """Where a configured remote directory actually lives on the target.

    SSH sees the whole filesystem, but a Jupyter server serves files from one
    root, so a path that a shell command can use has to be joined onto that
    root before it also means the same file to the contents API. Transports
    that have no such notion leave the path alone.
    """
    resolver = getattr(transport, "resolve_remote_dir", None)
    return resolver(remote_dir) if resolver is not None else remote_dir


def run_remote(
    transport,
    spec: TargetSpec,
    *,
    source: Path | None = None,
    probe_only: bool = False,
    run_file: tuple[str, str] | None = None,
) -> RemoteRun:
    """Detect the target, push Graft to it, and run the verification suite.

    ``source`` is the *local* tree to upload, which defaults to the installed
    package root. It is deliberately not ``spec.remote_dir``: that path only
    exists on the far side, so naming one parameter after the other invites
    packing whatever happens to sit at that path locally.

    ``run_file`` is an optional ``(filename, contents)`` pair to migrate on the
    target instead of the bundled example, which is what the UI's "verify on
    GPU" action uses so the report describes the operator's own source.

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

    archive = build_archive(source or config.PROJECT_ROOT)
    remote_dir = _resolve_dir(transport, spec.remote_dir)
    remote_archive = f"{remote_dir.rsplit('/', 1)[0] or '/'}/{ARCHIVE_NAME}"
    transport.push(archive, remote_archive)

    unpack = transport.run(
        f"rm -rf {quote(remote_dir)} && mkdir -p {quote(remote_dir)} "
        f"&& tar xzf {quote(remote_archive)} -C {quote(remote_dir)}",
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

    # Default to Graft's own example so a plain "verify" still exercises the
    # full toolchain. A selected file is placed beside the tree and migrated
    # instead, so the evidence describes the operator's source.
    pipeline_file = "examples/01_torch_cuda_softmax.py"
    filename = ""
    if run_file is not None:
        filename, contents = run_file
        name = Path(filename).name
        if Path(name).suffix.lower() not in {".py", ".cu"}:
            raise ValueError("Only .py and .cu source files can run on the target.")
        if not contents.strip():
            raise ValueError("The source file is empty; there is nothing to verify.")
        filename = name
        selected_dir = f"{remote_dir}/graft-selected"
        mkdir = transport.run(
            f"mkdir -p {quote(selected_dir)}", timeout_s=spec.command_timeout_s
        )
        if not mkdir.ok:
            return RemoteRun(
                spec_label=label,
                facts=facts,
                started_at=started,
                duration_s=time.monotonic() - started,
                pushed_bytes=len(archive),
                ok=False,
                error=f"Could not prepare a directory for the source file. {mkdir.summary()}",
            )
        pipeline_file = f"{selected_dir}/{name}"
        transport.push(contents.encode("utf-8"), pipeline_file)

    script = (
        REMOTE_SCRIPT.replace("__REMOTE_DIR__", quote(remote_dir))
        .replace("__MARKER__", RESULT_MARKER)
        .replace("__PIPELINE_FILE__", quote(pipeline_file))
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
            filename=filename,
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
            filename=filename,
            ok=False,
            error=str(payload["error"]),
        )

    steps = payload.get("steps", {})
    verify_output = (steps.get("verify_hip") or {}).get("output", "")
    verified = bool(_VERIFY_PASS.search(verify_output))

    # The pipeline step prints the same JSON summary the local UI renders.
    # Reuse it so a remote verify can show the report and the source diff
    # without re-running anything on this machine.
    summary = _parse_json_text((steps.get("pipeline") or {}).get("output", ""))
    pipeline_rc = int((steps.get("pipeline") or {}).get("returncode", 1))

    problems = list(facts.problems)
    if run_file is not None:
        # The operator's own file is the measurement that matters. The HIP
        # toolchain self-check is informational here, not a gate, because a
        # valid Python/ROCm target need not have a C++ compiler.
        if summary is None or pipeline_rc != 0:
            problems.append(
                "The migration did not complete on the target. See the pipeline "
                "step output for the failure."
            )
        ok = summary is not None and pipeline_rc == 0 and not problems
    else:
        if not verified:
            problems.append(
                "The ported HIP did not verify on the target. See the verify "
                "step output for the failing check."
            )
        ok = verified and not problems

    return RemoteRun(
        spec_label=label,
        facts=facts,
        started_at=started,
        duration_s=time.monotonic() - started,
        pushed_bytes=len(archive),
        steps=steps,
        ok=ok,
        error="; ".join(problems),
        filename=filename,
        verified=verified,
        summary=summary,
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

    run_dir = (
        f"{_resolve_dir(transport, spec.remote_dir)}/graft-runs/{uuid.uuid4().hex}"
    )
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
        # A missing PyTorch and a PyTorch that cannot see the GPU are different
        # problems with different fixes, but the interpreter reports both as a
        # failed probe. Say which one it is.
        if "no module named 'torch'" in gpu_probe.output.lower():
            error = (
                "The target has no PyTorch installed, so the converted file "
                "cannot run there. PyTorch (ROCm) is listed as missing in the "
                "dependency scan; install it, then try again."
            )
        else:
            error = (
                "The target has a GPU, but its PyTorch interpreter cannot use "
                "it. The converted script was not run."
            )
        return {
            **common,
            "ok": False,
            "stage": "pytorch_gpu_check",
            "uploaded_bytes": uploaded_bytes,
            "remote_path": remote_file,
            "gpu_probe": gpu_probe.output,
            "error": error,
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


def run_remote_ai_file(
    transport,
    spec: TargetSpec,
    filename: str,
    source: str,
) -> dict[str, Any]:
    """Run and verify a Gemma-produced file on the detected AMD target.

    Unlike :func:`run_remote_file`, this also uploads Graft itself and runs the
    deterministic pipeline against the AI file on the target, so the returned
    evidence (parity, benchmarks, report) comes from the same code path a local
    run uses rather than from the model's own claims. Execution and
    verification are reported separately.
    """
    started = time.monotonic()
    label = spec.safe_label()
    name = Path(filename).name
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in {"py", "cu"}:
        raise ValueError("Only .py and .cu converted files can run on the target.")
    if not source.strip():
        raise ValueError("The AI-converted file is empty; there is nothing to upload.")

    facts = detect(transport)
    common = {
        "spec_label": label,
        "facts": facts.as_dict(),
        "filename": name,
        "ai_converted": True,
    }
    if not facts.gpu_count:
        return {
            **common,
            "ok": False,
            "stage": "gpu_detection",
            "error": (
                "No AMD GPU is visible on the target. The AI-converted file was "
                "not uploaded or run."
            ),
            "duration_s": round(time.monotonic() - started, 2),
        }

    archive = build_archive(config.PROJECT_ROOT)
    remote_dir = _resolve_dir(transport, spec.remote_dir)
    remote_archive = f"{remote_dir.rsplit('/', 1)[0] or '/'}/{ARCHIVE_NAME}"
    transport.push(archive, remote_archive)
    unpack = transport.run(
        f"rm -rf {quote(remote_dir)} && mkdir -p {quote(remote_dir)} "
        f"&& tar xzf {quote(remote_archive)} -C {quote(remote_dir)}",
        timeout_s=spec.command_timeout_s,
    )
    if not unpack.ok:
        return {
            **common,
            "ok": False,
            "stage": "prepare",
            "error": f"Graft uploaded but would not unpack. {_unpack_hint(unpack, spec)}",
            "duration_s": round(time.monotonic() - started, 2),
        }

    run_dir = f"{remote_dir}/graft-ai-runs/{uuid.uuid4().hex}"
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

    python_path = facts.python_path or "python3"
    script = (
        AI_REMOTE_SCRIPT.replace("__REMOTE_DIR__", quote(remote_dir))
        .replace("__MARKER__", RESULT_MARKER)
        .replace("__PYTHON__", quote(python_path))
        .replace("__AI_FILE__", quote(remote_file))
        .replace("__AI_SUFFIX__", suffix)
        .replace("__AI_DIR__", quote(run_dir))
    )
    result = transport.run(script, timeout_s=spec.command_timeout_s)

    payload = _extract_json(result.stdout) or {}
    if not payload:
        return {
            **common,
            "ok": False,
            "stage": "run",
            "uploaded_bytes": uploaded_bytes,
            "remote_path": remote_file,
            "steps": {
                "remote_script": {
                    "returncode": result.returncode,
                    "output": result.summary(),
                }
            },
            "error": _script_failure_hint(result),
            "duration_s": round(time.monotonic() - started, 2),
        }
    if payload.get("error"):
        return {
            **common,
            "ok": False,
            "stage": "run",
            "uploaded_bytes": uploaded_bytes,
            "remote_path": remote_file,
            "error": str(payload["error"]),
            "duration_s": round(time.monotonic() - started, 2),
        }

    steps = payload.get("steps", {})
    run_step = steps.get("run") or {}
    verify_step = steps.get("verification") or {}
    report_step = steps.get("report") or {}

    verification = _parse_json_text(verify_step.get("output", ""))
    report_text = str(report_step.get("output", "")).strip()
    execution_output = str(run_step.get("output", "")).strip()

    run_ok = bool(run_step) and int(run_step.get("returncode", 1)) == 0
    verified = verify_step.get("returncode") == 0 and bool(verification)

    problems = list(facts.problems)
    if not run_ok:
        problems.append(
            "The AI-converted file did not run cleanly on the target; see the "
            "run step output."
        )
    if not verified:
        problems.append(
            "The deterministic pipeline could not verify the AI-converted file "
            "on the target; see the verification step output."
        )

    return {
        **common,
        "ok": run_ok and verified and not problems,
        "stage": "complete" if run_ok else "execute",
        "uploaded_bytes": uploaded_bytes,
        "remote_path": remote_file,
        "steps": steps,
        "verification": verification,
        "report_text": report_text,
        "execution_output": execution_output,
        "exit_code": run_step.get("returncode"),
        "error": "; ".join(problems),
        "duration_s": round(time.monotonic() - started, 2),
    }


def _parse_json_text(text: Any) -> dict[str, Any] | None:
    """Parse a JSON object the remote printed, tolerating leading noise."""
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = text.strip()
    start = candidate.find("{")
    if start == -1:
        return None
    try:
        parsed = json.loads(candidate[start:])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
