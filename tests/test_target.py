"""Target connection, detection, and remote orchestration.

Everything here runs against a fake transport. That is deliberate: these are
the tests most likely to be trusted about credentials, and they must be
runnable offline so they actually get run.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import threading
import types

import pytest
import requests

import graft.target.detect as detect_module
from graft.target import remote as remote_module
from graft.target import spec as spec_module
from graft.target.spec import REDACTED, TargetError, TargetSpec, parse_form
from graft.target.transport import CommandResult, TransportError, clip


class FakeTransport:
    """A scripted transport. Records commands so tests can assert on them."""

    def __init__(self, replies: dict[str, CommandResult] | None = None) -> None:
        self.replies = replies or {}
        self.commands: list[str] = []
        self.pushed: list[tuple[int, str]] = []
        self.default = CommandResult("echo", 0, "", "")

    def run(self, command, *, cwd=None, timeout_s=None):
        self.commands.append(command)
        for needle, reply in self.replies.items():
            if needle in command:
                return reply
        return self.default

    def push(self, local: bytes, remote_path: str) -> None:
        self.pushed.append((len(local), remote_path))

    def probe(self) -> None:
        return None




def test_empty_port_becomes_the_default_not_zero():
    """An omitted port must never survive as 0 and render as host:0."""
    ssh = parse_form(
        {
            "kind": "ssh",
            "host": "10.0.0.1",
            "username": "root",
            "key_path": "/tmp/id_ed25519",
        }
    )
    jupyter = parse_form({"kind": "jupyter", "host": "example.com", "token": "t"})
    assert ssh.port == 22
    assert jupyter.port == 80
    assert "host:0" not in ssh.safe_label()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://1.2.3.4/lab?token=abc", ("1.2.3.4", 443)),
        ("http://h:9000", ("h", 9000)),
        ("h:8888", ("h", 8888)),
        ("bare.example.com", ("bare.example.com", 80)),
        ("https://h", ("h", 443)),
    ],
)
def test_jupyter_accepts_pasted_urls(value, expected):
    spec = parse_form({"kind": "jupyter", "host": value, "token": "t"})
    assert (spec.host, spec.port) == expected


def test_an_explicit_port_beats_a_default_port_in_the_url():
    spec = parse_form(
        {"kind": "jupyter", "host": "https://h", "token": "t", "port": "443"}
    )
    assert spec.port == 443


@pytest.mark.parametrize(
    ("fields", "because"),
    [
        ({"kind": "ftp", "host": "h"}, "unknown kind"),
        ({"kind": "ssh", "host": ""}, "missing host"),
        ({"kind": "ssh", "host": "root@evil"}, "credentials in host"),
        (
            {"kind": "ssh", "host": "h", "key_path": "/tmp/id_ed25519", "port": "0x10"},
            "non-numeric port",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "port": "70000",
            },
            "port out of range",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "username": "a b",
            },
            "space in username",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "remote_dir": "relative/path",
            },
            "relative dir",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "remote_dir": "/a/../b",
            },
            "traversal in dir",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "remote_dir": "/a/",
            },
            "trailing slash",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "command_timeout_s": "1",
            },
            "timeout floor",
        ),
        (
            {
                "kind": "ssh",
                "host": "h",
                "key_path": "/tmp/id_ed25519",
                "key_passphrase": "line\nbreak",
            },
            "newline in secret",
        ),
        ({"kind": "jupyter", "host": "https://u:p@h"}, "credentials in URL"),
        (
            {"kind": "jupyter", "host": "h", "token": "t", "remote_dir": "//x"},
            "double slash",
        ),
    ],
)
def test_bad_input_is_rejected_with_a_readable_reason(fields, because):
    with pytest.raises(TargetError):
        parse_form(fields)


def test_only_the_relevant_secret_is_carried():
    """A form holding both secrets must not load both into the spec."""
    spec = parse_form(
        {
            "kind": "ssh",
            "host": "h",
            "key_path": "/tmp/id_ed25519",
            "key_passphrase": "ssh-secret",
            "token": "jupyter-secret",
        }
    )
    assert spec.key_passphrase == "ssh-secret"
    assert spec.token == ""




@pytest.mark.parametrize("secret_field", ["key_passphrase", "token"])
def test_a_secret_never_appears_in_any_rendered_form(secret_field):
    secret = "SUPERSECRETVALUE"
    spec = parse_form(
        {
            "kind": "jupyter" if secret_field == "token" else "ssh",
            "host": "h",
            "key_path": "/tmp/id_ed25519",
            secret_field: secret,
        }
    )
    surfaces = {
        "str": str(spec),
        "repr": repr(spec),
        "redacted": json.dumps(spec.redacted()),
        "label": spec.safe_label(),
        "as_dict": json.dumps(spec.redacted()),
    }
    for name, surface in surfaces.items():
        assert secret not in surface, f"secret leaked via {name}"


def test_redacted_marks_that_something_was_present():
    spec = parse_form({"kind": "jupyter", "host": "h", "token": "t"})
    assert spec.redacted()["secret"] == REDACTED
    assert spec.redacted()["host"] == "h"


def test_redacted_marks_an_absent_secret_as_empty():
    spec = parse_form(
        {"kind": "ssh", "host": "h", "username": "root", "key_path": "/tmp/id_ed25519"}
    )
    assert spec.redacted()["secret"] == ""
    assert spec.has_secret is False


def test_key_path_is_not_echoed_back():
    spec = parse_form(
        {"kind": "ssh", "host": "h", "key_path": "/home/someone/.ssh/id_ed25519"}
    )
    assert spec.redacted()["key_path"] == REDACTED




def test_clip_keeps_the_tail_where_diagnostics_live():
    """Compilers put the explanation at the end, so the tail must survive."""
    body = "x" * 100_000 + "THE REAL ERROR IS HERE\n"
    clipped = clip(body, limit=1000)
    assert len(clipped) <= 1000 + 64
    assert "THE REAL ERROR IS HERE" in clipped


def test_clip_leaves_short_output_alone():
    assert clip("hello", limit=100) == "hello"




GOOD_PROBE = """
<<<os>>>Ubuntu 24.04.4 LTS
<<<arch>>>x86_64
<<<rocm_path>>>/opt/rocm/core-10.0/bin/hipcc
<<<rocm>>>10.0.0
<<<agents>>>AMD Instinct MI300X VF|gfx942
<<<kfd_gpus>>>1
<<<vram>>>205822885888 B
<<<tool_hipcc>>>/opt/rocm/core-10.0/bin/hipcc
<<<tool_hipify-perl>>>/opt/rocm/core-10.0/bin/hipify-perl
<<<python>>>/usr/bin/python3
<<<torch>>>2.12.0+rocm10.0.0|AMD Instinct MI300X VF
<<<device_report>>>Agent 1
  Name:                    INTEL(R) XEON(R) PLATINUM 8568Y+
  Marketing Name:          INTEL(R) XEON(R) PLATINUM 8568Y+
  Vendor Name:             CPU
  Device Type:             CPU
Agent 2
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X VF
  Vendor Name:             AMD
  Device Type:             GPU
"""


@pytest.fixture
def source_tree(tmp_path):
    """A stand-in for the local tree that gets uploaded."""
    (tmp_path / "src" / "graft").mkdir(parents=True)
    (tmp_path / "src" / "graft" / "__init__.py").write_text("")
    (tmp_path / "README.md").write_text("graft")
    return tmp_path


def _probe_transport(stdout: str, returncode: int = 0) -> FakeTransport:
    return FakeTransport({"": CommandResult("probe", returncode, stdout, "")})


def test_detection_reads_a_healthy_target():
    facts = detect_module.detect(_probe_transport(GOOD_PROBE))
    assert facts.ok
    assert facts.gpu_count == 1
    assert facts.gfx_arch == "gfx942"
    assert facts.vram_mb == 196288
    assert facts.tools["hipify-perl"].endswith("hipify-perl")
    assert facts.torch_version == "2.12.0+rocm10.0.0"
    assert facts.torch_gpu == "AMD Instinct MI300X VF"
    assert facts.problems == []
    assert "MI300X" in facts.summary_line()


def test_detection_flags_missing_toolchain():
    probe = "<<<os>>>Ubuntu\n<<<python>>>/usr/bin/python3\n<<<gpus>>>1\n"
    facts = detect_module.detect(_probe_transport(probe))
    assert not facts.ok
    assert any("hipify-perl" in problem for problem in facts.problems)
    assert any("hipcc" in problem for problem in facts.problems)


def test_a_missing_gpu_is_a_note_not_a_failure():
    """Translation and cross-compiling work without a device present."""
    probe = GOOD_PROBE.replace(
        "<<<agents>>>AMD Instinct MI300X VF|gfx942\n", ""
    ).replace("<<<kfd_gpus>>>1\n", "<<<kfd_gpus>>>0\n")
    facts = detect_module.detect(_probe_transport(probe))
    assert facts.ok, facts.problems
    assert any("No AMD GPU" in note for note in facts.notes)


def test_torch_without_a_gpu_is_surfaced():
    probe = GOOD_PROBE.replace("AMD Instinct MI300X VF", "not available")
    facts = detect_module.detect(_probe_transport(probe))
    assert any("no usable GPU" in note for note in facts.notes)


def test_detection_raises_only_when_the_probe_returns_nothing():
    with pytest.raises(TransportError):
        detect_module.detect(_probe_transport("", returncode=1))


def test_facts_serialise_without_leaking_a_bulk_transcript():
    facts = detect_module.detect(_probe_transport(GOOD_PROBE + "z" * 10_000))
    assert len(facts.as_dict()["raw"]) <= detect_module.MAX_RAW_CHARS


def test_probe_script_is_bounded_and_syntactically_valid():
    """The probe must not walk the whole filesystem hunting for a Python.

    An unbounded `find /` can run for minutes on a remote host, which presents
    to the operator as a hung detection. Keep the search to bounded globs, and
    keep the script parseable after the shell line-continuations.
    """
    assert "find /" not in detect_module.PROBE_SCRIPT
    assert "containerd" not in detect_module.PROBE_SCRIPT
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    parsed = subprocess.run(
        ["bash", "-n"],
        input=detect_module.PROBE_SCRIPT,
        text=True,
        capture_output=True,
    )
    assert parsed.returncode == 0, parsed.stderr




def _remote_replies(verify_ok: bool = True) -> FakeTransport:
    verify = (
        "HIP VERIFICATION PASSED\n"
        if verify_ok
        else "  elementwise scale by 2  max_abs_error=1.0  FAIL\nHIP VERIFICATION FAILED\n"
    )
    body = (
        remote_module.RESULT_MARKER
        + "\n"
        + json.dumps(
            {
                "steps": {
                    "verify_hip": {
                        "returncode": 0 if verify_ok else 1,
                        "output": verify,
                    },
                    "pipeline": {"returncode": 0, "output": "{}"},
                }
            }
        )
    )
    return FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            remote_module.RESULT_MARKER: CommandResult("run", 0, body, ""),
        }
    )


def _spec() -> TargetSpec:
    return parse_form(
        {"kind": "jupyter", "host": "h", "token": "t", "remote_dir": "/root/graft"}
    )


def test_a_python_run_without_torch_names_pytorch():
    transport = FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            "import torch": CommandResult(
                "probe-torch",
                1,
                "",
                "ModuleNotFoundError: No module named 'torch'",
            ),
        }
    )
    result = remote_module.run_remote_file(
        transport, _spec(), "converted.py", "import torch\n"
    )
    assert result["ok"] is False
    assert result["stage"] == "pytorch_gpu_check"
    assert "PyTorch" in result["error"]
    assert "dependency scan" in result["error"]


def test_a_python_run_with_gpu_less_torch_keeps_the_generic_error():
    transport = FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            "import torch": CommandResult("probe-torch", 2, "GPU=unavailable", ""),
        }
    )
    result = remote_module.run_remote_file(
        transport, _spec(), "converted.py", "import torch\n"
    )
    assert result["ok"] is False
    assert result["stage"] == "pytorch_gpu_check"
    assert "cannot use it" in result["error"]


def test_archive_excludes_vcs_and_build_noise(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("x = 1")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret-ish")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    (tmp_path / "main.pyc").write_bytes(b"\x00")

    import io
    import tarfile

    with tarfile.open(
        fileobj=io.BytesIO(remote_module.build_archive(tmp_path))
    ) as archive:
        names = set(archive.getnames())

    assert "src/keep.py" in names
    assert not any(name.startswith(".git") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert "main.pyc" not in names


def test_probe_only_does_not_upload_or_run_anything(source_tree):
    transport = _probe_transport(GOOD_PROBE)
    run = remote_module.run_remote(
        transport, _spec(), source=source_tree, probe_only=True
    )
    assert run.ok
    assert transport.pushed == []
    assert not any(
        remote_module.RESULT_MARKER in command for command in transport.commands
    )


def test_a_probe_only_failure_reports_the_problems(source_tree):
    transport = _probe_transport("<<<os>>>Ubuntu\n<<<python>>>/usr/bin/python3\n")
    run = remote_module.run_remote(
        transport, _spec(), source=source_tree, probe_only=True
    )
    assert not run.ok
    assert "hipcc" in run.error


def test_a_successful_run_reports_verified(source_tree):
    transport = _remote_replies(verify_ok=True)
    transport.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    run = remote_module.run_remote(transport, _spec(), source=source_tree)
    assert run.ok, run.error
    assert "verify_hip" in run.steps
    assert run.pushed_bytes > 0


class _RootedTransport(FakeTransport):
    """A transport whose remote directory sits under a server root."""

    def resolve_remote_dir(self, remote_dir: str) -> str:
        return f"/shared-docker{remote_dir}"


def test_a_project_run_lands_under_the_transport_root(source_tree):
    base = _remote_replies(verify_ok=True)
    base.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    transport = _RootedTransport(base.replies)
    remote_module.run_remote(transport, _spec(), source=source_tree)
    pushed = [path for _, path in transport.pushed]
    assert any("/shared-docker/root/" in path for path in pushed)
    assert any("/shared-docker/root/graft" in command for command in transport.commands)




def test_verify_uploads_and_migrates_the_selected_file(source_tree):
    """The report for a remote verify must describe the operator's own file.

    A selected file is pushed beside the uploaded tree and named in the
    migration command, so the evidence comes from that source rather than the
    bundled example.
    """
    transport = _remote_replies(verify_ok=True)
    transport.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    run = remote_module.run_remote(
        transport,
        _spec(),
        source=source_tree,
        run_file=("kernel.py", "import torch\n"),
    )
    assert run.filename == "kernel.py"
    assert any(
        path.endswith("/graft-selected/kernel.py") for _, path in transport.pushed
    )
    assert any(
        "graft-selected/kernel.py" in command for command in transport.commands
    )


def test_verify_returns_the_pipeline_summary_for_the_report(source_tree):
    """The pipeline step's JSON summary rides back so the UI can render it."""
    summary = {
        "conversion": {"measured": True, "verdict": "production_ready"},
        "converted_source": "print('ported')\n",
    }
    body = json.dumps(
        {
            "steps": {
                "verify_hip": {
                    "returncode": 0,
                    "output": "HIP VERIFICATION PASSED",
                },
                "pipeline": {"returncode": 0, "output": json.dumps(summary)},
            }
        }
    )
    transport = FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            remote_module.RESULT_MARKER: CommandResult(
                "run", 0, f"{remote_module.RESULT_MARKER}\n{body}", ""
            ),
            "tar xzf": CommandResult("unpack", 0, "", ""),
        }
    )
    run = remote_module.run_remote(
        transport, _spec(), source=source_tree, run_file=("kernel.py", "import torch\n")
    )
    assert run.ok, run.error
    assert run.summary is not None
    assert run.summary["converted_source"] == "print('ported')\n"


def test_verify_rejects_an_unsupported_selected_suffix(source_tree):
    transport = _remote_replies(verify_ok=True)
    transport.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    with pytest.raises(ValueError):
        remote_module.run_remote(
            transport, _spec(), source=source_tree, run_file=("notes.txt", "hi")
        )


def test_verify_rejects_an_empty_selected_file(source_tree):
    transport = _remote_replies(verify_ok=True)
    transport.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    with pytest.raises(ValueError):
        remote_module.run_remote(
            transport, _spec(), source=source_tree, run_file=("kernel.py", "   ")
        )


def test_selected_file_verify_is_not_gated_on_the_hip_toolchain(source_tree):
    """A valid Python/ROCm target need not have a C++ compiler.

    The HIP self-check failing must not mark the operator's own Python
    migration as failed, so ``ok`` follows the pipeline result instead.
    """
    summary = {"conversion": {"measured": True, "verdict": "production_ready"}}
    body = json.dumps(
        {
            "steps": {
                "verify_hip": {"returncode": 1, "output": "HIP VERIFICATION FAILED"},
                "pipeline": {"returncode": 0, "output": json.dumps(summary)},
            }
        }
    )
    transport = FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            remote_module.RESULT_MARKER: CommandResult(
                "run", 0, f"{remote_module.RESULT_MARKER}\n{body}", ""
            ),
            "tar xzf": CommandResult("unpack", 0, "", ""),
        }
    )
    run = remote_module.run_remote(
        transport, _spec(), source=source_tree, run_file=("kernel.py", "import torch\n")
    )
    assert run.ok, run.error
    assert run.verified is False


def test_a_failed_selected_file_migration_is_not_ok(source_tree):
    body = json.dumps(
        {
            "steps": {
                "verify_hip": {"returncode": 0, "output": "HIP VERIFICATION PASSED"},
                "pipeline": {"returncode": 1, "output": "Traceback: boom"},
            }
        }
    )
    transport = FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            remote_module.RESULT_MARKER: CommandResult(
                "run", 0, f"{remote_module.RESULT_MARKER}\n{body}", ""
            ),
            "tar xzf": CommandResult("unpack", 0, "", ""),
        }
    )
    run = remote_module.run_remote(
        transport, _spec(), source=source_tree, run_file=("kernel.py", "import torch\n")
    )
    assert not run.ok
    assert "did not complete" in run.error


def test_a_file_migration_without_torch_names_the_missing_dependency(source_tree):
    """A host without PyTorch cannot run the pipeline, so say why.

    The pipeline imports torch, so a missing PyTorch kills it at import and no
    JSON report is produced. That has to read as a missing dependency, not as an
    unexplained pipeline failure the operator cannot act on.
    """
    body = json.dumps(
        {
            "steps": {
                "verify_hip": {"returncode": 0, "output": "HIP VERIFICATION PASSED"},
                "pipeline": {
                    "returncode": 1,
                    "output": "ModuleNotFoundError: No module named 'torch'",
                },
            }
        }
    )
    probe = GOOD_PROBE.replace("<<<torch>>>2.12.0+rocm10.0.0|AMD Instinct MI300X VF\n", "")
    transport = FakeTransport(
        {
            "emit os": CommandResult("probe", 0, probe, ""),
            remote_module.RESULT_MARKER: CommandResult(
                "run", 0, f"{remote_module.RESULT_MARKER}\n{body}", ""
            ),
            "tar xzf": CommandResult("unpack", 0, "", ""),
        }
    )
    run = remote_module.run_remote(
        transport, _spec(), source=source_tree, run_file=("kernel.py", "import torch\n")
    )
    assert not run.ok
    assert run.missing_torch
    assert "PyTorch was not found" in run.error
    assert "Jupyter" in run.error
    assert "did not complete" not in run.error


def test_a_file_run_lands_under_the_transport_root():
    transport = _RootedTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            "import torch": CommandResult("probe-torch", 2, "GPU=unavailable", ""),
        }
    )
    remote_module.run_remote_file(transport, _spec(), "converted.py", "import torch\n")
    assert any(
        "mkdir -p /shared-docker/root/graft/" in command
        for command in transport.commands
    )




_NO_GPU_PROBE = GOOD_PROBE.replace(
    "<<<agents>>>AMD Instinct MI300X VF|gfx942\n", ""
).replace("<<<kfd_gpus>>>1\n", "<<<kfd_gpus>>>0\n")


def _ai_body(run_rc: int = 0, verify_rc: int = 0) -> str:
    verification = {
        "conversion": {"verdict": "production_ready", "route": "pytorch"},
        "parity": {"measured": True, "passed": 4, "total": 4},
    }
    return json.dumps(
        {
            "steps": {
                "run": {"returncode": run_rc, "output": "ran the AI file"},
                "verification": {
                    "returncode": verify_rc,
                    "output": json.dumps(verification),
                },
                "report": {"returncode": 0, "output": "# AI report"},
            }
        }
    )


def _ai_transport(body: str) -> FakeTransport:
    return FakeTransport(
        {
            "emit os": CommandResult("probe", 0, GOOD_PROBE, ""),
            remote_module.RESULT_MARKER: CommandResult(
                "run", 0, f"{remote_module.RESULT_MARKER}\n{body}", ""
            ),
            "tar xzf": CommandResult("unpack", 0, "", ""),
        }
    )


def test_ai_file_reports_measured_verification_not_a_claim():
    result = remote_module.run_remote_ai_file(
        _ai_transport(_ai_body()), _spec(), "converted_gemma.py", "import torch\n"
    )
    assert result["ok"] is True
    assert result["ai_converted"] is True
    assert result["verification"]["conversion"]["verdict"] == "production_ready"
    assert result["verification"]["parity"]["passed"] == 4
    assert result["report_text"] == "# AI report"


def test_ai_file_refuses_to_upload_without_a_gpu():
    transport = FakeTransport(
        {"emit os": CommandResult("probe", 0, _NO_GPU_PROBE, "")}
    )
    result = remote_module.run_remote_ai_file(
        transport, _spec(), "converted_gemma.py", "import torch\n"
    )
    assert result["ok"] is False
    assert result["stage"] == "gpu_detection"
    assert transport.pushed == []


def test_ai_file_failed_run_is_not_reported_as_ok():
    result = remote_module.run_remote_ai_file(
        _ai_transport(_ai_body(run_rc=1)), _spec(), "converted_gemma.py", "x = 1\n"
    )
    assert result["ok"] is False
    assert "did not run cleanly" in result["error"]


def test_ai_file_failed_verification_is_not_reported_as_ok():
    result = remote_module.run_remote_ai_file(
        _ai_transport(_ai_body(verify_rc=1)), _spec(), "converted_gemma.py", "x = 1\n"
    )
    assert result["ok"] is False
    assert "could not verify" in result["error"]


def test_ai_file_rejects_an_unsupported_suffix():
    with pytest.raises(ValueError):
        remote_module.run_remote_ai_file(
            _ai_transport(_ai_body()), _spec(), "notes.txt", "hello"
        )


def test_ai_file_rejects_empty_source():
    with pytest.raises(ValueError):
        remote_module.run_remote_ai_file(
            _ai_transport(_ai_body()), _spec(), "converted_gemma.py", "   "
        )


def test_a_failed_verification_is_not_reported_as_ok(source_tree):
    transport = _remote_replies(verify_ok=False)
    transport.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    run = remote_module.run_remote(transport, _spec(), source=source_tree)
    assert not run.ok
    assert "did not verify" in run.error


def test_an_unpack_failure_is_reported_as_such(source_tree):
    transport = FakeTransport(
        {
            "tar xzf": CommandResult("unpack", 1, "", "tar: write error"),
        }
    )
    run = remote_module.run_remote(transport, _spec(), source=source_tree)
    assert not run.ok
    assert "would not unpack" in run.error
    assert "tar: write error" in run.error


def test_a_missing_result_marker_is_handled(source_tree):
    transport = FakeTransport({"": CommandResult("run", 0, "connection lost", "")})
    run = remote_module.run_remote(transport, _spec(), source=source_tree)
    assert not run.ok
    assert "verify_hip" not in run.steps
    assert "pipeline" not in run.steps
    assert "connection lost" in run.steps["remote_script"]["output"]
    assert "no result block" in run.error


def test_json_extraction_survives_surrounding_shell_noise():
    noisy = (
        "MOTD banner\nbash: warning: something\n"
        + remote_module.RESULT_MARKER
        + ('\n{"steps": {"a": {"returncode": 0, "output": "has } brace"}}}\ntrailing')
    )
    assert remote_module._extract_json(noisy) == {
        "steps": {"a": {"returncode": 0, "output": "has } brace"}}
    }


def test_json_extraction_handles_a_brace_inside_a_string():
    payload = remote_module.RESULT_MARKER + '\n{"error": "brace } in text"}'
    assert remote_module._extract_json(payload) == {"error": "brace } in text"}


def test_json_extraction_returns_none_when_absent():
    assert remote_module._extract_json("nothing here") is None


def test_the_remote_script_opens_no_doubled_braces():
    """The script is .replace()d, never .format()d, so {{ prints literally.

    A closing }} is legitimate, because the JSON wrapper really does end with
    two braces; only the opener betrays format-string thinking.
    """
    assert "{{" not in remote_module.REMOTE_SCRIPT


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_the_remote_script_emits_parsable_json(tmp_path):
    """Run the real emitter end to end.

    An unbalanced brace in the hand-written JSON only shows up as "no result
    block" after a round trip to a GPU host, which is an expensive place to
    find out. Here it fails in milliseconds.
    """
    project = tmp_path / "graft"
    (project / "src" / "graft").mkdir(parents=True)
    (project / "src" / "graft" / "__init__.py").write_text("")
    (project / "tools").mkdir()
    (project / "tools" / "verify_hip.py").write_text(
        'print("HIP VERIFICATION PASSED")\n'
    )

    script = tmp_path / "remote.sh"
    script.write_text(
        remote_module.REMOTE_SCRIPT.replace("__REMOTE_DIR__", f'"{project}"').replace(
            "__MARKER__", remote_module.RESULT_MARKER
        ).replace("__PIPELINE_FILE__", "examples/01_torch_cuda_softmax.py")
    )
    completed = subprocess.run(
        ["sh", str(script)], capture_output=True, text=True, timeout=120, check=False
    )

    payload = remote_module._extract_json(completed.stdout)
    assert payload is not None, (
        f"script wrote no parsable JSON.\nstdout: {completed.stdout!r}\n"
        f"stderr: {completed.stderr!r}"
    )
    steps = payload["steps"]
    assert set(steps) == {"verify_hip", "pipeline"}
    assert steps["verify_hip"]["returncode"] == 0
    assert "HIP VERIFICATION PASSED" in steps["verify_hip"]["output"]


def test_the_archive_source_is_the_local_tree_not_the_remote_dir(tmp_path):
    """`spec.remote_dir` exists only on the far side.

    Reading the two as the same path tars whatever happens to sit at that
    path locally, which is how the upload once grew to 11 MB.
    """
    local = tmp_path / "project"
    local.mkdir()
    (local / "README.md").write_text("graft")
    payload = remote_module.build_archive(local)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.getnames() == ["README.md"]


def test_remote_never_prints_the_secret(tmp_path, capsys):
    """The whole remote path must be silent about the credential."""
    secret = "UNIQUE-SECRET-STRING"
    spec = parse_form(
        {"kind": "jupyter", "host": "h", "token": secret, "remote_dir": "/root/graft"}
    )
    transport = _remote_replies()
    transport.replies["tar xzf"] = CommandResult("unpack", 0, "", "")
    run = remote_module.run_remote(transport, spec, source=tmp_path)
    rendered = json.dumps(run.as_dict()) + capsys.readouterr().out
    rendered += " ".join(transport.commands)
    assert secret not in rendered


def test_spec_module_exposes_no_stray_helper():
    """Guard against a secret-bearing helper creeping back in."""
    assert not hasattr(spec_module, "describe")




class TestJupyterUrlHandling:
    """A hosted notebook is reached over https. Treating every target as
    plaintext http on port 80 made the common case unreachable."""

    @pytest.mark.parametrize(
        ("raw", "scheme", "port"),
        [
            ("https://gpu.example.com", "https", 443),
            ("https://gpu.example.com:8443", "https", 8443),
            ("http://gpu.example.com", "http", 80),
            ("http://gpu.example.com:8080", "http", 8080),
            ("gpu.example.com", "http", 80),
            ("gpu.example.com:8888", "http", 8888),
            ("https://203.0.113.10", "https", 443),
        ],
    )
    def test_scheme_and_port_are_derived_from_the_url(self, raw, scheme, port):
        spec = parse_form({"kind": "jupyter", "host": raw})
        assert (spec.scheme, spec.port) == (scheme, port)

    def test_ssh_has_no_scheme(self):
        spec = parse_form({"kind": "ssh", "host": "203.0.113.5", "key_path": "/tmp/k"})
        assert spec.scheme == ""

    def test_the_transport_uses_https_and_wss(self):
        from graft.target.jupyter import JupyterTransport

        spec = parse_form({"kind": "jupyter", "host": "https://gpu.example.com"})
        transport = JupyterTransport(spec)
        assert transport.base == "https://gpu.example.com:443"
        assert transport._ws_url("1").startswith("wss://gpu.example.com:443/")

    def test_a_plain_target_still_uses_ws(self):
        from graft.target.jupyter import JupyterTransport

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com:8888"})
        transport = JupyterTransport(spec)
        assert transport.base == "http://gpu.example.com:8888"
        assert transport._ws_url("1").startswith("ws://gpu.example.com:8888/")

    def test_a_non_http_scheme_is_refused(self):
        with pytest.raises(TargetError):
            parse_form({"kind": "jupyter", "host": "ftp://gpu.example.com"})


class TestJupyterTokenPasting:
    """The token is usually copied from the address bar or from
    ``jupyter server list``, both of which wrap it in ``token=…``. Sending that
    prefix in the Authorization header makes Jupyter answer 403 (reported as
    "Check the token") even though the token itself is correct."""

    TOKEN = "O7NRmFuJbV5DGQGX01yKzbtYBJlbxqZxb7BuGKTZAivXdCzYy"

    @pytest.mark.parametrize(
        "raw",
        [
            TOKEN,
            f"token={TOKEN}",
            f"?token={TOKEN}",
            f"http://gpu.example.com/lab?token={TOKEN}",
            f"http://gpu.example.com:8888/?token={TOKEN}",
            f"  {TOKEN}\n",
        ],
    )
    def test_any_pasted_form_reduces_to_the_bare_token(self, raw):
        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com", "token": raw})
        assert spec.token == self.TOKEN

    def test_a_url_with_no_token_yields_an_empty_token(self):
        spec = parse_form(
            {"kind": "jupyter", "host": "gpu.example.com", "token": "http://h/lab"}
        )
        assert spec.token == ""

    def test_an_empty_token_stays_empty(self):
        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com", "token": ""})
        assert spec.token == ""


class TestJupyterVersionGate:
    @pytest.mark.parametrize(
        ("version", "supported"),
        [
            ("0.9.0", False),
            ("1.5.0", True),
            ("2.21.1", True),
            ("4.4.2", False),
            ("5.7.0", True),
            ("6.5.7", True),
            ("7.4.7", True),
            ("8.0.4", True),
            ("8.3.0", True),
            ("9.1.0", True),
            ("unknown", False),
        ],
    )
    def test_major_version_decides(self, version, supported):
        from graft.target.jupyter import supports_terminal_api

        assert supports_terminal_api(version) is supported


class TestJupyterOptionalDeps:
    """The Jupyter transport's dependencies are an opt-in extra.

    SSH and local runs work without them, so their absence must produce a
    clear, actionable message rather than an ImportError or a crash.
    """

    def test_a_missing_dep_names_itself_and_points_at_ssh(self, monkeypatch):
        from graft.target import jupyter

        monkeypatch.setattr(jupyter, "requests", None)
        with pytest.raises(TransportError) as excinfo:
            jupyter._require_deps()
        message = str(excinfo.value)
        assert "requests" in message
        assert "pip install requests" in message
        assert "SSH" in message

    def test_both_missing_are_both_named(self, monkeypatch):
        from graft.target import jupyter

        monkeypatch.setattr(jupyter, "requests", None)
        monkeypatch.setattr(jupyter, "websocket", None)
        with pytest.raises(TransportError) as excinfo:
            jupyter._require_deps()
        message = str(excinfo.value)
        assert "requests" in message and "websocket-client" in message
        assert "them with" in message

    def test_constructing_without_deps_fails_before_any_network(self, monkeypatch):
        from graft.target import jupyter

        monkeypatch.setattr(jupyter, "requests", None)
        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        with pytest.raises(TransportError):
            jupyter.JupyterTransport(spec)


class _SilentSocket:
    """A terminal socket that never produces the end marker.

    Stands in for a remote compile that prints nothing: the exact case that a
    fixed per-read timeout used to truncate.
    """

    def __init__(self):
        self.timeouts = []

    def send(self, _payload):
        return None

    def settimeout(self, value):
        self.timeouts.append(value)

    def recv(self):
        import websocket

        raise websocket.WebSocketTimeoutException("timed out")

    def close(self):
        return None


class _SlowSocket:
    """A terminal socket that dribbles output and burns wall-clock time."""

    def __init__(self, *, chunk: str, delay: float, frames: int):
        self.chunk = chunk
        self.delay = delay
        self.frames = frames
        self.timeouts: list[float] = []
        self.reads = 0

    def send(self, _payload):
        return None

    def settimeout(self, value):
        self.timeouts.append(value)

    def recv(self):
        import time as _time

        self.reads += 1
        if self.reads > self.frames:
            return ""
        _time.sleep(self.delay)
        return self.chunk

    def close(self):
        return None


class TestJupyterCommandDeadline:
    """`timeout_s` used to be accepted and then ignored, so every command was
    capped at the fixed per-read timeout however long the operator allowed."""

    def _transport(self, monkeypatch, socket_):
        import websocket

        from graft.target import jupyter

        monkeypatch.setattr(
            jupyter.websocket,
            "create_connection",
            lambda *args, **kwargs: socket_,
        )
        return websocket

    def test_a_stalled_command_is_cut_off_at_its_budget(self, monkeypatch):
        from graft.target.jupyter import JupyterTransport

        socket_ = _SilentSocket()
        self._transport(monkeypatch, socket_)
        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = JupyterTransport(spec)

        with pytest.raises(TransportError) as caught:
            transport._drive("1", "sleep 600", 5)
        assert "timed out" in str(caught.value).lower()
        assert "5s" in str(caught.value)
        assert "command timeout" in str(caught.value).lower()

    def test_the_socket_read_is_rearmed_with_the_remaining_budget(self, monkeypatch):
        """A read must never be armed for longer than the budget allows."""
        from graft.target.jupyter import JupyterTransport

        socket_ = _SlowSocket(chunk="working\r\n", delay=0.05, frames=4)
        self._transport(monkeypatch, socket_)
        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = JupyterTransport(spec)

        transport._drive("1", "make", 30)
        assert len(socket_.timeouts) >= 4
        assert socket_.timeouts == sorted(socket_.timeouts, reverse=True)
        assert all(value <= 30 for value in socket_.timeouts)

    def test_an_exhausted_budget_stops_before_another_read(self, monkeypatch):
        """Once the deadline passes the loop must raise, not keep reading."""
        from graft.target import jupyter

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = jupyter.JupyterTransport(spec)

        socket_ = _SlowSocket(chunk="working\r\n", delay=0.08, frames=100)
        monkeypatch.setattr(
            jupyter.websocket, "create_connection", lambda *a, **k: socket_
        )
        with pytest.raises(TransportError):
            transport._drive("1", "sleep 600", 0.2)
        assert socket_.reads < 100

    def test_no_budget_still_gets_a_default(self):
        from graft.target.jupyter import DEFAULT_RUN_TIMEOUT_S

        assert DEFAULT_RUN_TIMEOUT_S > 0


class _FakeResponse:
    """A minimal response that satisfies `_check` for the retry tests."""

    status_code = 200
    ok = True

    def json(self):
        return {"name": 1}


class TestJupyterConnectRetry:
    """A fresh server or a dropped packet must not fail the whole operation."""

    def _transport(self, monkeypatch):
        from graft.target import jupyter

        spec = parse_form(
            {"kind": "jupyter", "host": "gpu.example.com", "token": "t"}
        )
        transport = jupyter.JupyterTransport(spec)
        monkeypatch.setattr(jupyter, "CONNECT_BACKOFF_S", 0)
        return transport

    def test_a_transient_connection_error_is_retried_then_succeeds(self, monkeypatch):

        transport = self._transport(monkeypatch)
        calls: list[str] = []

        def flaky(method, url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                raise requests.exceptions.ConnectionError("connection refused")
            return _FakeResponse()

        monkeypatch.setattr(transport.session, "request", flaky)
        response = transport._request("GET", "/api/status", what="status", timeout=5)
        assert response.ok
        assert len(calls) == 2

    def test_a_persistent_connection_error_gives_up_after_the_retries(self, monkeypatch):
        from graft.target import jupyter

        transport = self._transport(monkeypatch)
        calls: list[str] = []

        def always_fail(method, url, **kwargs):
            calls.append(url)
            raise requests.exceptions.ConnectionError("connection refused")

        monkeypatch.setattr(transport.session, "request", always_fail)
        with pytest.raises(TransportError) as caught:
            transport._request("GET", "/api/status", what="status", timeout=5)
        assert "Could not reach" in str(caught.value)
        assert len(calls) == jupyter.CONNECT_RETRIES

    def test_a_read_timeout_is_not_retried(self, monkeypatch):

        transport = self._transport(monkeypatch)
        calls: list[str] = []

        def stalled(method, url, **kwargs):
            calls.append(url)
            raise requests.exceptions.ReadTimeout("read timed out")

        monkeypatch.setattr(transport.session, "request", stalled)
        with pytest.raises(TransportError) as caught:
            transport._request("GET", "/api/status", what="status", timeout=5)
        assert "Timed out" in str(caught.value)
        assert len(calls) == 1

    def test_a_dropped_handshake_is_retried_then_succeeds(self, monkeypatch):
        import websocket

        from graft.target import jupyter

        transport = self._transport(monkeypatch)
        attempts: list[int] = []

        class _Socket:
            def send(self, *_):
                return None

        def flaky(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise websocket.WebSocketException("handshake failed")
            return _Socket()

        monkeypatch.setattr(jupyter.websocket, "create_connection", flaky)
        socket_ = transport._open_socket("1", 30)
        assert isinstance(socket_, _Socket)
        assert len(attempts) == 2


class TestJupyterUploadMakesItsDirectory:
    """The contents API will not create a parent, so the upload has to.

    A PUT to a file whose directory is absent fails with an HTTP 500 from the
    filesystem, and a nested directory create does not make its own parents.
    The upload therefore walks the directory one component at a time first.
    """

    def _transport(self, monkeypatch):
        from graft.target import jupyter

        spec = parse_form(
            {"kind": "jupyter", "host": "gpu.example.com", "token": "t"}
        )
        transport = jupyter.JupyterTransport(spec)
        transport._server_root = "/shared-docker"
        return transport

    def test_push_creates_the_parent_before_the_file(self, monkeypatch):
        transport = self._transport(monkeypatch)
        calls: list[tuple[str, str, dict]] = []

        def record(method, url, **kwargs):
            calls.append((method, url, kwargs))
            return _FakeResponse()

        monkeypatch.setattr(transport.session, "request", record)
        transport.push(b"archive", "/shared-docker/root/graft-remote.tar.gz")

        assert calls[0][0] == "PUT"
        assert calls[0][1].endswith("/api/contents/root")
        assert calls[0][2]["data"] == '{"type": "directory"}'
        assert calls[-1][1].endswith("/api/contents/root/graft-remote.tar.gz")

    def test_a_nested_directory_is_created_one_component_at_a_time(self, monkeypatch):
        transport = self._transport(monkeypatch)
        urls: list[str] = []

        def record(method, url, **kwargs):
            urls.append(url)
            return _FakeResponse()

        monkeypatch.setattr(transport.session, "request", record)
        transport.push(b"x", "/shared-docker/a/b/c.bin")

        assert [url.rsplit("/api/", 1)[-1] for url in urls] == [
            "contents/a",
            "contents/a/b",
            "contents/a/b/c.bin",
        ]


class _RecordingSocket:
    """A terminal socket that records sends and replays canned frames."""

    def __init__(self, frames: list[str]) -> None:
        self.frames = list(frames)
        self.sent: list[str] = []
        self.timeouts: list[float] = []

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def recv(self) -> str:
        return self.frames.pop(0) if self.frames else ""

    def close(self) -> None:
        return None


def _stdout(text: str) -> str:
    return json.dumps(["stdout", text])


class TestJupyterTerminalFraming:
    """The terminal websocket speaks JSON frames, not a raw byte pipe.

    Sending the script as a bare string is accepted and then ignored by
    jupyter_server 2.x, so the command never ran and the reader blocked on
    output that never came. The mocked sockets in the other tests hid this,
    because they never looked at what was sent.
    """

    def test_stdin_is_wrapped_as_a_stdin_frame(self):
        from graft.target.jupyter import _stdin_frame

        assert json.loads(_stdin_frame("echo hi")) == ["stdin", "echo hi"]

    def test_a_stdout_frame_reduces_to_its_text(self):
        from graft.target.jupyter import _decode_frame

        assert _decode_frame(_stdout("hello\r\n")) == "hello\r\n"

    def test_terminal_control_sequences_are_stripped(self):
        from graft.target.jupyter import _decode_frame

        noisy = "\x1b[?2004hused=42\x1b]0;title\x07\r\n"
        assert _decode_frame(_stdout(noisy)) == "used=42\r\n"

    def test_control_frames_carry_no_output(self):
        from graft.target.jupyter import _decode_frame

        assert _decode_frame(json.dumps(["setup", {}])) == ""
        assert _decode_frame(json.dumps(["disconnect", 1])) == ""

    def test_a_non_stdout_json_frame_is_passed_through(self):
        from graft.target.jupyter import _decode_frame

        assert _decode_frame(json.dumps({"a": 1})) == json.dumps({"a": 1})

    def test_raw_text_frames_still_work(self):
        from graft.target.jupyter import _decode_frame

        assert _decode_frame("plain output") == "plain output"

    def test_drive_sends_a_framed_script(self, monkeypatch):
        from graft.target.jupyter import END, JupyterTransport

        socket_ = _RecordingSocket([_stdout(f"{END}\r\n0\r\n")])
        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = JupyterTransport(spec)
        monkeypatch.setattr(transport, "_open_socket", lambda *_: socket_)

        transport._drive("1", "echo hi", 5)

        assert json.loads(socket_.sent[0]) == ["stdin", "echo hi"]

    def test_stream_sends_a_framed_command(self, monkeypatch):
        from graft.target.jupyter import JupyterTransport

        socket_ = _RecordingSocket([_stdout("used=7\n")])
        stop = threading.Event()
        lines: list[str] = []

        def on_line(line: str) -> None:
            lines.append(line)
            stop.set()

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = JupyterTransport(spec)
        monkeypatch.setattr(transport, "_create_terminal", lambda: "1")
        monkeypatch.setattr(transport, "_open_socket", lambda *_: socket_)
        monkeypatch.setattr(transport, "_delete_terminal", lambda _: None)

        transport.stream("while :; do :; done", on_line, stop=stop)

        assert json.loads(socket_.sent[0])[0] == "stdin"
        assert lines == ["used=7"]


class TestJupyterServerRoot:
    """Files land under the server's root, which is not always "/".

    The terminal and the contents API share one filesystem, but only the
    contents API is confined to the server root. An image that starts Jupyter
    in /shared-docker served uploads from there while the shell looked under
    /root, so the archive was never where the unpack command expected it.
    """

    def _transport(self, monkeypatch, listing: str):
        from graft.target.jupyter import JupyterTransport

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = JupyterTransport(spec)
        monkeypatch.setattr(
            transport,
            "run",
            lambda *_, **__: CommandResult("jupyter server list", 0, listing, ""),
        )
        return transport

    def test_the_root_is_read_from_the_server_listing(self, monkeypatch):
        transport = self._transport(
            monkeypatch, "http://x:8888/?token=t :: /shared-docker\n"
        )
        assert transport.server_root == "/shared-docker"
        assert (
            transport.resolve_remote_dir("/root/graft") == "/shared-docker/root/graft"
        )

    def test_a_slash_root_leaves_paths_alone(self, monkeypatch):
        transport = self._transport(monkeypatch, ":: /\n")
        assert transport.server_root == "/"
        assert transport.resolve_remote_dir("/root/graft") == "/root/graft"

    def test_an_unreadable_listing_defaults_to_slash(self, monkeypatch):
        transport = self._transport(monkeypatch, "")
        assert transport.server_root == "/"

    def test_a_directory_already_under_the_root_is_left_alone(self, monkeypatch):
        transport = self._transport(monkeypatch, ":: /shared-docker\n")
        assert (
            transport.resolve_remote_dir("/shared-docker/graft")
            == "/shared-docker/graft"
        )

    def test_contents_path_strips_a_non_slash_root(self, monkeypatch):
        transport = self._transport(monkeypatch, ":: /shared-docker\n")
        assert (
            transport._contents_path("/shared-docker/root/graft/x.py")
            == "/api/contents/root/graft/x.py"
        )

    def test_contents_path_still_works_with_a_slash_root(self, monkeypatch):
        transport = self._transport(monkeypatch, ":: /\n")
        assert (
            transport._contents_path("/root/graft/x.py")
            == "/api/contents/root/graft/x.py"
        )


class _DeadChannel:
    """An SSH channel whose peer has gone: every read is an immediate EOF."""

    def settimeout(self, _value: float) -> None:
        return None

    def recv(self, _size: int) -> bytes:
        return b""

    def close(self) -> None:
        return None


class _DeadSSHClient:
    def exec_command(self, _command: str):
        return None, types.SimpleNamespace(channel=_DeadChannel()), None


class _ClosingSocket:
    """A terminal socket that reports the peer hung up."""

    def send(self, _payload):
        return None

    def settimeout(self, _value):
        return None

    def recv(self):
        return ""

    def close(self):
        return None


class TestLiveStreamLoss:
    """A vanished target must be reported as a lost connection.

    The stream command loops forever, so an EOF is the peer hanging up. Before
    this, the reader treated it as a clean finish and the last reading stayed
    on screen as if the GPU were still there.
    """

    def test_ssh_stream_raises_when_the_peer_closes(self):
        pytest.importorskip("paramiko")
        from graft.target.ssh import SSHTransport

        spec = parse_form(
            {"kind": "ssh", "host": "203.0.113.5", "key_path": "/tmp/k"}
        )
        transport = SSHTransport(spec)
        transport._ensure_connected = lambda: _DeadSSHClient()
        with pytest.raises(TransportError) as caught:
            transport.stream(
                "while :; do :; done", lambda line: None, stop=threading.Event()
            )
        assert "closed by the target" in str(caught.value)

    def test_ssh_enables_keepalives_so_a_silent_drop_is_noticed(self, monkeypatch):
        pytest.importorskip("paramiko")
        from graft.target import ssh

        recorded: dict = {}

        class _Transport:
            def set_keepalive(self, value):
                recorded["keepalive"] = value

        class _Client:
            def load_system_host_keys(self):
                return None

            def set_missing_host_key_policy(self, _policy):
                return None

            def connect(self, **_kwargs):
                return None

            def get_transport(self):
                return _Transport()

        monkeypatch.setattr(ssh.paramiko, "SSHClient", _Client)
        spec = parse_form(
            {"kind": "ssh", "host": "203.0.113.5", "key_path": "/tmp/k"}
        )
        ssh.SSHTransport(spec)._ensure_connected()
        assert recorded["keepalive"] == ssh.KEEPALIVE_S

    def test_jupyter_stream_raises_when_the_peer_closes(self, monkeypatch):
        pytest.importorskip("websocket")
        from graft.target.jupyter import JupyterTransport

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        transport = JupyterTransport(spec)
        monkeypatch.setattr(transport, "_create_terminal", lambda: "1")
        monkeypatch.setattr(
            transport, "_open_socket", lambda _name, _budget: _ClosingSocket()
        )
        monkeypatch.setattr(transport, "_delete_terminal", lambda _name: None)
        with pytest.raises(TransportError) as caught:
            transport.stream(
                "while :; do :; done", lambda line: None, stop=threading.Event()
            )
        assert "closed by the target" in str(caught.value)


class TestUnpackDiagnosis:
    """A failed unpack must name the likely cause, not just forward tar's text."""

    def _failed(self, stderr: str):
        from graft.target.transport import CommandResult

        return CommandResult(
            command="tar xzf a.tar.gz", returncode=2, stdout="", stderr=stderr
        )

    def test_a_jupyter_path_mismatch_is_explained(self):
        from graft.target.remote import _unpack_hint

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        message = _unpack_hint(
            self._failed("tar: /root/g.tar.gz: Cannot open: No such file or directory"),
            spec,
        )
        assert "root directory" in message

    def test_ssh_gets_the_raw_reason_only(self):
        from graft.target.remote import _unpack_hint

        spec = parse_form({"kind": "ssh", "host": "203.0.113.5", "key_path": "/tmp/k"})
        message = _unpack_hint(self._failed("tar: cannot open"), spec)
        assert "root directory" not in message
        assert "cannot open" in message

    def test_an_unrelated_failure_is_not_mislabelled(self):
        from graft.target.remote import _unpack_hint

        spec = parse_form({"kind": "jupyter", "host": "gpu.example.com"})
        message = _unpack_hint(
            self._failed("tar: graft/x: Cannot open: Permission denied"), spec
        )
        assert "root directory" not in message




def _deps_facts(**overrides):
    from graft.target.detect import TargetFacts

    base: dict = {
        "ok": True,
        "os_name": "Ubuntu 24.04.5 LTS",
        "rocm_version": "6.2.0",
        "gfx_arch": "gfx942",
        "gpu_count": 1,
        "gpu_name": "AMD Instinct MI300X",
        "tools": {
            "hipcc": "/opt/rocm/bin/hipcc",
            "hipify-perl": "/opt/rocm/bin/hipify-perl",
        },
        "python_path": "/usr/bin/python3",
    }
    base.update(overrides)
    return TargetFacts(**base)


class TestDependencyScan:
    def test_a_complete_target_has_nothing_actionable(self):
        from graft.target.dependencies import actionable_dependencies

        facts = _deps_facts(torch_version="2.4.0+rocm6.2")
        assert actionable_dependencies(facts) == []

    def test_torch_is_actionable_but_not_required(self):
        from graft.target.dependencies import (
            actionable_dependencies,
            missing_dependencies,
        )

        facts = _deps_facts(torch_version="")
        assert [dep.key for dep in actionable_dependencies(facts)] == ["torch"]
        assert missing_dependencies(facts) == []

    def test_torch_install_uses_the_rocm_wheel_index(self):
        from graft.target.dependencies import scan_dependencies

        torch = next(
            dep for dep in scan_dependencies(_deps_facts()) if dep.key == "torch"
        )
        assert "https://download.pytorch.org/whl/rocm6.2" in torch.command
        assert "ROCm 6.2" in torch.detail

    def test_torch_install_bootstraps_pip_and_installs_vision(self):
        from graft.target.dependencies import scan_dependencies

        torch = next(
            dep for dep in scan_dependencies(_deps_facts()) if dep.key == "torch"
        )
        assert "apt-get install -y python3-pip" in torch.command
        assert torch.command.index("python3-pip") < torch.command.index("pip3 install")
        assert "pip3 install" in torch.command
        assert "torch torchvision" in torch.command
        assert "--break-system-packages" in torch.command

    def test_pip_break_flag_is_added_only_where_pep_668_is_enforced(self):
        from graft.target.dependencies import _torch_command

        assert "--break-system-packages" in _torch_command(
            _deps_facts(os_name="Ubuntu 24.04.5 LTS")
        )[0]
        assert "--break-system-packages" not in _torch_command(
            _deps_facts(os_name="Ubuntu 22.04.5 LTS")
        )[0]
        assert "--break-system-packages" in _torch_command(
            _deps_facts(os_name="Debian GNU/Linux 12 (bookworm)")
        )[0]
        assert "--break-system-packages" not in _torch_command(
            _deps_facts(os_name="Debian GNU/Linux 11 (bullseye)")
        )[0]

    def test_torch_install_prefers_pytorch_own_rocm_series(self):
        from graft.target.dependencies import scan_dependencies

        facts = _deps_facts(rocm_version="10.1.0", torch_hip="7.14.60850")
        torch = next(dep for dep in scan_dependencies(facts) if dep.key == "torch")
        assert "https://download.pytorch.org/whl/rocm7.14" in torch.command
        assert "rocm10.1" not in torch.command
        assert "ROCm 7.14" in torch.detail

    def test_an_unknown_rocm_version_falls_back_to_pypi_with_a_warning(self):
        from graft.target.dependencies import scan_dependencies

        facts = _deps_facts(rocm_version="")
        torch = next(dep for dep in scan_dependencies(facts) if dep.key == "torch")
        assert "rocm" not in torch.command.split("index-url")[-1]
        assert "unknown" in torch.detail.lower()

    def test_a_non_debian_target_gets_a_manual_hint_not_a_command(self):
        from graft.target.dependencies import (
            missing_dependencies,
            scan_dependencies,
        )

        facts = _deps_facts(os_name="Red Hat Enterprise Linux 9", tools={})
        hipcc = next(dep for dep in scan_dependencies(facts) if dep.key == "hipcc")
        assert hipcc.present is False
        assert hipcc.command == ""
        assert "not apt-based" in hipcc.detail
        assert "hipcc" in [dep.key for dep in missing_dependencies(facts)]






class TestMetrics:
    def test_parse_reads_the_known_keys(self):
        from graft.target.metrics import parse_metrics

        snapshot = parse_metrics(
            "vram_total_bytes=100\nvram_used_bytes=40\n"
            "gpu_util_pct=12\ntemp_c=52\npower_w=143\n"
            "disk_total_bytes=1000\ndisk_used_bytes=250\n"
            "gpu_name=AMD Instinct MI300X\n"
        )
        assert snapshot.vram_used_bytes == 40
        assert snapshot.vram_total_bytes == 100
        assert snapshot.gpu_util_pct == 12
        assert snapshot.temp_c == 52
        assert snapshot.disk_used_bytes == 250
        assert snapshot.gpu_name == "AMD Instinct MI300X"

    def test_unknown_and_blank_values_stay_unmeasured(self):
        from graft.target.metrics import parse_metrics

        snapshot = parse_metrics("vram_total_bytes=\ngpu_util_pct=nonsense\n")
        assert snapshot.vram_total_bytes == -1
        assert snapshot.gpu_util_pct == -1

    def test_the_sample_command_is_a_single_reading(self):
        from graft.target.metrics import sample_command

        command = sample_command()
        assert "while :" not in command
        assert "mem_info_vram_used" in command
        assert "df -P -B1" in command
        assert "temp*_input" in command
        assert "power1_input" in command

    def test_the_stream_command_loops_and_marks_each_reading(self):
        from graft.target.metrics import SENTINEL, stream_command

        command = stream_command(0.5)
        assert "while :" in command
        assert SENTINEL in command
        assert "sleep 0.5" in command

    def test_the_stream_interval_has_a_floor(self):
        from graft.target.metrics import stream_command

        assert "sleep 0.2" in stream_command(0.0)

    def test_local_snapshot_always_reports_disk(self):
        from graft.target.metrics import local_snapshot

        usage = shutil.disk_usage("/")
        snapshot = local_snapshot()
        assert snapshot.source == "local"
        assert snapshot.disk_total_bytes == usage.total





class TestLocalTarget:
    def test_local_target_facts_is_targetfacts_shaped(self):
        from graft.target.detect import TargetFacts
        from graft.target.local import local_target_facts

        facts = TargetFacts(**local_target_facts())
        assert facts.ok is True
        assert facts.python_path
        assert isinstance(facts.tools, dict)

    def test_local_transport_runs_a_command_here(self):
        from graft.target.local import LocalTransport

        with LocalTransport() as transport:
            result = transport.run("echo local-ok")
        assert result.ok
        assert "local-ok" in result.stdout

    def test_a_failing_local_command_is_reported_not_raised(self):
        from graft.target.local import LocalTransport

        with LocalTransport() as transport:
            result = transport.run("exit 3")
        assert result.ok is False
        assert result.returncode == 3

    def test_local_stream_and_push_say_they_do_not_apply(self):
        from graft.target.local import LocalTransport

        transport = LocalTransport()
        with pytest.raises(TransportError):
            transport.stream("while :; do echo x; done", lambda line: None, stop=None)
        with pytest.raises(TransportError):
            transport.push(b"data", "/tmp/x")



class TestLocalSysfsMetrics:
    def test_sysfs_readings_are_parsed_from_a_fake_card(self, tmp_path, monkeypatch):
        from graft.target import metrics as metrics_module

        card = tmp_path / "card0" / "device"
        hwmon = card / "hwmon" / "hwmon0"
        hwmon.mkdir(parents=True)
        (card / "mem_info_vram_total").write_text("17179869184\n")
        (card / "mem_info_vram_used").write_text("4294967296\n")
        (card / "gpu_busy_percent").write_text("41\n")
        (hwmon / "temp1_input").write_text("52000\n")
        (hwmon / "power1_average").write_text("143000000\n")

        real_glob = metrics_module.glob.glob

        def fake_glob(pattern: str):
            if pattern.startswith("/sys/class/drm/"):
                return [str(card)] if pattern == "/sys/class/drm/card*/device" else []
            return real_glob(pattern)

        monkeypatch.setattr(metrics_module.glob, "glob", fake_glob)
        fields = metrics_module._local_sysfs_fields()
        assert fields["vram_total_bytes"] == "17179869184"
        assert fields["vram_used_bytes"] == "4294967296"
        assert fields["gpu_util_pct"] == "41"
        assert fields["temp_c"] == "52"
        assert fields["power_w"] == "143"

    def test_a_machine_without_amdgpu_sysfs_reports_nothing(self, monkeypatch):
        from graft.target import metrics as metrics_module

        monkeypatch.setattr(metrics_module.glob, "glob", lambda pattern: [])
        assert metrics_module._local_sysfs_fields() == {}

    def test_parse_jupyter_token_extracts_known_formats(self):
        from graft.target import token

        assert token.parse_jupyter_token("http://h:8888/?token=abc123DEF") == "abc123DEF"
        assert token.parse_jupyter_token('"token": "TK/xyz123"') == "TK/xyz123"
        assert token.parse_jupyter_token("ServerApp.token='ABCD1234abcd'") == "ABCD1234abcd"
        assert token.parse_jupyter_token("no token here") == ""
        assert token.parse_jupyter_token('"token": "disabled"') == ""

    def test_discover_jupyter_token_runs_script_and_returns_first_match(self):
        from graft.target.token import TOKEN_DISCOVERY_SCRIPT, discover_jupyter_token

        class T:
            def __init__(self):
                self.cmds = []

            def run(self, command, *, cwd=None, timeout_s=None):
                self.cmds.append(command)
                return CommandResult("s", 0, '{"token":"ABCDEF123456"}', "")

        t = T()
        got = discover_jupyter_token(t, timeout_s=5)
        assert got == "ABCDEF123456"
        assert t.cmds[0] == TOKEN_DISCOVERY_SCRIPT
        assert len(t.cmds) == 1

    def test_discover_jupyter_token_raises_when_nothing_found(self):
        from graft.target import TargetError
        from graft.target.token import discover_jupyter_token

        class T:
            def run(self, command, *, cwd=None, timeout_s=None):
                return CommandResult("s", 0, "", "")

        with pytest.raises(TargetError):
            discover_jupyter_token(T())
