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

import pytest

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


# --- spec ------------------------------------------------------------------


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
        ("https://h", ("h", 443)),  # an https URL with no port means 443
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


# --- secret hygiene --------------------------------------------------------


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


# --- transport output bounds ----------------------------------------------


def test_clip_keeps_the_tail_where_diagnostics_live():
    """Compilers put the explanation at the end, so the tail must survive."""
    body = "x" * 100_000 + "THE REAL ERROR IS HERE\n"
    clipped = clip(body, limit=1000)
    assert len(clipped) <= 1000 + 64
    assert "THE REAL ERROR IS HERE" in clipped


def test_clip_leaves_short_output_alone():
    assert clip("hello", limit=100) == "hello"


# --- detection -------------------------------------------------------------


# Captured from the MI300X host on 2026-10-04, including the CPU agent that
# rocminfo lists beside the GPU. The probe reports one line per GPU agent.
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


# --- orchestration ---------------------------------------------------------


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
    # Nothing was parsed, so nothing may be presented as a verification step;
    # the raw transcript is kept under an explicit diagnostic name instead.
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
        )
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
    # Both steps must be present: a missing key means a truncated JSON object.
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


# --- Jupyter transport details --------------------------------------------


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
        # SSH negotiates its own transport security; a scheme here would be
        # meaningless and misleading.
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


class TestJupyterVersionGate:
    @pytest.mark.parametrize(
        ("version", "supported"),
        [
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
        # The message has to be actionable: it names the budget and the fix.
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
        # Each re-arm must shrink: the read budget follows the deadline down.
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
        # It stopped because the clock ran out, not because the socket died.
        assert socket_.reads < 100

    def test_no_budget_still_gets_a_default(self):
        from graft.target.jupyter import DEFAULT_RUN_TIMEOUT_S

        assert DEFAULT_RUN_TIMEOUT_S > 0


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
        # Permission denied is not a path mismatch, so no misleading hint.
        assert "root directory" not in message


# --- dependencies -----------------------------------------------------------


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
        # PyTorch is optional, so it is not a blocking gap.
        assert missing_dependencies(facts) == []

    def test_torch_install_uses_the_rocm_wheel_index(self):
        from graft.target.dependencies import scan_dependencies

        torch = next(
            dep for dep in scan_dependencies(_deps_facts()) if dep.key == "torch"
        )
        assert "https://download.pytorch.org/whl/rocm6.2" in torch.command
        assert "ROCm 6.2" in torch.detail

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


class TestDependencyInstall:
    def test_only_the_missing_installable_commands_run(self):
        from graft.target.dependencies import install_dependencies

        # hipcc is present; hipify-perl and torch are missing and installable.
        facts = _deps_facts(tools={"hipcc": "/opt/rocm/bin/hipcc"})
        transport = FakeTransport()
        spec = parse_form({"kind": "ssh", "host": "10.0.0.9", "key_path": "/tmp/k"})
        results = install_dependencies(transport, spec, facts)

        assert len(transport.commands) == 2
        assert "apt-get install -y rocm-hip-sdk" in transport.commands[0]
        assert any("install torch" in c for c in transport.commands)
        assert all(item["ok"] for item in results)

    def test_a_manual_gap_is_reported_and_nothing_else_runs(self):
        from graft.target.dependencies import install_dependencies

        facts = _deps_facts(
            os_name="Red Hat Enterprise Linux 9",
            tools={"hipify-perl": "/x"},
        )
        transport = FakeTransport()
        spec = parse_form({"kind": "ssh", "host": "10.0.0.9", "key_path": "/tmp/k"})
        results = install_dependencies(transport, spec, facts)

        # hipcc cannot be auto-installed here, but torch still can.
        assert all("apt-get" not in c for c in transport.commands)
        manual = [item for item in results if item["key"] == "hipcc"]
        assert manual and manual[0]["ok"] is False
        assert "not apt-based" in manual[0]["error"]

    def test_a_failed_command_is_reported_not_raised(self):
        from graft.target.dependencies import install_dependencies

        facts = _deps_facts(tools={})  # hipify-perl and hipcc missing
        transport = FakeTransport(
            {"apt-get": CommandResult("apt", 100, "", "E: Unable to locate package")}
        )
        spec = parse_form({"kind": "ssh", "host": "10.0.0.9", "key_path": "/tmp/k"})
        results = install_dependencies(transport, spec, facts)

        failed = [item for item in results if not item["ok"]]
        assert failed
        assert "Unable to locate package" in failed[0]["error"]

    def test_progress_is_reported_per_dependency(self):
        from graft.target.dependencies import install_dependencies

        seen: list[str] = []
        facts = _deps_facts(tools={"hipcc": "/x"})
        spec = parse_form({"kind": "ssh", "host": "10.0.0.9", "key_path": "/tmp/k"})
        install_dependencies(
            FakeTransport(), spec, facts, on_progress=lambda key, _: seen.append(key)
        )
        assert "torch" in seen


# --- metrics ----------------------------------------------------------------


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
        # The sensor node names differ per card, so both families are globbed.
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

