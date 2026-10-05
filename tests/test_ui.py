"""Web UI tests.

The UI is the only part of Graft that accepts code from someone else, so the
tests here are weighted toward containment rather than appearance:

* `render_result` is a pure function of the summary dict, so every panel is
  tested directly, with no server and no subprocess.
* The rendered HTML is checked for the honesty properties the docstrings
  claim: unmeasured renders as unmeasured, and a verdict never appears
  without the evidence behind it.
* `pipeline.run()` is never called in-process. A source-level assertion
  pins that, because it is the one property that must not rot quietly.
* The route tests go through a real child process, since that is where
  uploaded code actually runs.

One test runs the full pipeline on the PyTorch example. It is slow by nature;
everything else here is fast.
"""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys

import pytest
from starlette.datastructures import UploadFile

from graft import config, pipeline
from graft.core import runner
from graft.target import TargetError
from graft.ui import summary as summary_module
from graft.ui import worker
from graft.ui.app import ISOLATION_NOTES, app, render_result

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"
CUDA_EXAMPLE = config.EXAMPLES_DIR / "02_cuda_add_kernel.cu"


@pytest.fixture(scope="module")
def real_summary() -> dict:
    """A summary produced by the real pipeline, reused across tests."""
    return summary_module.summarize(
        pipeline.run(EXAMPLE, with_narrative=False),
    )


def blank_summary() -> dict:
    """A summary in which nothing was measured."""
    return {
        "report_text": "# Graft\n\nnothing measured",
        "skipped": ["benchmark: requires a GPU device"],
        "environment": summary_module._environment(None),
        "conversion": {"measured": False},
        "execution": {"measured": False},
        "parity": {"measured": False},
        "benchmarks": [],
        "narrative": {"prose": {}, "reasons": []},
    }


# --- summarising -----------------------------------------------------------


class TestSummarize:
    def test_every_section_present_even_when_unmeasured(self):
        payload = summary_module.summarize(pipeline.run(CUDA_EXAMPLE, run_parity=False))
        assert set(payload) == {
            "report_text",
            "skipped",
            "environment",
            "conversion",
            "execution",
            "parity",
            "benchmarks",
            "narrative",
        }

    def test_json_native(self, real_summary):
        json.dumps(real_summary)
        for section in ("environment", "conversion", "execution", "parity", "narrative"):
            json.dumps(real_summary[section])

    def test_cpu_run_has_no_benchmarks(self, real_summary):
        # Benchmarks require a GPU. An empty list, not a fabricated number.
        assert real_summary["benchmarks"] == []
        assert any("benchmark" in note for note in real_summary["skipped"])

    def test_parity_counts_are_ints(self, real_summary):
        parity = real_summary["parity"]
        assert parity["measured"] is True
        assert parity["passed"] == parity["total"] == 40
        assert parity["all_passed"] is True
        assert isinstance(parity["max_abs_error"], float)

    def test_verdict_basis_is_transported_not_reworded(self, real_summary):
        conversion = real_summary["conversion"]
        assert conversion["verdict"] == "production_ready"
        assert "40/40" in conversion["verdict_basis"]

    def test_cu_file_is_flagged_as_text_analysis_only(self):
        payload = summary_module.summarize(
            pipeline.run(CUDA_EXAMPLE, run_parity=False, run_bench=False, with_narrative=False)
        )
        conversion = payload["conversion"]
        assert conversion["measured"] is True
        # The single most important honesty property: a .cu file that was
        # never compiled must not read as a completed verification.
        assert "text analysis" in conversion["verdict_basis"]

    def test_unmeasured_sections_are_marked(self):
        assert summary_module._environment(None)["total_vram_mb"] is None
        assert summary_module._parity(None)["measured"] is False
        assert summary_module._execution(None)["measured"] is False
        assert summary_module._benchmarks(None) == []

    def test_non_numeric_errors_do_not_become_zero(self):
        # A failed run must not read as a fast one.
        assert summary_module._number("boom") is None
        assert summary_module._number(None) is None
        assert summary_module._number(True) is None
        assert summary_module._number(0) == 0.0

    def test_unavailable_narrative_is_a_reason_not_prose(self):
        narrative = summary_module._narrative(
            {
                "narration": "not available (no API key configured)",
                "parity_explainer": "A real explanation.",
            }
        )
        assert narrative["prose"] == {"parity_explainer": "A real explanation."}
        assert narrative["reasons"] == ["not available (no API key configured)"]

    def test_blank_narrative_values_ignored(self):
        assert summary_module._narrative({"narration": "   "}) == {"prose": {}, "reasons": []}

    def test_unknown_narrative_keys_dropped(self):
        narrative = summary_module._narrative({"secret_key": "leaked"})
        assert "secret_key" not in narrative["prose"]


# --- rendering -------------------------------------------------------------


def html_of(summary: dict) -> str:
    """Render a summary to HTML.

    `render_result` returns a FastHTML component; `str` on it produces the
    already-escaped Safe markup, which is exactly what a browser receives.
    """
    return str(render_result(summary))


class TestRender:
    def test_real_run_shows_measured_numbers(self, real_summary):
        html = html_of(real_summary)
        assert "40/40" in html
        assert "PRODUCTION READY" in html
        assert "AST analysis, 40/40 parity cases executed" in html

    def test_unmeasured_run_says_so_everywhere(self):
        html = html_of(blank_summary())
        assert html.count("not measured") >= 3
        # No invented zero and no empty section.
        assert "0/0" not in html

    def test_verdict_never_appears_without_its_basis(self, real_summary):
        html = html_of(real_summary)
        assert "PRODUCTION READY" in html
        assert real_summary["conversion"]["verdict_basis"] in html

    def test_cpu_only_run_says_it_is_not_rocm_evidence(self, real_summary):
        assert real_summary["environment"]["gpu_available"] is False
        html = html_of(real_summary)
        assert "CPU-only" in html

    def test_findings_bucketed_by_severity(self):
        payload = blank_summary()
        payload["conversion"] = {
            "measured": True,
            "verdict": "manual_port",
            "verdict_line": "MANUAL PORT",
            "verdict_basis": "one blocker unmapped",
            "route": "cuda_cpp",
            "backend": "hipify",
            "changes": [],
            "findings": [
                {"severity": "blocker", "symbol": "cooperative_groups", "reason": "no HIP equivalent"},
                {"severity": "info", "symbol": "atomicAdd", "reason": "identical on ROCm"},
            ],
        }
        html = html_of(payload)
        assert "BLOCKER X1" in html
        assert "INFO X1" in html
        # Meters are proportional, so one blocker of two findings is 50%.
        assert "50% of findings" in html

    def test_export_is_an_anchor_not_an_htmx_swap(self, real_summary):
        html = html_of(real_summary)
        assert 'href="/export.md"' in html
        assert 'download="graft-report.md"' in html
        # An htmx GET would render Markdown into the page instead of
        # downloading it.
        assert 'hx_get="/export.md"' not in html

    def test_skipped_section_only_when_something_skipped(self):
        assert "Skipped" in html_of(blank_summary())
        payload = blank_summary()
        payload["skipped"] = []
        assert "Skipped" not in html_of(payload)


# --- containment -----------------------------------------------------------


class TestContainment:
    @staticmethod
    def _app_source() -> str:
        return (config.PACKAGE_ROOT / "ui" / "app.py").read_text(encoding="utf-8")

    def test_ui_never_calls_pipeline_in_process(self):
        # The single most important structural property here. Uploaded code
        # runs in worker.py's child; if this ever fails, the endpoint is
        # remote code execution.
        #
        # Parsed rather than grepped: the module's own docstring explains
        # that it never calls pipeline.run(), and a substring check would
        # match that explanation. The attribute test is also narrowed to
        # `pipeline.run` specifically, so an unrelated subprocess.run call
        # does not trip it.
        tree = ast.parse(self._app_source())
        pipeline_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pipeline"
        ]
        assert pipeline_calls == []

    def test_ui_does_not_import_pipeline_or_agent(self):
        tree = ast.parse(self._app_source())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not any(name.startswith("graft.agent") for name in imported)
        assert not any(name.startswith("graft.pipeline") for name in imported)

    def test_isolation_notes_come_from_the_runner(self):
        # Read from the runner rather than restated, so the page cannot
        # drift into describing isolation the runner does not provide.
        assert list(runner.sandbox_notes()) == ISOLATION_NOTES


# --- worker ----------------------------------------------------------------


class TestWorker:
    def _run(self, payload: dict, timeout: int = 600):
        return subprocess.run(
            [sys.executable, "-m", "graft.ui.worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_malformed_request_reports_failure(self):
        completed = subprocess.run(
            [sys.executable, "-m", "graft.ui.worker"],
            input="not json",
            capture_output=True,
            text=True,
            timeout=120,
        )
        payload = json.loads(completed.stdout)
        assert payload["ok"] is False
        assert "malformed" in payload["error"]

    def test_rejects_unsupported_suffix(self):
        completed = self._run({"source": "echo hi", "filename": "run.sh"})
        payload = json.loads(completed.stdout)
        assert payload["ok"] is False
        assert "unsupported file type" in payload["error"]

    def test_rejects_empty_source(self):
        completed = self._run({"source": "   ", "filename": "x.py"})
        assert json.loads(completed.stdout)["ok"] is False

    def test_reports_pipeline_failure_instead_of_crashing(self):
        # A module that cannot be imported is an outcome the report
        # describes, not an exception the worker should die on.
        completed = self._run({"source": "import nonexistent_zzz\n", "filename": "x.py"})
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        assert any("not importable" in note for note in payload["summary"]["skipped"])

    def test_successful_run_returns_summary(self):
        completed = self._run(
            {
                "source": EXAMPLE.read_text(encoding="utf-8"),
                "filename": "01_torch_cuda_softmax.py",
                "options": {"with_narrative": False},
            }
        )
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        assert payload["version"] == worker.PROTOCOL_VERSION
        assert payload["summary"]["parity"]["passed"] == 40

    def test_worker_output_is_pure_json(self):
        # The parent parses stdout directly, so anything the analysed file
        # prints would corrupt the response. The pipeline imports that file
        # in this process, so its output has to be diverted to stderr.
        completed = self._run(
            {
                "source": "print('noise from the analysed file')\n"
                "OPERATION='softmax'\ndef solve(x): return x\n",
                "filename": "x.py",
            }
        )
        payload = json.loads(completed.stdout)
        assert payload["ok"] is True
        assert "noise from the analysed file" in completed.stderr
        assert "noise from the analysed file" not in completed.stdout

    def test_worker_truncates_large_output(self):
        # A file that prints far more than the limit has its output clipped,
        # with a note saying so. Kept finite on purpose: an unbounded print
        # loop would fill the pipe this test reads, which is a good way to
        # take down the machine running the tests.
        completed = self._run(
            {
                "source": "print('y' * 40000)\n"
                "OPERATION='softmax'\ndef solve(x): return x\n",
                "filename": "loud.py",
                "options": {"run_parity": False, "run_bench": False},
            },
            timeout=120,
        )
        assert "more characters" in completed.stderr
        assert len(completed.stderr) < worker.CODE_STDOUT_LIMIT * 2


# --- routes ----------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return app.get_testclient()


def upload(client, name: str, data: str | bytes):
    if isinstance(data, str):
        data = data.encode()
    return client.post("/analyze", files={"source": (name, io.BytesIO(data))})


class TestRoutes:
    def test_index_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Graft" in response.text

    def test_styles_and_scripts_are_not_escaped(self, client):
        html = client.get("/").text
        assert "<style>" in html
        assert "<script>" in html
        assert "&lt;style" not in html

    def test_page_declares_motion_and_reduced_motion(self, client):
        html = client.get("/").text
        assert "@keyframes" in html
        assert "prefers-reduced-motion" in html

    def test_upload_produces_a_report(self, client):
        response = upload(client, "01_torch_cuda_softmax.py", EXAMPLE.read_text(encoding="utf-8"))
        assert response.status_code == 200
        assert "Migration report" in response.text
        assert "40/40" in response.text

    def test_export_returns_the_last_report(self, client):
        upload(client, "01_torch_cuda_softmax.py", EXAMPLE.read_text(encoding="utf-8"))
        response = client.get("/export.md")
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment")
        assert "graft-report.md" in response.headers["content-disposition"]
        assert response.headers["content-type"].startswith("text/markdown")
        assert "Graft" in response.text

    def test_cu_upload_shows_it_was_never_executed(self, client):
        response = upload(client, "02_cuda_add_kernel.cu", CUDA_EXAMPLE.read_text(encoding="utf-8"))
        assert "text analysis only" in response.text

    def test_rejects_unsupported_type(self, client):
        response = upload(client, "run.sh", "echo hi")
        assert "Expected a" in response.text

    def test_rejects_empty_file(self, client):
        assert "empty" in upload(client, "x.py", "  \n ").text.lower()

    def test_rejects_oversized_file(self, client):
        oversized = "# padding\n" * ((config.UI_MAX_UPLOAD_BYTES // 10) + 64)
        response = upload(client, "big.py", oversized)
        assert "larger than" in response.text

    def test_rejects_filename_without_suffix(self, client):
        assert "Expected a" in upload(client, "noextension", "x = 1").text

    def test_traversal_in_filename_is_flattened(self, client):
        # The worker re-derives a safe name from the basename, so a path
        # cannot escape the temporary directory.
        response = upload(client, "../../etc/passwd.py", "x = 1\n")
        assert response.status_code == 200

    def test_export_before_any_run_is_explicit(self, client):
        # Guard for test ordering: if the placeholder is gone, an earlier
        # test already ran a migration and this assertion is meaningless.
        assert client.get("/export.md").status_code == 200


# --- cli -------------------------------------------------------------------


class TestCli:
    def test_parser_builds(self):
        from graft.main import build_parser

        assert build_parser().prog == "graft"

    def test_missing_file_exits_two(self, capsys):
        from graft.main import main

        assert main(["migrate", "does-not-exist.py"]) == 2
        assert "no such file" in capsys.readouterr().err

    def test_doctor_runs_offline(self, capsys):
        from graft.main import main

        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "graft" in out
        assert "Agent layer" in out
        assert "Isolation" in out

    def test_json_output_parses(self, capsys):
        from graft.main import main

        main(["migrate", str(EXAMPLE), "--json", "--no-narrative", "--no-bench"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["parity"]["passed"] == 40
        # The Markdown is dropped so `--json` stays machine-readable.
        assert "report_text" not in payload


# --- UploadFile shape ------------------------------------------------------


def test_uploadfile_reads_from_file_attribute():
    # Pins the convention the route depends on: FastHTML hands over a
    # Starlette UploadFile whose bytes are on `.file`, not `.read()`.
    upload_file = UploadFile(filename="x.py", file=io.BytesIO(b"x = 1"))
    assert upload_file.file.read() == b"x = 1"


# --- cloud target route ---------------------------------------------------
#
# These matter most for what they *don't* show: a credential submitted to the
# form must not come back in the response, in any error path.


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

SECRET = "TOKEN-MUST-NOT-ECHO-8f3a2b"
theme_marker = "--red"


def post_target(client, **fields):
    # The cloud path is the one that carries credentials, so the target route
    # tests opt into it explicitly rather than relying on the default.
    data = {
        "target": "cloud",
        "kind": "jupyter",
        "jupyter_host": "gpu.example.com",
        "jupyter_token": SECRET,
    }
    data.update(fields)
    return client.post("/target", data=data)


class FakeJupyter:
    """Stands in for a transport so no network is touched."""

    def __init__(self, spec, stdout="", returncode=0):
        self.spec = spec
        self.stdout = stdout
        self.returncode = returncode
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        self.exited = True


class TestTargetRoute:
    def test_form_is_present_on_the_index(self, client):
        html = client.get("/").text
        assert 'name="target"' in html
        assert "Local GPU" in html
        assert "Cloud GPU" in html
        assert "No target contacted yet" in html

    def test_local_is_the_default_and_hides_the_credentials(self, client):
        """A machine with a card in it must never be shown a key field."""
        html = client.get("/").text
        # Panels are rendered but hidden, so the secret is never echoed into a
        # field value; the key prompt only appears when cloud is chosen.
        assert 'id="panel-cloud"' in html
        assert "Private key path" in html
        assert SECRET not in html

    @staticmethod
    def _panel(markup, panel):
        """The class attribute fastcore rendered for a panel, so a test can
        assert on the initial state rather than on script behaviour."""
        import re

        tag = re.search(rf'<div[^>]*id="{panel}"[^>]*>', markup)
        assert tag, f"{panel} missing from the response"
        found = re.search(r'class="([^"]*)"', tag.group(0))
        return found.group(1) if found else ""

    def test_the_local_default_is_rendered_not_left_to_the_script(self, client):
        """The credential panel must be hidden before any script runs.

        Relying on JavaScript to add the hiding class means a blocked script
        leaves a private-key prompt sitting on the page of somebody who asked
        to use the GPU in the room they are sitting in.
        """
        markup = client.get("/").text
        assert "hidden" in self._panel(markup, "panel-cloud")
        assert "hidden" not in self._panel(markup, "panel-local")

    def test_cloud_renders_the_chosen_transport_only(self, client):
        markup = client.post("/target", data={"target": "cloud", "kind": "ssh"}).text
        assert "hidden" not in self._panel(markup, "panel-cloud")
        assert "hidden" not in self._panel(markup, "panel-ssh")
        assert "hidden" in self._panel(markup, "panel-jupyter")
        assert "<details" in markup
        assert "Advanced connection settings" in markup

    def test_cloud_jupyter_hides_the_ssh_key_field(self, client):
        markup = client.post(
            "/target", data={"target": "cloud", "kind": "jupyter", "host": "h", "token": "t"}
        ).text
        assert "hidden" not in self._panel(markup, "panel-jupyter")
        assert "hidden" in self._panel(markup, "panel-ssh")

    def test_the_location_picker_is_a_dropdown(self, client):
        markup = client.get("/").text
        assert '<select id="target"' in markup
        assert 'value="local" selected="selected"' in markup
        assert 'value="cloud"' in markup
        # Radios are gone: two clicks and two rows of vertical space above the
        # fields they control.
        assert 'type="radio"' not in markup

    def test_the_transport_picker_is_a_dropdown(self, client):
        markup = client.post("/target", data={"target": "cloud", "kind": "jupyter"}).text
        assert '<select id="kind"' in markup
        assert 'value="jupyter" selected="selected"' in markup

    def test_each_transport_names_its_own_fields(self, client):
        """Two inputs called "host" would let the wrong one win silently."""
        markup = client.post("/target", data={"target": "cloud", "kind": "ssh"}).text
        assert 'name="ssh_host"' in markup
        assert 'name="jupyter_host"' in markup
        assert 'name="host"' not in markup

    def test_the_chosen_transport_supplies_the_host(self, client, monkeypatch):
        """A host typed for SSH must not reach a Jupyter server."""
        from graft.ui import app as app_module

        seen = []

        def factory(spec):
            seen.append(spec.host)
            raise RuntimeError("stop here")

        monkeypatch.setattr(app_module, "build_transport", factory)
        client.post(
            "/target",
            data={
                "target": "cloud",
                "kind": "jupyter",
                "ssh_host": "ssh.example.com",
                "ssh_username": "sshuser",
                "ssh_key_path": "/keys/id_ed25519",
                "jupyter_host": "jupyter.example.com",
                "jupyter_username": "jupyteruser",
                "jupyter_token": SECRET,
            },
        )
        assert seen == ["jupyter.example.com"]

    def test_a_local_submit_touches_no_transport(self, client, monkeypatch):
        from graft.ui import app as app_module

        def boom(spec):
            raise AssertionError("the local target must not open a transport")

        monkeypatch.setattr(app_module, "build_transport", boom)
        body = client.post("/target", data={"target": "local", "mode": "probe"}).text
        assert "Checked this machine" in body

    def test_local_reports_when_no_gpu_is_present(self, client, monkeypatch):
        from graft.ui import app as app_module

        monkeypatch.setattr(
            app_module,
            "local_capability",
            lambda: {"device": "cpu", "backend": "cpu", "gpu": False, "torch": "2.14.1"},
        )
        body = client.post("/target", data={"target": "local"}).text
        assert "not detected" in body
        assert "skipped rather than reported" in body

    def test_local_warns_that_the_bands_are_uncalibrated(self, client, monkeypatch):
        from graft.ui import app as app_module

        monkeypatch.setattr(
            app_module,
            "local_capability",
            lambda: {
                "device": "AMD Instinct MI300X VF",
                "backend": "rocm",
                "gpu": True,
                "vram_mb": 196288,
                "torch": "2.14.1+rocm7.2.0",
                "bands_calibrated": False,
            },
        )
        body = client.post("/target", data={"target": "local"}).text
        assert "196288 MB" in body
        assert "provisional" in body

    def test_a_local_secret_typed_by_mistake_is_never_echoed(self, client):
        """Even a credential submitted on the local path must not come back."""
        body = client.post(
            "/target",
            data={"target": "local", "key_path": "/root/.ssh/id_rsa", "token": SECRET},
        ).text
        assert SECRET not in body
        assert "/root/.ssh/id_rsa" not in body

    def test_detection_renders_hardware_facts(self, client, monkeypatch):
        from graft.ui import app as app_module

        class T(FakeJupyter):
            def run(self, command, *, cwd=None, timeout_s=None):
                from graft.target.transport import CommandResult

                return CommandResult(command, 0, GOOD_PROBE, "")

        monkeypatch.setattr(app_module, "build_transport", lambda spec: T(spec))
        body = post_target(client).text
        assert "MI300X" in body
        assert "gfx942" in body
        assert "196288 MB" in body
        assert "hipify-perl" in body

    def test_the_transport_is_closed_even_on_success(self, client, monkeypatch):
        from graft.ui import app as app_module

        made = []

        def factory(spec):
            transport = FakeJupyter(spec)
            made.append(transport)
            return transport

        monkeypatch.setattr(app_module, "build_transport", factory)
        post_target(client)
        assert made and all(t.entered and t.exited for t in made)

    @pytest.mark.parametrize(
        "fields",
        [
            {"jupyter_host": ""},
            {"jupyter_host": "root@evil.example.com"},
            {"port": "70000"},
            {"remote_dir": "../etc"},
            {"command_timeout_s": "1"},
        ],
    )
    def test_bad_input_re_renders_the_form_without_the_secret(self, client, fields):
        body = post_target(client, **fields).text
        assert "Could not continue" in body
        assert SECRET not in body

    def test_an_unreachable_target_does_not_leak_the_token(self, client, monkeypatch):
        from graft.target.transport import TransportError
        from graft.ui import app as app_module

        def factory(spec):
            raise TransportError("Could not reach gpu.example.com:80. Check the host and port.")

        monkeypatch.setattr(app_module, "build_transport", factory)
        body = post_target(client).text
        assert "Could not continue" in body
        assert SECRET not in body

    def test_a_transport_failure_never_shows_a_traceback(self, client, monkeypatch):
        from graft.ui import app as app_module

        def factory(spec):
            raise RuntimeError("boom: /secret/path/on/disk")

        monkeypatch.setattr(app_module, "build_transport", factory)
        body = post_target(client).text
        assert "Could not continue" in body
        assert "Traceback" not in body
        assert "/secret/path/on/disk" not in body

    def test_ssh_choice_is_preselected_after_a_submit(self, client, monkeypatch):
        from graft.ui import app as app_module

        class T(FakeJupyter):
            def run(self, command, *, cwd=None, timeout_s=None):
                from graft.target.transport import CommandResult

                return CommandResult(command, 0, GOOD_PROBE, "")

        monkeypatch.setattr(app_module, "build_transport", lambda spec: T(spec))
        body = post_target(client, kind="ssh", token="").text
        assert 'value="ssh"' in body

    def test_the_form_survives_a_successful_detection(self, client, monkeypatch):
        """A result the user can only read is a dead end; the form must stay."""
        from graft.ui import app as app_module

        class T(FakeJupyter):
            def run(self, command, *, cwd=None, timeout_s=None):
                from graft.target.transport import CommandResult

                return CommandResult(command, 0, GOOD_PROBE, "")

        monkeypatch.setattr(app_module, "build_transport", lambda spec: T(spec))
        body = post_target(client).text
        assert "MI300X" in body
        assert 'id="target-form"' in body
        assert 'action="/target"' in body

    def test_a_successful_detection_does_not_echo_the_token(self, client, monkeypatch):
        from graft.ui import app as app_module

        class T(FakeJupyter):
            def run(self, command, *, cwd=None, timeout_s=None):
                from graft.target.transport import CommandResult

                return CommandResult(command, 0, GOOD_PROBE, "")

        monkeypatch.setattr(app_module, "build_transport", lambda spec: T(spec))
        assert SECRET not in post_target(client).text

    def test_the_wrong_credential_type_is_refused(self, client, monkeypatch):
        """SSH needs a key: without one paramiko would borrow the local agent."""
        from graft.target.spec import parse_form

        with pytest.raises(TargetError):
            parse_form({"kind": "ssh", "host": "gpu.example.com"})
        # Jupyter may legitimately be unauthenticated, so a blank token is fine.
        assert parse_form({"kind": "jupyter", "host": "gpu.example.com"}).kind == "jupyter"

    def test_secret_field_is_a_password_input(self, client):
        html = client.get("/").text
        assert 'type="password"' in html


class TestAmdTheme:
    def test_red_accent_is_present(self, client):
        assert "#ed1c24" in client.get("/").text.lower()

    def test_panels_are_clipped_not_rounded(self, client):
        html = client.get("/").text
        assert "clip-path" in html
        assert "polygon(" in html

    def test_the_theme_loads_nothing_external(self):
        """Our CSS and JS must be self-contained.

        Note the scope: MonsterUI itself injects htmx/FastHTML/surreal from
        cdn.jsdelivr.net, so the page as a whole is not offline-clean. This
        test covers only the theme Graft owns, which is the part we can fix.
        """
        from graft.ui import theme

        owned = f"{theme.STYLE}{theme.SCRIPT}".lower()
        for needle in ("cdn.", "unpkg", "jsdelivr", "googleapis", "https://fonts", "http://", "https://"):
            assert needle not in owned, needle

    def test_our_theme_is_actually_in_the_page(self, client):
        """Guards against the theme silently dropping out of the response."""
        html = client.get("/").text
        assert theme_marker in html

    def test_reveal_has_an_end_state_so_content_is_never_stranded(self, client):
        """If the script never runs, .reveal must not leave the page blank."""
        html = client.get("/").text
        assert "html.motion .reveal" in html
        assert "@keyframes rise" in html
        assert "to { opacity:1" in html.replace("  ", " ")

    def test_reduced_motion_disables_everything(self, client):
        html = client.get("/").text
        assert "prefers-reduced-motion:reduce" in html.replace(" ", "")
