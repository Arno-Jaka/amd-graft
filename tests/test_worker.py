"""The disposable worker child (`python -m graft.worker`).

The parent parses the child's stdout directly, so the contract tested here is
strict: stdout is exactly one JSON document, and anything the analyzed file
prints is diverted to stderr and clipped.
"""

from __future__ import annotations

import json
import subprocess
import sys

from graft import config, worker

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"


class TestWorker:
    def _run(self, payload: dict, timeout: int = 600):
        return subprocess.run(
            [sys.executable, "-m", "graft.worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_malformed_request_reports_failure(self):
        completed = subprocess.run(
            [sys.executable, "-m", "graft.worker"],
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
        completed = self._run(
            {"source": "import nonexistent_zzz\n", "filename": "x.py"}
        )
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
