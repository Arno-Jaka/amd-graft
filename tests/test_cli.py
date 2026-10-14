"""Command-line entry points."""

from __future__ import annotations

import json

from graft import config

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"


class TestCli:
    def test_parser_builds(self):
        from graft.main import build_parser

        assert build_parser().prog == "graft"

    def test_serve_is_an_alias_for_tui(self):
        from graft.main import COMMANDS, build_parser

        assert COMMANDS["serve"] is COMMANDS["tui"]
        assert build_parser().parse_args(["serve"]).command == "serve"
        assert build_parser().parse_args(["tui"]).command == "tui"

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
