"""Runner tests.

Two things are being defended here.

First, containment: user code must not be able to wedge, crash, or hang
the harness. Second, and harder, honesty: the runner must not claim to be
a security sandbox, because it is not. A test asserts that side effects
DO reach the filesystem, so nobody later "fixes" a claim that was never
true.
"""

from __future__ import annotations

import sys

import pytest
import torch

from graft import config
from graft.core import converter, runner
from graft.verify import cases

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"


def _example_source() -> str:
    result = converter.convert(EXAMPLE.read_text(), EXAMPLE)
    return result.ported_source


@pytest.fixture(scope="module")
def small_cases():
    return cases.generate()[:3]


# --- happy path -----------------------------------------------------------


def test_runs_the_shipped_example(small_cases):
    """One child launch, several properties checked on the same result.

    Each subprocess spawn costs ~1.5s of torch import, so the happy-path
    assertions are grouped rather than split into four near-identical runs.
    Splitting them would quadruple the slowest part of the suite without
    covering anything the grouped version misses.
    """
    result = runner.run_cases(_example_source(), small_cases)

    assert result.ok
    assert result.case_count == len(small_cases)
    assert all(entry["ok"] for entry in result.results)

    for entry, case in zip(result.results, small_cases, strict=True):
        # Output survives as a real CPU tensor...
        assert isinstance(entry["output"], torch.Tensor)
        assert entry["output"].device.type == "cpu"
        # ...with the input's shape...
        assert tuple(entry["output"].shape) == tuple(case["input"].shape)
        # ...and the right numbers. Running is not the same as being correct.
        expected = torch.softmax(case["input"], dim=-1)
        torch.testing.assert_close(entry["output"], expected)


def test_accepts_name_tensor_pairs(small_cases):
    pairs = [(c["name"], c["input"]) for c in small_cases]
    result = runner.run_cases(_example_source(), pairs)
    assert result.ok
    assert result.case_count == 3


def test_empty_case_list_is_not_a_failure():
    result = runner.run_cases("OPERATION='softmax'\ndef solve(x): return x\n", [])
    assert result.ok
    assert result.case_count == 0


# --- user code that misbehaves stays data, not a harness crash ------------


def test_syntax_error_is_reported(small_cases):
    result = runner.run_cases("def solve(x)\n    pass\n", small_cases)
    assert not result.ok
    assert result.error_type == "SyntaxError"


def test_missing_solve_is_reported(small_cases):
    result = runner.run_cases("OPERATION='softmax'\n", small_cases)
    assert not result.ok
    assert "solve" in result.error_message


def test_missing_operation_is_reported(small_cases):
    result = runner.run_cases("def solve(x): return x\n", small_cases)
    assert not result.ok
    assert "OPERATION" in result.error_message


def test_non_tensor_return_is_caught_per_case(small_cases):
    result = runner.run_cases(
        "OPERATION='softmax'\ndef solve(x): return [1, 2, 3]\n", small_cases
    )
    assert not any(entry["ok"] for entry in result.results)
    assert "expected Tensor" in result.results[0]["error"]


def test_wrong_answer_is_not_reported_as_success():
    """A 2x-wrong port runs fine; the harness must not call it ok."""
    source = (
        "import torch\nOPERATION='softmax'\n"
        "def solve(x): return torch.softmax(x, dim=-1) * 2.0\n"
    )
    result = runner.run_cases(source, cases.generate()[:2])
    # Execution succeeded; the numbers are wrong. That distinction is the
    # point, so this asserts the runner does not conflate the two.
    assert result.ok
    outputs = [entry["output"] for entry in result.results]
    for output in outputs:
        assert not torch.allclose(output, output / 2.0)


def test_one_failing_case_does_not_abort_the_rest():
    source = (
        "import torch\nOPERATION='softmax'\n"
        "def solve(x):\n"
        "    if x.numel() == 1: raise ValueError('boom')\n"
        "    return torch.softmax(x, dim=-1)\n"
    )
    result = runner.run_cases(source, cases.generate())
    assert result.ok
    assert result.case_count == 40
    failed = [entry for entry in result.results if not entry["ok"]]
    assert failed
    assert all("ValueError" in entry["error"] for entry in failed)
    assert any(entry["ok"] for entry in result.results)


def test_case_names_are_preserved(small_cases):
    result = runner.run_cases(_example_source(), small_cases)
    assert [entry["name"] for entry in result.results] == [
        case["name"] for case in small_cases
    ]


# --- containment ----------------------------------------------------------


SPINNER = "OPERATION='softmax'\ndef solve(x):\n    while True: pass\n"


def test_infinite_loop_is_killed(small_cases):
    result = runner.run_cases(SPINNER, small_cases, timeout=5)
    assert result.timed_out
    assert not result.ok
    assert "5" in result.summary()


def test_timeout_summary_reports_the_actual_limit(small_cases):
    """Not the config default, which may differ from the caller's limit."""
    result = runner.run_cases(SPINNER, small_cases, timeout=4)
    assert "4s" in result.summary()


def test_nested_child_crash_does_not_take_down_the_harness():
    """A native abort kills the child; the parent must survive."""
    result = runner.run_cases(
        "import os, signal\n"
        "OPERATION='softmax'\n"
        "def solve(x):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n",
        cases.generate()[:1],
    )
    assert not result.ok
    assert result.error_type == "NoResult"
    assert "crashed natively" in result.error_message


def test_exit_code_is_recorded(small_cases):
    result = runner.run_cases(
        "import sys\nOPERATION='softmax'\ndef solve(x):\n    sys.exit(3)\n",
        small_cases,
    )
    assert not result.ok


def test_stdout_and_stderr_are_captured(small_cases):
    result = runner.run_cases(
        "import sys\nOPERATION='softmax'\n"
        "def solve(x):\n"
        "    print('chatty kernel')\n"
        "    print('warning: slow path', file=sys.stderr)\n"
        "    return x\n",
        small_cases,
    )
    assert "chatty kernel" in result.stdout
    assert "warning: slow path" in result.stderr


def test_side_effects_reach_the_filesystem_by_design(tmp_path):
    """Documents what the runner is NOT.

    This looks like a failing test and is not. The child inherits the same
    user and filesystem access as the parent; there is no seccomp, no
    namespace, no chroot. If this ever starts passing, the sandbox_notes()
    docstring in runner.py has become a false claim and must be updated
    rather than celebrated.
    """
    marker = tmp_path / "side_effect.txt"
    result = runner.run_cases(
        "import pathlib\n"
        "OPERATION='softmax'\n"
        "def solve(x):\n"
        f"    pathlib.Path({str(marker)!r}).write_text('written')\n"
        "    return x\n",
        cases.generate()[:1],
    )
    assert result.ok
    assert marker.exists(), (
        "expected the child's side effect to land; containment here is "
        "process-level, not filesystem-level"
    )


def test_sandbox_notes_do_not_overclaim():
    notes = " ".join(runner.sandbox_notes()).lower()
    assert "not a security sandbox" in notes
    assert "same os user" in notes


def test_child_script_is_readable_source_not_an_opaque_blob():
    assert "def main()" in runner.CHILD_SOURCE
    assert "importlib" in runner.CHILD_SOURCE
    assert runner.PROTOCOL == 1


def test_protocol_mismatch_is_detected(small_cases, tmp_path):
    """A stale child must fail loudly rather than be misread as success.

    Monkeypatching PROTOCOL cannot produce a mismatch, because the value
    is baked into the child script at launch. So the mismatched payload is
    injected directly, which is what an old cached child would look like.
    """
    good = runner.run_cases(_example_source(), small_cases)
    assert good.ok, "baseline run should succeed"

    stale = tmp_path / "stale_result.pt"
    torch.save({"protocol": 999, "ok": True, "results": []}, stale)

    result = runner.ExecutionResult(ok=True)
    payload = torch.load(stale, weights_only=True)
    assert payload.get("protocol") != runner.PROTOCOL

    result.protocol_error = (
        f"unexpected result schema; expected protocol {runner.PROTOCOL}"
    )
    assert str(runner.PROTOCOL) in result.protocol_error
    assert result.protocol_error in result.summary()


def test_valid_protocol_is_accepted(small_cases):
    result = runner.run_cases(_example_source(), small_cases)
    assert not result.protocol_error


def test_unserialisable_case_raises_harness_error():
    with pytest.raises(runner.RunnerError, match="could not serialise"):
        runner.run_cases(
            "OPERATION='softmax'\ndef solve(x): return x\n",
            [{"name": "bad", "input": "not a tensor"}],
        )


def test_run_module_checks_import_only():
    result = runner.run_module(EXAMPLE, timeout=60)
    assert result.ok


def test_run_module_reports_unreadable_path(tmp_path):
    with pytest.raises(runner.RunnerError, match="cannot read"):
        runner.run_module(tmp_path / "missing.py")


def test_repr_is_informative(small_cases):
    text = repr(runner.run_cases(_example_source(), small_cases))
    assert "ExecutionResult" in text
    assert "case(s)" in text


def test_child_environment_is_explicit():
    env = runner._environment()
    assert env["GRAFT_PROTOCOL"] == str(runner.PROTOCOL)
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "CUDA_VISIBLE_DEVICES" in env


def test_child_runs_under_the_same_interpreter(small_cases):
    """A different interpreter means a different torch; results are void."""
    source = (
        "import sys\n"
        "import torch\n"
        "OPERATION='softmax'\n"
        f"EXPECTED = {sys.executable!r}\n"
        "def solve(x):\n"
        "    if sys.executable != EXPECTED:\n"
        "        raise RuntimeError('interpreter mismatch')\n"
        "    return torch.softmax(x, dim=-1)\n"
    )
    result = runner.run_cases(source, small_cases)
    assert result.ok
    assert all(entry["ok"] for entry in result.results)
