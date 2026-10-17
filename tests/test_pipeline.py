"""Pipeline tests.

`pipeline.run()` is the only function that touches every stage, so it is
the only place where an ordering mistake or a missing handoff can hide.
These tests assert the sequence and the degraded paths, not the individual
stages, which the per-stage test files already cover.

No test here contacts a model or a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graft import config, pipeline
from graft.report import report

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"
CUDA_EXAMPLE = config.EXAMPLES_DIR / "02_cuda_add_kernel.cu"


@pytest.fixture(scope="module")
def result():
    return pipeline.run(EXAMPLE, with_narrative=False)


# --- the sequence ---------------------------------------------------------


def test_every_deterministic_stage_is_present(result):
    assert result.inputs.migration is not None
    assert result.inputs.execution is not None
    assert result.inputs.parity is not None
    assert result.inputs.telemetry_snapshot is not None


def test_report_carries_the_verdict(result):
    assert "## Conversion" in result.report_text
    assert "## Execution" in result.report_text
    assert "## Parity" in result.report_text
    assert "## Readiness" in result.report_text


def test_parity_passes_on_the_reference_example(result):
    assert result.all_passed is True
    assert result.inputs.parity["passed"] == result.inputs.parity["total"]


def test_execution_runs_the_ported_source_not_the_original(result):
    """The child must receive ported text.

    Running the original would make execution look clean while telling us
    nothing about the port.
    """
    assert result.inputs.execution.ok
    assert result.inputs.execution.case_count > 0


def test_readiness_is_derived_not_asserted(result):
    """CPU-only means the port is clean but not production ready.

    Conversion and parity both pass, yet readiness is still downgraded,
    because a CPU run cannot demonstrate ROCm behaviour. That distinction
    is the whole point of the report.
    """
    assert result.verdict is not None
    assert "PRODUCTION READY" in result.report_text  # conversion verdict
    assert "**NEEDS REVIEW**" in result.report_text  # overall readiness
    assert "No AMD GPU was detected" in result.report_text


# --- degradation ----------------------------------------------------------


def test_narrative_is_absent_when_no_endpoint(result):
    assert result.inputs.narratives == {}
    assert "No narrative generated" in result.report_text


def test_missing_benchmark_is_reported_not_faked(result):
    if not result.inputs.benchmarks:
        assert result.skipped
        assert any("benchmark" in note for note in result.skipped)
        assert "not measured" in result.report_text


def test_skip_notes_reach_the_reader(result):
    for note in result.skipped:
        assert note in result.report_text


def test_cuda_source_converts_and_reports(result):
    """A .cu file cannot be executed or parity-checked.

    It must still produce an honest report saying so, rather than raising
    or implying it was verified.
    """
    cuda = pipeline.run(CUDA_EXAMPLE, with_narrative=False)
    assert cuda.inputs.migration is not None
    assert cuda.inputs.migration.backend in {"table", "hipify-perl"}
    assert "## Conversion" in cuda.report_text
    assert cuda.inputs.parity is None
    assert "not measured" in cuda.report_text


def test_agent_failure_cannot_break_the_report(monkeypatch, result):
    """Even a narrative that raises must leave every measured value intact."""
    inputs = report.ReportInputs(
        migration=result.inputs.migration, parity=result.inputs.parity
    )
    texts = pipeline._narrative(inputs)
    assert set(texts) == {
        "narration",
        "parity_explainer",
        "benchmark_explainer",
        "remediation",
    }
    assert all(isinstance(value, str) for value in texts.values())
    assert "## Parity" in report.render(inputs)


def test_narrative_helper_survives_a_missing_agent_layer(monkeypatch):
    """Simulates deleting agent/ or breaking its import."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("graft.agent"):
            raise ImportError("no agent here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    texts = pipeline._narrative(report.ReportInputs())
    assert all(value.startswith("not available") for value in texts.values())


# --- module loading -------------------------------------------------------


def test_load_module_does_not_leak_sys_path(tmp_path: Path):
    import sys

    before = list(sys.path)
    module = pipeline.load_module(EXAMPLE)
    assert module.OPERATION == "softmax"
    assert sys.path == before


def test_load_module_reports_a_syntax_error(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("def solve(x)\n  pass\n")
    with pytest.raises(SyntaxError):
        pipeline.load_module(bad)


# --- example contract -----------------------------------------------------


def test_example_exports_the_benchmark_contract():
    """The example must keep exporting everything the benchmark harness needs.

    Worth pinning because the failure is silent and looks like a feature:
    `_benchmark` looks up `kernel`, `reference`, and `arg` with getattr and
    skips the whole stage when any is missing, so deleting a module-level
    `arg` turns "no GPU here" into "this module cannot be benchmarked" with
    no error anywhere.
    """
    module = pipeline.load_module(EXAMPLE)
    for name in ("OPERATION", "solve", "kernel", "reference", "arg"):
        assert hasattr(module, name), f"example no longer exports {name}"
    assert module.arg.shape[0] == 64


def test_missing_arg_is_reported_as_a_skip(tmp_path: Path):
    """A module without `arg` is skipped with a reason naming it."""
    partial = tmp_path / "partial.py"
    partial.write_text(
        "import torch\n"
        "OPERATION = 'softmax'\n"
        "def solve(x): return x\n"
        "def kernel(x): return x\n"
        "def reference(x): return x\n"
    )
    result = pipeline.run(
        partial, run_parity=False, run_bench=True, with_narrative=False
    )
    assert any("does not export arg" in note for note in result.skipped)
    assert result.inputs.benchmarks == []


# --- reviewing a run measured elsewhere -----------------------------------


def _summary_payload(**overrides):
    payload = {
        "conversion": {
            "measured": True,
            "verdict": "production_ready",
            "verdict_line": "converted, production ready",
            "verdict_basis": "40/40 parity cases matched",
            "route": "pytorch",
            "findings": [],
        },
        "execution": {
            "measured": True,
            "ok": True,
            "case_count": 40,
            "timed_out": False,
            "exit_code": 0,
            "error_type": None,
            "error_message": None,
        },
        "parity": {
            "measured": True,
            "passed": 40,
            "total": 40,
            "all_passed": True,
            "worst_case": "softmax",
            "max_abs_error": 1e-7,
        },
        "benchmarks": [{"label": "softmax", "ratio": 1.1, "verdict": "ok"}],
        "environment": {"gpu_name": "MI300X", "backend": "rocm"},
    }
    payload.update(overrides)
    return payload


def test_evidence_from_summary_matches_the_in_process_projection(result):
    """A remote run's summary must rebuild the same evidence a local run sends.

    The model may only cite measured numbers; rebuilding the projection from a
    transported summary keeps that guarantee across the process boundary. The
    conversion and execution evidence must be identical, and parity must carry
    the same measured counts.
    """
    from graft import summary as summary_module

    transported = summary_module.summarize(result)
    from_summary = pipeline._review_evidence_from_summary(transported)
    in_process = pipeline._review_evidence(result.inputs)
    assert from_summary["conversion"] == in_process["conversion"]
    assert from_summary["execution"] == in_process["execution"]
    assert from_summary["benchmarks"] == in_process["benchmarks"]
    assert from_summary["parity"]["passed"] == in_process["parity"]["passed"]
    assert from_summary["parity"]["total"] == in_process["parity"]["total"]


def test_evidence_from_summary_leaves_unmeasured_sections_absent():
    evidence = pipeline._review_evidence_from_summary(
        {
            "conversion": {"measured": False},
            "execution": {"measured": False},
            "parity": {"measured": False},
            "benchmarks": [],
            "environment": {},
        }
    )
    assert evidence["conversion"]["verdict"] == report.NOT_MEASURED
    assert evidence["execution"] is None
    assert evidence["parity"] is None


def test_review_with_ai_sends_the_summary_evidence_and_source(monkeypatch):
    review_module = pytest.importorskip("graft.agent.review")
    captured: dict = {}

    def fake_review(original, converted, evidence, *, base_url, model_id, api_key):
        captured.update(
            original=original,
            converted=converted,
            evidence=evidence,
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
        )
        return {"available": True, "recommendation": "engineer_review_required"}

    monkeypatch.setattr(review_module, "review_conversion", fake_review)

    review = pipeline.review_with_ai(
        "print('original')",
        "print('converted')",
        _summary_payload(),
        {"base_url": "https://example.test/v1", "model": "gemma", "api_key": "k"},
    )
    assert review["available"] is True
    assert captured["converted"] == "print('converted')"
    assert captured["evidence"]["conversion"]["route"] == "pytorch"
    assert captured["base_url"] == "https://example.test/v1"


def test_review_with_ai_degrades_when_the_agent_layer_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("graft.agent"):
            raise ImportError("no agent here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    review = pipeline.review_with_ai("a", "b", _summary_payload(), {})
    assert review["available"] is False
    assert "reason" in review
