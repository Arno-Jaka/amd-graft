"""Report rendering tests.

The report is the artifact a judge reads instead of trusting the tool, so
the priority is that it never overstates. These tests check absent data
reads as unmeasured, that failures cannot be smoothed over, and that
no-GPU runs are labelled CPU-only rather than implying ROCm results.
"""

from __future__ import annotations

import copy

import pytest
import torch

from graft import config
from graft.bench import bench, telemetry
from graft.core import converter, runner
from graft.core.converter import Verdict
from graft.report import report
from graft.verify import cases, parity

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"


@pytest.fixture(scope="module")
def full_inputs():
    """One real end-to-end run, built once.

    Each call previously re-ran the parity harness, the benchmark, and a
    subprocess. With ~15 report tests that dominated the suite runtime.
    Tests that mutate the result use deepcopy rather than rebuilding.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    all_cases = cases.generate()
    results = parity.check(module, all_cases)
    summary = parity.summarize(results)

    device = config.resolve_device()
    x = torch.randn(16, 128)
    port = bench.measure("port", module.kernel, x, device, iters=20)
    ref = bench.measure(
        "reference", lambda t: torch.softmax(t, dim=-1), x, device, iters=20
    )
    comparison = bench.compare_port_vs_reference(port, ref)
    comparison["label"] = "port vs fused reference"

    migration = converter.convert(EXAMPLE.read_text(), EXAMPLE)
    execution = runner.run_cases(migration.ported_source, all_cases[:3], timeout=60)
    return report.ReportInputs(
        migration=migration,
        parity=summary,
        parity_results=results,
        benchmarks=[comparison],
        telemetry_snapshot=telemetry.device_snapshot(),
        execution=execution,
    )


def test_empty_inputs_render_without_crashing():
    text = report.render(report.ReportInputs())
    assert "# Graft migration report" in text


def test_missing_stages_are_marked_not_measured():
    text = report.render(report.ReportInputs())
    assert report.NOT_MEASURED in text
    assert "Conversion was not run." in text
    assert "Ported code was not executed." in text
    assert "Parity was not measured." in text
    assert "No benchmarks were recorded." in text


def test_unmeasured_never_renders_as_production_ready():
    text = report.render(report.ReportInputs())
    assert "PRODUCTION READY" not in text
    assert "**NEEDS REVIEW**" in text or "**UNSUPPORTED**" in text


def test_real_run_renders_every_section(full_inputs):
    text = report.render(copy.deepcopy(full_inputs))
    for heading in (
        "## Environment",
        "## Conversion",
        "## Parity",
        "## Benchmarks",
        "## Readiness",
    ):
        assert heading in text, heading


def test_parity_counts_and_error_appear(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    text = report.render(inputs)
    assert f"{inputs.parity['passed']}/{inputs.parity['total']} cases" in text
    assert "Max absolute error: 5.960e-08" in text
    assert inputs.parity["worst_case"] in text


def test_changes_are_listed_with_symbol_and_reason(full_inputs):
    """Every applied change must be traceable, not just counted."""
    inputs = copy.deepcopy(full_inputs)
    text = report.render(inputs)
    changes = inputs.migration.changes
    assert changes, "example should produce at least one change"
    for change in changes:
        assert change["symbol"] in text
        assert change["reason"] in text
    assert "change" not in text.split("## Conversion")[1].split("###")[0]


def test_absence_of_errors_is_not_printed_as_a_zero(full_inputs):
    """A null error must read as unmeasured, never as a perfect 0."""
    inputs = copy.deepcopy(full_inputs)
    inputs.parity = {
        "passed": 0,
        "total": 0,
        "all_passed": False,
        "worst_case": None,
        "max_abs_error": None,
    }
    text = report.render(inputs)
    assert "Max absolute error: not measured" in text


def test_failing_parity_case_is_listed_individually(full_inputs):
    inputs = copy.deepcopy(full_inputs)

    class Result:
        name = "nan_handling/normal"
        passed = False
        error = "tolerance exceeded"

    inputs.parity_results = [Result()]
    inputs.parity = {
        "passed": 0,
        "total": 1,
        "all_passed": False,
        "worst_case": "nan_handling/normal",
        "max_abs_error": 9.9e-3,
    }
    text = report.render(inputs)
    assert "FAIL" in text
    assert "`nan_handling/normal`" in text
    assert "tolerance exceeded" in text
    assert "One or more parity cases exceeded tolerance." in text


def test_cpu_only_run_does_not_claim_rocm(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    inputs.telemetry_snapshot = {
        "backend": "CPU",
        "gpu_available": False,
        "gpu_name": "CPU",
        "total_vram_mb": None,
        "peak_vram_mb": None,
    }
    text = report.render(inputs)
    assert "No AMD GPU was detected" in text
    assert "CPU-only" in text
    assert "PRODUCTION READY" not in text


def test_manual_port_verdict_is_carried_through(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    inputs.migration.coverage.verdict = Verdict.MANUAL_PORT
    text = report.render(inputs)
    assert "MANUAL PORT REQUIRED" in text


def test_unmapped_symbols_are_surfaced(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    inputs.migration.coverage.unmapped = [
        {"symbol": "custom_triton_kernel", "reason": "Triton unsupported"}
    ]
    text = report.render(inputs)
    assert "`custom_triton_kernel`" in text
    assert "Triton unsupported" in text


def test_findings_are_grouped_by_severity(full_inputs):
    """The reader needs to see how much work is left, not a flat list."""
    from graft.core.coverage import Finding, Severity

    inputs = copy.deepcopy(full_inputs)
    inputs.migration.coverage.unmapped = [
        Finding(
            Severity.BLOCKER,
            "ptx",
            "inline PTX",
            4,
            "no HIP equivalent",
            "rewrite with MFMA intrinsics",
        ).to_dict(),
        Finding(
            Severity.REVIEW,
            "shuffle",
            "warp shuffle",
            9,
            "masks differ",
            "verify lane indices",
        ).to_dict(),
        Finding(
            Severity.INFO,
            "nvcc_flags",
            "gencode flag",
            1,
            "flags differ",
            "use --offload-arch",
        ).to_dict(),
    ]
    text = report.render(inputs)
    assert "#### Blockers" in text
    assert "#### Needs review" in text
    assert "#### Notes" in text
    assert "rewrite with MFMA intrinsics" in text


def test_finding_without_severity_is_still_shown(full_inputs):
    """Missing metadata must not silently hide a finding."""
    inputs = copy.deepcopy(full_inputs)
    inputs.migration.coverage.unmapped = [
        {"symbol": "mystery", "line": 3, "reason": "unclassified risk"}
    ]
    text = report.render(inputs)
    assert "`mystery`" in text
    assert "unclassified risk" in text


def test_skipped_notes_become_blockers(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    inputs.skipped_notes = ["HIPIFY backend not wired up."]
    text = report.render(inputs)
    assert "HIPIFY backend not wired up." in text
    assert "PRODUCTION READY" not in text


def test_execution_failure_is_not_hidden_by_good_parity(full_inputs):
    """A run that crashed cannot be reported as verified."""
    inputs = copy.deepcopy(full_inputs)
    inputs.execution = runner.ExecutionResult(
        ok=False, error_type="SyntaxError", error_message="expected ':' (line 1)"
    )
    text = report.render(inputs)
    assert "Execution failed" in text
    assert "expected ':'" in text
    assert "PRODUCTION READY" not in text


def test_partial_case_failures_count_as_blockers(full_inputs):
    inputs = copy.deepcopy(full_inputs)

    class Entry:
        def __init__(self, ok):
            self.ok = ok

    inputs.execution.results = [Entry(True), Entry(False), Entry(False)]
    text = report.render(inputs)
    assert "2 case(s) raised during execution." in text


def test_timeout_is_visible_in_the_report(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    inputs.execution = runner.ExecutionResult(
        ok=False, timed_out=True, timeout_seconds=30
    )
    text = report.render(inputs)
    assert "timed out after 30s" in text
    assert "Timed out: True" in text


def test_isolation_caveat_appears_even_on_a_clean_run(full_inputs):
    """A green report must still not imply a real sandbox."""
    text = report.render(copy.deepcopy(full_inputs))
    assert "not a security sandbox" in text
    assert "## Isolation" in text


def test_report_and_bench_quote_the_same_noise_band(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    text = report.render(inputs)
    percent = round((config.BENCH_NOISE_BAND - 1) * 100)
    assert f"±{percent:.0f}%" in text
    assert bench.NOISE_BAND == config.BENCH_NOISE_BAND


def test_speedup_comparison_renders(full_inputs):
    inputs = copy.deepcopy(full_inputs)
    inputs.benchmarks = [
        {
            "label": "device vs CPU",
            "speedup": 12.5,
            "verdict": "12.50x faster than the CPU reference",
        }
    ]
    text = report.render(inputs)
    assert "12.5x" in text
    assert "device vs CPU" in text
