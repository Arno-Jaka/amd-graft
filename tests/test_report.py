"""Report rendering tests.

The report is the artifact a judge reads instead of trusting the tool, so
the priority is that it never overstates. These tests check absent data
reads as unmeasured, that failures cannot be smoothed over, and that
no-GPU runs are labelled CPU-only rather than implying ROCm results.
"""

from __future__ import annotations

import torch

from graft import config
from graft.bench import bench, telemetry
from graft.core import converter
from graft.core.converter import Verdict
from graft.report import report
from graft.verify import cases, parity

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"


def _full_inputs():
    """A genuinely successful run: converted, verified, benchmarked."""
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
    return report.ReportInputs(
        migration=migration,
        parity=summary,
        parity_results=results,
        benchmarks=[comparison],
        telemetry_snapshot=telemetry.device_snapshot(),
    )


def test_empty_inputs_render_without_crashing():
    text = report.render(report.ReportInputs())
    assert "# Graft migration report" in text


def test_missing_stages_are_marked_not_measured():
    text = report.render(report.ReportInputs())
    assert report.NOT_MEASURED in text
    assert "Conversion was not run." in text
    assert "Parity was not measured." in text
    assert "No benchmarks were recorded." in text


def test_unmeasured_never_renders_as_production_ready():
    text = report.render(report.ReportInputs())
    assert "PRODUCTION READY" not in text
    assert "**NEEDS REVIEW**" in text or "**UNSUPPORTED**" in text


def test_real_run_renders_every_section():
    text = report.render(_full_inputs())
    for heading in (
        "## Environment",
        "## Conversion",
        "## Parity",
        "## Benchmarks",
        "## Readiness",
    ):
        assert heading in text, heading


def test_parity_counts_and_error_appear():
    inputs = _full_inputs()
    text = report.render(inputs)
    assert f"{inputs.parity['passed']}/{inputs.parity['total']} cases" in text
    assert "Max absolute error: 5.960e-08" in text
    assert inputs.parity["worst_case"] in text


def test_changes_are_listed_with_symbol_and_reason():
    """Every applied change must be traceable, not just counted."""
    inputs = _full_inputs()
    text = report.render(inputs)
    changes = inputs.migration.changes
    assert changes, "example should produce at least one change"
    for change in changes:
        assert change["symbol"] in text
        assert change["reason"] in text
    assert "change" not in text.split("## Conversion")[1].split("###")[0]


def test_absence_of_errors_is_not_printed_as_a_zero():
    """A null error must read as unmeasured, never as a perfect 0."""
    inputs = _full_inputs()
    inputs.parity = {
        "passed": 0,
        "total": 0,
        "all_passed": False,
        "worst_case": None,
        "max_abs_error": None,
    }
    text = report.render(inputs)
    assert "Max absolute error: not measured" in text


def test_failing_parity_case_is_listed_individually():
    inputs = _full_inputs()

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


def test_cpu_only_run_does_not_claim_rocm():
    inputs = _full_inputs()
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


def test_manual_port_verdict_is_carried_through():
    inputs = _full_inputs()
    inputs.migration.coverage.verdict = Verdict.MANUAL_PORT
    text = report.render(inputs)
    assert "MANUAL PORT REQUIRED" in text


def test_unmapped_symbols_are_surfaced():
    inputs = _full_inputs()
    inputs.migration.coverage.unmapped = [
        {"symbol": "custom_triton_kernel", "reason": "Triton unsupported"}
    ]
    text = report.render(inputs)
    assert "`custom_triton_kernel`" in text
    assert "Triton unsupported" in text


def test_skipped_notes_become_blockers():
    inputs = _full_inputs()
    inputs.skipped_notes = ["HIPIFY backend not wired up."]
    text = report.render(inputs)
    assert "HIPIFY backend not wired up." in text
    assert "PRODUCTION READY" not in text


def test_report_and_bench_quote_the_same_noise_band():
    inputs = _full_inputs()
    text = report.render(inputs)
    percent = round((config.BENCH_NOISE_BAND - 1) * 100)
    assert f"±{percent:.0f}%" in text
    assert bench.NOISE_BAND == config.BENCH_NOISE_BAND


def test_speedup_comparison_renders():
    inputs = _full_inputs()
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
