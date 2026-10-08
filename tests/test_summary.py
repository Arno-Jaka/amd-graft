"""`graft.summary` flattens a PipelineResult for transport across a process.

The summary is what a UI actually renders, so these tests are about honesty:
every section is always present, and an unmeasured value stays unmeasured
rather than becoming a zero or an empty string.
"""

from __future__ import annotations

import json

import pytest

from graft import config, pipeline
from graft import summary as summary_module

EXAMPLE = config.EXAMPLES_DIR / "01_torch_cuda_softmax.py"
CUDA_EXAMPLE = config.EXAMPLES_DIR / "02_cuda_add_kernel.cu"


@pytest.fixture(scope="module")
def real_summary() -> dict:
    """A summary produced by the real pipeline, reused across tests."""
    return summary_module.summarize(
        pipeline.run(EXAMPLE, with_narrative=False),
    )


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
        for section in (
            "environment",
            "conversion",
            "execution",
            "parity",
            "narrative",
        ):
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
            pipeline.run(
                CUDA_EXAMPLE, run_parity=False, run_bench=False, with_narrative=False
            )
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
        assert summary_module._narrative({"narration": "   "}) == {
            "prose": {},
            "reasons": [],
        }

    def test_unknown_narrative_keys_dropped(self):
        narrative = summary_module._narrative({"secret_key": "leaked"})
        assert "secret_key" not in narrative["prose"]
