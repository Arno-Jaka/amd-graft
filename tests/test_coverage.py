"""Coverage scoring tests.

Two properties are being defended. First, the scanner must not miss a
construct that genuinely cannot be ported. Second, and harder, it must
not cry wolf: a verdict that cries manual_port on a file that translates
cleanly trains people to ignore it.
"""

from __future__ import annotations

from graft.core import converter, coverage
from graft.core.converter import Verdict

PTX_SOURCE = """
#include <cuda_runtime.h>
__global__ void k(float* p) {
    asm volatile("mma.sync.aligned.m16n8k8.f32.f32.f32.f32 {%0,%1}, {%2}, {%3};" : "=f"(d) : "f"(a), "f"(b));
    p[threadIdx.x] = d;
}
"""

SHFL_SOURCE = """
#include <cuda_runtime.h>
__global__ void k(float* out) {
    float v = __shfl_xor_sync(0xffffffff, 1.0f, 0);
    out[threadIdx.x] = v;
}
"""

WARP_SOURCE = """
#include <cuda_runtime.h>
#define WARP_SIZE 32
__global__ void k(int* out) { out[0] = threadIdx.x % WARP_SIZE; }
"""

CUTLASS_SOURCE = '#include "cutlass/cutlass.h"\nint main(){ return 0; }\n'

CLEAN_SOURCE = """
#include <hip/hip_runtime.h>

__global__ void add_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * 2.0f;
}

int main() {
    float* p = nullptr;
    hipMalloc(&p, 1024);
    add_kernel<<<1, 1>>>(p, p, 1);
    hipDeviceSynchronize();
    hipFree(p);
    return 0;
}
"""




def test_inline_ptx_is_a_blocker():
    findings = coverage.scan(PTX_SOURCE)
    assert any(f.severity is coverage.Severity.BLOCKER for f in findings)


def test_ptx_caps_verdict_at_manual_port():
    assert coverage.assess(PTX_SOURCE).verdict == "manual_port"


def test_shuffle_is_review_not_blocker():
    assessment = coverage.assess(SHFL_SOURCE)
    assert assessment.verdict == "needs_review"
    assert assessment.blockers == []
    assert len(assessment.reviews) == 1


def test_warp_size_literal_is_review():
    assessment = coverage.assess(WARP_SOURCE)
    assert "warp_size" in {f.category for f in assessment.reviews}


def test_cutlass_is_a_blocker():
    assert coverage.assess(CUTLASS_SOURCE).verdict == "manual_port"


def test_clean_hip_file_has_no_findings():
    assessment = coverage.assess(CLEAN_SOURCE)
    assert assessment.findings == []
    assert assessment.verdict == "production_ready"
    assert not assessment.manual_work_required


def test_blocker_outranks_review():
    """The worst severity present decides; it must not average out."""
    source = SHFL_SOURCE + PTX_SOURCE
    assert coverage.assess(source).verdict == "manual_port"


def test_verdict_never_improves_on_a_lower_fallback():
    """A backend that already said 'needs review' is not upgraded by a scan."""
    assert coverage.verdict_for([], fallback="needs_review") == "needs_review"
    assert coverage.verdict_for([], fallback="manual_port") == "manual_port"


def test_info_does_not_cap_the_verdict():
    assessment = coverage.assess('cmd = "nvcc -gencode arch=compute_90"\n')
    assert assessment.counts["info"] >= 1
    assert assessment.verdict == "production_ready"




def test_every_finding_has_a_line_number():
    """A finding with no line tells the reader nothing they can act on."""
    for source in (PTX_SOURCE, SHFL_SOURCE, WARP_SOURCE, CUTLASS_SOURCE):
        for finding in coverage.scan(source):
            assert finding.line >= 1, finding


def test_line_numbers_point_at_real_lines():
    findings = coverage.scan(PTX_SOURCE)
    assert findings[0].line == 4


def test_finding_records_the_literal_text_it_matched():
    """A human label is prose; the literal is what you search for."""
    findings = coverage.scan(SHFL_SOURCE)
    shuffle = [f for f in findings if f.category == "shuffle"]
    assert shuffle
    assert shuffle[0].matched_text == "__shfl_xor_sync"
    assert shuffle[0].symbol == "warp shuffle intrinsic"


def test_matched_text_is_serialised_for_the_agent():
    payload = coverage.scan(SHFL_SOURCE)[0].to_dict()
    assert payload["matched_text"]
    assert payload["matched_text"] != payload["symbol"] or payload["symbol"]


def test_every_finding_carries_a_reason_and_a_fix():
    for source in (PTX_SOURCE, SHFL_SOURCE, WARP_SOURCE, CUTLASS_SOURCE):
        for finding in coverage.scan(source):
            assert finding.reason
            assert finding.remediation
            assert finding.category
            assert finding.symbol


def test_remediation_names_the_amd_equivalent():
    """A blocker that says only 'not supported' is not actionable."""
    ptx_finding = coverage.scan(PTX_SOURCE)[0]
    assert (
        "amd" in ptx_finding.remediation.lower()
        or "hip" in ptx_finding.remediation.lower()
    )


def test_duplicate_matches_on_one_line_collapse():
    """A gencode string matches two patterns; one fix, one finding."""
    findings = coverage.scan('cmd = "nvcc -gencode arch=compute_90"\n')
    nvcc = [f for f in findings if f.category == "nvcc_flags"]
    assert len(nvcc) == 1


def test_same_construct_on_different_lines_are_separate_findings():
    source = "int a = __shfl_xor_sync(1, 2, 0);\nint b = __shfl_down_sync(1, 2, 0);\n"
    assert len(coverage.scan(source)) == 2


def test_findings_are_sorted_by_line():
    findings = coverage.scan(PTX_SOURCE + SHFL_SOURCE)
    assert [f.line for f in findings] == sorted(f.line for f in findings)


def test_to_dict_is_renderable_by_the_report():
    for finding in coverage.scan(PTX_SOURCE):
        payload = finding.to_dict()
        assert payload["severity"] in {"blocker", "review", "info"}
        assert payload["note"] == payload["reason"]
        assert payload["line"] >= 1




def test_summary_counts_by_severity():
    summary = coverage.assess(PTX_SOURCE).summary()
    assert "blocker" in summary
    assert "review" not in summary


def test_clean_summary_says_so():
    assert "nothing needs human attention" in coverage.assess(CLEAN_SOURCE).summary()


def test_counts_sum_to_finding_count():
    assessment = coverage.assess(PTX_SOURCE + SHFL_SOURCE)
    assert sum(assessment.counts.values()) == len(assessment.findings)


def test_manual_work_flag_tracks_blockers_only():
    assert coverage.assess(PTX_SOURCE).manual_work_required is True
    assert coverage.assess(SHFL_SOURCE).manual_work_required is False
    assert coverage.assess(CLEAN_SOURCE).manual_work_required is False




def test_converter_routes_verdict_through_coverage():
    result = converter.convert(PTX_SOURCE)
    assert result.coverage.verdict is Verdict.MANUAL_PORT


def test_converter_keeps_clean_file_production_ready():
    assert converter.convert(CLEAN_SOURCE).coverage.verdict is Verdict.PRODUCTION_READY


def test_converter_reports_findings_on_the_ported_source():
    result = converter.convert(SHFL_SOURCE)
    assert result.coverage.unmapped
    assert result.coverage.findings


def test_findings_are_attached_to_conversion_result():
    result = converter.convert(WARP_SOURCE)
    assert result.coverage.verdict is Verdict.NEEDS_REVIEW
    assert any("wavefront" in f for f in result.coverage.findings)


def test_no_ported_source_leaves_verdict_untouched():
    """An unconverted file has no ported text to scan; verdict stays default."""
    result = converter.convert("@triton.jit\ndef k(x): pass\n")
    assert result.coverage.verdict is Verdict.UNSUPPORTED


def test_scanner_does_not_fire_on_hip_naming():
    """hipMalloc must never be reported as CUDA surface."""
    assert coverage.assess(CLEAN_SOURCE).findings == []


def test_backend_notes_survive_coverage_merging():
    """The backend knows intent a regex cannot recover; it must not be lost."""
    result = converter.convert(SHFL_SOURCE)
    assert any("wavefront" in note for note in result.notes)


def test_no_duplicate_findings_on_the_same_line():
    """Two scanners looking at one construct must not double-report it.

    Distinct lines are distinct fixes and are kept; only same-line repeats
    collapse. WARP_SOURCE uses WARP_SIZE twice, on two different lines.
    """
    result = converter.convert(WARP_SOURCE)
    located = [(item["symbol"], item["line"]) for item in result.coverage.unmapped]
    assert len(located) == len(set(located)), located
    assert len({item["symbol"] for item in result.coverage.unmapped}) == 1


def test_repeated_construct_on_distinct_lines_survives():
    result = converter.convert(WARP_SOURCE)
    lines = {
        item["line"]
        for item in result.coverage.unmapped
        if item["category"] == "warp_size"
    }
    assert len(lines) >= 2


def test_route_notes_are_not_lost_for_pytorch_sources():
    result = converter.convert("import torch\nx = 1  # arch=compute_90\n", path=None)
    assert result.coverage.findings is not None
