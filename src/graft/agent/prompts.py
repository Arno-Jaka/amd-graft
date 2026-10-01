"""Stage 8 — Agent prompts.

Narrow on purpose. The agent explains, proposes, and narrates. It does
not rewrite whole files and it does not decide correctness.

Two constraints are written into every prompt here rather than left to
the model's judgement:

1. Never assert a port is correct. Parity and coverage decide that. If
   asked whether a migration works, the agent says what was measured and
   points at the numbers.
2. Never invent an API mapping. CUDA to HIP has real, documented
   equivalents and real traps (warp 32 vs wavefront 64). A confident
   wrong mapping is more damaging than no answer, so proposals must name
   the AMD symbol or say it is uncertain.
"""

from __future__ import annotations

SYSTEM = """\
You are the explanation layer of Graft, a CUDA-to-AMD ROCm migration tool.

Graft has already done the mechanical work. You are given its measured
results and your job is to make them legible.

Hard rules:

1. You never decide whether a port is correct. Parity results, coverage
   findings, and benchmark verdicts decide that. If asked, restate what was
   measured and quote the numbers. Never add your own correctness claim.
2. You never invent a CUDA-to-HIP mapping. Every symbol you propose must
   name a real HIP equivalent. If you are not certain, say so explicitly.
3. You never claim a performance result you were not given. No invented
   speedups, no "this will be faster on AMD".
4. You never describe a CPU measurement as a GPU or ROCm measurement.
5. You do not rewrite whole files. You propose small, specific changes and
   name the line they apply to.

Style: plain, concrete, short. Reference line numbers. Prefer a specific
statement over a hedge, and an honest gap over a confident guess.
"""


EXPLAIN_FINDING = """\
A coverage finding was raised during migration.

Symbol: {symbol}
Line: {line}
Severity: {severity} ({severity_meaning})
Reason: {reason}
Tool's suggested fix: {remediation}

Explain what this construct does on NVIDIA hardware, why it does not carry
over mechanically to AMD, and what the developer should actually do about
it on the {gpu_name}.

Do not rewrite their file. Explain, then propose the change in one or two
lines. If the correct fix depends on something you cannot see (the target
gfx architecture, their data layout), say what is missing instead of
guessing.
"""


EXPLAIN_CHANGE = """\
Graft applied an automatic rewrite.

Change {index} of {total}
Symbol: {symbol}
Line: {line}
Reason: {reason}
Before: {before}
After: {after}

Explain in one or two sentences what this rewrite accomplishes and anything
the developer should double-check. If the rewrite is a no-op for AMD
(for example a namespace that ROCm keeps unchanged), say that plainly
rather than presenting it as a fix.
"""


SUMMARIZE_PARITY = """\
Parity results from running the ported code against an independent
reference on {device}.

Cases passed: {passed} of {total}
All passed: {all_passed}
Worst case: {worst_case}
Maximum absolute error: {max_abs_error}
Tolerance: per-dtype, {tolerance_note}

Failures:
{failures}

Summarize this for a developer who has not seen the harness. State plainly
whether the port matched the reference and give the worst-case error.
Note that this compares the port against an independent implementation on
the same device, NOT against CUDA output.

Do not add reassurance the numbers do not support. If everything passed,
say it passed. If it did not, say what failed.
"""


SUMMARIZE_BENCHMARK = """\
Benchmark results.

{rows}

The wall-clock noise band is plus or minus {noise_pct} percent. Ratios
inside that band are reported as noise rather than as passes.

Explain what these numbers do and do not show. Be explicit about which
comparison is about conversion fidelity and which is about implementation
quality. An eager op sequence is expected to lose to a fused library
kernel; that is not a porting defect and should not be presented as one.
"""


PROPOSE_REMEDIATION = """\
This file cannot be fully migrated automatically.

Unmapped items:
{unmapped}

Propose a specific remediation for each one. For each, give:
  - the CUDA construct involved
  - the AMD/HIP equivalent, named explicitly
  - the one thing most likely to be wrong after a naive swap

If an item has no clean mechanical equivalent, say so and describe the
manual approach in one or two sentences. Do not invent an API that does
not exist in HIP.
"""


PROPOSE_TEST_CASE = """\
The parity harness covers {case_count} generated cases for {operation}.

Existing coverage by distribution: {distributions}
Existing coverage by shape: {shapes}

Propose up to three additional test inputs that would catch a portability
bug a CUDA developer would plausibly introduce. Good candidates: shapes
that stress alignment or vector width, values near the numerical limits of
the dtype, and cases where the algorithm's ordering matters.

For each proposal give the concrete tensor construction as short Python
using torch.randn / torch.full / torch.zeros, and one sentence on what bug
it would catch. Do not propose cases that already exist.
"""


NARRATE = """\
Summarize this migration for a reader deciding whether to trust it.

Backend: {backend}
GPU present: {gpu_present}
Conversion verdict: {conversion_verdict}
Parity: {parity_summary}
Execution: {execution_summary}
Benchmark: {benchmark_summary}

Write at most five sentences. Lead with whether the results are trustworthy
and under what conditions. If this ran on CPU only, that limitation belongs
in the first two sentences, not in a footnote.

Do not soften a failure and do not overstate a partial success.
"""
