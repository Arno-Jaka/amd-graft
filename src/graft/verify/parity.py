"""Stage 2 — The parity harness.

Runs an independent reference and the ported implementation on identical
inputs, then reports per-case pass/fail carrying actual error magnitudes.
Per-case rather than aggregate: a single aggregate hides one broken shape.
"""

from __future__ import annotations

import torch

from graft.core.reference import get_reference
from graft.verify.tolerances import tolerance_for


class CaseResult:
    __slots__ = ("error", "max_abs", "max_rel", "name", "passed")

    def __init__(self, name, passed, max_abs=None, max_rel=None, error=""):
        self.name = name
        self.passed = passed
        self.max_abs = max_abs
        self.max_rel = max_rel
        self.error = error

    def __repr__(self) -> str:
        if not self.passed:
            detail = self.error or "tolerance exceeded"
            return f"CaseResult({self.name}, FAIL, {detail})"
        return f"CaseResult({self.name}, PASS, max_abs={self.max_abs:.3e})"


def compare(expected, actual, dtype):
    """Return (passed, max_abs_err, max_rel_err, note).

    Arity is fixed on every path: callers unpack positionally, so a
    length-varying return here would surface as an opaque ValueError
    instead of a reported failure.
    """
    rtol, atol = tolerance_for(dtype)

    if expected.shape != actual.shape:
        return (
            False,
            None,
            None,
            (
                f"shape mismatch: expected {tuple(expected.shape)}, "
                f"got {tuple(actual.shape)}"
            ),
        )

    comparable = ~torch.isnan(expected)
    diff = (expected - actual).abs()
    finite = diff[comparable]

    if finite.numel() == 0:
        max_abs = 0.0
        max_rel = 0.0
    else:
        max_abs = finite.max().item()
        denom = expected[comparable].abs().clamp_min(atol)
        max_rel = (finite / denom).max().item()

    within = torch.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True)
    return bool(within), max_abs, max_rel, ""


def check(module, cases, operation=None) -> list[CaseResult]:
    op = operation or module.OPERATION
    reference = get_reference(op)

    results = []
    for case in cases:
        try:
            expected = reference(case["input"])
            actual = module.solve(case["input"])
        except Exception as exc:
            results.append(
                CaseResult(case["name"], False, error=f"{type(exc).__name__}: {exc}")
            )
            continue

        passed, max_abs, max_rel, note = compare(expected, actual, case["input"].dtype)
        results.append(CaseResult(case["name"], passed, max_abs, max_rel, note))
    return results


def summarize(results) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    scored = [r for r in results if r.max_abs is not None]
    worst = max(scored, key=lambda r: r.max_abs, default=None)
    return {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "worst_case": worst.name if worst else None,
        "max_abs_error": worst.max_abs if worst else None,
    }
