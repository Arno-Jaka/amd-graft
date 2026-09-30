"""Smoke check: the demo example converts and passes parity.

Run in CI so a converter or harness regression surfaces as a red build
rather than a broken live demo. Delegates to graft.verify.parity so the
comparison logic lives in exactly one place.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "01_torch_cuda_softmax.py"


def load_example():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("example_01", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from graft.core.converter import convert
    from graft.verify import cases, parity

    result = convert(EXAMPLE.read_text(), EXAMPLE)
    assert result.converted, f"conversion failed: {result.coverage.verdict}"
    assert result.coverage.verdict.value == "production_ready", (
        f"demo example must stay production-ready, got "
        f"{result.coverage.verdict.value}"
    )

    results = parity.check(load_example(), cases.generate())
    for r in results:
        if not r.passed:
            print(f"  {r.name:24} FAIL  {r.error}")
            continue
        print(f"  {r.name:24} PASS  max_abs={r.max_abs:.3e}")

    summary = parity.summarize(results)
    if not summary["all_passed"]:
        print(f"\nFAILED: {summary['passed']}/{summary['total']}")
        return 1

    print(
        f"\nOK - {summary['total']} cases, conversion + parity clean "
        f"(worst {summary['worst_case']}, max_abs {summary['max_abs_error']:.3e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())