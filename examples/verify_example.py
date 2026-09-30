"""Smoke check: the demo example converts and still matches its reference.

Run in CI so a regression in the converter surfaces as a red build rather
than as a broken demo.
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "01_torch_cuda_softmax.py"

SHAPES = [(1, 1), (4, 8), (16, 128), (7, 33), (2, 3, 65), (1, 1024)]


def load_example():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("example_01", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from graft.core.converter import convert

    source = EXAMPLE.read_text()
    result = convert(source, EXAMPLE)

    assert result.converted, f"conversion failed: {result.coverage.verdict}"
    assert result.coverage.verdict.value == "production_ready", (
        f"demo example must stay production-ready, got "
        f"{result.coverage.verdict.value}"
    )

    module = load_example()
    failures = []
    for shape in SHAPES:
        x = torch.randn(*shape)
        got, expected = module.solve(x), module.reference(x)
        err = (got - expected).abs().max().item()
        ok = torch.allclose(got, expected, rtol=1e-5, atol=1e-6)
        print(f"  {str(shape):12} max_abs_err={err:.3e}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append((shape, err))

    if failures:
        print(f"\nFAILED: {failures}")
        return 1

    print(f"\nOK - {len(SHAPES)} shapes, conversion + parity clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())