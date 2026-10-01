# Graft

[![CI](https://github.com/Arno-Jaka/graft/actions/workflows/ci.yml/badge.svg)](https://github.com/Arno-Jaka/graft/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

CUDA to ROCm migration with correctness verification.

## Pitch

> Porting CUDA to AMD used to take a week. Graft takes 60 seconds and
> tells you if it actually works.

Deterministic conversion (HIPIFY + AST rewriting), a numerical parity
harness proving the port computes the right answer, and a benchmark
verdict against a reference implementation on the same device.

## Scope

Single file in. No repository ingestion.

**Non-goals**, stated up front:
- No PTX translation
- No CUTLASS
- No performance parity claim against NVIDIA hardware (none is available)
- Correctness is proven against a naive PyTorch reference on the same device,
  not against CUDA output

## Install

    uv sync
    uv run pytest tests -q

Containerization is not set up. Run against the interpreter directly.

## Architecture

    input -> detector -> converter (hipify | ast) -> runner
          -> parity (reference vs port) -> bench -> report -> UI

The `agent/` layer handles the long tail deterministic conversion misses
and is removable: delete it and the product still works.

Converted code is executed in a separate process under a wall-clock
timeout. That is process containment, not a security sandbox: there is no
seccomp, no namespace, and the child inherits the same user and filesystem
access. It stops accidental damage, not adversarial code.

## Status

Deterministic stages are done and tested on CPU:

| Stage | Module | State |
| --- | --- | --- |
| Detection | `core/detector.py` | done |
| CUDA C++ to HIP | `core/hipify_backend.py` | done, table fallback verified |
| PyTorch AST rewriting | `core/torch_rewriter.py` | done |
| Parity harness | `verify/parity.py` | 40/40 cases pass on CPU |
| Benchmarking | `bench/bench.py` | done, noise-calibrated verdicts |
| Sandboxed execution | `core/runner.py` | done, process-isolated with timeout |
| Report | `report/report.py` | done |

Not yet done: `core/coverage.py`, the `agent/` layer, and `ui/app.py`.

ROCm has not been executed. Everything above was verified on a CPU-only
PyTorch build, so the AMD-specific claims are unproven until the code runs
on real hardware.