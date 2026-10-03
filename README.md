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

## Use

    uv run graft doctor                        # what this machine can do
    uv run graft migrate examples/01_torch_cuda_softmax.py
    uv run graft serve                         # web UI on 127.0.0.1:8000

`migrate` writes Markdown to stdout, or to a file with `-o`, and prints the
same numbers the UI shows. `--json` emits the UI's summary instead, so the
two front ends cannot disagree. `--no-parity`, `--no-bench`, and
`--no-narrative` skip stages; each skip is recorded in the report rather
than silently dropped.

`doctor` exists so that "it didn't work" has a first question with a definite
answer. On a machine without an AMD GPU it says so up front and explains
which claims are therefore unavailable.

## Web UI

MonsterUI on FastHTML, one screen, no build step. Upload a `.py` or `.cu`
file and the page shows what the pipeline actually measured: the conversion
verdict together with its evidence, findings grouped by severity, execution,
parity, benchmarks, and the optional narrative.

The presentation rules are the same ones the report follows. A verdict never
appears without the basis behind it, so a `.cu` file that was only read and
never compiled reads as *"PRODUCTION READY (text analysis only)"* rather than
as a completed verification. Anything unmeasured says `not measured` instead
of rendering a zero or an empty panel. Motion is CSS-first and everything
degrades under `prefers-reduced-motion`.

The visual language is AMD-adjacent — near-black surfaces, one red accent,
2px radii, their 326deg diagonal gradient. These are general stylistic
tendencies. No AMD logo, wordmark, font file, or other brand asset is
included or imitated.

**Security.** The web endpoint accepts code, so it is built as if that
mattered:

- The server process never calls `pipeline.run()`. Uploads run in a
  disposable child (`ui/worker.py`) which is killed on timeout.
- Anything the analysed file prints is captured, clipped, and forwarded to
  stderr, so it cannot corrupt the protocol stream or exhaust memory.
- Uploads are size-limited and restricted to `.py` and `.cu`; filenames are
  reduced to a basename inside a temporary directory.
- `serve` binds to `127.0.0.1` by default and warns if asked to bind
  elsewhere.

The honest limit: the child shares the serving user's filesystem and network
access. This is process containment, not a security sandbox, and the UI
prints that admission in its own Execution panel rather than only in this
README. A test asserts the isolation notes on the page are the runner's own,
so the two cannot drift.

## Architecture

    input -> detector -> converter (hipify | ast) -> runner
          -> parity (reference vs port) -> bench -> report -> UI

Every finding carries a severity: `blocker` (no mechanical translation
exists, a human must rewrite it), `review` (a translation exists but the
semantics are not provably equivalent on AMD), or `info` (worth knowing,
changes no verdict). The worst severity present caps the verdict.

The `agent/` layer handles the long tail deterministic conversion misses
and is removable: delete it and the product still works. That claim is
tested rather than asserted — `test_product_works_with_agent_directory_deleted`
copies the tree, removes `agent/`, and runs the suite plus an end-to-end
migration in the copy. The copy is forced onto its own `src` path first,
because otherwise the subprocess imports the original checkout through the
editable install and the deletion silently never happens.

The agent never decides correctness. It receives measured results through
six read-only tools and narrates them; it cannot execute user code, and its
output is rendered under a heading that says a language model wrote it.
With no endpoint configured the section reads "no narrative generated" and
every measured section is unaffected. Prose the agent could not obtain is
never printed under a heading claiming a model wrote it.

`pipeline.run()` is the single entry point: convert, execute, verify,
benchmark, narrate, render. Every stage is skippable and every skip is
recorded in the report rather than silently dropped.

Set `LLM_MODEL` and `LLM_BASE_URL` to enable the agent. Values come from
the environment or from a `.env` file at the project root; see
`.env.example`. vLLM on the AMD instance is the intended target, which
makes a self-hosted demo possible without an API key.

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
| Coverage scoring | `core/coverage.py` | done, blocker/review/info severities |
| Report | `report/report.py` | done |
| Pipeline | `pipeline.py` | done, single entry point |
| Explanation layer | `agent/` | done, optional, needs an LLM endpoint |
| Web UI | `ui/app.py` | done, MonsterUI on FastHTML |
| Upload containment | `ui/worker.py` | done, disposable child per upload |
| CLI | `main.py` | done, `migrate` / `serve` / `doctor` |

Not yet done: nothing in the plan. Next is verification on real AMD
hardware, which cannot be done from here.

ROCm has not been executed. Everything above was verified on a CPU-only
PyTorch build, so the AMD-specific claims are unproven until the code runs
on real hardware. In particular, `hipify-perl` has never been invoked and
`bench/` has never compared two runs on a GPU, so its noise bands are
calibrated from CPU measurements and are probably too loose.