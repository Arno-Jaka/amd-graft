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

One line, no clone:

    curl -fsSL https://raw.githubusercontent.com/Arno-Jaka/graft/main/install.sh | sh

The script installs `uv` if it is missing, puts a checkout in `~/.graft`,
builds its virtualenv, and links `graft` onto your PATH. Re-run it to update,
or `sh install.sh --uninstall` to remove everything. To work from a checkout
instead:

    uv sync
    uv run pytest tests -q

A `Dockerfile` and `compose.yaml` are included; `docker build -t graft .`
produces an image whose entrypoint is the `graft` CLI.

## Use

    graft serve                                # launch the terminal UI
    graft doctor                               # what this machine can do
    graft migrate examples/01_torch_cuda_softmax.py

`serve` and `tui` are the same command. From a development checkout, prefix
each with `uv run`.

`migrate` writes Markdown to stdout, or to a file with `-o`, and prints the
same numbers the TUI shows. `--json` emits the UI's summary instead, so the
two front ends cannot disagree. `--no-parity`, `--no-bench`, and
`--no-narrative` skip stages; each skip is recorded in the report rather
than silently dropped.

`doctor` exists so that "it didn't work" has a first question with a definite
answer. On a machine without an AMD GPU it says so up front and explains
which claims are therefore unavailable.

## Terminal UI

A Textual app, `graft serve` (alias `graft tui`), one screen, no build step. Choose a `.py` or
`.cu` file, or probe a GPU target, and the panes show what the pipeline
actually measured: the conversion verdict together with its evidence,
findings grouped by severity, execution, parity, benchmarks, and the
optional narrative.

The GPU panel and its live readings are a fixed dock at the upper right,
present on every tab rather than parked inside the workspace. It steps down
to a narrower gauge on smaller terminals and stands aside entirely below 64
columns, where it would cost more than it shows. Source previews and the
original/converted comparison render with a line-number gutter, like an IDE.

Highlighting text and right-clicking it opens a copy popup that puts the
selection on the clipboard. A right-click on its own copies nothing: the
popup has to be confirmed, because a terminal write to the clipboard can
overwrite a password manager's buffer.

The presentation rules are the same ones the report follows. A verdict never
appears without the basis behind it, so a `.cu` file that was only read and
never compiled reads as *"PRODUCTION READY (text analysis only)"* rather than
as a completed verification. Anything unmeasured says `not measured` instead
of rendering a zero or an empty panel.

Under the GPU panel, live readings stream from whichever machine is selected:
VRAM, GPU utilization, temperature, power, and disk, drawn as bars. Remote
readings arrive over the connection already open — an SSH channel or the
Jupyter websocket — rather than a reconnect per refresh. A local AMD machine
gets the same gauges: utilization, temperature, and power are read from the
amdgpu sysfs nodes that the remote script reads, so a local target is not a
second-class citizen. A sensor that cannot be read shows `not measured`,
never a zero that reads as idle.

The moment a target connects — local probe or remote detect — an overlay
checks it for the tools the pipeline needs and names what is missing, each
row showing the exact install command. For the pieces Graft can install, the
overlay's button runs them with live per-tool progress and re-detects the
target afterwards; it installs nothing until pressed, because the installs
are system-wide and can be large. The same overlay is reachable from the
workspace's install button, which appears only when a checked target is
actually missing something. Starting a run without installing first stops
with the missing tool named in red on the results screen, instead of letting
a cryptic remote error through. Any results shown can be copied to the
clipboard with a single button. A connection that fails authentication — or
that a host's fail2ban has banned — opens a short help panel with the unban
command instead of a bare error.

**Security.** The UI accepts code, so it is built as if that mattered:

- The TUI process never calls `pipeline.run()`. A migration runs in a
  disposable child (`worker.py`) which is killed on timeout.
- Anything the analysed file prints is captured, clipped, and forwarded to
  stderr, so it cannot corrupt the protocol stream or exhaust memory.
- Source files are restricted to `.py` and `.cu`; filenames are reduced to a
  basename inside a temporary directory.

The honest limit: the child shares the TUI user's filesystem and network
access. This is process containment, not a security sandbox.

## Remote GPU targets

The UI can run the pipeline against a `local` interpreter, an `ssh` host, or a
`jupyter` server. Jupyter is the fallback for the many cloud GPU providers
that expose port 80 and block inbound SSH; commands run in a terminal session
over the Jupyter websocket, and files move through the contents API.

Jupyter support depends on two packages that nothing else uses, so they are an
opt-in extra rather than a base requirement:

    uv sync --extra jupyter        # or: pip install 'graft[jupyter]'

Without them, choosing the Jupyter transport fails with a message naming the
missing package and pointing at the SSH option; SSH and local runs are
unaffected.

SSH has been validated end to end against an AMD Instinct MI300X (gfx942,
ROCm 6.x): the probe read the card, `hipify-perl` converted the examples,
`hipcc --offload-arch=gfx942` compiled them, and the HIP kernels ran within
tolerance. Jupyter is implemented but has not been validated against a live
server. It is the least-exercised path, so treat a first connection as a test
of the transport as much as of the code.

## Architecture

    input -> detector -> converter (hipify | ast) -> runner
          -> parity (reference vs port) -> bench -> report -> summary -> TUI

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

The agent never decides correctness. It receives measured results and
read-only context (`agent/tools.py`) and narrates them; it cannot execute
user code, and its output is rendered under a heading that says a language
model wrote it. With no endpoint configured the section reads "no narrative
generated" and every measured section is unaffected. Prose the agent could
not obtain is never printed under a heading claiming a model wrote it.

`pipeline.run()` is the single entry point: convert, execute, verify,
benchmark, narrate, render. Every stage is skippable and every skip is
recorded in the report rather than silently dropped.

The model is Google Gemma, reached through AMD's OpenAI-compatible Radeon
"token factory" endpoint (`POST {base}/chat/completions`) by a stdlib-only
client in `agent/gemma.py` — no provider SDK is required, and the `agent/`
layer stays removable. `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` come
from the environment or a `.env` at the project root (see `.env.example`);
all three ship with defaults so the AI report works out of the box. The TUI
lets you override the endpoint, model, and key per run without saving the
key.

When a migration needs manual work — a blocker, a failing parity case, or a
verdict below production ready — the TUI offers to finish the port with
Gemma. That produces a separate AI-converted file with inline `GRAFT-AI`
comments, previewable in a popup and runnable on a remote GPU. The
deterministic verdict is unchanged by it, and the run step re-runs the full
pipeline on the AI file so parity and benchmarks are measured by Graft, not
claimed by the model.

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
| Explanation layer | `agent/` | done, optional, Google Gemma |
| Terminal UI | `tui/` | done, Textual |
| Upload containment | `worker.py` | done, disposable child per migration |
| CLI | `main.py` | done, `migrate` / `tui` / `doctor` |
| Remote targets | `target/` | SSH verified on an MI300X; Jupyter implemented, unvalidated live |
| Dependency scan | `target/dependencies.py` | done, install is opt-in |
| Live target readings | `target/metrics.py` | local via torch; remote over the transport stream |

Not yet done: Jupyter against a live server, and recalibrating the benchmark
noise bands on a real GPU.

The HIP path has been exercised on real hardware: on an AMD Instinct MI300X,
`hipify-perl` converted the examples, `hipcc --offload-arch=gfx942` compiled
them, and the kernels ran within tolerance. The remaining gap is breadth, not
the smoke test — a single architecture was tested, and PyTorch was not
installed on that host, so the benchmark bands are still the CPU-calibrated
ones and are probably too loose.
