"""Views for pointing Graft at a remote AMD instance.

The form here is the answer to a fair objection: verification needs real
hardware, and handing a tool standing credentials for a cloud machine is a bad
trade. So the operator supplies connection details at request time and Graft
forgets them the moment the response is sent.

Two rules govern everything in this module:

* The secret fields are never echoed back into the HTML. A masked placeholder
  is rendered instead, so a failed submit cannot leak a passphrase or token
  into a page view, a browser cache, or a screenshot.
* Only ``TargetSpec.redacted()`` output is ever displayed, never the spec.
"""

from __future__ import annotations

from typing import Any

from fastcore.xml import (
    Button,
    Code,
    Details,
    Div,
    Form,
    Input,
    Label,
    Option,
    Script,
    Select,
    Small,
    Strong,
    Summary,
)

from graft.target.spec import REDACTED
from graft.ui.theme import FAIL, INFO, PASS, WARN

KIND_SSH = "ssh"
KIND_JUPYTER = "jupyter"

LABEL = "text-[.6rem] font-extrabold tracking-[.14em] uppercase text-[#6b6b70]"
HINT = "hint"


def _field(name: str, label: str, kind: str, placeholder: str, **kwargs: Any) -> Any:
    return Div(
        Label(label, for_=name, cls=LABEL),
        Input(id=name, name=name, type=kind, placeholder=placeholder, autocomplete="off", **kwargs),
        cls="field",
    )


TARGET_LOCAL = "local"
TARGET_CLOUD = "cloud"


def _select(
    name: str,
    label: str,
    options: tuple[tuple[str, str], ...],
    selected: str,
    help_text: str = "",
) -> Any:
    """A labelled dropdown of ``(value, text)`` pairs.

    A dropdown rather than radios because the first question has only two
    answers and picking one should not cost two clicks of vertical space above
    the fields it controls.
    """
    chosen = [
        Option(text, value=value, selected="selected" if value == selected else "")
        for value, text in options
    ]
    return Div(
        Label(label, for_=name, cls=LABEL),
        Select(*chosen, id=name, name=name),
        Div(help_text, cls="mt-1 text-[.68rem] leading-relaxed text-[#a3a3a6]"),
        cls="field",
    )


def _field_grid(*fields: Any, cls: str = "") -> Any:
    return Div(*fields, cls=("grid g3 " + cls).strip())


def connection_form(
    *,
    error: str = "",
    kind: str = KIND_SSH,
    target: str = TARGET_LOCAL,
    local: dict | None = None,
) -> Any:
    """The target form.

    Two dropdowns and only the fields that apply. Picking the local GPU hides
    every credential prompt; picking a cloud host reveals the transport, and the
    transport reveals its own three fields. Port, remote directory, timeout and
    the key passphrase sit in a native disclosure, so the common case is three
    boxes rather than eight.

    The panels are all present in the markup and the script shows or hides them.
    Their initial visibility is rendered server-side, so a blocked script leaves
    the right panels showing instead of flashing a private-key prompt at
    somebody who only wanted to use the GPU in the room they are sitting in.

    Field names are prefixed per transport (``ssh_host``, ``jupyter_host``).
    Two inputs sharing the name ``host`` would make ``form.get`` return
    whichever arrived first, so switching transport after typing an SSH host
    would quietly send that host to a Jupyter server.
    """
    is_local = target == TARGET_LOCAL
    is_ssh = kind == KIND_SSH

    target_select = _select(
        "target",
        "GPU location",
        ((TARGET_LOCAL, "Local GPU - this machine"), (TARGET_CLOUD, "Cloud GPU - a remote server")),
        selected=target,
        help_text="A local AMD device needs no credentials at all.",
    )

    kind_select = _select(
        "kind",
        "How to reach it",
        ((KIND_SSH, "SSH"), (KIND_JUPYTER, "Jupyter")),
        selected=kind,
        help_text="SSH takes a private key. Jupyter takes a browser URL and token.",
    )

    ssh_panel = Div(
        _field_grid(
            _field("ssh_host", "Host or IP", "text", "203.0.113.10"),
            _field("ssh_username", "Username", "text", "root"),
            _field("ssh_key_path", "Private key path", "text", "~/.ssh/id_ed25519"),
        ),
        Div(
            Small(
                "The key is read for this request only. Graft never writes it, "
                "logs it, or sends it back to the browser.",
                cls=HINT,
            ),
            cls="mt-3",
        ),
        id="panel-ssh",
        cls="" if is_ssh else "hidden",
    )

    jupyter_panel = Div(
        _field_grid(
            _field("jupyter_host", "Server URL", "text", "https://gpu.example.com/lab"),
            _field("jupyter_username", "Username", "text", "root"),
            _field("jupyter_token", "Token", "password", REDACTED),
        ),
        id="panel-jupyter",
        cls="hidden" if is_ssh else "",
    )

    advanced = Details(
        Summary("Advanced connection settings", cls="eyebrow cursor-pointer"),
        _field_grid(
            _field("port", "Port", "text", "22 for SSH, 443 for https", inputmode="numeric"),
            _field("remote_dir", "Remote directory", "text", "/root/graft"),
            _field("command_timeout_s", "Command timeout (s)", "text", "900", inputmode="numeric"),
            _field("key_passphrase", "Key passphrase", "password", REDACTED),
            cls="mt-4",
        ),
        cls="mt-6 border-t border-white/10 pt-4",
    )

    cloud_panel = Div(
        kind_select,
        ssh_panel,
        jupyter_panel,
        advanced,
        id="panel-cloud",
        cls="" if not is_local else "hidden",
    )

    local_panel = Div(
        local_capability(local or {}),
        id="panel-local",
        cls="" if is_local else "hidden",
    )

    notice = ""
    if error:
        # Build the notice into a plain list before the Div exists. Mutating
        # `Div.children` would fail: fastcore stores children as a tuple.
        notice = Div(
            Div("Could not continue", cls="eyebrow"),
            Div(error, cls="mt-1 text-[.8rem] leading-relaxed text-[#f87171]"),
            cls="notice mb-4",
            style=f"--tone:{FAIL}",
        )

    script = Script(
        """
        (function () {
          function pick(name) {
            var el = document.getElementById(name);
            return el ? el.value : null;
          }
          function show(id, on) {
            var el = document.getElementById(id);
            if (el) el.classList.toggle('hidden', !on);
          }
          function apply() {
            var cloud = pick('target') === 'cloud';
            show('panel-cloud', cloud);
            show('panel-local', !cloud);
            if (!cloud) return;
            var jupyter = pick('kind') === 'jupyter';
            show('panel-jupyter', jupyter);
            show('panel-ssh', !jupyter);
          }
          document.addEventListener('change', apply);
          apply();
        })();
        """,
        cls="hidden",
    )

    if is_local:
        actions = Div(
            Button("Check this machine", type="submit", name="mode", value="probe", cls="btn ghost"),
            cls="mt-6 flex flex-wrap items-center gap-4",
        )
        footer = Small(
            "Uploading a file above runs it on this machine's device. Graft "
            "says so honestly when there is no AMD GPU: conversion and parity "
            "still run, and benchmarks are skipped rather than faked.",
            cls="mt-2 block " + HINT,
        )
    else:
        actions = Div(
            Button("Detect target", type="submit", name="mode", value="probe", cls="btn ghost"),
            Button(
                "Detect and verify on GPU",
                type="submit",
                name="mode",
                value="verify",
                cls="btn",
            ),
            cls="mt-6 flex flex-wrap items-center gap-4",
        )
        footer = Small(
            "Detect only reads what the machine already reports and uploads "
            "nothing. Detect and verify additionally uploads this Graft "
            "checkout and runs the HIP compile and numerical checks on the GPU.",
            cls="mt-2 block " + HINT,
        )

    return Form(
        Div(
            Div(
                Div("Target", cls="plate-title"),
                Div("02", cls="plate-idx"),
                cls="plate-head",
            ),
            _field_grid(target_select, cls="mb-6"),
            notice,
            local_panel,
            cloud_panel,
            actions,
            footer,
            script,
        ),
        action="/target",
        method="post",
        cls="plate",
        id="target-form",
    )


def local_capability(facts: dict) -> Any:
    """What this machine can do, from the same snapshot `doctor` reports."""
    if not facts:
        return Div(
            Small("Checking this machine has not run yet.", cls=HINT),
            cls="mt-4",
        )
    tiles = [
        _tile("Device", str(facts.get("device", "unknown"))),
        _tile("Backend", str(facts.get("backend", "unknown"))),
        _tile("GPU", "available" if facts.get("gpu") else "not detected"),
    ]
    vram = facts.get("vram_mb")
    if vram:
        tiles.append(_tile("VRAM", f"{vram} MB"))
    tiles.append(_tile("Torch", str(facts.get("torch", "unknown"))))
    out = [Div(*tiles, cls="grid g3 mt-4")]
    if not facts.get("gpu"):
        out.append(
            _note(
                "No AMD GPU on this machine",
                "Conversion, execution and parity still run, on CPU. "
                "Benchmarks are skipped rather than reported from CPU timings, "
                "and nothing here would demonstrate ROCm behaviour. Choose "
                "Cloud GPU to verify on real silicon.",
                WARN,
            )
        )
    elif not facts.get("bands_calibrated"):
        out.append(
            _note(
                "Verdict bands uncalibrated",
                "The speedup bands were measured on CPU. Treat any timing "
                "verdict from this machine as provisional until they are "
                "re-measured here.",
                WARN,
            )
        )
    return Div(*out, cls="mt-2")


def _tile(label: str, value: str) -> Any:
    return Div(
        Div(label, cls="figure-label"),
        Div(value, cls="figure-value text-[1.15rem]"),
        cls="figure",
    )


def _note(label: str, text: str, tone: str) -> Any:
    return Div(
        Div(label, cls="eyebrow", style=f"color:{tone}"),
        Strong(text, cls="mt-1 block text-[.79rem] font-semibold leading-relaxed"),
        cls="notice mt-3",
        style=f"--tone:{tone}",
    )


def _step(name: str, step: dict[str, Any]) -> Any:
    code = int(step.get("returncode") or 0)
    tone = PASS if code == 0 else FAIL
    tail = str(step.get("output") or "").strip()[-1600:]
    return Div(
        Div(
            Div(
                Div(name.replace("_", " "), cls="eyebrow", style=f"color:{tone}"),
                Div(
                    "exit 0" if code == 0 else f"exit {code}",
                    cls="mono text-[.62rem] text-[#6b6b70]",
                ),
                cls="flex items-start justify-between gap-3",
            ),
            Code(
                tail or "(no output)",
                cls="mt-3 block overflow-x-auto text-[.66rem] leading-relaxed text-[#a3a3a6]",
            ),
            Small("Remote output, last 1600 characters.", cls=HINT),
        ),
        cls="plate-body",
    )


def detection_result(run: dict[str, Any], *, kind: str = KIND_JUPYTER) -> Any:
    """Render a completed remote run: what was found, and what came back."""
    facts = run.get("facts") or {}
    ok = bool(run.get("ok"))
    tone = PASS if ok else WARN

    parts = [
        Div(
            Div(
                Div("Target result", cls="plate-title"),
                Div("OK" if ok else "ATTENTION", cls="tag", style=f"--tone:{tone}"),
                cls="plate-head",
            )
        ),
        Div(
            Div(
                _tile("Device", facts.get("gpu_name") or "none detected"),
                _tile("Architecture", facts.get("gfx_arch") or "not reported"),
                _tile("GPUs", str(facts.get("gpu_count") or 0)),
                _tile("VRAM", f"{facts.get('vram_mb')} MB" if facts.get("vram_mb") else "not reported"),
                _tile("OS", facts.get("os_name") or "unknown"),
                _tile("ROCm", facts.get("rocm_version") or "not reported"),
                _tile("PyTorch", facts.get("torch_version") or "not installed"),
                _tile("Uploaded", f"{run.get('pushed_bytes', 0) // 1024} KB"),
                cls="grid g4",
            ),
            cls="plate-body",
        ),
    ]

    tools = facts.get("tools") or {}
    if tools:
        parts.append(
            Div(
                Div("Toolchain", cls="eyebrow"),
                Div(
                    *(
                        Div(
                            Div(name, cls="kv-label"),
                            Div(path, cls="kv-value mono text-[.72rem]"),
                            cls="kv",
                        )
                        for name, path in sorted(tools.items())
                    ),
                    cls="grid g2 mt-3",
                ),
            )
        )

    for problem in facts.get("problems") or []:
        parts.append(_note("Blocking", problem, FAIL))
    for note in facts.get("notes") or []:
        parts.append(_note("Note", note, INFO))
    if run.get("error"):
        # Shown whether or not the run succeeded. Suppressing it on failure
        # hides the only part that says why, which is the opposite of useful.
        parts.append(_note("Result" if ok else "Failed", run["error"], WARN if ok else FAIL))

    report = facts.get("device_report") or ""
    if report:
        # Shown by default. A wrong device name is the failure mode that is
        # hardest to notice from the parsed numbers alone, so the machine's own
        # words belong next to them.
        parts.append(
            Div(
                Div("What the host reported", cls="eyebrow"),
                Code(
                    report,
                    cls="mt-2 block overflow-x-auto text-[.62rem] leading-relaxed text-[#a3a3a6]",
                ),
                cls="plate-body",
            )
        )

    for name, step in (run.get("steps") or {}).items():
        parts.append(_step(name, step))

    return Div(*parts, cls="plate", id="target-outcome")


def outcome_placeholder() -> Any:
    """Shown before the first attempt, so the swap target always exists."""
    return Div(
        Div("No target contacted yet", cls="plate-title"),
        Div(
            "Supply connection details above. Graft reports what the machine "
            "is before it asks the machine to do anything.",
            cls="mt-2 text-[.8rem] leading-relaxed text-[#a3a3a6]",
        ),
        cls="plate",
        id="target-outcome",
    )


def idle(message: str) -> Any:
    """A neutral panel for a target that could not be reached at all."""
    return Div(
        Div(
            Div("Target unreachable", cls="plate-title"),
            Div("ERROR", cls="tag", style=f"--tone:{FAIL}"),
            cls="plate-head",
        ),
        Div(message, cls="plate-body text-[.8rem] leading-relaxed text-[#a3a3a6]"),
        cls="plate",
        id="target-outcome",
    )
