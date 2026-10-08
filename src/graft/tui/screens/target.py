"""Workspace screen for choosing a source and a verification target."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
)
from textual_fspicker import FileOpen, Filters

from graft import config
from graft.target import (
    TargetError,
    TransportError,
    parse_form,
    run_remote,
    run_remote_file,
)
from graft.tui.screens.results import ResultsScreen
from graft.tui.widgets import GPUStatusPanel


def _field(label: str, control: Input) -> ComposeResult:
    """Return the labelled input pair used in responsive field grids."""
    yield Label(label, classes="field-label")
    yield control


class TargetScreen(Vertical):
    """Run a local migration or check and verify a remote GPU target."""

    AI_MODELS: ClassVar[dict[str, str]] = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-3.8-flash",
        "groq": "openai/gpt-oss-120b",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._converted_source = ""
        self._converted_filename = ""
        self._preview_timer: Timer | None = None

    _LOGO = r""" ________    ________     ________    ________   _________
 |\   ____\  |\   __  \   |\   __  \  |\  _____\ |\___   __\
 \ \  \___|  \ \  \|\  \  \ \  \|\  \ \ \  \__/  \|___ \  \_|
  \ \  \  ___ \ \   _  _\  \ \   __  \ \ \   __\      \ \  \
   \ \  \|\  \ \ \  \\  \|  \ \  \ \  \ \ \  \_|       \ \  \
    \ \_______\ \ \__\\ _\   \ \__\ \__\ \ \__\         \ \__\
     \|_______|  \|__|\|__|   \|__|\|__|  \|__|          \|__| V1.0.0 | powered by AMD"""
    _COMPACT_LOGO = "╭─┬─┬─┬─┬─╮\n│G│R│A│F│T│\n╰─┴─┴─┴─┴─╯"

    def compose(self) -> ComposeResult:
        yield Static(
            self._LOGO,
            id="graft-ascii",
            classes="ascii-logo",
            markup=False,
        )
        yield Static("GRAFT  /  MIGRATION WORKSPACE", classes="eyebrow")
        yield Static(
            "Turn a CUDA source file into measured ROCm evidence.",
            classes="hero-title",
        )
        yield Static(
            "Choose a local file for conversion and parity, or connect to a remote AMD machine for hardware detection and HIP verification.",
            classes="hero-copy",
        )

        with Grid(id="workspace-grid"):
            with VerticalScroll(classes="card", id="source-card"):
                yield Static("SOURCE FILE", classes="eyebrow")
                yield Static("Run a local migration", classes="card-title")
                yield Label("Python or CUDA source path", classes="field-label")
                with Horizontal(id="source-path-row"):
                    yield Input(
                        placeholder="path/to/kernel.py or path/to/kernel.cu",
                        id="source-path",
                    )
                    yield Button("Browse", id="browse-source")
                    yield Button("Preview", id="preview-source")
                yield Static(
                    "The migration runs through Graft’s disposable worker. It accepts one .py or .cu file.",
                    classes="hint",
                )
                with Collapsible(title="AI review settings", collapsed=True):
                    yield Label("AI review provider", classes="field-label")
                    yield Select(
                        [("OpenAI", "openai"), ("Gemini", "gemini"), ("Groq", "groq")],
                        value="openai",
                        allow_blank=False,
                        id="ai-provider",
                    )
                    yield Label("Model ID (optional)", classes="field-label")
                    yield Input(placeholder="e.g., gpt-4o-mini", id="ai-model")
                    yield Label("Provider API key (optional)", classes="field-label")
                    yield Input(
                        password=True,
                        placeholder="Used for this review only",
                        id="ai-api-key",
                    )
                    yield Static(
                        "When a key is provided, the original and complete converted source are sent to the selected provider. The key is not saved. AI review is advisory; GPU execution and parity are reported separately.",
                        classes="hint",
                    )
                with Horizontal(classes="button-row", id="local-actions"):
                    yield Button("Convert & review", variant="primary", id="run-local")
                    yield Button("Check this machine", id="probe-local")
                with Horizontal(classes="button-row", id="remote-actions"):
                    yield Button("Detect target", id="probe-remote")
                    yield Button("Verify on GPU", variant="primary", id="verify-remote")
                    yield Button("Disconnect GPU", id="disconnect-remote")
                    yield Button(
                        "Upload & run converted file",
                        id="run-converted",
                        disabled=True,
                    )
                yield Static(
                    "Ready · local conversion and parity are available on CPU; GPU timings require an AMD device.",
                    id="status",
                    classes="status",
                    markup=False,
                )

            with VerticalScroll(classes="card", id="target-card"):
                yield Static("COMPUTE TARGET", classes="eyebrow")
                yield GPUStatusPanel(id="gpu-visual")
                yield Static("Where should Graft run?", classes="card-title")
                yield Label("GPU location", classes="field-label")
                yield Select(
                    [("Local workstation", "local"), ("Remote AMD target", "cloud")],
                    value="local",
                    allow_blank=False,
                    id="location-select",
                )
                yield Static(
                    "Local migrations need no connection details.",
                    id="local-info",
                    classes="hint",
                )

                with Vertical(id="cloud-config", classes="hidden"):
                    yield Label("Connection method", classes="field-label")
                    yield Select(
                        [("SSH private key", "ssh"), ("Jupyter server", "jupyter")],
                        value="ssh",
                        allow_blank=False,
                        id="transport-select",
                    )

                    with Vertical(id="ssh-config"):
                        with Grid(id="ssh-fields", classes="fields"):
                            with Vertical(classes="field"):
                                yield from _field(
                                    "Host or IP",
                                    Input(
                                        placeholder="GPU droplet public IPv4",
                                        id="ssh-host",
                                    ),
                                )
                            with Vertical(classes="field"):
                                yield from _field(
                                    "Username",
                                    Input(
                                        placeholder="root",
                                        id="ssh-username",
                                    ),
                                )
                            with Vertical(classes="field"):
                                yield from _field(
                                    "Private key path",
                                    Input(
                                        placeholder="~/.ssh/id_ed25519",
                                        id="ssh-key-path",
                                    ),
                                )
                        yield Static(
                            "The named key is used for this request and is not saved by Graft.",
                            classes="hint",
                        )

                    with (
                        Vertical(id="jupyter-config", classes="hidden"),
                        Grid(id="jupyter-fields", classes="fields"),
                    ):
                        with Vertical(classes="field"):
                            yield from _field(
                                "Server URL",
                                Input(
                                    placeholder="https://gpu.example.com/lab",
                                    id="jupyter-host",
                                ),
                            )
                        with Vertical(classes="field"):
                            yield from _field(
                                "Username",
                                Input(
                                    placeholder="root",
                                    id="jupyter-username",
                                ),
                            )
                        with Vertical(classes="field"):
                            yield from _field(
                                "Token",
                                Input(
                                    password=True,
                                    placeholder="Jupyter token",
                                    id="jupyter-token",
                                ),
                            )

                    with (
                        Collapsible(
                            title="Advanced connection settings", collapsed=True
                        ),
                        Grid(id="advanced-fields", classes="fields"),
                    ):
                        with Vertical(classes="field"):
                            yield from _field(
                                "Port (optional)",
                                Input(
                                    placeholder="22 for SSH · 443 for HTTPS",
                                    id="remote-port",
                                ),
                            )
                        with Vertical(classes="field"):
                            yield from _field(
                                "Remote workspace",
                                Input(placeholder="/root/graft", id="remote-dir"),
                            )
                        with Vertical(classes="field"):
                            yield from _field(
                                "Command timeout (seconds)",
                                Input(placeholder="900", id="remote-timeout"),
                            )
                        with Vertical(classes="field"):
                            yield from _field(
                                "SSH key passphrase",
                                Input(
                                    password=True,
                                    placeholder="Optional",
                                    id="ssh-passphrase",
                                ),
                            )

                    yield Static(
                        "Detect only reads target capabilities. Verify uploads this project, recreates the remote workspace, compiles the HIP smoke test, and runs Graft’s bundled softmax example.",
                        classes="hint",
                    )

    def _gpu_panel(self) -> GPUStatusPanel | None:
        panel = self.query(GPUStatusPanel)
        return next(iter(panel), None)

    def _gpu_status(self, method: str, *args: Any) -> None:
        """Drive the window-level GPU overlay if it is mounted yet."""
        panel = self._gpu_panel()
        if panel is not None:
            getattr(panel, method)(*args)

    def on_mount(self) -> None:
        self._sync_target_controls()
        self.call_after_refresh(self._apply_layout)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_layout()

    def _apply_layout(self) -> None:
        self._set_layout(self.app.size.width, self.app.size.height)

    def _set_layout(self, width: int, height: int) -> None:
        if not self.is_mounted:
            return
        compact = width < 108
        self.set_class(compact, "compact")
        self.query_one("#workspace-grid", Grid).set_class(compact, "compact")
        short = height < 36
        self.set_class(short, "short")
        self.query_one("#workspace-grid", Grid).set_class(short, "short")
        logo = self.query_one("#graft-ascii", Static)
        logo_compact = width < 86 or short
        logo.update(self._COMPACT_LOGO if logo_compact else self._LOGO)
        logo.set_class(logo_compact, "compact")
        self.query_one(".hero-copy", Static).display = not short
        self.query_one("#source-path-row", Horizontal).set_class(width < 90, "stacked")
        fields_compact = width < 92
        for selector in ("#ssh-fields", "#jupyter-fields", "#advanced-fields"):
            self.query_one(selector, Grid).set_class(fields_compact, "compact")
        actions_stacked = width < 94
        for selector in ("#local-actions", "#remote-actions"):
            self.query_one(selector, Horizontal).set_class(actions_stacked, "stacked")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in {"location-select", "transport-select"}:
            self._sync_target_controls()
            self._gpu_status("set_waiting", "DETECT THE SELECTED TARGET")
        elif event.select.id == "ai-provider":
            provider = str(event.value)
            model = self.query_one("#ai-model", Input)
            model.value = self.AI_MODELS.get(provider, "")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in {"ssh-host", "ssh-username", "jupyter-host"}:
            self._gpu_status("set_waiting", "DETECT AFTER CHANGING TARGET")
        elif event.input.id == "source-path":
            if self._preview_timer is not None:
                self._preview_timer.stop()
            # Avoid reading the disk on every keystroke; pasted and selected
            # paths still preview automatically once the user pauses.
            path_text = event.input.value
            self._invalidate_converted_file()
            self._preview_timer = self.set_timer(
                0.35, lambda: self._preview_source_path(path_text)
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "source-path":
            self._preview_source_path(event.input.value, switch_to_results=True)

    def _sync_target_controls(self) -> None:
        if not self.is_mounted:
            return
        is_remote = self.query_one("#location-select", Select).value == "cloud"
        self.query_one("#local-info", Static).display = not is_remote
        self.query_one("#cloud-config", Vertical).display = is_remote
        self.query_one("#local-actions", Horizontal).display = not is_remote
        self.query_one("#remote-actions", Horizontal).display = is_remote
        is_jupyter = self.query_one("#transport-select", Select).value == "jupyter"
        self.query_one("#ssh-config", Vertical).display = is_remote and not is_jupyter
        self.query_one("#jupyter-config", Vertical).display = is_remote and is_jupyter

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "disconnect-remote":
            self._gpu_status("set_disconnected", "TARGET DISCONNECTED")
            self._set_status("Remote GPU disconnected", tone="info")
            return
        if button_id == "run-local":
            self._start_local_migration()
        elif button_id == "browse-source":
            self._browse_source_file()
        elif button_id == "preview-source":
            self._preview_source_path(
                self.query_one("#source-path", Input).value,
                switch_to_results=True,
            )
        elif button_id == "probe-local":
            self._set_busy("Checking this machine’s PyTorch and GPU capabilities…")
            self.run_worker(
                self._probe_local,
                name="local-capability-check",
                group="graft-operation",
                exclusive=True,
                thread=True,
            )
        elif button_id == "probe-remote":
            self._start_remote("probe")
        elif button_id == "verify-remote":
            self._start_remote("verify")
        elif button_id == "run-converted":
            self._start_remote_file()

    def _browse_source_file(self) -> None:
        current = Path(
            self.query_one("#source-path", Input).value or config.EXAMPLES_DIR
        ).expanduser()
        location = current if current.is_dir() else current.parent
        if not location.is_dir():
            location = config.EXAMPLES_DIR
        picker = FileOpen(
            location,
            title="Choose a Python or CUDA source file",
            filters=Filters(
                (
                    "Python and CUDA source",
                    lambda path: path.suffix.lower() in {".py", ".cu"},
                ),
            ),
        )
        self.app.push_screen(picker, callback=self._source_file_selected)

    def _source_file_selected(self, path: Path | None) -> None:
        if path is None:
            return
        self.query_one("#source-path", Input).value = str(path)
        self._preview_source_path(str(path), switch_to_results=True)

    def _invalidate_converted_file(self) -> None:
        self._converted_source = ""
        self._converted_filename = ""
        if not self.is_mounted:
            return
        self.query_one("#run-converted", Button).disabled = True

    def _preview_source_path(
        self, path_text: str, *, switch_to_results: bool = False
    ) -> None:
        if not self.is_mounted:
            return
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None
        path_text = path_text.strip()
        self._invalidate_converted_file()
        results = self.app.query_one("#results-view", ResultsScreen)
        if not path_text:
            message = "Enter or browse to a .py or .cu source file."
            results.show_source_preview("", "", message=message)
            self._set_status(message, tone="warning")
        else:
            path = Path(path_text).expanduser()
            try:
                if path.suffix.lower() not in {".py", ".cu"}:
                    raise ValueError("Choose a .py or .cu source file.")
                if not path.is_file():
                    raise FileNotFoundError(f"No such source file: {path}")
                if path.stat().st_size > config.UI_MAX_UPLOAD_BYTES:
                    limit_kb = config.UI_MAX_UPLOAD_BYTES // 1024
                    raise ValueError(
                        f"The source file is larger than the {limit_kb} KB limit."
                    )
                source = path.read_text(encoding="utf-8")
                if not source.strip():
                    raise ValueError("The selected source file is empty.")
            except (OSError, UnicodeError, ValueError) as exc:
                results.show_source_preview(path.name, str(path), message=str(exc))
                self._set_status(f"Source file not loaded · {exc}", tone="error")
            else:
                results.show_source_preview(path.name, str(path), source=source)
                self._set_status(
                    f"Source loaded · {path.name} · ready to convert and review",
                    tone="info",
                )
        if switch_to_results:
            self.app.query_one("#main-tabs", TabbedContent).active = "results"

    def convert_selected_file(self) -> None:
        """Allow the comparison page to start a conversion from its editor."""
        self._start_local_migration()

    def _start_local_migration(self) -> None:
        path_text = self.query_one("#source-path", Input).value.strip()
        if not path_text:
            self._show_error("Enter a source file path.")
            return
        path = Path(path_text).expanduser().resolve()
        if path.suffix.lower() not in {".py", ".cu"}:
            self._show_error("Choose a .py or .cu source file.")
            return
        try:
            if not path.is_file():
                self._show_error(f"No such source file: {path}")
                return
            if path.stat().st_size > config.UI_MAX_UPLOAD_BYTES:
                limit_kb = config.UI_MAX_UPLOAD_BYTES // 1024
                self._show_error(
                    f"The source file is larger than the {limit_kb} KB limit."
                )
                return
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self._show_error(f"Could not read {path.name}: {exc}")
            return
        if not source.strip():
            self._show_error("The source file is empty.")
            return

        provider = str(self.query_one("#ai-provider", Select).value)
        model_id = self.query_one("#ai-model", Input).value.strip()
        api_key = self.query_one("#ai-api-key", Input).value.strip()
        options: dict[str, Any] = {
            "run_parity": True,
            "run_bench": True,
            # This screen uses the explicit provider/key form below. Do not
            # call an ambient environment-configured model when the user has
            # not requested review here.
            "with_narrative": False,
        }
        if api_key:
            # The model is optional: fall back to a sensible per-provider
            # default so a user who does not know a model name can still get
            # an advisory review.
            resolved_model = model_id or self.AI_MODELS.get(provider, "")
            if not resolved_model:
                self._show_error("Enter a model ID for the selected AI provider.")
                return
            options["ai_review"] = {
                "provider": provider,
                "model": resolved_model,
                "api_key": api_key,
            }

        self._set_busy(f"Migrating {path.name} in a disposable worker…")
        self.run_worker(
            partial(self._run_local_migration, source, path.name, str(path), options),
            name="local-migration",
            group="graft-operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_local_migration(
        self,
        source: str,
        filename: str,
        source_path: str,
        options: dict[str, Any],
    ) -> None:
        try:
            # Reuse the same child-process boundary as the browser UI. The
            # analyzed module never imports into the long-lived TUI process.
            from graft.worker import run_worker

            summary = run_worker(
                source,
                filename,
                options,
            )
            self.app.call_from_thread(
                self._local_migration_done, summary, filename, source_path, source, ""
            )
        except Exception as exc:
            self.app.call_from_thread(
                self._local_migration_done,
                None,
                filename,
                source_path,
                source,
                str(exc),
            )

    def _probe_local(self) -> None:
        try:
            from graft.target import local_capability

            facts = local_capability()
            self.app.call_from_thread(self._local_probe_done, facts, "")
        except Exception as exc:
            self.app.call_from_thread(self._local_probe_done, {}, str(exc))

    def _start_remote(self, mode: str) -> None:
        fields = self._remote_fields()
        message = (
            "Detecting the remote target. No files will be uploaded…"
            if mode == "probe"
            else "Uploading this project and starting remote HIP verification…"
        )
        self._set_busy(message)
        self.run_worker(
            partial(self._run_remote, mode, fields),
            name=f"remote-{mode}",
            group="graft-operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _remote_fields(self) -> dict[str, Any]:
        kind = str(self.query_one("#transport-select", Select).value)
        fields: dict[str, Any] = {
            "kind": kind,
            "port": self.query_one("#remote-port", Input).value,
            "remote_dir": self.query_one("#remote-dir", Input).value,
            "command_timeout_s": self.query_one("#remote-timeout", Input).value,
        }
        if kind == "ssh":
            fields.update(
                host=self.query_one("#ssh-host", Input).value,
                username=self.query_one("#ssh-username", Input).value,
                key_path=self.query_one("#ssh-key-path", Input).value,
                key_passphrase=self.query_one("#ssh-passphrase", Input).value,
            )
        else:
            fields.update(
                host=self.query_one("#jupyter-host", Input).value,
                username=self.query_one("#jupyter-username", Input).value,
                token=self.query_one("#jupyter-token", Input).value,
            )
        return fields

    def _start_remote_file(self) -> None:
        if not self._converted_source or not self._converted_filename:
            self._show_error(
                "Run a migration first so there is a converted file to upload."
            )
            return
        fields = self._remote_fields()
        filename = self._converted_filename
        source = self._converted_source
        self._set_busy(
            f"Detecting the GPU, uploading {filename}, and running it remotely…"
        )
        self.run_worker(
            partial(self._run_remote_file, fields, filename, source),
            name="remote-converted-file",
            group="graft-operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_remote_file(
        self, fields: dict[str, Any], filename: str, source: str
    ) -> None:
        try:
            spec = parse_form(fields)
            if spec.kind == "ssh":
                from graft.target.ssh import SSHTransport

                transport = SSHTransport(spec)
            else:
                from graft.target.jupyter import JupyterTransport

                transport = JupyterTransport(spec)
            with transport:
                result = run_remote_file(transport, spec, filename, source)
            self.app.call_from_thread(self._remote_done, result, "file", "")
        except (TargetError, TransportError, ValueError) as exc:
            self.app.call_from_thread(self._remote_done, None, "file", str(exc))
        except Exception:
            self.app.call_from_thread(
                self._remote_done,
                None,
                "file",
                "Graft could not upload or run the converted file. Check the target details and try again.",
            )

    def _run_remote(self, mode: str, fields: dict[str, Any]) -> None:
        try:
            spec = parse_form(fields)
            if spec.kind == "ssh":
                from graft.target.ssh import SSHTransport

                transport = SSHTransport(spec)
            else:
                from graft.target.jupyter import JupyterTransport

                transport = JupyterTransport(spec)
            with transport:
                result = run_remote(
                    transport,
                    spec,
                    source=config.PROJECT_ROOT,
                    probe_only=(mode == "probe"),
                )
            self.app.call_from_thread(self._remote_done, result.as_dict(), mode, "")
        except (TargetError, TransportError) as exc:
            self.app.call_from_thread(self._remote_done, None, mode, str(exc))
        except Exception:
            # Connection libraries can include request details in arbitrary
            # exceptions, so keep their contents out of the terminal UI.
            self.app.call_from_thread(
                self._remote_done,
                None,
                mode,
                "Graft could not complete the remote request. Check the target details and try again.",
            )

    def _set_busy(self, message: str) -> None:
        status = self.query_one("#status", Static)
        status.update(message)
        for name in ("running", "success", "warning", "error", "info"):
            status.set_class(name == "running", name)
        self._toggle_buttons(True)
        self.app.show_loading(message)

    def _toggle_buttons(self, disabled: bool) -> None:
        for selector in (
            "#run-local",
            "#probe-local",
            "#probe-remote",
            "#verify-remote",
            "#run-converted",
        ):
            self.query_one(selector, Button).disabled = disabled
        self.query_one("#run-converted", Button).disabled = disabled or not bool(
            self._converted_source
        )

    def _set_status(
        self, message: str, *, is_error: bool = False, tone: str | None = None
    ) -> None:
        status = self.query_one("#status", Static)
        status.update(message)
        selected_tone = "error" if is_error else tone or "success"
        for name in ("running", "success", "warning", "error", "info"):
            status.set_class(name == selected_tone, name)
        self._toggle_buttons(False)
        self.app.hide_loading()

    def _show_error(self, message: str) -> None:
        self._set_status(message, is_error=True)
        self.app.query_one("#results-view", ResultsScreen).show_error(
            "Could not start", message
        )
        self.app.query_one("#main-tabs", TabbedContent).active = "results"

    def _local_migration_done(
        self,
        summary: dict[str, Any] | None,
        filename: str,
        source_path: str,
        original_source: str,
        error: str,
    ) -> None:
        self.query_one("#ai-api-key", Input).value = ""
        if error or summary is None:
            message = error or "The migration worker returned no summary."
            self._converted_source = ""
            self._converted_filename = ""
            self.query_one("#run-converted", Button).disabled = True
            self._set_status(f"Migration failed · {message}", is_error=True)
            self.app.query_one("#results-view", ResultsScreen).show_migration_error(
                f"Migration failed / {filename}", message, original_source
            )
        else:
            self._converted_source = str(summary.get("converted_source") or "")
            self._converted_filename = filename
            conversion = summary.get("conversion") or {}
            parity = summary.get("parity") or {}
            verdict = str(conversion.get("verdict") or "").lower()
            parity_failed = bool(parity.get("measured")) and int(
                parity.get("passed", 0)
            ) < int(parity.get("total", 0))
            if (
                verdict == "production_ready"
                and parity.get("measured")
                and not parity_failed
            ):
                tone = "success"
            elif verdict == "unsupported" or parity_failed:
                tone = "error"
            else:
                tone = "warning"
            self._set_status(f"Migration complete · {filename}", tone=tone)
            self.app.query_one("#results-view", ResultsScreen).show_local(
                summary, filename, source_path, original_source
            )
            self.query_one("#run-converted", Button).disabled = not bool(
                self._converted_source
            )
        self.app.query_one("#main-tabs", TabbedContent).active = "results"

    def _local_probe_done(self, facts: dict[str, Any], error: str) -> None:
        if error:
            self._gpu_status("set_disconnected", "LOCAL GPU CHECK FAILED")
            self._set_status(f"Capability check failed · {error}", is_error=True)
            self.app.query_one("#results-view", ResultsScreen).show_error(
                "Local capability check failed", error
            )
        else:
            self._gpu_status("set_local_facts", facts)
            device = facts.get("device", "local machine")
            gpu = "GPU detected" if facts.get("gpu") else "CPU-only"
            self._set_status(
                f"Local machine checked · {device} · {gpu}",
                tone="success" if facts.get("gpu") else "info",
            )
            self.app.query_one("#results-view", ResultsScreen).show_local_probe(facts)
        self.app.query_one("#main-tabs", TabbedContent).active = "results"

    def _remote_done(
        self, result: dict[str, Any] | None, mode: str, error: str
    ) -> None:
        # Passwords and tokens are request-only. Clear them even after a failed
        # attempt so they do not remain visible in the form after completion.
        self.query_one("#ssh-passphrase", Input).value = ""
        self.query_one("#jupyter-token", Input).value = ""
        if error or result is None:
            self._gpu_status("set_disconnected", "TARGET CONNECTION FAILED")
            message = error or "The remote request returned no result."
            self._set_status(f"Remote request failed · {message}", is_error=True)
            self.app.query_one("#results-view", ResultsScreen).show_error(
                "Remote request failed", message
            )
        else:
            self._gpu_status("set_target_facts", result.get("facts"))
            state = "ready" if result.get("ok") else "needs attention"
            self._set_status(
                f"Remote {mode} complete · target {state}",
                tone="success" if result.get("ok") else "warning",
            )
            results = self.app.query_one("#results-view", ResultsScreen)
            if mode == "file":
                results.show_remote_file(result)
            else:
                results.show_remote(result, mode)
        self.app.query_one("#main-tabs", TabbedContent).active = "results"


__all__ = ["TargetScreen"]
