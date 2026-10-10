"""Workspace screen for choosing a source and a verification target."""

from __future__ import annotations

import contextlib
import threading
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
    actionable_dependencies,
    missing_dependencies,
    parse_form,
    run_remote,
    run_remote_file,
    scan_dependencies,
)
from graft.target.detect import TargetFacts, detect
from graft.target.metrics import (
    SENTINEL,
    MetricsSnapshot,
    local_snapshot,
    snapshot_from_fields,
    stream_command,
)
from graft.tui.screens.deps import DependencyOverlay
from graft.tui.screens.results import ResultsScreen
from graft.tui.screens.tips import classify_connection_failure
from graft.tui.widgets import GPUMetrics, GPUStatusPanel


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
    # Seconds to wait before each automatic reconnect attempt after a live
    # connection drops. Three tries, spaced out: a brief network blip is ridden
    # out, while a genuinely gone host is not hammered hard enough to trip
    # fail2ban. A manual "Reconnect" is always available.
    RECONNECT_BACKOFF_S: ClassVar[tuple[int, ...]] = (5, 20, 60)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._converted_source = ""
        self._converted_filename = ""
        self._preview_timer: Timer | None = None
        # The last successful remote connection, kept so the dependency scan
        # and the live metrics can re-open the same target without the user
        # retyping (or re-pasting a token) the form. It survives a dropped
        # connection so the reconnect flow can reuse it.
        self._connected_fields: dict[str, Any] | None = None
        # Capabilities from the last *live* connection. None means the target
        # is not currently reachable, which is what gates the reconnect button
        # and hides the install button.
        self._connected_facts: dict[str, Any] | None = None
        self._pending_fields: dict[str, Any] | None = None
        self._metrics_stop: threading.Event | None = None
        self._metrics_thread: threading.Thread | None = None
        self._reconnect_attempts = 0
        self._reconnect_timer: Timer | None = None
        self._operation_active = False

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
                    yield Button("Reconnect", id="reconnect-remote", disabled=True)
                    yield Button("Disconnect GPU", id="disconnect-remote")
                    yield Button(
                        "Upload & run converted file",
                        id="run-converted",
                        disabled=True,
                    )
                    yield Button(
                        "Install missing dependencies",
                        id="install-deps",
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
        """The app-level GPU dock panel, if the app hosts one."""
        panel = self.app.query(GPUStatusPanel)
        return next(iter(panel), None)

    def _gpu_status(self, method: str, *args: Any) -> None:
        """Drive the app-level GPU dock, which is fixed on every screen."""
        show = getattr(self.app, "show_gpu_facts", None)
        if show is not None:
            show(method, *args)
            return
        panel = self._gpu_panel()
        if panel is not None:
            getattr(panel, method)(*args)

    def on_mount(self) -> None:
        self._sync_target_controls()
        # is_mounted is False inside on_mount, so defer the style-only
        # updates until after the first refresh.
        self.call_after_refresh(self._apply_layout)
        self.call_after_refresh(self._refresh_dependency_button)
        self.call_after_refresh(self._refresh_reconnect_button)
        # The workspace opens on the local machine, so its readings are the
        # first thing the panel can show without any connection.
        self._start_metrics("local")

    def on_unmount(self) -> None:
        self._stop_reconnect_timer()
        self._stop_metrics()

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
            if event.select.id == "location-select":
                self._on_location_changed(str(event.value))
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
            self._cancel_reconnect()
            self._gpu_status("set_disconnected", "TARGET DISCONNECTED")
            self._set_status("Remote GPU disconnected", tone="info")
            self._connected_fields = None
            self._connected_facts = None
            self._refresh_dependency_button()
            self._refresh_reconnect_button()
            self._start_metrics("local")
            return
        if button_id == "reconnect-remote":
            self._start_reconnect(manual=True)
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
        elif button_id == "install-deps":
            self._start_dependency_install()

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
        if mode == "verify":
            blocking = self._blocking_dependencies(None)
            if blocking:
                self._show_dependency_block("Remote verification", blocking)
                return
        fields = self._remote_fields()
        self._pending_fields = dict(fields)
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
        suffix = Path(self._converted_filename).suffix.lower()
        blocking = self._blocking_dependencies(suffix)
        if blocking:
            self._show_dependency_block("Running the converted file", blocking)
            return
        fields = self._remote_fields()
        filename = self._converted_filename
        source = self._converted_source
        self._pending_fields = dict(fields)
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
        self._operation_active = True
        self._toggle_buttons(True)
        self.query_one("#install-deps", Button).disabled = True
        self.app.show_loading(message)

    def _toggle_buttons(self, disabled: bool) -> None:
        for selector in (
            "#run-local",
            "#probe-local",
            "#probe-remote",
            "#verify-remote",
            "#reconnect-remote",
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
        self._operation_active = False
        self._toggle_buttons(False)
        self._refresh_dependency_button()
        self._refresh_reconnect_button()
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
            # A local machine is a target too: check it for the tools the
            # pipeline needs the moment it is inspected, but only interrupt
            # with the overlay when something actually wants attention.
            try:
                from graft.target.local import local_target_facts

                local_facts = local_target_facts()
                if self._has_missing_dependencies(local_facts):
                    self._show_dependency_overlay(
                        local_facts,
                        fields=None,
                        label="this machine",
                    )
            except Exception:
                pass
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
            self._pending_fields = None
            # The target is not live, even if a previous connection worked, so
            # the reconnect button becomes available.
            self._connected_facts = None
            self._refresh_reconnect_button()
            self._maybe_show_tips(message)
        else:
            self._cancel_reconnect()
            self._connected_fields = self._pending_fields
            self._connected_facts = result.get("facts")
            self._pending_fields = None
            self._gpu_status("set_target_facts", result.get("facts"))
            state = "ready" if result.get("ok") else "needs attention"
            self._set_status(
                f"Remote {mode} complete · target {state}",
                tone="success" if result.get("ok") else "warning",
            )
            # The moment a target connects, check it for the tools the
            # pipeline needs and say what is missing, rather than waiting
            # for the first operation to fail on one. Verification is an
            # explicit "show me the result" action, so it never has an
            # overlay thrown over its findings, and a fully-equipped target
            # needs no overlay at all.
            if mode == "probe" and self._has_missing_dependencies(
                self._connected_facts
            ):
                self._show_dependency_overlay(
                    self._connected_facts,
                    fields=self._connected_fields,
                    label=str((self._connected_fields or {}).get("host") or "target"),
                )
            results = self.app.query_one("#results-view", ResultsScreen)
            if mode == "file":
                blocked = []
                if result.get("stage") == "pytorch_gpu_check":
                    suffix = Path(result.get("filename", "")).suffix.lower()
                    blocked = self._blocking_dependencies(suffix)
                if blocked:
                    self._show_dependency_block(
                        "Running the converted file", blocked
                    )
                else:
                    results.show_remote_file(result)
            else:
                results.show_remote(result, mode)
            self._refresh_dependency_button()
            if self._connected_fields:
                self._start_metrics("remote", self._connected_fields)
        self.app.query_one("#main-tabs", TabbedContent).active = "results"

    def _refresh_reconnect_button(self) -> None:
        """Offer a reconnect only for a target that is known but not live."""
        if not self.is_mounted:
            return
        button = self.query_one("#reconnect-remote", Button)
        on_cloud = self.query_one("#location-select", Select).value == "cloud"
        can_reconnect = (
            on_cloud
            and self._connected_fields is not None
            and self._connected_facts is None
        )
        button.display = can_reconnect
        button.disabled = not can_reconnect

    def _stop_reconnect_timer(self) -> None:
        if self._reconnect_timer is not None:
            self._reconnect_timer.stop()
            self._reconnect_timer = None

    def _cancel_reconnect(self) -> None:
        self._stop_reconnect_timer()
        self._reconnect_attempts = 0

    def _start_reconnect(self, *, manual: bool = False) -> None:
        """Probe the remembered target again, now or on the retry timer."""
        self._stop_reconnect_timer()
        if not self.is_mounted:
            return
        if self._connected_fields is None or self._connected_facts is not None:
            return
        if manual:
            self._reconnect_attempts = 0
        if self._operation_active:
            # Something the user started is running; try again shortly rather
            # than cancelling their work or colliding with its status.
            self._reconnect_timer = self.set_timer(5, self._start_reconnect)
            return
        self._reconnect_attempts += 1
        fields = dict(self._connected_fields)
        self._set_busy("Reconnecting to the target…")
        self.run_worker(
            partial(self._run_reconnect, fields),
            name="reconnect",
            group="graft-operation",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_reconnect(self, fields: dict[str, Any]) -> None:
        try:
            spec = parse_form(fields)
            if spec.kind == "ssh":
                from graft.target.ssh import SSHTransport

                transport = SSHTransport(spec)
            else:
                from graft.target.jupyter import JupyterTransport

                transport = JupyterTransport(spec)
            with transport:
                facts = detect(transport).as_dict()
            self.app.call_from_thread(self._reconnect_done, fields, facts, "")
        except (TargetError, TransportError) as exc:
            self.app.call_from_thread(self._reconnect_done, fields, None, str(exc))
        except Exception:
            self.app.call_from_thread(
                self._reconnect_done,
                fields,
                None,
                "Graft could not reach the target again. Check the connection "
                "details and try again.",
            )

    def _reconnect_done(
        self,
        fields: dict[str, Any],
        facts: dict[str, Any] | None,
        error: str,
    ) -> None:
        if not self.is_mounted:
            return
        if self.query_one("#location-select", Select).value != "cloud":
            # The user moved away from the target while the probe was in
            # flight; drop the result rather than yanking them back.
            self._cancel_reconnect()
            return
        if error or facts is None:
            # A rejected key or an active ban will not fix itself, and retrying
            # only makes a ban worse; stop and let the tips explain. Everything
            # else (a dropped route, a rebooting host) gets the next attempt.
            category = classify_connection_failure(error, str(fields.get("kind", "")))
            if category in {"auth", "banned"}:
                self._cancel_reconnect()
                self._set_status(f"Reconnect failed · {error}", is_error=True)
                self._maybe_show_tips(error)
                return
            self._schedule_reconnect(error)
            return
        self._reconnect_attempts = 0
        self._connected_fields = dict(fields)
        self._connected_facts = dict(facts)
        self._gpu_status("set_target_facts", facts)
        self._set_status("Reconnected · live target readings resumed", tone="success")
        self._start_metrics("remote", self._connected_fields)

    def _schedule_reconnect(self, reason: str) -> None:
        """Arrange the next automatic attempt, or give up after the budget."""
        if self._connected_fields is None:
            return
        attempted = self._reconnect_attempts
        total = len(self.RECONNECT_BACKOFF_S)
        if attempted >= total:
            self._set_status(
                f"Target unreachable · reconnect gave up · {reason}",
                is_error=True,
            )
            return
        delay = self.RECONNECT_BACKOFF_S[attempted]
        self._set_status(
            f"Connection lost · retrying in {delay}s "
            f"(attempt {attempted + 1}/{total})",
            tone="warning",
        )
        self._reconnect_timer = self.set_timer(delay, self._start_reconnect)

    def _refresh_dependency_button(self) -> None:
        """Show the install button only when a checked target is missing tools.

        Gating on the facts means the button cannot appear before a target has
        been inspected, and it stays hidden when there is nothing to install.
        """
        if not self.is_mounted:
            return
        button = self.query_one("#install-deps", Button)
        actionable: list[Any] = []
        if self._connected_facts:
            try:
                actionable = actionable_dependencies(
                    TargetFacts(**self._connected_facts)
                )
            except TypeError:
                actionable = []
        button.disabled = not actionable
        button.display = bool(actionable)

    def _has_missing_dependencies(self, facts: dict[str, Any] | None) -> bool:
        """True when the connected target lacks at least one checked tool."""
        if not facts:
            return False
        try:
            parsed = TargetFacts(**facts)
        except TypeError:
            return False
        return any(not dep.present for dep in scan_dependencies(parsed))

    def _blocking_dependencies(self, suffix: str | None) -> list[Any]:
        """Tools absent on the connected target that would fail this operation.

        ``suffix`` is the file being run, or ``None`` for the HIP verification.
        A converted Python file runs under the target's PyTorch, so there the
        optional ``torch`` becomes blocking rather than merely installable.
        """
        if not self._connected_facts:
            return []
        try:
            facts = TargetFacts(**self._connected_facts)
        except TypeError:
            return []
        blocking = missing_dependencies(facts)
        if suffix is None or suffix == ".cu":
            return blocking
        return blocking + [
            dep for dep in actionable_dependencies(facts) if dep.key == "torch"
        ]

    def _show_dependency_block(self, action: str, missing: list[Any]) -> None:
        """Stop an operation and say, in red, which target tool is missing."""
        names = ", ".join(getattr(dep, "label", dep.key) for dep in missing)
        message = (
            f"{action} needs {names} on the target, but it is not installed. "
            "Install it with “Install missing dependencies”, or install it on "
            "the target directly, then try again."
        )
        self._set_status(f"Blocked · {names} missing on the target", is_error=True)
        self.app.query_one("#results-view", ResultsScreen).show_missing_dependencies(
            action, missing, message
        )
        self.app.query_one("#main-tabs", TabbedContent).active = "results"

    def _show_dependency_overlay(
        self,
        facts: dict[str, Any],
        *,
        fields: dict[str, Any] | None,
        label: str,
    ) -> None:
        """Open the dependency checklist for a target that just connected.

        The overlay is advisory: it names what is missing and offers to
        install it, but nothing runs until its button is pressed.
        """
        self.app.push_screen(
            DependencyOverlay(
                facts,
                fields=fields,
                label=label,
                on_refreshed=self._dependency_facts_refreshed,
            )
        )

    def _dependency_facts_refreshed(self, facts: dict[str, Any]) -> None:
        """Adopt facts re-detected after an install from the overlay."""
        if not self.is_mounted:
            return
        self._connected_facts = facts
        self._gpu_status("set_target_facts", facts)
        self._refresh_dependency_button()

    def _start_dependency_install(self) -> None:
        if not self._connected_fields or not self._connected_facts:
            self._show_error("Connect to a target before installing dependencies.")
            return
        self._show_dependency_overlay(
            self._connected_facts,
            fields=self._connected_fields,
            label=str(self._connected_fields.get("host") or "target"),
        )

    def _metrics_widget(self) -> GPUMetrics | None:
        node = self.app.query(GPUMetrics)
        return next(iter(node), None)

    def _on_location_changed(self, location: str) -> None:
        """Local readings are always available; remote ones need a connection."""
        # Leaving or returning to the target invalidates a pending retry; the
        # user can restart it with the Reconnect button.
        self._stop_reconnect_timer()
        self._refresh_reconnect_button()
        if location == "local":
            self._start_metrics("local")
        else:
            self._stop_metrics()
            widget = self._metrics_widget()
            if widget is not None:
                widget.clear("CONNECT TO SEE LIVE TARGET READINGS")

    def _start_metrics(
        self, source: str, fields: dict[str, Any] | None = None
    ) -> None:
        """(Re)start the live-readings loop for a source on a background thread."""
        self._stop_metrics()
        stop = threading.Event()
        self._metrics_stop = stop
        thread = threading.Thread(
            target=self._metrics_loop,
            args=(source, fields, stop),
            name="graft-metrics",
            daemon=True,
        )
        self._metrics_thread = thread
        thread.start()

    def _stop_metrics(self) -> None:
        stop, thread = self._metrics_stop, self._metrics_thread
        self._metrics_stop = None
        self._metrics_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            # The stream loops wake within a second, so this join is short and
            # keeps a torn-down screen from leaving a thread attached to it.
            thread.join(timeout=2.0)

    def _metrics_loop(
        self,
        source: str,
        fields: dict[str, Any] | None,
        stop: threading.Event,
    ) -> None:
        if source == "remote" and fields is not None:
            self._remote_metrics_loop(fields, stop)
        else:
            self._local_metrics_loop(stop)

    def _local_metrics_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self._post_metrics(local_snapshot())
            stop.wait(2.0)

    def _remote_metrics_loop(
        self, fields: dict[str, Any], stop: threading.Event
    ) -> None:
        buffer: dict[str, str] = {}

        def on_line(line: str) -> None:
            nonlocal buffer
            stripped = line.strip()
            if stripped == SENTINEL:
                if buffer:
                    self._post_metrics(
                        snapshot_from_fields(buffer, source="remote")
                    )
                    buffer = {}
                return
            key, separator, value = stripped.partition("=")
            if separator:
                buffer[key.strip()] = value.strip()

        try:
            spec = parse_form(fields)
            if spec.kind == "ssh":
                from graft.target.ssh import SSHTransport

                transport = SSHTransport(spec)
            else:
                from graft.target.jupyter import JupyterTransport

                transport = JupyterTransport(spec)
            with transport:
                transport.stream(stream_command(1.0), on_line, stop=stop)
        except (TargetError, TransportError) as exc:
            self._report_metrics_loss(str(exc))
        except Exception as exc:
            self._report_metrics_loss(
                str(exc).strip() or "Live readings unavailable"
            )

    def _report_metrics_loss(self, message: str) -> None:
        """Post the failed reading, then drop the target state."""
        self._post_metrics(MetricsSnapshot(source="remote", error=message))
        with contextlib.suppress(Exception):
            self.app.call_from_thread(self._handle_metrics_loss, message)

    def _post_metrics(self, snapshot: MetricsSnapshot) -> None:
        # The screen may be gone while a stream is still winding down; a
        # dropped reading is not worth surfacing.
        with contextlib.suppress(Exception):
            self.app.call_from_thread(self._metrics_update, snapshot)

    def _handle_metrics_loss(self, message: str) -> None:
        """A live reading failed: the "connected" state is no longer true.

        The streaming thread has already stopped (it only reports a loss after
        its stream ended), so this must not join it: the caller is that thread,
        parked in ``call_from_thread``. Flip the panel, keep the connection
        details for the retry, and schedule the first reconnect.
        """
        if not self.is_mounted:
            return
        panel = self._gpu_panel()
        if panel is not None:
            panel.set_disconnected("TARGET CONNECTION LOST")
        # Facts describe a live target, so they are now stale; the fields still
        # say how to reach it, so the reconnect flow can reuse them.
        self._connected_facts = None
        self._stop_reconnect_timer()
        self._set_status(f"Target connection lost · {message}", is_error=True)
        self._schedule_reconnect(message)

    def _metrics_update(self, snapshot: MetricsSnapshot) -> None:
        if not self.is_mounted:
            return
        widget = self._metrics_widget()
        if widget is not None:
            widget.set_metrics(snapshot)

    def _maybe_show_tips(self, error: str) -> None:
        """Offer troubleshooting only when the failure looks like a ban or auth."""
        kind = str(self.query_one("#transport-select", Select).value)
        category = classify_connection_failure(error, kind)
        if category in {"banned", "auth"}:
            self.app.show_tips(category, kind, error)


__all__ = ["TargetScreen"]
