"""Textual terminal application for Graft."""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, TabbedContent, TabPane

from graft.tui.screens.instructions import InstructionsScreen
from graft.tui.screens.loading import LoadingOverlay
from graft.tui.screens.results import ResultsScreen
from graft.tui.screens.target import TargetScreen
from graft.tui.screens.tips import TipsScreen
from graft.tui.widgets import GPUDock


class GraftTUI(App):
    """Migration workspace for local files and remote AMD targets."""

    TITLE = "Graft"
    SUB_TITLE = "CUDA → ROCm migration workspace"
    BINDINGS: ClassVar = [
        ("ctrl+1", "show_workspace", "Workspace"),
        ("ctrl+2", "show_results", "Results"),
        ("ctrl+3", "show_instructions", "Instructions"),
    ]

    CSS = """
    Screen {
        background: #080d15;
        color: #e5edf6;
    }

    #gpu-dock {
        dock: right;
        width: 56;
        height: auto;
        max-height: 100%;
        margin: 1 1 0 0;
        padding: 1 1 1 1;
        background: #0a111c;
        border: round #26384d;
    }

    #gpu-dock.narrow {
        width: 44;
    }

    #gpu-dock.tiny {
        width: 32;
    }

    #gpu-dock.hidden {
        display: none;
    }

    #gpu-dock.short #gpu-metrics {
        display: none;
    }

    #gpu-dock-title {
        width: 100%;
        height: 1;
        margin-bottom: 1;
        color: #70ddc8;
        text-style: bold;
        text-align: center;
    }

    AIConversionPopup {
        align: center middle;
        background: #050a12 78%;
    }

    #ai-popup-card {
        width: 96;
        max-width: 96%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round #c9a4ff;
        background: #101a29;
    }

    #ai-popup-eyebrow {
        color: #c9a4ff;
        text-style: bold;
    }

    #ai-popup-title {
        margin-top: 1;
        color: #dbe9f4;
    }

    #ai-popup-disclaimer {
        margin-top: 1;
        color: #e6c07b;
    }

    #ai-popup-scroll {
        height: auto;
        max-height: 26;
        margin-top: 1;
        padding: 0 1;
        background: #0a111c;
        scrollbar-color: #4a3a63;
        scrollbar-background: #0a111c;
    }

    #ai-popup-code {
        color: #dbe9f4;
    }

    .ai-popup-section {
        margin-top: 1;
        color: #c9a4ff;
        text-style: bold;
    }

    .ai-popup-comment-title {
        margin-top: 1;
        color: #9aabbd;
    }

    .ai-popup-comment {
        color: #dbe9f4;
    }

    #ai-popup-actions {
        height: 3;
        margin-top: 1;
        layout: horizontal;
    }

    #ai-assist-actions,
    #ai-result-actions,
    #compare-actions {
        height: 3;
        margin-top: 1;
        layout: horizontal;
    }

    SourceComparePopup {
        align: center middle;
        background: #050a12 78%;
    }

    #compare-popup-card {
        width: 96;
        max-width: 96%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round #83bfff;
        background: #101a29;
    }

    #compare-popup-eyebrow {
        color: #83bfff;
        text-style: bold;
    }

    #compare-popup-title {
        margin-top: 1;
        color: #dbe9f4;
    }

    #compare-popup-note {
        margin-top: 1;
        color: #e6c07b;
    }

    #compare-popup-scroll {
        height: auto;
        max-height: 30;
        margin-top: 1;
        padding: 0 1;
        background: #0a111c;
        scrollbar-color: #35526a;
        scrollbar-background: #0a111c;
    }

    .compare-popup-section {
        margin-top: 1;
        color: #83bfff;
        text-style: bold;
    }

    .compare-popup-path {
        color: #9aabbd;
    }

    .compare-popup-code {
        color: #dbe9f4;
    }

    #compare-popup-actions {
        height: 3;
        margin-top: 1;
        layout: horizontal;
    }

    .ai-flag {
        color: #c9a4ff;
    }

    DependencyOverlay {
        align: center middle;
        background: #050a12 78%;
    }

    #deps-card {
        width: 84;
        max-width: 96%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round #70ddc8;
        background: #101a29;
    }

    #deps-eyebrow {
        color: #70ddc8;
        text-style: bold;
    }

    #deps-title {
        margin-top: 1;
        color: #edf5fb;
    }

    #deps-list {
        height: auto;
        max-height: 20;
        margin-top: 1;
        scrollbar-color: #35526a;
        scrollbar-background: #101a29;
    }

    .dep-row {
        height: auto;
        color: #c9d9e5;
    }

    .dep-present {
        color: #79e5b0;
    }

    .dep-missing {
        color: #ffb86b;
    }

    .dep-manual {
        color: #ff9ba1;
    }

    #deps-status {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: #9aabbd;
    }

    LoadingOverlay {
        align: center middle;
        background: #050a12 78%;
    }

    #loading-card {
        width: 64;
        max-width: 90%;
        height: auto;
        padding: 1 3;
        border: round #70ddc8;
        background: #101a29;
        align: center middle;
    }

    #loading-eyebrow {
        color: #70ddc8;
        text-style: bold;
    }

    #loading-spinner {
        width: 100%;
        height: 5;
        margin-top: 1;
        color: #70ddc8;
    }

    #loading-message {
        width: 100%;
        margin-top: 1;
        color: #edf5fb;
        text-style: bold;
        text-align: center;
    }

    #loading-hint {
        width: 100%;
        margin-top: 1;
        color: #9aabbd;
        text-align: center;
    }

    TipsScreen {
        align: center middle;
        background: #050a12 78%;
    }

    #tips-card {
        width: 82;
        max-width: 94%;
        height: auto;
        max-height: 90%;
        padding: 1 3;
        border: round #f4bd55;
        background: #1a160e;
        align: center middle;
    }

    #tips-eyebrow {
        color: #f4bd55;
        text-style: bold;
    }

    #tips-title {
        margin-top: 1;
        color: #ffe0a0;
        text-style: bold;
    }

    #tips-error {
        margin-top: 1;
        color: #ff9ba1;
    }

    #tips-list {
        height: auto;
        max-height: 22;
        margin-top: 1;
        scrollbar-color: #6a5326;
        scrollbar-background: #1a160e;
    }

    .tip {
        height: auto;
        margin-top: 1;
        color: #e5edf6;
    }

    #tips-close {
        width: 100%;
        margin-top: 1;
    }

    Header {
        background: #101a29;
        color: #ecf8f6;
    }

    Footer {
        background: #0d1420;
        color: #92a3b7;
    }

    TabbedContent {
        height: 1fr;
        background: #080d15;
    }

    TabPane {
        height: 1fr;
        padding: 1 2;
    }

    #workspace-scroll {
        height: 1fr;
        scrollbar-color: #35526a;
        scrollbar-background: #0b121d;
    }

    #target-screen {
        height: 1fr;
    }

    #target-screen.compact,
    #target-screen.short {
        height: auto;
    }

    #workspace-grid {
        height: 1fr;
        grid-size: 2;
        grid-columns: 1fr 1.08fr;
        grid-rows: 1fr;
        grid-gutter: 1 2;
        padding-bottom: 1;
    }

    #workspace-grid.compact {
        height: auto;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: auto;
        grid-gutter: 1 0;
    }

    #workspace-grid.short {
        height: auto;
        grid-rows: auto;
    }

    .card {
        height: 1fr;
        min-height: 0;
        padding: 1 2;
        border: round #26384d;
        background: #101a29;
        scrollbar-color: #35526a;
        scrollbar-background: #101a29;
    }

    #workspace-grid.compact .card,
    #workspace-grid.short .card {
        height: auto;
        min-height: 12;
    }

    .eyebrow {
        color: #70ddc8;
        text-style: bold;
    }

    .hero-title {
        margin-top: 1;
        color: #f2f7fb;
        text-style: bold;
    }

    .hero-copy {
        margin-top: 1;
        color: #a5b4c5;
    }

    .ascii-logo {
        height: 7;
        margin-bottom: 1;
        color: #70ddc8;
        text-style: bold;
    }

    .ascii-logo.compact {
        height: 3;
        color: #70ddc8;
    }

    .card-title {
        margin-top: 1;
        color: #edf5fb;
        text-style: bold;
    }

    .field-label {
        margin-top: 1;
        color: #90a3b8;
        text-style: bold;
    }

    .hint {
        margin-top: 1;
        color: #9aabbd;
    }

    #gpu-visual {
        width: 1fr;
        min-width: 0;
        height: auto;
        margin-top: 1;
        padding: 1;
        border: round #26384d;
        background: #0a111c;
        color: #97a9bc;
        text-style: bold;
    }

    #gpu-visual.active {
        border: round #3b9c86;
        background: #0b1919;
        color: #70ddc8;
    }

    #gpu-visual.offline, #gpu-visual.disconnected {
        border: round #9b4853;
        background: #1a1019;
        color: #ff8992;
    }

    #gpu-metrics {
        width: 1fr;
        min-width: 0;
        height: auto;
        margin-top: 1;
        padding: 0 1;
        color: #8fd8c8;
        background: transparent;
        text-align: left;
    }

    #gpu-metrics Static {
        margin-bottom: 1;
    }

    Input {
        width: 1fr;
        height: 3;
        margin-top: 1;
        border: round #344961;
        background: #0a111c;
        color: #e5edf6;
    }

    Select {
        width: 1fr;
        height: 3;
        margin-top: 1;
        color: #e5edf6;
        background: transparent;
    }

    SelectCurrent {
        width: 1fr;
        height: 3;
        padding: 0 1;
        border: round #344961;
        background: #0a111c;
        color: #e5edf6;
    }

    SelectCurrent > Static#label {
        width: 1fr;
        color: #e5edf6;
        background: transparent;
        content-align: left middle;
    }

    SelectCurrent .arrow {
        color: #70ddc8;
        background: transparent;
    }

    Input:focus {
        border: round #70ddc8;
    }

    Select:focus > SelectCurrent {
        border: round #70ddc8;
        background: #101a29;
    }

    Select:focus > SelectCurrent > Static#label {
        color: #f2f7fb;
    }

    Grid.fields {
        height: auto;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 0 1;
    }

    Grid.fields.compact {
        grid-size: 1;
        grid-columns: 1fr;
    }

    #cloud-config,
    #ssh-config,
    #jupyter-config {
        height: auto;
    }

    .field {
        height: auto;
    }

    #jupyter-token-row {
        height: auto;
        layout: horizontal;
    }

    #detect-token {
        width: auto;
        min-width: 15;
        margin: 1 0 0 1;
    }

    .button-row {
        height: auto;
        margin-top: 1;
        layout: horizontal;
    }

    .button-row.stacked {
        layout: vertical;
    }

    Button {
        min-width: 14;
        width: 1fr;
        margin: 0 1 0 0;
        border: round #38526a;
        background: #172638;
        color: #dbe9f4;
    }

    Button.-primary {
        border: round #70ddc8;
        background: #70ddc8;
        color: #07130f;
        text-style: bold;
    }

    Button:hover {
        background: #233a50;
    }

    Button.-primary:hover {
        background: #93ebdb;
    }

    Collapsible {
        margin-top: 1;
        border-top: solid #28394b;
        background: #0d1623;
    }

    CollapsibleTitle {
        color: #70ddc8;
        text-style: bold;
    }

    .status {
        min-height: 3;
        margin-top: 1;
        padding: 0 1;
        border-left: tall #70ddc8;
        background: #0b1720;
        color: #c9d9e5;
        content-align: left middle;
    }

    .status.running {
        border-left: tall #83bfff;
        color: #c8e2ff;
    }

    .status.error {
        border-left: tall #ff7d86;
        color: #ffd0d3;
    }

    .status.success {
        border-left: tall #4de0a1;
        color: #a8f5cc;
    }

    .status.warning {
        border-left: tall #f4bd55;
        color: #ffe0a0;
    }

    .status.info {
        border-left: tall #83bfff;
        color: #c8e2ff;
    }

    #results-view {
        height: 1fr;
        padding: 0 1;
    }

    #result-heading, .result-heading {
        margin: 0 0 1 0;
        color: #70ddc8;
        text-style: bold;
    }

    #result-heading.success, .result-heading.success { color: #4de0a1; }
    #result-heading.warning, .result-heading.warning { color: #f4bd55; }
    #result-heading.error, .result-heading.error { color: #ff7d86; }
    #result-heading.info, .result-heading.info { color: #83bfff; }

    #result-summary, .result-summary {
        height: auto;
        margin-bottom: 1;
        padding: 1 2;
        border: round #26384d;
        background: #101a29;
        color: #e5edf6;
    }

    #result-summary.success, .result-summary.success {
        border: round #2e755d;
        background: #0b1917;
        color: #a8f5cc;
    }

    #result-summary.warning, .result-summary.warning {
        border: round #87642b;
        background: #1a160e;
        color: #ffe0a0;
    }

    #result-summary.error, .result-summary.error {
        border: round #87444d;
        background: #1a1014;
        color: #ffd0d3;
    }

    #result-summary.info, .result-summary.info {
        border: round #355d7d;
        background: #0e1721;
        color: #c8e2ff;
    }

    #results-body {
        height: auto;
    }

    .result-section {
        height: auto;
        margin-top: 1;
        color: #70ddc8;
        text-style: bold;
    }

    .result-kv {
        height: auto;
        color: #c9d9e5;
    }

    .finding-group {
        height: auto;
        margin-top: 1;
        color: #ffb86b;
        text-style: bold;
    }

    .finding {
        height: auto;
        color: #c9d9e5;
    }

    .source-pane {
        height: auto;
        max-height: 40;
        margin-top: 1;
        padding: 0 1;
        border: round #293d54;
        background: #0a111c;
        color: #dbe9f4;
        overflow-y: auto;
    }

    .editor-title {
        height: auto;
        margin-top: 1;
        padding: 0 1;
        background: #152438;
        color: #9eb3c8;
        text-style: bold;
    }

    .result-card {
        height: auto;
        margin-bottom: 1;
        padding: 1 2;
        border: round #26384d;
        background: #101a29;
    }

    .result-card.error {
        border: round #87444d;
        background: #1a1019;
        color: #ff9ba1;
    }

    #source-editor-grid {
        height: auto;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 0 1;
        margin-top: 1;
    }

    #source-editor-grid.stacked {
        grid-size: 1;
        grid-columns: 1fr;
    }

    .editor-pane {
        height: auto;
        min-width: 0;
        border: round #293d54;
        background: #0a111c;
    }

    .editor-title {
        height: 3;
        padding: 0 1;
        background: #152438;
        color: #9eb3c8;
        text-style: bold;
        content-align: left middle;
    }

    #original-source, #converted-source {
        height: 22;
        border: none;
        background: #0a111c;
    }

    #save-row {
        height: 3;
        margin-top: 1;
        layout: horizontal;
    }

    #save-row.stacked {
        height: auto;
        layout: vertical;
    }

    #output-path {
        width: 1fr;
        margin-right: 1;
    }

    #save-output {
        width: auto;
        min-width: 24;
    }

    #save-row.stacked #output-path {
        width: 1fr;
        margin-right: 0;
    }

    #save-row.stacked #save-output {
        width: 1fr;
        margin-top: 1;
    }

    .inline-status {
        height: auto;
        margin-top: 1;
        color: #9aabbd;
    }

    .inline-status.error {
        color: #ff9ba1;
    }

    .inline-status.success { color: #79e5b0; }
    .inline-status.warning { color: #f4c66d; }
    .inline-status.info { color: #83bfff; }

    #source-path-row {
        height: 3;
        layout: horizontal;
        align: left middle;
    }

    #source-path-row Input {
        width: 1fr;
        margin-right: 1;
    }

    #source-path-row Button {
        width: auto;
        min-width: 10;
    }

    #source-path-row.stacked {
        height: auto;
        layout: vertical;
    }

    #source-path-row.stacked Input,
    #source-path-row.stacked Button {
        width: 1fr;
        margin-right: 0;
    }

    #comparison-actions {
        height: 3;
        margin-top: 1;
        layout: horizontal;
        align: left middle;
    }

    #comparison-actions #comparison-status {
        width: 1fr;
        margin-top: 0;
    }

    #comparison-actions Button {
        width: auto;
        min-width: 22;
    }

    #comparison-actions.stacked {
        height: auto;
        layout: vertical;
    }

    #comparison-actions.stacked Button {
        width: 1fr;
        margin-top: 1;
    }

    #ai-review-view {
        height: auto;
        padding: 0 1;
        background: #0d1623;
        color: #d5e0ec;
    }

    #ai-review-view.success { color: #a8f5cc; }
    #ai-review-view.warning { color: #ffe0a0; }
    #ai-review-view.error { color: #ffd0d3; }
    #ai-review-view.info { color: #c8e2ff; }

    #report-view {
        height: auto;
        padding: 1 2;
        border: round #26384d;
        background: #0e1724;
        color: #d5e0ec;
    }

    #report-view.success {
        border: round #2e755d;
        color: #d0f3df;
    }

    #report-view.warning {
        border: round #87642b;
        color: #f4e2bd;
    }

    #report-view.error {
        border: round #87444d;
        color: #f6d0d2;
    }

    #report-view.info {
        border: round #355d7d;
        color: #d2e8fa;
    }

    #instructions-view {
        height: 1fr;
        padding: 0 1;
        scrollbar-color: #35526a;
        scrollbar-background: #0b121d;
    }

    #instructions-title {
        margin-top: 1;
        color: #f2f7fb;
        text-style: bold;
    }

    #instructions-intro {
        margin-top: 1;
        margin-bottom: 1;
        color: #a5b4c5;
    }

    .instructions-section {
        margin-top: 1;
        color: #70ddc8;
        text-style: bold;
    }

    .instructions-step {
        height: auto;
        margin-top: 1;
        color: #c9d9e5;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        self._loading_overlay: LoadingOverlay | None = None
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Header()
        yield GPUDock(id="gpu-dock")
        with TabbedContent(initial="workspace", id="main-tabs"):
            with (
                TabPane("Workspace", id="workspace"),
                VerticalScroll(id="workspace-scroll"),
            ):
                yield TargetScreen(id="target-screen")
            with TabPane("Results", id="results"):
                yield ResultsScreen(id="results-view")
            with TabPane("Instructions", id="instructions"):
                yield InstructionsScreen(id="instructions-view")
        yield Footer()

    def on_mount(self) -> None:
        self._resize_gpu_dock(self.size)

    def on_resize(self, event: events.Resize) -> None:
        self._resize_gpu_dock(event.size)

    def _resize_gpu_dock(self, size) -> None:
        try:
            dock = self.query_one("#gpu-dock")
        except Exception:
            return
        width, height = size.width, size.height
        dock.set_class(width < 64, "hidden")
        dock.set_class(64 <= width < 88, "tiny")
        dock.set_class(88 <= width < 112, "narrow")
        dock.set_class(height < 28, "short")

    def show_gpu_facts(self, method: str, *args) -> None:
        """Drive the fixed GPU dock from any screen.

        The dock lives at app level so it survives tab switches; screens
        reach it through this method rather than reaching into the widget
        tree, which keeps their own queries local.
        """
        from graft.tui.widgets import GPUMetrics, GPUStatusPanel

        try:
            if method == "set_metrics":
                widget = next(iter(self.query(GPUMetrics)), None)
                if widget is not None:
                    widget.set_metrics(*args)
                return
            panel = next(iter(self.query(GPUStatusPanel)), None)
            if panel is not None:
                getattr(panel, method)(*args)
        except Exception:
            pass

    def action_show_workspace(self) -> None:
        self.query_one(TabbedContent).active = "workspace"

    def action_show_results(self) -> None:
        self.query_one(TabbedContent).active = "results"

    def action_show_instructions(self) -> None:
        self.query_one(TabbedContent).active = "instructions"

    def show_loading(self, message: str) -> None:
        """Push or refresh the blocking overlay for a long-running action."""
        if self._loading_overlay is not None:
            self._loading_overlay.set_message(message)
            return
        overlay = LoadingOverlay(message)
        self._loading_overlay = overlay
        self.push_screen(overlay)

    def hide_loading(self) -> None:
        """Dismiss the active operation overlay after its worker completes."""
        overlay = self._loading_overlay
        self._loading_overlay = None
        if overlay is not None and self.screen is overlay:
            self.pop_screen()

    def show_tips(self, category: str, kind: str, error: str) -> None:
        """Open the connection help panel for a classified failure."""
        self.push_screen(TipsScreen(category, kind, error))


def run() -> None:
    """Run the terminal application."""
    GraftTUI().run()


if __name__ == "__main__":
    run()
