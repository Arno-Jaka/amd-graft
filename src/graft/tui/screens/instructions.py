"""The Instructions tab: how to use Graft, in the order it is meant to happen.

The guide is plain data so its wording can be checked without a terminal, and
the screen is only the layout of it. Keeping the steps here rather than in the
app also keeps the buttons they name next to the tests that assert they match.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

GUIDE_TITLE = "How to use Graft"
GUIDE_INTRO = (
    "Graft turns a CUDA source file into a port plus the measured evidence that "
    "the port behaves. Work down this list: each step says what it needs and "
    "what it produces."
)

GUIDE_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "1 · Convert a file on this machine",
        (
            "Enter a .py or .cu path under SOURCE FILE, or press Browse to pick one.",
            "Press Convert & review. Graft rewrites the CUDA call surface and runs the parity cases on the CPU.",
            "The port and its evidence appear on the Results tab.",
        ),
    ),
    (
        "2 · Read the evidence",
        (
            "Open the Results tab with Ctrl+2: verdict, parity totals, benchmark ratios and findings.",
            "A stage that did not run reads \u201cnot measured\u201d rather than borrowing a number.",
            "Press Save converted file to write the port to disk.",
        ),
    ),
    (
        "3 · Connect an AMD GPU",
        (
            "Under COMPUTE TARGET set GPU location to Remote AMD target.",
            "Choose SSH private key or Jupyter server and fill in the connection fields. Detect token finds a Jupyter token over SSH.",
            "Press Detect target to read the machine capabilities. Nothing is uploaded yet.",
        ),
    ),
    (
        "4 · Run on the GPU",
        (
            "Verify on GPU uploads this project and runs the HIP verification suite on the remote device.",
            "Upload & run converted file runs the file from step 1 on the GPU and reports its output.",
            "If the dependency scan reports a gap, the overlay names what is missing so you can install it on the target.",
        ),
    ),
    (
        "5 · When something is refused",
        (
            "A refused connection opens a help panel naming the likely cause and its fix.",
            "Ctrl+1 returns to the workspace, Ctrl+2 to the results, Ctrl+3 to this guide.",
        ),
    ),
)


class InstructionsScreen(VerticalScroll):
    """A scrollable walkthrough of the whole workflow."""

    def compose(self) -> ComposeResult:
        yield Static("GETTING STARTED", classes="eyebrow")
        yield Static(GUIDE_TITLE, id="instructions-title")
        yield Static(GUIDE_INTRO, id="instructions-intro")
        for heading, steps in GUIDE_SECTIONS:
            yield Static(heading, classes="instructions-section")
            for index, step in enumerate(steps, start=1):
                yield Static(
                    f"  {index}. {step}",
                    classes="instructions-step",
                    markup=False,
                )


__all__ = ["GUIDE_SECTIONS", "GUIDE_TITLE", "InstructionsScreen"]
