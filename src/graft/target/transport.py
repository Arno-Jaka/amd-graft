"""The interface every remote target implements.

Keeping this abstract is what lets the detection and orchestration logic be
tested without a network, and what lets the UI offer SSH and Jupyter without
either implementation leaking into the view layer.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Protocol

# Remote output is captured, so it has to be bounded. A chatty toolchain can
# emit megabytes of warnings, and an unbounded read is how a UI request turns
# into an out-of-memory kill.
MAX_OUTPUT_CHARS = 200_000
TRUNCATION_NOTICE = "\n... [output truncated] ...\n"


def clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Bound captured output, keeping the head and tail.

    The tail matters more than it looks: compilers put diagnostics at the end,
    so discarding the end of a log throws away the part that explains the
    failure.
    """
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + TRUNCATION_NOTICE + text[-tail:]


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of one remote command."""

    command: str
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """stdout and stderr together, which is what a log usually means."""
        if not self.stderr.strip():
            return self.stdout
        if not self.stdout.strip():
            return self.stderr
        return f"{self.stdout}\n{self.stderr}"

    def summary(self, limit: int = 400) -> str:
        """A short tail, for surfacing a failure in the UI."""
        body = self.output.strip() or "(no output)"
        return body if len(body) <= limit else f"...{body[-limit:]}"


class TransportError(RuntimeError):
    """The remote could not be reached or driven. Safe to show the user.

    Implementations must never put a credential in the message: these are
    rendered in the UI and written to reports.
    """


class Transport(Protocol):
    """Everything the rest of Graft needs from a remote machine."""

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
    ) -> CommandResult:
        """Run a shell command and capture its output."""

    def stream(
        self,
        command: str,
        on_line: Callable[[str], None],
        *,
        stop: Event,
    ) -> None:
        """Run a long-lived command, handing each output line to ``on_line``.

        Returns once the command ends or ``stop`` is set. Used for live
        readings, where a websocket or a persistent channel avoids reopening a
        connection on every refresh.
        """

    def push(self, local: bytes, remote_path: str) -> None:
        """Write a file to an absolute path on the remote."""

    def resolve_remote_dir(self, remote_dir: str) -> str:
        """Map a configured remote directory to an absolute path on the target.

        An SSH login sees the whole filesystem and returns the path unchanged.
        A Jupyter server serves files from one root, so the path is joined onto
        that root; otherwise an upload and the shell that consumes it name
        different files. Callers that may not have it fall back to the path.
        """

    def probe(self) -> None:
        """Raise :class:`TransportError` unless the remote is usable."""


def quote(value: str) -> str:
    """Shell-quote a single value for interpolation into a command."""
    return shlex.quote(value)
