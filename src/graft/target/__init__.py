"""Remote AMD targets: connect, detect, and run Graft on someone else's GPU.

The UI collects connection details per request and hands them straight to a
transport. Nothing in this package writes a credential to disk, a log, or a
report; :meth:`~graft.target.spec.TargetSpec.redacted` is the only supported
way to render a target for display.
"""

from __future__ import annotations

from graft.target.dependencies import (
    Dependency,
    actionable_dependencies,
    install_dependencies,
    missing_dependencies,
    scan_dependencies,
)
from graft.target.detect import TargetFacts
from graft.target.local import LocalTransport, local_capability, local_target_facts
from graft.target.remote import RemoteRun, run_remote, run_remote_file
from graft.target.spec import TargetError, TargetSpec, parse_form
from graft.target.token import discover_jupyter_token, parse_jupyter_token
from graft.target.transport import CommandResult, TransportError, clip, quote

# graft.target.detect.detect is deliberately not re-exported here. A bare
# `detect` would shadow the submodule of the same name, so
# `from graft.target import detect` would silently yield the function.
# Callers wanting detection use run_remote(..., probe_only=True).
__all__ = [
    "CommandResult",
    "Dependency",
    "LocalTransport",
    "RemoteRun",
    "TargetError",
    "TargetFacts",
    "TargetSpec",
    "TransportError",
    "actionable_dependencies",
    "clip",
    "discover_jupyter_token",
    "install_dependencies",
    "local_capability",
    "local_target_facts",
    "missing_dependencies",
    "parse_form",
    "parse_jupyter_token",
    "quote",
    "run_remote",
    "run_remote_file",
    "scan_dependencies",
]
