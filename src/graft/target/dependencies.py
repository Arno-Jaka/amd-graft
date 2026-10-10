"""Decide which tools a connected target is missing, and install them on request.

The probe already reports what it found; this module turns that into a short
checklist and, for the pieces Graft can install without guessing, the exact
command to install them. Nothing runs automatically: the UI shows what is
missing and a person presses a button, because the installs are system-wide
and can be large.

Every install command is built here from a fixed table and the probe's own
paths, never from free-form input, and the paths are shell-quoted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from graft.target.detect import TargetFacts
from graft.target.spec import TargetSpec
from graft.target.transport import quote

# Only these OS families are assumed to have apt. Anything else gets a manual
# hint rather than a command that would fail confusingly.
_DEBIAN_LIKE = ("ubuntu", "debian", "linux mint", "pop!_os")


@dataclass(frozen=True, slots=True)
class Dependency:
    """One tool Graft needs on the target, and whether it is there."""

    key: str
    label: str
    present: bool
    required: bool
    detail: str = ""
    # A shell command to install it, or "" when it has to be installed by hand.
    command: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "present": self.present,
            "required": self.required,
            "detail": self.detail,
            "installable": bool(self.command),
        }


def _torch_command(facts: TargetFacts) -> tuple[str, str]:
    """Return (command, detail) for installing PyTorch that can see the GPU.

    The ROCm wheel index is named after the ROCm series, so it is derived from
    the reported version. When that version cannot be mapped to a known index
    the plain PyPI wheel is used, which is honest about being CUDA-or-CPU; the
    detail says so rather than pretending it is the right build.
    """
    python = quote(facts.python_path or "python3")
    version = facts.rocm_version or ""
    parts = version.split(".")
    # Prefer known ROCm series if version looks like X.Y or X.Y.Z
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        rocm_series = f"{parts[0]}.{parts[1]}"
        index = f"https://download.pytorch.org/whl/rocm{rocm_series}"
        return (
            f"{python} -m pip install --upgrade pip && "
            f"{python} -m pip install torch --index-url {quote(index)}",
            f"PyTorch from the ROCm {rocm_series} wheel index.",
        )
    # Keep behavior consistent with tests: when unknown, fall back to PyPI
    return (
        f"{python} -m pip install --upgrade pip && "
        f"{python} -m pip install torch torchvision",
        "ROCm version unknown; installing the default PyPI wheel. Install a "
        "matching ROCm wheel by hand if GPU support is needed.",
    )


def _rocm_tool_command(facts: TargetFacts, package: str) -> tuple[str, str]:
    """Return (command, detail) for an apt-based ROCm tool install."""
    if any(name in facts.os_name.lower() for name in _DEBIAN_LIKE):
        return (
            "apt-get update && DEBIAN_FRONTEND=noninteractive "
            f"apt-get install -y {package}",
            f"Installed with apt ({package}).",
        )
    return (
        "",
        f"{facts.os_name or 'This OS'} is not apt-based; install {package} "
        "with the system package manager.",
    )


def scan_dependencies(facts: TargetFacts) -> list[Dependency]:
    """Build the checklist of tools Graft wants on a target."""
    deps: list[Dependency] = []

    python_ok = bool(facts.python_path) and facts.python_path != "none"
    deps.append(
        Dependency(
            key="python",
            label="Python 3",
            present=python_ok,
            required=True,
            detail=facts.python_path or "not found",
        )
    )

    hipify_present = "hipify-perl" in facts.tools
    hipify_command, hipify_detail = (
        ("", "Present.")
        if hipify_present
        else _rocm_tool_command(facts, "rocm-hip-sdk")
    )
    deps.append(
        Dependency(
            key="hipify-perl",
            label="hipify-perl",
            present=hipify_present,
            required=True,
            detail=hipify_detail,
            command=hipify_command,
        )
    )

    hipcc_present = "hipcc" in facts.tools
    hipcc_command, hipcc_detail = (
        ("", "Present.") if hipcc_present else _rocm_tool_command(facts, "hipcc")
    )
    deps.append(
        Dependency(
            key="hipcc",
            label="hipcc (ROCm compiler)",
            present=hipcc_present,
            required=True,
            detail=hipcc_detail,
            command=hipcc_command,
        )
    )

    deps.append(
        Dependency(
            key="rocm",
            label="ROCm runtime",
            present=bool(facts.rocm_version) or bool(facts.gfx_arch),
            required=False,
            detail=facts.rocm_version or "version not reported",
        )
    )

    torch_present = bool(facts.torch_version)
    if torch_present:
        torch_command, torch_detail = "", f"torch {facts.torch_version}"
    else:
        torch_command, torch_detail = _torch_command(facts)
    deps.append(
        Dependency(
            key="torch",
            label="PyTorch (ROCm)",
            present=torch_present,
            required=False,
            detail=torch_detail,
            command=torch_command,
        )
    )
    return deps


def missing_dependencies(facts: TargetFacts) -> list[Dependency]:
    """Only the required tools that are absent, the actionable subset."""
    return [
        dep
        for dep in scan_dependencies(facts)
        if dep.required and not dep.present
    ]


def actionable_dependencies(facts: TargetFacts) -> list[Dependency]:
    """Absent tools the user can do something about.

    That is a required gap with no automatic install (a manual hint) or any
    gap Graft can install, including the optional PyTorch.
    """
    return [
        dep
        for dep in scan_dependencies(facts)
        if not dep.present and (dep.command or dep.required)
    ]


ProgressHook = Callable[[str, str], None]
ResultHook = Callable[[dict[str, Any]], None]

# Local installs have no TargetSpec to take a timeout from. Two minutes per
# command matches the remote default's spirit without making a slow apt run
# look like a hang.
_LOCAL_TIMEOUT_S = 1800


def install_dependencies(
    transport,
    spec: TargetSpec | None,
    facts: TargetFacts,
    *,
    on_progress: ProgressHook | None = None,
    on_result: ResultHook | None = None,
) -> list[dict[str, Any]]:
    """Install the missing, installable dependencies and return the outcomes.

    ``spec`` may be ``None`` for a local target, where the transport runs the
    same fixed commands on this machine. ``on_progress`` fires before each
    install and ``on_result`` after each one, so a UI can show live progress
    rather than one silent batch at the end.

    Non-installable gaps are returned with ``ok`` false and their manual hint,
    so the caller can show the whole story rather than a silent no-op.
    """
    timeout_s = spec.command_timeout_s if spec is not None else _LOCAL_TIMEOUT_S
    results: list[dict[str, Any]] = []
    for dep in scan_dependencies(facts):
        if dep.present:
            continue
        if not dep.command:
            # Nothing we can run. Only report the gaps that matter.
            if dep.required:
                outcome = {
                    "key": dep.key,
                    "label": dep.label,
                    "ok": False,
                    "error": dep.detail or "Install by hand.",
                }
                results.append(outcome)
                if on_result is not None:
                    on_result(outcome)
            continue
        if on_progress is not None:
            on_progress(dep.key, f"Installing {dep.label}…")
        result = transport.run(dep.command, timeout_s=timeout_s)
        outcome = {
            "key": dep.key,
            "label": dep.label,
            "ok": result.ok,
            "output": result.summary(800) if result.ok else "",
            "error": "" if result.ok else result.summary(800),
        }
        results.append(outcome)
        if on_result is not None:
            on_result(outcome)
    return results


__all__ = [
    "Dependency",
    "ProgressHook",
    "ResultHook",
    "actionable_dependencies",
    "install_dependencies",
    "missing_dependencies",
    "scan_dependencies",
]
