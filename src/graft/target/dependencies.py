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

import re
from dataclasses import dataclass
from typing import Any

from graft.target.detect import TargetFacts
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


def _is_debian_like(facts: TargetFacts) -> bool:
    return any(name in facts.os_name.lower() for name in _DEBIAN_LIKE)


# A bare cloud image often ships python3 without pip, so the ROCm wheel install
# would fail with "No module named pip" before it ever reached PyTorch. On
# apt-based targets the command installs python3-pip first, then runs pip3.
_APT_INSTALL = "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y"


def _rocm_series(version: str) -> str:
    """Return the "X.Y" series from a ROCm/torch HIP version, or ""."""
    parts = (version or "").split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return ""


def _python_is_externally_managed(facts: TargetFacts) -> bool:
    """Whether the target's pip enforces PEP 668.

    Ubuntu 23.04+, Debian 12+, and their derivatives mark the system Python as
    externally managed, so a bare ``pip install`` refuses with
    "externally-managed-environment" and nothing gets installed. On those
    releases pip accepts ``--break-system-packages``, which is what AMD's own
    ROCm install notes use. Older releases ship a pip that does not understand
    the flag, so passing it there would replace a working install with an
    "unknown option" error; they are left alone.
    """
    name = facts.os_name.lower()
    match = re.search(r"(\d+)(?:\.(\d+))?", name)
    if not match:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if "linux mint" in name:
        # Mint 22 is based on Ubuntu 24.04 and enforces PEP 668; 21 and older
        # are based on 22.04 and do not.
        return major >= 22
    if "debian" in name:
        return major >= 12
    if "ubuntu" in name or "pop" in name:
        return (major, minor) >= (23, 4)
    return False


def _pip_break_flag(facts: TargetFacts) -> str:
    """The PEP 668 override as a leading space, or "" when it is not needed."""
    return " --break-system-packages" if _python_is_externally_managed(facts) else ""


def _torch_command(facts: TargetFacts) -> tuple[str, str]:
    """Return (command, detail) for installing PyTorch that can see the GPU.

    The ROCm wheel index is named after the ROCm series. PyTorch's own HIP
    version is the best source: a host can carry more than one ROCm stack, and
    the wheel that will work is the one matching the PyTorch the GPU already
    runs on. Only when that is unavailable does the detected ROCm version get
    used. pip is not guaranteed to exist on a fresh image, so on apt-based
    targets the command installs ``python3-pip`` first. When no series can be
    mapped the plain PyPI wheel is used, which is honest about being
    CUDA-or-CPU; the detail says so rather than pretending it is the right
    build.
    """
    python = quote(facts.python_path or "python3")
    break_flag = _pip_break_flag(facts)
    rocm_series = _rocm_series(facts.torch_hip) or _rocm_series(facts.rocm_version)
    if rocm_series:
        index = f"https://download.pytorch.org/whl/rocm{rocm_series}"
        wheels = f"torch torchvision --index-url {quote(index)}"
        if _is_debian_like(facts):
            command = (
                f"{_APT_INSTALL} python3-pip && "
                f"pip3 install{break_flag} {wheels}"
            )
        else:
            command = f"{python} -m pip install{break_flag} {wheels}"
        return command, f"PyTorch from the ROCm {rocm_series} wheel index."
    # Keep behavior consistent with tests: when unknown, fall back to PyPI
    return (
        f"{python} -m pip install{break_flag} --upgrade pip && "
        f"{python} -m pip install{break_flag} torch torchvision",
        "ROCm version unknown; installing the default PyPI wheel. Install a "
        "matching ROCm wheel by hand if GPU support is needed.",
    )


def _rocm_tool_command(facts: TargetFacts, package: str) -> tuple[str, str]:
    """Return (command, detail) for an apt-based ROCm tool install."""
    if _is_debian_like(facts):
        return (
            f"{_APT_INSTALL} {package}",
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
