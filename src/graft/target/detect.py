"""Work out what a remote machine actually is, before trusting it.

The point of this module is to answer one question honestly: can this host
really run ported HIP, and if so on what. It reports what it found and what
it could not find, rather than a single pass/fail, because a partially
usable target is common and the operator needs to see which half is missing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from graft.target.transport import CommandResult, TransportError

# Probes are cheap and independent, so they are issued as one script and
# marked rather than run one at a time. Each prints "<<<KEY>>>value".
PROBE_SCRIPT = r"""
set +e
emit() { printf '<<<%s>>>%s\n' "$1" "$2"; }

if [ -r /etc/os-release ]; then
  . /etc/os-release 2>/dev/null
  emit os "${PRETTY_NAME:-${NAME:-unknown}}"
else
  emit os "unknown"
fi
emit arch "$(uname -m 2>/dev/null || echo unknown)"

# PATH is not trustworthy on a host we did not configure: a non-login shell
# routinely misses a versioned ROCm bin directory, and reporting "hipcc not
# found" on a fully installed ROCm is worse than useless. Search the install
# layout directly.
find_tool() {
  for candidate in $(command -v "$1" 2>/dev/null) \
                   /opt/venv/bin/"$1" \
                   /opt/rocm/bin/"$1" \
                   /opt/rocm/core-*/bin/"$1" \
                   /opt/rocm-*/bin/"$1"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

rocmcmd="$(find_tool hipcc)"
if [ -n "$rocmcmd" ]; then emit rocm_path "$rocmcmd"; fi

# ROCm version. rocminfo prints a "Version:" line for *every* agent, including
# the CPU, so scraping the first match reports whatever the CPU reported. Read
# the version file instead and fall back to hipcc's own banner.
rocm_ver=""
for version_file in /opt/rocm/.info/version /opt/rocm/core-*/.info/version; do
  if [ -r "$version_file" ]; then
    rocm_ver="$(head -1 "$version_file" 2>/dev/null | tr -d '\r')"
    [ -n "$rocm_ver" ] && break
  fi
done
if [ -z "$rocm_ver" ] && [ -n "$rocmcmd" ]; then
  rocm_ver="$("$rocmcmd" --version 2>/dev/null | sed -n 's/.*ROCm version: *\([0-9][0-9.]*\).*/\1/p' | head -1)"
fi
if [ -n "$rocm_ver" ]; then emit rocm "$rocm_ver"; fi

for tool in hipcc hipify-perl hipify-clang nvcc rocminfo rocm-smi; do
  found="$(find_tool "$tool")"
  if [ -n "$found" ]; then emit "tool_$tool" "$found"; fi
done

# GPU agents only. rocminfo lists a CPU agent beside the GPUs and its Name
# comes first, so counting every agent yields the CPU as "GPU 1".
# One line per GPU agent: "<marketing name>|<gfx>". rocminfo reports a CPU
# agent alongside the GPUs, and within an agent the Device Type line comes
# after Name and Marketing Name, so the block has to be buffered before it can
# be classified. The gfx target is taken from the agent's own Name rather than
# scraped from the ISA strings, which list several targets (gfx942 and the
# gfx9-4-generic compatibility alias) and sort to the wrong one.
rocminfo_bin="$(find_tool rocminfo)"
if [ -n "$rocminfo_bin" ]; then
  agents="$("$rocminfo_bin" 2>/dev/null | awk '
    function flush() {
      if (isgpu == 1 && mname != "") print mname "|" gname
      mname = ""; gname = ""; isgpu = 0
    }
    /^Agent[[:space:]]/ { flush(); next }
    /^[[:space:]]*Device Type:[[:space:]]*GPU/ { isgpu = 1; next }
    /^[[:space:]]*Marketing Name:/ {
      t = $0; sub(/^[^:]*:[[:space:]]*/, "", t); sub(/[[:space:]]+$/, "", t)
      mname = t; next
    }
    /^[[:space:]]*Name:/ {
      if (gname == "") { t = $0; sub(/^[^:]*:[[:space:]]*/, "", t); sub(/[[:space:]]+$/, "", t); gname = t }
      next
    }
    END { flush() }
  ')"
  if [ -n "$agents" ]; then emit agents "$agents"; fi
  emit device_report "$("$rocminfo_bin" 2>/dev/null | grep -E '^Agent|Name:|Device Type|Marketing|Vendor' | head -40)"
fi

# KFD topology comes straight from sysfs and needs no rocminfo. Gating it on
# rocminfo made a missing diagnostic tool look like a missing GPU.
kfd_nodes="$(grep -l 'gfx_target_version' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l | tr -d ' ')"
emit kfd_gpus "$kfd_nodes"
emit node_report "$(grep -h 'gfx' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | sort -u | head -12)"
emit kfd_device "$( [ -e /dev/kfd ] && echo present || echo absent )"

if [ -n "$rocmcmd" ]; then
  emit toolchain_banner "$("$rocmcmd" --version 2>&1 | head -8)"
fi

# VRAM arrives in different units across ROCm releases: newer rocm-smi prints a
# byte count, older prints a suffixed value. Send the number with its unit and
# let the parser convert, rather than assuming one shape.
vram=""
if command -v rocm-smi >/dev/null 2>&1; then
  vram="$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2 { gsub(/[ \t]/, "", $2); print $2 }')"
  if [ -z "$vram" ]; then
    vram="$(rocm-smi --showmeminfo vram 2>/dev/null \
      | sed -n 's/.*Total.*Memory[^0-9]*\([0-9][0-9]*\).*/\1/p' | head -1)"
  fi
fi
if [ -n "$vram" ]; then emit vram "$vram"; fi

# Same trap as the toolchain: a ROCm image that ships PyTorch usually puts it
# in a virtualenv, and the system interpreter cannot see it. Asking one
# interpreter reports "no PyTorch" on a host that has it.
find_python() {
  # The GPU image usually puts PyTorch in a virtualenv or conda env, and a
  # non-login shell's PATH does not point at it. Check the interpreter on PATH
  # and the common install layouts, then fall back to shallow globs of typical
  # venv roots. Globs stay bounded; a whole-filesystem find can run for minutes
  # on a remote host and returns interpreters the host itself cannot use.
  for candidate in \
      "$(command -v python3 2>/dev/null)" \
      "$(command -v python 2>/dev/null)" \
      /opt/venv/bin/python3 \
      /opt/conda/bin/python3 \
      /opt/conda3/bin/python3 \
      /opt/*/bin/python3 \
      /root/*/bin/python3 \
      /home/*/*/bin/python3 \
      /usr/local/bin/python3 \
      /usr/bin/python3.[0-9]* \
      /usr/bin/python3; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      if "$candidate" -c 'import torch' >/dev/null 2>&1; then
        printf '%s' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

pybin="$(find_python)"
emit python "${pybin:-$(command -v python3 2>/dev/null || echo none)}"
emit torch "$(if [ -n "$pybin" ]; then "$pybin" -c 'import torch;v=torch.__version__;ok=torch.cuda.is_available();n=torch.cuda.get_device_name(0) if ok else "none";print(v+chr(124)+n)' 2>/dev/null || true; fi)"
"""


# How much of the raw probe output is kept for the diagnostics panel. The
# tail matters more than the head: the interesting lines are the last ones a
# broken toolchain printed.
MAX_RAW_CHARS = 4000


@dataclass(frozen=True, slots=True)
class TargetFacts:
    """What was found on a remote host, plus what it means for a run."""

    ok: bool
    os_name: str = "unknown"
    arch: str = "unknown"
    rocm_version: str = ""
    rocm_path: str = ""
    gfx_arch: str = ""
    gpu_count: int = 0
    gpu_name: str = ""
    vram_mb: int = 0
    tools: dict[str, str] = field(default_factory=dict)
    torch_version: str = ""
    torch_gpu: str = ""
    device_report: str = ""
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: str = ""
    python_path: str = ""

    def summary_line(self) -> str:
        """One line for a heading or a log."""
        if not self.gpu_count:
            return f"no GPU detected on {self.os_name}"
        name = self.gpu_name or self.torch_gpu
        if name and "not available" not in name.lower():
            return f"{name} on {self.os_name}"
        where = self.gfx_arch or "unknown arch"
        return f"{self.gpu_count}x GPU ({where}) on {self.os_name}"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # The raw transcript can be long and may echo an environment; it is
        # kept for debugging but trimmed for anything user-facing.
        data["raw"] = self.raw[-MAX_RAW_CHARS:]
        return data


_MARKER = re.compile(r"^<<<([A-Za-z0-9_-]+)>>>(.*)$")


def _parse_probe(stdout: str) -> dict[str, str]:
    """Collect ``<<<key>>>value`` pairs, letting a value span several lines.

    A marker delimits the field that follows it, so text up to the next marker
    belongs to it. Matching each line independently instead would silently keep
    only the first line of any multi-line value and scatter the rest into no
    field at all, which is how a truncated device report still looked plausible.
    """
    fields: dict[str, str] = {}
    key: str | None = None
    buffer: list[str] = []

    for line in stdout.splitlines():
        found = _MARKER.match(line)
        if found:
            if key is not None:
                fields[key] = "\n".join(buffer).strip()
            key = found.group(1)
            first = found.group(2)
            buffer = [first] if first.strip() else []
        elif key is not None:
            buffer.append(line)
    if key is not None:
        fields[key] = "\n".join(buffer).strip()
    return fields


def _as_int(value: str | None) -> int:
    try:
        return int((value or "").strip())
    except (TypeError, ValueError):
        return 0


# rocm-smi has reported VRAM both as a bare byte count and as a suffixed value
# depending on the release. A byte count read as MB turns a 192 GB card into
# "205 petabytes", which is worse than reporting nothing.
_MEMORY_UNITS = {
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
}


def _to_mb(value: str | None) -> int:
    """Convert a reported memory figure to whole MB.

    A bare number is treated as bytes, which is what current rocm-smi prints;
    a suffixed number is converted from its own unit.
    """
    text = (value or "").strip().replace(",", "")
    if not text:
        return 0
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)$", text)
    if not match:
        return 0
    number, unit = match.group(1), match.group(2).lower()
    if not unit:
        return int(float(number) / (1024**2))
    factor = _MEMORY_UNITS.get(unit)
    if factor is None:
        return 0
    return int(float(number) * factor / (1024**2))


def parse_gpu_agents(agents: str) -> list[tuple[str, str]]:
    """Split the probe's GPU agent lines into (name, gfx) pairs."""
    found: list[tuple[str, str]] = []
    for line in agents.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, gfx = line.partition("|")
        name = name.strip()
        if name:
            found.append((name, gfx.strip()))
    return found


def detect(transport) -> TargetFacts:
    """Probe a connected transport and interpret what comes back.

    Never raises for a merely unhelpful host: an unusable target is reported
    through ``ok`` and ``problems`` so the UI can explain it. Only a broken
    connection raises, as :class:`TransportError`.
    """
    result: CommandResult = transport.run(PROBE_SCRIPT)
    if not result.ok and not result.output.strip():
        raise TransportError(
            "The target accepted the connection but returned nothing for the "
            "hardware probe."
        )

    fields = _parse_probe(result.stdout)
    problems: list[str] = []
    notes: list[str] = []

    tools = {
        name[len("tool_") :]: path
        for name, path in fields.items()
        if name.startswith("tool_")
    }

    # A CPU-only host can still translate CUDA to HIP, because hipify-perl is
    # a text rewriter and hipcc cross-compiles without a device present. Only
    # running kernels needs a GPU, so a missing one is a note, not a failure.
    agents = parse_gpu_agents(fields.get("agents", ""))
    if agents:
        gpu_count = len(agents)
        gpu_name = agents[0][0]
        gfx_arch = next((gfx for _, gfx in agents if gfx), "")
    else:
        # KFD exposes one node per GPU device, and only GPU nodes carry a
        # gfx_target_version property.
        gpu_count = _as_int(fields.get("kfd_gpus"))
        gpu_name = ""
        gfx_arch = ""
    vram_mb = _to_mb(fields.get("vram"))
    if not gpu_count:
        # rocm-smi reports a total independent of whether any GPU agent was
        # found, so a host with no visible device still yields a plausible
        # byte count. "0 GPUs, 192 GB of VRAM" is a self-contradiction, and a
        # reader skimming the panel would take the VRAM as proof a card is
        # there. With no device there is no VRAM to report.
        if vram_mb:
            notes.append(
                f"Ignored a reported {vram_mb} MB VRAM figure: no GPU device is "
                "visible, so that number cannot describe this host."
            )
        vram_mb = 0
        kfd = fields.get("kfd_device", "")
        if kfd:
            notes.append(
                f"/dev/kfd is {kfd}. Without it the ROCm compute stack cannot "
                "reach a GPU even when one is fitted."
            )
        notes.append(
            "No AMD GPU visible. CUDA-to-HIP translation and cross-compiling "
            "still work here, but executing the ported kernel and measuring "
            "benchmarks will not."
        )

    if "hipify-perl" not in tools:
        problems.append(
            "hipify-perl not found. Official CUDA-to-HIP translation needs it; "
            "without it Graft falls back to its own symbol table."
        )
    if "hipcc" not in tools:
        problems.append(
            "hipcc not found. Graft cannot compile the translated source to a "
            "runnable binary on this target."
        )
    if fields.get("python") in (None, "", "none"):
        problems.append("No python3 on the target, so Graft cannot run there.")

    rocm_version = fields.get("rocm", "")
    if not rocm_version:
        notes.append(
            "ROCm version not reported. Translation and compilation can still "
            "be checked, but GPU execution needs a working ROCm runtime."
        )

    torch_version = fields.get("torch", "")
    torch_gpu = ""
    if torch_version:
        # torch_version carries "<version>|<device>" from the inline probe.
        raw_version, _, torch_gpu = torch_version.partition("|")
        torch_version = raw_version.strip()
        if torch_gpu and "not available" in torch_gpu.lower():
            notes.append(
                "PyTorch reports no usable GPU. Its benchmarks will be "
                "skipped rather than silently measured on CPU."
            )

    return TargetFacts(
        ok=not problems,
        os_name=fields.get("os", "unknown"),
        arch=fields.get("arch", "unknown"),
        rocm_version=rocm_version,
        rocm_path=fields.get("rocm_path", ""),
        gfx_arch=gfx_arch,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        tools=tools,
        torch_version=torch_version,
        torch_gpu=torch_gpu,
        device_report=fields.get("device_report", "")[:2000],
        problems=problems,
        notes=notes,
        raw=result.output,
        python_path=fields.get("python", ""),
    )
