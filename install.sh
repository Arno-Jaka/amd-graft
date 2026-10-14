#!/bin/sh
# Graft installer.
#
#   curl -fsSL https://raw.githubusercontent.com/Arno-Jaka/graft/main/install.sh | sh
#
# What it does, in order:
#   1. checks the platform and the tools it needs (curl, or git as a bonus)
#   2. installs uv if it is not already present
#   3. puts a Graft checkout in $GRAFT_HOME (default ~/.graft)
#   4. creates the project virtualenv with `uv sync`
#   5. links a `graft` launcher onto your PATH
#   6. verifies the launcher and tells you what to run next
#
# It is deliberately boring: POSIX sh, no dependencies beyond curl and uv, and
# every step is printed so a failure names the step that broke.

set -eu

REPO="Arno-Jaka/graft"
REPO_URL="https://github.com/${REPO}.git"
REF="${GRAFT_REF:-main}"
GRAFT_HOME="${GRAFT_HOME:-$HOME/.graft}"
RAW_URL="https://raw.githubusercontent.com/${REPO}/${REF}/install.sh"

# ---- pretty output ---------------------------------------------------------

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$(printf '\033[1m')
    DIM=$(printf '\033[2m')
    GREEN=$(printf '\033[32m')
    CYAN=$(printf '\033[36m')
    YELLOW=$(printf '\033[33m')
    RED=$(printf '\033[31m')
    RESET=$(printf '\033[0m')
else
    BOLD= DIM= GREEN= CYAN= YELLOW= RED= RESET=
fi

step() { printf '%s==>%s %s\n' "$CYAN" "$RESET" "$1"; }
ok()   { printf '%s ok %s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '%swarn%s %s\n' "$YELLOW" "$RESET" "$1" >&2; }
die()  { printf '%serr %s %s\n' "$RED" "$RESET" "$1" >&2; exit 1; }

banner() {
    printf '%s' "$CYAN"
    cat <<'ASCII'
  ██████╗ ██████╗  █████╗ ███████╗████████╗
 ██╔════╝ ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
 ██║  ███╗██████╔╝███████║█████╗     ██║
 ██║   ██║██╔══██╗██╔══██║██╔══╝     ██║
 ╚██████╔╝██║  ██║██║  ██║██║        ██║
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝
ASCII
    printf '%s' "$RESET"
    printf '  %sCUDA → ROCm migration with correctness verification%s\n\n' "$DIM" "$RESET"
}

usage() {
    cat <<EOF
Graft installer

Usage:
  curl -fsSL $RAW_URL | sh
  sh install.sh [options]

Options:
  --ref <name>     Git ref to install (branch or tag). Default: ${REF}
  --dir <path>     Where to put the checkout. Default: \$GRAFT_HOME or ~/.graft
  --bin <path>     Where to link the launcher. Default: first writable of
                   /usr/local/bin, ~/.local/bin
  --no-modify-path Do not print PATH instructions
  --uninstall      Remove the checkout and the launcher, then exit
  -h, --help       Show this help

Environment:
  GRAFT_HOME       Same as --dir
  GRAFT_REF        Same as --ref
  GRAFT_BIN_DIR    Same as --bin
  NO_COLOR         Set to disable colour
EOF
}

# ---- options ---------------------------------------------------------------

BIN_DIR="${GRAFT_BIN_DIR:-}"
DO_UNINSTALL=0
MODIFY_PATH=1

while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="$2"; shift 2 ;;
        --ref=*) REF="${1#*=}"; shift ;;
        --dir) GRAFT_HOME="$2"; shift 2 ;;
        --dir=*) GRAFT_HOME="${1#*=}"; shift ;;
        --bin) BIN_DIR="$2"; shift 2 ;;
        --bin=*) BIN_DIR="${1#*=}"; shift ;;
        --no-modify-path) MODIFY_PATH=0; shift ;;
        --uninstall) DO_UNINSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (try --help)" ;;
    esac
done

# Recompute the raw URL if a non-default ref was supplied.
RAW_URL="https://raw.githubusercontent.com/${REPO}/${REF}/install.sh"

# ---- platform --------------------------------------------------------------

detect_platform() {
    os=$(uname -s 2>/dev/null || echo unknown)
    case "$os" in
        Linux)  ;;
        Darwin) ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            die "Windows is not supported directly. Use WSL and re-run inside it." ;;
        *) warn "untested platform: $os. Continuing." ;;
    esac
}

ensure_curl() {
    command -v curl >/dev/null 2>&1 || die "curl is required. Install it and re-run."
}

# ---- uninstall -------------------------------------------------------------

uninstall() {
    step "Removing Graft"
    link="${BIN_DIR:-$HOME/.local/bin}/graft"
    if [ -L "$link" ] || [ -f "$link" ]; then
        rm -f "$link" && ok "removed $link"
    fi
    if [ -d "$GRAFT_HOME" ]; then
        rm -rf "$GRAFT_HOME" && ok "removed $GRAFT_HOME"
    fi
    printf '\nGraft has been uninstalled.\n'
    exit 0
}

# ---- uv --------------------------------------------------------------------

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
        return
    fi
    step "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh \
        || die "uv installation failed."
    # The uv installer drops binaries in ~/.local/bin and may provide env.
    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.local/bin/env"
    fi
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    command -v uv >/dev/null 2>&1 || die "uv installed but not found on PATH."
    ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
}

# ---- source ----------------------------------------------------------------

fetch_tarball() {
    if command -v git >/dev/null 2>&1; then
        git clone --depth 1 --branch "$REF" "$REPO_URL" "$GRAFT_HOME" \
            || die "git clone failed."
    else
        warn "git not found; downloading a tarball instead."
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        tarball="$tmp/graft.tar.gz"
        url="https://github.com/${REPO}/archive/${REF}.tar.gz"
        curl -fsSL "$url" -o "$tarball" || die "download failed: $url"
        mkdir -p "$GRAFT_HOME"
        tar -xzf "$tarball" -C "$GRAFT_HOME" --strip-components=1 \
            || die "failed to unpack the archive."
    fi
}

fetch_source() {
    if [ -d "$GRAFT_HOME/.git" ]; then
        step "Updating existing checkout at $GRAFT_HOME"
        git -C "$GRAFT_HOME" fetch --depth 1 origin "$REF" \
            || die "git fetch failed."
        git -C "$GRAFT_HOME" checkout -q FETCH_HEAD \
            || die "git checkout failed."
        ok "updated to $REF"
    elif [ -f "$GRAFT_HOME/pyproject.toml" ]; then
        step "Using existing project at $GRAFT_HOME"
    elif [ -e "$GRAFT_HOME" ] && [ -n "$(ls -A "$GRAFT_HOME" 2>/dev/null)" ]; then
        die "$GRAFT_HOME exists and is not a Graft checkout. Set GRAFT_HOME to another path."
    else
        step "Downloading Graft ($REF)"
        fetch_tarball
        ok "source ready"
    fi
}

# ---- install ---------------------------------------------------------------

build_env() {
    step "Creating the virtualenv (this downloads PyTorch, give it a minute)"
    if ! uv sync --project "$GRAFT_HOME" --no-dev; then
        warn "--no-dev unsupported by this uv; retrying with dev dependencies."
        uv sync --project "$GRAFT_HOME" || die "uv sync failed."
    fi
    [ -x "$GRAFT_HOME/.venv/bin/graft" ] || die "the graft launcher is missing after sync."
    ok "virtualenv ready"
}

path_has() {
    case ":${PATH}:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

choose_bin_dir() {
    if [ -n "$BIN_DIR" ]; then
        :
    elif path_has "/usr/local/bin" && [ -w "/usr/local/bin" ]; then
        BIN_DIR="/usr/local/bin"
    else
        BIN_DIR="$HOME/.local/bin"
    fi
    mkdir -p "$BIN_DIR" || die "cannot create $BIN_DIR"
}

link_launcher() {
    step "Linking the launcher"
    choose_bin_dir
    ln -sf "$GRAFT_HOME/.venv/bin/graft" "$BIN_DIR/graft"
    ok "$BIN_DIR/graft"
}

verify() {
    step "Verifying"
    "$GRAFT_HOME/.venv/bin/graft" --version >/dev/null 2>&1 \
        || die "the launcher does not run. See $GRAFT_HOME for logs."
    ok "graft $("$GRAFT_HOME/.venv/bin/graft" --version | awk '{print $2}')"
}

next_steps() {
    printf '\n%sGraft is installed.%s\n\n' "$BOLD$GREEN" "$RESET"

    if path_has "$BIN_DIR"; then
        printf '  Start the app:   %sgraft serve%s\n' "$BOLD" "$RESET"
        printf '  Check this box:  %sgraft doctor%s\n' "$BOLD" "$RESET"
        printf '  Port one file:   %sgraft migrate path/to/file.py%s\n' "$BOLD" "$RESET"
    else
        printf '  %sAdd the launcher to your PATH first:%s\n\n' "$YELLOW" "$RESET"
        printf '    export PATH="%s:$PATH"\n\n' "$BIN_DIR"
        printf '  Then:\n\n'
        printf '    %sgraft serve%s    launch the terminal UI\n' "$BOLD" "$RESET"
        printf '    %sgraft doctor%s   check this machine\n' "$BOLD" "$RESET"
        if [ "$MODIFY_PATH" -eq 1 ]; then
            printf '\n  Add this line to your shell profile to make it permanent:\n'
            printf '    export PATH="%s:$PATH"\n' "$BIN_DIR"
        fi
    fi
    printf '\n  Docs: %shttps://github.com/%s%s\n\n' "$DIM" "$REPO" "$RESET"
}

# ---- main ------------------------------------------------------------------

main() {
    banner
    detect_platform
    ensure_curl
    [ "$DO_UNINSTALL" -eq 1 ] && uninstall
    ensure_uv
    fetch_source
    build_env
    link_launcher
    verify
    next_steps
}

main "$@"
