#!/usr/bin/env bash
# Create .venv-CPU-Linux with the CPU-only PyTorch build and PyInstaller.
#
# The pinned torch and pygame-ce wheels need Python 3.12 or 3.13. Pass a
# specific interpreter with PYTHON=/path/to/python3.12, or install uv and let it
# provide Python 3.12.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv-CPU-Linux"
VENV_PYTHON="$VENV_DIR/bin/python"
cd "$PROJECT_ROOT"

# uv's installer puts it in ~/.local/bin, which is only on PATH in new shells.
UV="$(command -v uv || true)"
for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -z "$UV" && -x "$candidate" ]]; then
        UV="$candidate"
    fi
done

if [[ -e "$VENV_DIR" ]]; then
    echo "$VENV_DIR already exists; installing requirements into it."
elif [[ -n "${PYTHON:-}" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
elif [[ -n "$UV" ]]; then
    "$UV" venv --python 3.12 --seed "$VENV_DIR"
elif command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$VENV_DIR"
elif command -v python3.13 >/dev/null 2>&1; then
    python3.13 -m venv "$VENV_DIR"
else
    echo "No Python 3.12 or 3.13 was found." >&2
    echo "Install uv (https://docs.astral.sh/uv/) or set PYTHON=/path/to/python3.12." >&2
    exit 1
fi

"$VENV_PYTHON" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 13) else 1)' || {
    echo "$VENV_PYTHON is not Python 3.12 or 3.13; the pinned wheels need one of those." >&2
    exit 1
}

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements-cpuai.txt
"$VENV_PYTHON" -m pip install -r requirements-build-tools.txt

echo "Ready: $VENV_PYTHON"
