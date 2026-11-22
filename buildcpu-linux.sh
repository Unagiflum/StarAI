#!/usr/bin/env bash
# Build the CPU-only Linux application into build/StarAI_Linux.
#
# Usage: ./buildcpu-linux.sh [--skip-tests]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_NAME="StarAI_Linux"
PYTHON="$PROJECT_ROOT/.venv-CPU-Linux/bin/python"
DIST_ROOT="$PROJECT_ROOT/build"
WORK_ROOT="$PROJECT_ROOT/build/pyinstaller-linux"
SMOKE_DATA="$PROJECT_ROOT/build/smoke-data-linux"

SKIP_TESTS=0
for argument in "$@"; do
    case "$argument" in
        --skip-tests) SKIP_TESTS=1 ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "CPU Linux Python was not found: $PYTHON" >&2
    echo "Create it with: ./setup-linux-cpu-venv.sh" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

if ! "$PYTHON" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"; then
    echo "PyInstaller is not installed. Run: $PYTHON -m pip install -r requirements-build-tools.txt" >&2
    exit 1
fi

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    if ! "$PYTHON" -m unittest discover -s tests; then
        echo "WARNING: Tests failed; continuing with the build." >&2
    fi
fi

"$PYTHON" -m PyInstaller --noconfirm --clean \
    --distpath "$DIST_ROOT" \
    --workpath "$WORK_ROOT" \
    "$BUILD_NAME.spec"

EXECUTABLE="$DIST_ROOT/$BUILD_NAME/$BUILD_NAME"
if ! SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy STARAI_DATA_DIR="$SMOKE_DATA" \
        "$EXECUTABLE" --smoke-test; then
    if [[ -f "$SMOKE_DATA/smoke-test-error.log" ]]; then
        cat "$SMOKE_DATA/smoke-test-error.log" >&2
    fi
    echo "Packaged application smoke test failed." >&2
    exit 1
fi

ARCHIVE="$DIST_ROOT/$BUILD_NAME-linux-x64.tar.gz"
tar -czf "$ARCHIVE" -C "$DIST_ROOT" "$BUILD_NAME"

echo "Build complete: $EXECUTABLE"
echo "Shareable archive: $ARCHIVE"
