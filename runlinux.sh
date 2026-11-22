#!/usr/bin/env bash
# Run StarAI from source on Linux with the CPU environment
# (create it once with ./setup-linux-cpu-venv.sh).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_ROOT/.venv-CPU-Linux/bin/python" "$PROJECT_ROOT/src/main.py" "$@"
