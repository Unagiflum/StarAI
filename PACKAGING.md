# Packaging StarAI for Windows

StarAI is built as a PyInstaller one-folder application. Build it on Windows;
PyInstaller does not cross-compile applications for other operating systems.

## Initial setup

From the repository root, using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
```

## Build

```powershell
.\build.cmd
```

The script runs the test suite, removes stale PyInstaller intermediates, builds
the application, and runs a headless packaged smoke test that loads every asset
and dynamic ship/ability module. It creates `dist\StarAI\StarAI.exe` and the
shareable `dist\StarAI-windows-x64.zip`. Use `-SkipTests` only when tests have
already run in the same revision:

```powershell
.\build.cmd -SkipTests
```

`build.cmd` launches the underlying PowerShell script with a process-scoped
execution-policy bypass. It does not change the machine or user policy.

Distribute `dist\StarAI-windows-x64.zip`. The executable depends on the adjacent
`_internal` directory and will not work if only `StarAI.exe` is copied.

Player settings and fleets are stored outside the installation at
`%LOCALAPPDATA%\StarAI`. A new build therefore does not overwrite player data.
Set `STARAI_DATA_DIR` to use a different location, which is also useful for
isolated smoke tests.

## Linux CPU build

The Linux build bundles the CPU-only PyTorch wheel. Build it on Linux. Create
its virtual environment once; it needs Python 3.12 or 3.13, found through `uv`,
`python3.12`/`python3.13`, or `PYTHON=/path/to/python`:

```bash
./setup-linux-cpu-venv.sh
```

Then build:

```bash
./buildcpu-linux.sh            # or ./buildcpu-linux.sh --skip-tests
```

This creates `build/StarAI_Linux/StarAI_Linux` and the shareable
`build/StarAI_Linux-linux-x64.tar.gz`. PyInstaller intermediates go in
`build/pyinstaller-linux`. As on Windows, the executable depends on the
adjacent `_internal` directory. Player data is stored in
`$XDG_DATA_HOME/StarAI` (default `~/.local/share/StarAI`).

Full builds and packaged smoke tests run on a separate packaging schedule. Do
not use them for routine source-level iteration, and do not invoke packaging
from the normal test suite. Run `build.cmd` only when explicitly performing
scheduled packaging verification.
