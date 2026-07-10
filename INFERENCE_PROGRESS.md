# Battle AI Inference Progress

## Phase 1: Dependency And Build Split

Status: Implemented, verification passed.

Completed:

- Added `requirements-cpuai.txt` for the CPU inference build environment.
- Added `buildcpuai.cmd` targeting `StarAI_CPUAI`.
- Added `StarAI_CPUAI.spec` based on the lightweight spec while allowing PyTorch to be bundled and excluding `torchvision` and `torchaudio`.
- Left `build.cmd`, `buildtrain.cmd`, and `build.ps1` behavior unchanged.
- Added static unittest coverage for the build/spec split.

Verification:

- Passed: `.venv\Scripts\python.exe -m unittest tests.test_cpuai_build`
- Passed: `.venv\Scripts\python.exe -m unittest discover -s tests`

Notes:

- CPU PyTorch is installed from the official PyTorch CPU wheel index by `requirements-cpuai.txt`.
- PyInstaller will bundle the installed torch wheel from the active build environment; it does not convert a GPU torch install into CPU torch.
