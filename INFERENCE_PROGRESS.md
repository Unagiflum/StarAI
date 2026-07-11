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

## Phase 2: Battle AI Runtime Skeleton

Status: Implemented, focused verification passed.

Completed:

- Added `src/Battle/battle_ai.py`.
- Implemented `BattleAIManager` with per-player AI ownership, round binding, controller creation, HUD label values, and model-load failure tracking.
- Implemented model slot resolution using `TrainingModelRepository`, including default-slot priority, first-loadable fallback, non-empty checkpoint checks, CPU checkpoint loading, schema checks, and per-match model caching.
- Kept PyTorch optional through `src.training.torch_backend`; missing PyTorch resolves to fallback controls and `None found`.
- Implemented trained-model inference using `encode_observation()`, greedy `select_action_epsilon_greedy(..., epsilon=0.0)`, and the shared action schema.
- Implemented stateful fallback controls that face the enemy, hold forward, and toggle A1/A2 with simulation-RNG-driven probabilities.
- Added focused unit coverage in `tests/test_battle_ai.py`.

Verification:

- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai`
- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai tests.test_training_models tests.test_training_orchestration`
- Attempted: `.venv\Scripts\python.exe -m unittest discover -s tests`
  - Failed in pre-existing areas outside Phase 2: `test_collision_pipeline` KzerZa/Orz special-object crew-loss attributes, and `test_train_ai_ui` display-off console `previous_opponent` test doubles.

Notes:

- Phase 2 does not yet wire the manager into the live battle loop; that is Phase 3.
- The existing `INFERENCE_SPEC.md` had local modifications before this phase and was not changed.
