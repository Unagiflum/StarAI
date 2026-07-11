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

## Phase 3: Battle Loop Integration

Status: Implemented, focused verification passed.

Completed:

- Extended `battle.run()` with `player1_ai` and `player2_ai` flags while preserving existing callers.
- Created and bound `BattleAIManager` immediately after `BattleSimulation` creation.
- Fed AI action dictionaries into `BattleSimulation.step()` on each fixed physics step.
- Filtered accumulated player action key changes so AI-owned sides ignore human movement/action inputs while F1 pause and Escape/end-match remain handled before filtering.
- Cleared stale key state and ship controls for AI-owned ships when AI ownership starts.
- Rebound AI controllers and cleared AI-owned input state after `select_next_round()`.
- Added focused unit coverage for AI input filtering and stale-control clearing.

Verification:

- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai`
- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai tests.test_battle_entry`
- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai tests.test_match_ui`

Notes:

- Fleet-selection AI toggle handoff is now covered by Phase 4.
- HUD label drawing remains Phase 5.

## Phase 4: Ship Selection Automation

Status: Implemented, focused verification passed.

Completed:

- Extended `pick_ship.run()` with `player1_ai` and `player2_ai` flags.
- Passed saved fleet AI toggle values from `pick_fleet.run()` into `pick_ship.run()`.
- Passed AI ownership flags from `pick_ship.run()` into `battle.run()`.
- Preserved AI ownership flags when `battle.run()` reopens ship selection after a round ends.
- Added `const.AI_SHIP_SELECTION_DELAY_SECONDS` for the 0.5 second AI selection/continue delay.
- Added `ShipSelectionAutomation` to drive non-blocking wall-clock AI random selection and both-AI auto-Continue.
- AI automation uses existing `ShipSelectionState` alive-ship, survivor-lock, random-lock, and forced-order rules.
- Added focused unit coverage for delayed AI random selection, alive-only selection, both-AI auto-Continue, one-human/no-auto-Continue, and forced-order timing.

Verification:

- Passed: `.venv\Scripts\python.exe -m unittest tests.test_fleet_picker`
- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai tests.test_match_ui`
- Passed: `.venv\Scripts\python.exe -m unittest tests.test_menu_state tests.test_fleet_picker tests.test_battle_ai tests.test_match_ui tests.test_configuration_registry`

Notes:

- The automation does not block the event loop and Escape/end-match handling remains in the normal event path.
- HUD label drawing remains Phase 5.

## Phase 5: HUD Status Labels

Status: Implemented, focused verification passed.

Completed:

- Extended `BattleDrawOptions` and `draw_battle()` with optional per-player AI labels.
- Threaded `BattleAIManager.label_for_player()` values from `battle.run()` into the battle draw path each rendered frame.
- Rendered AI status text as `AI: <label>` below each AI-controlled player's HUD panel while keeping labels clipped to the player's HUD region.
- Preserved human-controlled behavior by omitting labels when the AI manager returns no label.
- Added focused unit coverage for HUD label text formatting/routing and live battle label handoff.

Verification:

- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai tests.test_match_ui`
- Passed: `.venv\Scripts\python.exe -m unittest tests.test_battle_ai tests.test_match_ui tests.test_fleet_picker tests.test_battle_entry`

Notes:

- The AI manager continues to expose raw label values such as `Earthling-01` or `None found`; the HUD draw layer owns the visible `AI: ` prefix.
- Pixel-level layout tests were not added because the focused rendering test asserts the label text requested by the HUD path, matching the Phase 5 acceptance criteria.
