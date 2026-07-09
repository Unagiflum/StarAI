# AI Training Implementation Progress

## Current Phase

Phase 8: UI integration and hardening. Complete.

## Completed

### Phase 1

- Added versioned observation/action constants for the 533-input schema and 24-action output schema.
- Added the stable action-index table excluding simultaneous left and right controls.
- Added explicit ship catalog ordering for training metadata.
- Added guarded PyTorch access helpers and a configurable fully connected ReLU value network.
- Added prediction, Adam optimizer construction, and selected-action Huber regression helpers.
- Extended model metadata to persist schema versions, action ordering, architecture, current game settings, and progress counters.
- Updated focused tests for contracts, model construction/regression, optional PyTorch behavior, and metadata round-tripping.
- Added CUDA PyTorch to `requirements.txt`.
- New user AI model slots now require a non-empty description before creation.
- Upgraded the active interpreter to CUDA PyTorch `2.7.0+cu128`; `.venv` already has CUDA PyTorch `2.5.1+cu121`.
- Installed `pygame-ce==2.5.2` for the active interpreter; both the active interpreter and `.venv` now expose `pygame.draw.aacircle`.

### Phase 2

- Added canonical observation field names, ship-block field names, object-slot field names, and offset constants for the 533-input schema.
- Added a pure/read-only training observation encoder in `src.training.observation`.
- Encoded enemy ship type one-hot values using the stable 25-ship catalog order.
- Encoded both 45-value ship blocks for general runtime state: crew, battery, movement waits, timers, velocity, angle, trackability, held controls, repeat countdowns, form flags, limpets, damage-shield state, cloak transition, and zero defaults for not-yet-implemented ship-specific counters.
- Left the Phase 3 object-slot section explicitly zero-filled behind the canonical object-slot boundary.
- Added focused tests for exact observation size, finite values, object-slot zero fill, one-hot enemy type, cooldown/timer/velocity fields, held-key repeat countdowns, form/limpet-adjusted movement values, and missing-state defaults.

### Phase 3

- Completed the ship-specific observation fields for live Orz marines, boarded Orz marines, Ur-Quan fighters, Chmmr satellites, Chenjesu Dogis, and Kohr-Ah saws.
- Extended the read-only observation encoder with optional `game_objects` input while preserving the existing no-world call path.
- Added canonical object-slot population for enemy ship, planet, enemy/friendly A1 and non-A1 ability objects, asteroids, and Syreen crew.
- Added toroidal nearest-object ordering, deterministic tie-breaking, relative bearing, inverse-distance, relative-velocity, expiration, and contact-effect encoding.
- Added explicit object classification rules for A1 versus non-A1 objects, Syreen crew, asteroids, planets, and Chmmr satellite positional-slot exclusion.
- Added focused Phase 3 tests for object slot ordering, zero masking, toroidal boundary geometry, expiration encoding, contact effects, satellite exclusion, Syreen crew slots, and ship-specific live counters.

### Phase 4

- Added read-only A1/A2 pointing and effective-range predicates in `src.training.combat_adapters`.
- Characterized conventional directional weapons, tracking/omnidirectional abilities, configured laser/area ranges, toroidal bearing, projectile lifetime/speed range, and unsupported actions.
- Added an opt-in typed training battle event ledger in `src.training.event_ledger`.
- Wired `BattleSimulation` and `World` to bind an optional ledger, stamp frame IDs, and record spawned ability objects, destruction, and natural expiration without changing behavior when no ledger is attached.
- Instrumented crew loss/gain, battery deltas, ship deaths, limpet applications, Orz marine boarding, Melnorme confusion applications, Chenjesu/Dogi drains, Syreen crew pickup, and Pkunk rebirth attempts/completions.
- Added focused Phase 4 tests for representative pointing/range predicates, loss followed by healing, launched crew-unit transfer suppression, debuff refreshes, object spawn/removal typing, and Pkunk rebirth exclusion from crew gain.

### Phase 5

- Added canonical reward component constants in `src.training.rewards` and wired the Training UI reward labels to that shared contract.
- Added read-only reward decision/outcome snapshots that capture observation, selected action, endpoint state, shaping predicates, and frame events.
- Implemented all Phase 5 reward components, including normalized pointing/range shaping, spawn flags, high-speed endpoint detection, ledger-based crew/debuff/death/object-removal rewards, endpoint battery delta rewards, and battery-at-zero detection.
- Added a rolling finite-horizon return pipeline that holds pending decisions until their prediction window matures or a terminal frame flushes them.
- Mature samples now include the original observation/action, total weighted return, unweighted component totals, weighted component totals, actual window length, frame range, and terminal-truncation status.
- Added focused Phase 5 tests for normalized shaping, endpoint-vs-ledger reward separation, natural expiration exclusion, exact window maturation, and terminal truncation.

### Phase 6

- Added `src.training.replay` with a capacity-bound FIFO replay buffer and validated replay sample contract.
- Added deterministic minibatch sampling with injectable RNG for reproducible tests and future training orchestration.
- Added epsilon-greedy action selection over the Phase 1 action-value network, including explicit exploration diagnostics.
- Added replay-backed minibatch optimization using selected-action Huber regression.
- Added atomic `.pth` training checkpoint saves and load validation for model weights, optimizer state, optional replay state, and small extra training state.
- Added focused Phase 6 tests for replay eviction, seeded sampling, replay state round-tripping, epsilon-greedy action selection, synthetic loss reduction, and checkpoint prediction round-tripping.

### Phase 7

- Added `src.training.orchestration` for headless training rounds and batches.
- Added deterministic simple-opponent scheduling with exactly `rounds_per_batch * 25` matches in the stable ship-catalog order.
- Added existing-AI opponent discovery that freezes loadable model snapshots at batch start, skips empty/unloadable slots, and reports skipped entries.
- Added one-frame action application through the normal battle control API using the Phase 1 action table.
- Added simple-opponent behavior controls for continuous thrust, A1, A2, fixed turning, and face/face-away turning.
- Wired training rounds to `BattleSimulation(None, ...)` so initial two-ship entry is omitted without consuming simulation frames or rendering work.
- Bound the Phase 4 event ledger, Phase 2/3 observation encoder, Phase 5 rolling-return pipeline, and Phase 6 replay buffer into the round loop.
- Added terminal flushing for pending rolling-return windows on resolved outcomes and timeouts, while pending Pkunk rebirths keep the round non-terminal.
- Fully armed AI-controlled Shofixti ships before the first actionable training frame.
- Added end-of-batch replay optimization scheduling.
- Added focused Phase 7 tests for opponent scheduling, action-control mapping, terminal flushing, Shofixti arming, pending-rebirth non-terminal handling, headless deterministic rounds, existing-AI discovery, and existing-AI batch counts.

### Phase 8

- Added `src.training.session` to own UI-facing training sessions, worker-thread safe status snapshots, model/checkpoint loading, safe stop requests, progress persistence, batch metrics, bounded batch-log history, grouped CSV metrics export, and compatibility validation.
- Connected the Training UI Start/Stop flow to real training batches for writable user slots while preserving bundled-slot protection.
- Start now creates new user models, continues compatible existing models, warns on game-setting differences, reports structural incompatibilities, and resets stale weights when the user confirms an architecture overwrite.
- Stop and Back request a documented safe stop after the current batch instead of abandoning an active worker.
- Added live UI status for batch/round progress, current opponent, frame, replay occupancy, recent loss, exploration versus greedy action, weighted return, and top weighted reward component totals.
- Added a display-off monospace scrollable batch log with selectable/copyable text and bounded recent history.
- Added display-on battle visualization by publishing the latest training battle render snapshot from orchestration/session state and drawing the cropped battle arena through the existing battle renderer.
- Added grouped CSV metrics beside the trainee model sidecar using the existing model filename convention, including header creation and rolling averages over the selected Batch grouping window.
- Kept training display state presentation-only; training rounds now omit star creation regardless of display toggle so display-on/display-off settings do not alter simulation setup, RNG consumption, replay contents, or returns.
- Added focused Phase 8 tests for metrics formatting, outcome aggregation, rolling averages, CSV append behavior, compatibility errors/warnings, session checkpoint/progress persistence, UI config adaptation, selectable log text, display-toggle simulation equivalence, battle-view snapshot publishing, and display-on arena drawing.

## Verification

- Passed: `python -m unittest tests.test_training_models tests.test_train_ai_ui`
  - 22 tests passed.
  - PyTorch-specific tests now run in this environment.
- Passed: `python -m unittest discover tests`
  - 622 tests passed.
- Passed in `.venv`: `.\.venv\Scripts\python.exe -m unittest tests.test_laser_drawing tests.test_training_models tests.test_train_ai_ui`
  - 23 tests passed.
- Passed: `python -m unittest tests.test_training_observation tests.test_training_models`
  - 18 tests passed.
- Passed: `python -m unittest discover tests`
  - 627 tests passed.
- Passed: `python -m unittest tests.test_training_observation`
  - 10 tests passed.
- Passed: `python -m unittest tests.test_training_observation tests.test_training_models tests.test_train_ai_ui`
  - 32 tests passed.
- Passed: `python -m unittest discover tests`
  - 632 tests passed.
- Passed: `python -m unittest tests.test_training_phase4 tests.test_training_observation tests.test_training_models tests.test_train_ai_ui`
  - 41 tests passed.
- Passed: `python -m unittest discover tests`
  - 641 tests passed.
- Passed: `python -m unittest tests.test_training_rewards`
  - 6 tests passed.
- Passed: `python -m unittest tests.test_training_rewards tests.test_training_phase4 tests.test_training_observation tests.test_training_models tests.test_train_ai_ui`
  - 47 tests passed.
- Passed: `python -m unittest discover tests`
  - 647 tests passed.
- Passed: `python -m unittest tests.test_training_replay`
  - 7 tests passed.
- Passed: `python -m unittest tests.test_training_replay tests.test_training_rewards tests.test_training_phase4 tests.test_training_observation tests.test_training_models tests.test_train_ai_ui`
  - 54 tests passed.
- Passed: `python -m unittest discover tests`
  - 654 tests passed.
- Passed: `python -m unittest tests.test_training_orchestration`
  - 8 tests passed.
- Passed: `python -m unittest tests.test_training_orchestration tests.test_training_replay tests.test_training_rewards tests.test_training_phase4 tests.test_training_observation tests.test_training_models tests.test_train_ai_ui`
  - 62 tests passed.
- Passed: `python -m unittest discover tests`
  - 662 tests passed.
- Passed: `python -m unittest tests.test_training_session tests.test_training_orchestration tests.test_train_ai_ui`
  - 27 tests passed.
- Passed: `python -m unittest discover tests`
  - 672 tests passed.
- Passed: `python -m unittest tests.test_train_ai_ui tests.test_training_orchestration tests.test_training_session`
  - 29 tests passed.
- Passed: `python -m unittest discover tests`
  - 674 tests passed.

## Next Phase

Initial implementation complete. Deferred work remains listed in `TRAINING_IMPLEMENTATION_SPEC.md`.
