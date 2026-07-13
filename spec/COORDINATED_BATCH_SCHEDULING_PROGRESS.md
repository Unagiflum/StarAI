# Coordinated Batch Scheduling Progress

Last updated: 2026-07-13

## Phase 1: Spec-To-Code Scaffolding

Status: Completed for current implementation scope.

Completed:

- Added `TrainingBatchSchedulingState` to `TrainingInstanceManager` for the
  Batch tab coordination-scope checkbox state.
- Added shared helpers for the Batch-tab-controlled settings:
  `match_time_limit`, `rounds_per_batch`, `batch_grouping`, `minibatch_size`,
  `replay_updates_per_batch`, and `learning_rate`.
- Added coordinated architecture-signature comparison over the required fields:
  observation input size, action output count, hidden layer width, and hidden
  layer count.
- Added structured `Start All` validation with blocking codes/reasons for:
  running or stopping instances, too few instances, incomplete selections,
  read-only slots, empty slots without descriptions, duplicate writer targets,
  missing or incompatible metadata, mixed architectures, missing CUDA/PyTorch,
  mixed or non-GPU resolved devices, and display-on instances.
- Added Batch tab UI with the required initial controls, coordination-scope
  checkbox/status area, and `Start All` button.
- Added active-only Batch-tab editing when coordinated scope is unavailable.
- Added apply-to-all behavior that copies Batch tab values to every open
  instance after confirmation is requested by the UI.
- Added `Start All` settings-drift confirmation before launching coordinated
  runtime work.
- Added focused UI tests for settings copying, mismatch detection,
  architecture signatures, validation success, and representative validation
  blockers.

Verified:

- `python -m unittest tests.test_train_ai_ui`
- `python -m unittest tests.test_training_session tests.test_training_orchestration tests.test_training_models`

Remaining:

- Add more focused UI tests for the rendered Batch tab control states and all
  validation blockers listed in the spec.
- Manual smoke of the new Batch tab remains pending.

## Phase 2: Coordinated Runtime Skeleton

Status: Completed.

Completed:

- Added `src/training/coordinated.py` with:
  - `CoordinatedTrainingRecord`
  - `CoordinatedRuntimeComponents`
  - `CoordinatedTrainingStatusProxy`
  - `CoordinatedTrainingSession`
- Added scheduler lifecycle methods: `start`, `request_stop`, `join`, status
  accessors, history/log accessors, and stable per-instance status proxies.
- Added per-record component loading that mirrors the independent session
  component path: normalized architecture, value network, optimizer, replay
  buffer, checkpoint load, and optimizer-state device move.
- Implemented the Phase 2 no-op worker behavior: the coordinated worker loads
  all records, exposes running status, remains headless/idle, and exits when
  Stop All is requested.
- Integrated coordinated startup into `TrainingInstanceManager` and the Batch
  tab `Start All` path.
- Replaced the Phase 1 scaffold notice with real coordinated scheduler
  startup.
- Added atomic multi-writer reservation for coordinated starts and ensured
  reservations are released after coordinated stop/error cleanup.
- Attached coordinated status proxies to each included instance so existing UI
  status/log rendering can continue to use `instance.session.status`,
  `history`, and `log_lines`.
- Added coordinated-run UI lockout for add, close, individual start/stop,
  display, load, device edits, and Batch-tab sliders while the coordinated
  worker is active.
- Added `Stop All` behavior through the Batch tab button and existing back/stop
  routing.

Verified:

- `python -m unittest tests.test_coordinated_training`
- `python -m unittest tests.test_train_ai_ui`
- `python -m unittest tests.test_training_session tests.test_training_orchestration tests.test_training_models`

Remaining:

- Manual smoke of the Batch tab `Start All` / `Stop All` flow remains pending.
- Phase 2 intentionally does not simulate battles, optimize, or save completed
  progress because no coordinated batches are completed yet.
- Phase 3 should replace the no-op idle loop with fixed-frame battle windows.
