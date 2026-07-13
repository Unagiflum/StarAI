# Coordinated Batch Scheduling Progress

Last updated: 2026-07-13

## Phase 1: Spec-To-Code Scaffolding

Status: In progress.

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
- Added `Start All` settings-drift confirmation scaffolding. The click path now
  validates and applies shared settings, but intentionally stops before
  launching a scheduler because the runtime is Phase 2+ work.
- Added focused UI tests for settings copying, mismatch detection,
  architecture signatures, validation success, and representative validation
  blockers.

Verified:

- `python -m unittest tests.test_train_ai_ui`
- `python -m unittest tests.test_training_session tests.test_training_orchestration tests.test_training_models`

Remaining:

- Add more focused UI tests for the rendered Batch tab control states and all
  validation blockers listed in the spec.
- Replace the `Start All` scaffold notice with coordinated scheduler startup
  after Phase 2 introduces the runtime object.
- Add writer reservation and release tests for the real scheduler startup path.
- Manual smoke of the new Batch tab remains pending.

## Phase 2: Coordinated Runtime Skeleton

Status: Planned.

Next work:

- Create `src/training/coordinated.py`.
- Add a `CoordinatedTrainingSession` or scheduler object with `start`,
  `request_stop`, and `join` lifecycle methods.
- Build per-instance runtime records and status proxies.
- Integrate scheduler startup/cleanup with `TrainingInstanceManager`.
- Reserve all writer keys before scheduler start and release them after exit.
