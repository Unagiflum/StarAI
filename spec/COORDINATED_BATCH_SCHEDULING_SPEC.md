# Coordinated Batch Scheduling Specification

## Goal

Add a coordinated "Start All" training mode that runs every eligible open
training instance under one scheduler instead of independent training-session
threads.

The purpose is to improve GPU utilization by batching the small per-frame
PyTorch work that currently dominates multi-instance training time. The first
implementation should prove the core throughput idea with intentionally narrow
lifecycle rules:

- `Start All` is available only when no training instance is running.
- Every included instance must be GPU-backed.
- Every included model must have the same value-network architecture.
- During a coordinated run, only `Stop All` is available.
- Individual stop, close, add, remove, and training-setting edits are disabled
  until the coordinated run exits.

This document is the implementation source of truth for the coordinated
scheduler feature.

## Motivation

Existing multi-instance training runs each `TrainingSession` independently on
its own worker thread. Timing diagnostics show that this does not improve
aggregate throughput on the measured GPU workload. Most of the slowdown is in
many tiny PyTorch calls:

- Trainee inference per frame.
- Existing-AI opponent inference per frame when the opponent is a loaded model.
- End-of-batch optimizer updates.

The current thread-per-session design lets those tiny GPU operations contend
instead of becoming larger GPU jobs. A central scheduler can collect the active
frame observations across instances, run larger batched inference calls, then
step each simulation with the returned actions.

## Current System Summary

Relevant current surfaces:

- `src/Menus/train_ai.py`
  - Owns `TrainingInstance` and `TrainingInstanceManager`.
  - Manages add/select/close/start/stop/display behavior for multiple open
    instances.
  - Builds `TrainingOrchestrationConfig` from `TrainingUIState`.
  - Starts one `TrainingSession` per active training instance.
- `src/training/session.py`
  - `TrainingSession` owns one trainable model, optimizer, replay buffer,
    status, worker thread, checkpoint saves, and batch metrics.
  - The UI reads only `status`, `history`, and `log_lines`.
- `src/training/orchestration.py`
  - `run_training_batch()` runs one model against a schedule of opponents.
  - `run_training_round()` creates one `BattleSimulation`, steps until terminal
    or `match_time_limit`, emits frame progress, and appends mature samples to
    one replay buffer.
- `src/training/replay.py`
  - `optimize_from_replay()` samples one model's replay buffer and applies one
    optimizer step.
- `src/training/value_network.py`
  - `predict_action_values()` already accepts a batch of observations for one
    model.

The coordinated feature should not break independent single-instance training.
The existing `Start` button should continue to start only the active instance
using the current `TrainingSession` path.

## User Experience Requirements

### Batch Scheduling Tab

Add a new top-level training tab named `Batch`.

The tab is shared by all open instances. It should hold settings that define a
coordinated run and should make it clear through control state, not explanatory
text, that these settings are global.

Initial controls:

- `Match frame limit`
- `Rounds per batch`
- `Batch grouping`
- `Minibatch size`
- `Gradient steps`
- `Start All` / `Stop All`

The existing regimen tab may keep per-instance controls for independent
training. Shared batch controls can initially mirror the same value ranges as
the regimen controls.

If implementation simplicity requires it, the first version may use the active
instance's shared batch values as the global source and apply them to all
included instances at coordinated-run start. The preferred design is a small
global batch-scheduling state owned by the UI, separate from each
`TrainingUIState`.

### Start All Enablement

`Start All` is enabled only when all of these are true:

- No instance is running or stopping.
- At least two instances are open and startable.
- Each included instance has a selected trainee ship and user-writable slot.
- No included slot is bundled or empty without a valid description to create.
- No duplicate `(ship, slot)` writer target exists among included instances.
- All included models are metadata-compatible.
- All included models have the same architecture:
  - observation input size
  - action output count
  - hidden layer width
  - hidden layer count
- Every included instance resolves to the CUDA/GPU device.
- PyTorch and CUDA are available.
- Display is off for all instances.

If any check fails after the user clicks `Start All`, show one concise
user-facing notice identifying the first blocking reason.

Recommended first-inclusion rule:

- Include every open instance that has a selected trainee ship and selected
  slot.
- If an open instance is incomplete, treat that as a blocking validation error
  rather than silently skipping it.

This avoids ambiguity about which models are being trained.

### During Coordinated Run

While the coordinated scheduler is running:

- `Start All` becomes `Stop All`.
- Individual `Start` / `Stop` is disabled.
- `Close` is disabled.
- `Add` is disabled.
- Instance selection remains allowed for status inspection only.
- Training-defining controls are disabled.
- Display On is disabled.
- Back behaves like `Stop All` or is disabled until stop completes.

No individual stop or close behavior is required for the first implementation.
Stopping is all-or-nothing.

### Status And Console

Each instance should continue to expose status through the same UI concepts:

- running/stopping/error
- completed batch count
- current round
- total rounds
- current opponent
- current frame
- replay size
- recent loss
- current epsilon
- current batch seconds
- batches/hour
- display message such as `Applying gradient descent`

The active instance's display-off console can continue to show only the active
instance. The instance strip should still show which instances are running.

Live battle rendering is out of scope for the first coordinated scheduler. The
coordinated run should be headless.

## Functional Requirements

### Coordinated Session Type

Introduce a new training runtime object rather than forcing coordinated behavior
into `TrainingSession`.

Suggested names:

```python
class CoordinatedTrainingSession:
    def start(self) -> None: ...
    def request_stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
```

or:

```python
class CoordinatedTrainingScheduler:
    def start(self) -> None: ...
    def request_stop(self) -> None: ...
```

The object should own one worker thread for the entire coordinated run. It
should own per-instance runtime records containing:

- `TrainingInstance` or stable instance id.
- Repository slot and metadata.
- `TrainingOrchestrationConfig`.
- Model.
- Optimizer.
- Replay buffer.
- Current epsilon.
- Current opponent schedule position.
- Current battle simulation.
- Rolling return pipeline.
- Simple opponent controller.
- Batch metrics/history/log state.
- Save/checkpoint bookkeeping.

The UI should not read mutable internals directly. Provide the same thread-safe
read surfaces as individual sessions, either by attaching lightweight status
proxies to each instance or by having the manager resolve coordinated status
through the scheduler.

### UI Manager Integration

`TrainingInstanceManager` needs a coordinated-run state:

- no coordinated run
- coordinated run starting/running/stopping
- coordinated run stopped/error

The manager should be responsible for:

- Validating Start All eligibility.
- Reserving all writer keys before scheduler start.
- Releasing all writer keys after scheduler stop.
- Disabling display on all instances before start.
- Routing `Stop All` to the coordinated scheduler.
- Preventing add/close/start/edit operations during the coordinated run.
- Cleaning up after scheduler completion.

Existing independent training behavior must continue to use per-instance
`TrainingSession`.

### Fixed Frame Windows

Coordinated runs should advance each included instance for exactly
`match_frame_limit` frames per round window.

If a terminal event occurs before the frame limit:

- Record the episode result.
- Flush/settle any terminal reward samples needed by the reward pipeline.
- Start a new battle for that same opponent/window stream.
- Continue advancing that instance until the fixed frame budget is consumed.

This means one fixed frame window may contain multiple terminal episodes. The
batch result must be able to report all terminal outcomes or a summarized
equivalent.

Recommended data model:

```python
@dataclass(frozen=True)
class TrainingEpisodeResult:
    opponent: OpponentSpec
    frames: int
    terminal_reason: str
    mature_samples: int
    total_return: float
    win: bool
    loss: bool
    draw: bool
    component_totals: Mapping[str, float]
```

Then either replace `TrainingRoundResult` for coordinated mode or let a fixed
frame window contain `episode_results`.

Important: rewards for a terminal episode should not leak into the next battle
after reset. Each new battle should get a fresh `RollingReturnPipeline`.

### Batched Trainee Inference

Per scheduler frame:

1. Collect one observation for every active instance.
2. Group observations by trainable model if needed.
3. Run GPU inference as larger batches.
4. Apply epsilon-greedy selection per instance.
5. Step each `BattleSimulation` with its selected controls.
6. Feed outcomes into that instance's reward pipeline and replay buffer.

Because each instance has a separate model, true single-call inference across
different model weights is not automatic. The first implementation can choose
one of these approaches:

- Preferred MVP: use `torch.func` or an equivalent stacked-parameter approach
  only if available and straightforward.
- Conservative MVP: batch observations per model only where a model has
  multiple observations in the same frame, then synchronize GPU calls in one
  scheduler thread.
- Practical middle ground: run model inference sequentially inside the single
  scheduler thread first, then add true multi-model batched inference behind a
  helper after correctness is proven.

The spec goal is to make the scheduler central and ready for batched GPU work.
If true multi-model batched inference is deferred, instrumentation must make
that explicit so throughput results are interpreted correctly.

### Existing-AI Opponent Inference

Existing-AI opponent mode must continue to work, but it is not required to be
fully batched in the first implementation.

Minimum behavior:

- Existing-AI opponent models are loaded at coordinated batch boundaries.
- Checkpoint readers see only saved model state, as they do today.
- A model being trained in memory is not used as an opponent until it has been
  saved.
- Opponent inference can initially run through the existing read-only predictor.

Future optimization:

- Batch opponent observations by shared opponent model.
- Keep opponent models CPU-backed if GPU memory becomes the limiting factor.
- Share loaded opponent-model cache across all coordinated records.

### Optimization Phase

After every included instance completes its scheduled fixed-frame windows for a
batch, enter a synchronized optimization phase.

For each included instance:

- Emit/update status as `Applying gradient descent`.
- Run `gradient_steps` calls against that instance's replay buffer.
- Use that instance's optimizer and model.
- Record losses per instance.

The first implementation does not need one fused optimizer call across models.
Each model can perform its own optimizer steps inside the central scheduler
thread. The important lifecycle property is that optimization happens together
at the same scheduler phase, not interleaved with other instances' frame
simulation threads.

### Saving And Batch Grouping

Each included instance keeps its own completed batch count and save cadence.

At coordinated batch completion:

- Increment each instance's completed batch count.
- Update each instance's metrics/history/log line.
- If `completed_batches % batch_grouping == 0`, save that model state.
- On coordinated stop, save any instance with unsaved completed progress.

Saves should be serialized in the scheduler thread for the first
implementation. Concurrent final save I/O was measured as expensive, and
serial saves are simpler and safer.

### Epsilon Handling

Each instance keeps independent epsilon state:

- Load/resume current epsilon from its metadata when possible.
- Use the coordinated run's shared epsilon decay settings if those remain part
  of the batch tab.
- Decay epsilon once per completed coordinated batch for that instance.
- Persist current epsilon in that instance's metadata on save.

If the batch tab initially excludes epsilon controls, use each instance's
existing regimen epsilon settings.

### Error Handling

If scheduler startup fails before the worker thread starts:

- Release all writer reservations.
- Restore all instances to stopped/editable state.
- Show a user-facing notice.

If one instance fails during a coordinated run:

- Request stop for the whole coordinated run.
- Mark the failing instance with the error.
- Preserve other instances' most recent completed progress.
- Save completed unsaved progress where possible.
- Release all writer reservations after the scheduler exits.

Do not leave the UI in a partial coordinated-running state after failure.

## Non-Goals

The first implementation should not include:

- Individual stop during coordinated run.
- Individual close during coordinated run.
- Adding instances during coordinated run.
- Live battle display during coordinated run.
- Mixed CPU/GPU coordinated training.
- Mixed architecture coordinated training.
- Distributed training across processes or machines.
- Shared optimizer state across models.
- Merged replay buffers across instances.
- Guaranteeing faster throughput before measurement.

## Implementation Phases

### Phase 1: Spec-To-Code Scaffolding

Goals:

- Add global batch-scheduling state and tab UI.
- Add `Start All` validation without starting a scheduler.
- Add tests for enablement and blocking reasons.

Work:

- Add a `TrainingBatchSchedulingState` dataclass or equivalent.
- Add `Batch` tab and controls.
- Add validation helper returning structured errors.
- Validate no running instances, complete selections, unique writer keys,
  same architecture, GPU device, CUDA availability, and writable slots.

Verification:

- Focused `tests.test_train_ai_ui` coverage for tab state and enablement.
- Existing independent start/stop tests still pass.

### Phase 2: Coordinated Runtime Skeleton

Goals:

- Add a scheduler that can start, stop all, expose statuses, and save/cleanup
  without changing training behavior yet.

Work:

- Create `src/training/coordinated.py` or similar.
- Build per-instance runtime records.
- Load model/optimizer/replay buffer per record.
- Reserve/release writers through manager.
- Provide status snapshots compatible with existing UI expectations.
- Run a simple worker loop that immediately exits or performs no-op batches for
  testability.

Verification:

- Unit tests for lifecycle, stop all, writer release, error cleanup.

### Phase 3: Fixed-Frame Battle Windows

Goals:

- Implement headless fixed-frame windows for each coordinated record.
- Handle terminal reset inside a window.
- Populate replay buffers and episode/window metrics.

Work:

- Extract reusable pieces from `run_training_round()` where practical.
- Avoid broad rewrites of independent training.
- Add episode result data structures.
- Ensure terminal pipelines are flushed and recreated on reset.

Verification:

- Unit tests with fake simulation factories for timeout, terminal reset, and
  multiple episodes in one fixed frame window.
- Existing `tests.test_training_orchestration` remains green.

### Phase 4: Central Scheduler Frame Loop

Goals:

- Advance all included records frame by frame in one worker thread.
- Keep statuses synchronized by scheduler frame.

Work:

- For each global scheduler frame, collect observations, select actions, step
  simulations, process rewards, and emit per-record status.
- Keep display/battle-view construction disabled.
- Respect stop-all requests between frames.

Verification:

- Deterministic tests with fake policies/simulations show same frame advancement
  for all records.
- Stop all exits cleanly without individual session leakage.

### Phase 5: GPU Inference Batching Helper

Goals:

- Introduce a clear helper boundary for batched trainee inference.
- Keep a conservative fallback if true multi-model batching is not yet in place.

Work:

- Add a helper such as `select_actions_for_records(records, observations)`.
- Implement per-record epsilon-greedy behavior.
- Use batched tensor operations where possible.
- Instrument whether inference ran as true batched multi-model, per-model
  batched, or sequential fallback.

Verification:

- Unit tests for action selection, epsilon behavior, and result routing.
- Manual timing comparison against independent sessions.

### Phase 6: Synchronized Optimization And Saving

Goals:

- Complete coordinated batch lifecycle.
- Run per-model optimization at the synchronized optimization phase.
- Save models at grouping boundaries and on stop.

Work:

- Port metrics/log/csv behavior from `TrainingSession`.
- Serialize saves.
- Notify opponent model cache after saves where applicable.
- Update metadata progress per instance.

Verification:

- Unit tests for completed batch counts, grouped saves, final save on stop,
  epsilon decay, metrics, and loss recording.
- Focused `tests.test_training_session` and `tests.test_train_ai_ui` pass.

### Phase 7: Performance Characterization

Goals:

- Measure whether coordinated mode improves aggregate throughput.
- Identify whether true multi-model batched inference is needed immediately.

Work:

- Add timing buckets for coordinated scheduler:
  - observation
  - trainee inference
  - opponent inference
  - simulation
  - reward
  - optimization
  - save
- Compare:
  - one independent instance
  - N independent instances
  - N coordinated instances with sequential fallback
  - N coordinated instances with batched inference, if implemented

Verification:

- Produce a progress note with batches/hour and timing bucket shares.

## Testing Strategy

Add or update tests for:

- Batch tab layout and shared state.
- `Start All` disabled while any instance is running.
- `Start All` disabled for incomplete instances.
- `Start All` disabled for duplicate writer targets.
- `Start All` disabled for bundled slots.
- `Start All` disabled for mixed architectures.
- `Start All` disabled when GPU/CUDA is unavailable.
- `Start All` reserves every writer key before scheduler start.
- Failed scheduler startup releases all writer keys.
- During coordinated run, add/close/individual start/individual stop/display are
  disabled.
- `Stop All` requests scheduler stop.
- Coordinated status snapshots populate active-instance console fields.
- Fixed frame windows continue after terminal reset.
- Terminal reset starts a fresh reward pipeline.
- Synchronized optimization records per-instance losses.
- Grouped saves and final saves occur per instance.
- Independent single-instance training remains unchanged.

Focused suites expected during development:

- `python -m unittest tests.test_train_ai_ui`
- `python -m unittest tests.test_training_session`
- `python -m unittest tests.test_training_orchestration`
- `python -m unittest tests.test_training_models`

## Implementation Notes And Dependencies

Before implementation, inspect these files carefully:

- `src/Menus/train_ai.py`
- `src/training/session.py`
- `src/training/orchestration.py`
- `src/training/replay.py`
- `src/training/value_network.py`
- `src/training/opponent_cache.py`
- `tests/test_train_ai_ui.py`
- `tests/test_training_session.py`
- `tests/test_training_orchestration.py`

Preferred code organization:

- Put scheduler runtime in a new training module rather than expanding
  `train_ai.py` or overloading `TrainingSession`.
- Keep UI validation helpers small and unit-testable.
- Reuse existing metadata, checkpoint, replay, reward, and model-building
  helpers.
- Extract shared metrics formatting only if duplication becomes meaningful.
- Do not refactor independent training paths except where reuse is necessary.

## Risks And Mitigations

### True Multi-Model Batching May Be Nontrivial

Risk:

Each instance has distinct model weights, so stacking observations alone does
not create one forward pass through all models.

Mitigation:

Build the central scheduler first with an explicit inference helper and
instrumented fallback. Add true multi-model batching behind that helper once
correctness and lifecycle are stable.

### Fixed-Frame Windows Change Training Semantics

Risk:

Continuing after terminal events changes the meaning of one round/batch and may
alter reward distribution.

Mitigation:

Model terminal episodes explicitly and reset reward pipelines at battle reset.
Compare metrics before and after the change.

### GPU Memory

Risk:

All coordinated models and possibly opponent models must be resident on GPU.

Mitigation:

Start All requires GPU and same architecture but should still fail cleanly on
out-of-memory. Future work can keep opponent models on CPU or add shared caches.

### UI Lockout Bugs

Risk:

Allowing editing or closing during coordinated execution could corrupt scheduler
state or writer reservations.

Mitigation:

The first implementation disables all individual lifecycle operations during
coordinated runs. Keep this strict until scheduler lifecycle is proven.

### Save Latency

Risk:

Saving many models at once can pause training.

Mitigation:

Serialize saves in the scheduler thread first. Optimize later only if save time
becomes a measured bottleneck.

## Open Questions

- Should the Batch tab values be persisted in model metadata, app settings, or
  remain only in memory?
- Should coordinated runs require every open instance to be complete, or should
  there eventually be an explicit include/exclude checkbox per instance?
- Should `auto` count as GPU if it resolves to CUDA, or should Start All require
  explicit `GPU` selection?
- Should existing-AI opponent mode be allowed in the first coordinated
  implementation, or should Start All initially require simple-opponent mode?
- Should coordinated-mode metrics count fixed-frame windows, terminal episodes,
  or both in the user-facing batch summary?
- Should the independent regimen tab and shared Batch tab eventually merge?

