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
- During a coordinated run, `Stop All`, instance selection, and the global
  `Display` viewer toggle remain available.
- Individual stop, close, add, remove, and training-setting edits are disabled
  until the coordinated run exits.

This document is the implementation source of truth for the coordinated
scheduler feature.

## July 2026 Coordinated Runtime Contract

The following requirements supersede earlier sections that describe a separate
Batch/Setup tab, independent coordinated epsilon state, or saved-slot opponent
discovery:

- Batch and Regimen controls are presented on one scrollable `Regimen` tab in
  this order: match frame length, rounds per batch, batch grouping, replay size,
  starting epsilon, epsilon floor, epsilon decay, epsilon frame span, gamma,
  minibatch size, gradient steps, learning rate, hidden layer size, and hidden
  layer count.
- Every displayed Regimen value must match across a coordinated run. All models
  must also use the same slot number, CUDA device, opponent mode, and AI-opponent
  frequency.
- A mismatch in persisted current epsilon is allowed. Start All initializes the
  coordinated current epsilon to the arithmetic mean across included instances,
  clamped to the shared epsilon floor and `[0, 1]`.
- One scheduler-owned epsilon gate decides explore versus greedy once per
  epsilon span. All trainees explore or all trainees infer together. During an
  exploration span, each trainee receives its own independently sampled random
  action and holds it for the span unless its episode resets.
- Greedy trainee frames use the fixed full tuple of coordinated models. The
  packed trainee parameter tensor is therefore reusable throughout a simulation
  batch. If a record finishes a round before its peers, pad its observation and
  discard its output until the round finishes so the tensor width does not
  shrink.
- Existing-AI selection is coordinated per opponent ship at the batch boundary.
  Only the live in-memory models participating in that coordinated run are
  eligible. Missing ship models fall back to each record's configured simple
  behavior. Saved or bundled models outside the coordinated population are not
  loaded as coordinated opponents.
- All instances face the same selected AI model for a ship. Its observations
  are evaluated as one conventional observation batch, padded to the full
  coordinated width after early finishes. Trainee and opponent inference remain
  separate GPU calls.
- Independent `Start` training retains per-model Regimen/opponent settings and
  discovery across all compatible available AI slots.

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

### Regimen Tab

Use one top-level `Regimen` tab for both ordinary and coordinated setup. It is
an active-instance editor; apply-to-all can propagate changes to other open
instances. Start All validation requires complete matching Regimen values.

Initial controls:

- `Match frame length`
- `Rounds per batch`
- `Batch grouping`
- `Minibatch size`
- `Gradient steps`
- `Learning rate`
- coordination scope control
- `Start All` / `Stop All`

The existing regimen tab may keep per-instance controls for independent
training. Shared batch controls can initially mirror the same value ranges as
the regimen controls.

The Batch tab has two operating modes:

- Coordinated-compatible mode.
- Individual-only mode.

Coordinated-compatible mode is available only when coordinated scope is
available. In this mode, Batch tab values may be edited for only the active
instance or for all open instances, depending on the coordination scope
checkbox. `Start All` may be enabled if the remaining start validation also
passes.

Individual-only mode is used when coordinated scope is unavailable. In this
mode, `Start All` is disabled and every Batch tab change applies only to the
active instance. This keeps incompatible behavior simple and prevents the tab
from silently synchronizing settings when coordinated training cannot run.

The architecture comparison must include:

- observation input size
- action output count
- hidden layer width
- hidden layer count

`Minibatch size`, `Gradient steps`, and `Learning rate` are coordinated
optimization parameters. When the Batch tab is in coordinated-compatible mode,
those values must be treated as shared coordinated-run settings.

### Coordination Scope Control

The Batch tab should reserve one control area for coordination scope. It has two
render states.

When coordinated scope is available, show a checkbox labeled exactly:

```text
Apply to all open instances
```

When the checkbox is unchecked:

- Batch tab edits apply only to the active instance.
- `Start All` can still be enabled if all Start All validation passes.
- If other instances have different batch-controlled settings, `Start All`
  must confirm before synchronizing them.

When the checkbox is checked:

- Batch tab edits immediately apply to every open instance.
- The checkbox remains checked until the user unchecks it or coordinated scope
  becomes unavailable.
- If enabling the checkbox would change one or more open instances, show a
  confirmation popup before enabling. On cancel, leave the checkbox unchecked
  and make no state changes.

When coordinated scope is unavailable, do not show a checkbox glyph. Instead,
show disabled status text. If the blocker is architecture incompatibility, the
text must be:

```text
Coordinated mode disabled; incompatible instances
```

Other blockers may use similarly direct status text, for example:

```text
Coordinated mode disabled; training in progress
```

When the status text is shown:

- The checkbox state is forced false.
- Batch tab edits apply only to the active instance.
- `Start All` is disabled.

Coordinated scope is unavailable if any of these are true:

- Any instance is running or stopping.
- Fewer than two open instances are eligible.
- Any open instance is incomplete: no selected ship or no selected slot.
- Any selected slot is bundled or otherwise read-only.
- Any selected slot cannot be created or started because it lacks required
  description or metadata.
- Duplicate `(ship, slot)` writer targets exist.
- Any model metadata is incompatible.
- Model architectures differ.
- Resolved training devices differ.
- Any resolved training device is not CUDA/GPU.
- PyTorch or CUDA is unavailable.

Device compatibility must compare resolved device keys, not only UI strings.
For example, `auto` and explicit `GPU` are compatible if both resolve to the
same CUDA device. If multiple CUDA devices become selectable, mixed resolved
devices such as `cuda:0` and `cuda:1` are incompatible for the first
implementation.

When the checkbox is checked, these Batch tab values are copied into every open
instance's in-memory training state:

- match frame limit
- rounds per batch
- batch grouping
- minibatch size
- gradient steps
- learning rate

The checkbox should update UI state only. It should not force an immediate
model metadata save unless the surrounding training UI already persists
comparable setting changes immediately.

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
- Every included instance resolves to the same CUDA/GPU device.
- PyTorch and CUDA are available.

If any check fails after the user clicks `Start All`, show one concise
user-facing notice identifying the first blocking reason.

Recommended first-inclusion rule:

- Include every open instance that has a selected trainee ship and selected
  slot.
- If an open instance is incomplete, treat that as a blocking validation error
  rather than silently skipping it.

This avoids ambiguity about which models are being trained.

Before starting, compare every included instance's batch-controlled settings to
the Batch tab values:

- match frame limit
- rounds per batch
- batch grouping
- minibatch size
- gradient steps
- learning rate

If one or more included instances differ, `Start All` must show a confirmation
popup warning that the coordinated run will apply Batch tab settings to those
open instances. On confirm, apply the Batch tab values and start. On cancel,
leave all instance state unchanged and do not start.

`Start All` must never silently train with mixed batch-controlled settings.

### During Coordinated Run

While the coordinated scheduler is running:

- `Start All` becomes `Stop All`.
- Individual `Start` / `Stop` is disabled.
- `Close` is disabled.
- `Add` is disabled.
- Instance selection remains allowed for status and live-view selection.
- Training-defining controls are disabled.
- Display remains enabled and transfers the live view to the selected instance.
- While Display is on, synchronized physics-frame groups are capped at 24 Hz;
  slower simulation groups are presented without additional delay.
- The UI applies the configured video-frame interpolation between physics
  snapshots. Display Off restores unrestricted coordinated throughput.
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

Live battle rendering publishes only the selected instance. In-process runs
publish stable render snapshots. Multi-process runs render configured
interpolation subframes in the selected worker and transfer them through shared
memory so pygame surfaces are not serialized through the worker command pipe.
Battle music and effects likewise follow only the selected displayed instance;
multi-process workers relay its normalized audio events to the parent process.
Turning Display off stops coordinated battle music.

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

Frame accounting must use a scheduler-local counter for each fixed frame
window. Do not use `BattleSimulation.frame_id` as the window counter after a
reset, because each new `BattleSimulation` starts with local `frame_id == 0`.

The current training step convention is:

- `BattleSimulation.step()` increments local `simulation.frame_id` before
  processing the frame.
- The decision snapshot for a frame uses `simulation.frame_id + 1` before the
  step.
- The outcome snapshot uses the returned post-step `state["frame_id"]`.
- A frame limit of `N` should consume exactly frames `1..N`, not `0..N` and not
  `1..N+1`.

The coordinated scheduler must preserve that convention for each underlying
match while separately counting `window_frames_consumed`.

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

Timeout at the end of a fixed frame window is a window boundary, not a
simulation terminal event. When a window ends by frame budget:

- Flush pending reward samples exactly once with the current simulation's local
  `frame_id`.
- Record a timeout/draw episode or window result according to the coordinated
  result model.
- Do not advance an extra frame to create a terminal marker.

Permanent death is terminal as soon as `BattleSimulation.aftermath` is not
`None`, matching the current training behavior. The coordinated scheduler must
not consume explosion, victory-ditty, or ship-selection aftermath frames after a
permanent terminal event.

Rebirth/reincarnation is not terminal while
`BattleSimulation.aftermath.pending_rebirths` is non-empty. Those aftermath,
rebirth, and re-entry frames remain part of the same match and consume the same
fixed frame window. In particular:

- A rebirth event must not start a new battle.
- A rebirth event must not reset `window_frames_consumed`.
- Re-entry frames after rebirth can consume frame budget even though entering
  ships may be excluded from normal input processing.
- If the reborn ship dies permanently later in the same fixed frame window,
  only that later permanent death should end the episode and trigger a battle
  reset.

Because `RollingReturnPipeline` rejects frames after a terminal outcome,
terminal handling must be precise:

- For a permanent terminal event, add the terminal outcome, collect matured
  samples, record the episode, then create a fresh pipeline for the next battle.
- For a timeout/window boundary, call `flush_pending()` once, record the window
  result, then create a fresh pipeline for the next window.
- For rebirth/pending-rebirth frames, continue using the same pipeline.

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

Existing-AI selection is shared across coordinated records for the duration of
a batch. Only live models in the coordinated population are eligible. For each
opponent ship, select AI versus simple once using the shared configured
frequency. If AI is selected and that ship is being trained, apply its one
in-memory model to the observation batch for every record. Otherwise use each
record's simple behavior configuration. Do not consult the saved opponent-model
cache from the coordinated path.

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

The scheduler owns one coordinated epsilon state:

- Starting epsilon, floor, decay, and frame span must match.
- Initialize current epsilon from the mean persisted value across included
  instances and clamp it to the shared floor and `[0, 1]`.
- Draw one explore/greedy decision per span for the entire population.
- Draw and hold a separate random exploratory action for every trainee.
- Infer the full coordinated model tuple on greedy frames and skip trainee
  inference on exploration frames.
- Decay once per completed coordinated batch and persist the same current value
  in every model's metadata.

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
- Add the coordination scope checkbox/status control.
- Add validation helper returning structured errors.
- Add architecture comparison helper for Batch tab mode.
- Validate no running instances, complete selections, unique writer keys,
  same architecture, same resolved CUDA device, CUDA availability, writable
  slots, compatible metadata, and display-off state.
- Implement forced active-instance-only mode when coordinated scope is
  unavailable.
- Implement Start All confirmation when Batch values differ from one or more
  included instances.

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
- Add scheduler-local fixed-window frame counters that survive battle resets.
- Ensure terminal pipelines are flushed and recreated on permanent terminal
  reset.
- Preserve pending-rebirth behavior as non-terminal.

Verification:

- Unit tests with fake simulation factories for timeout, terminal reset,
  rebirth, and multiple episodes in one fixed frame window.
- Unit tests prove `match_frame_limit=N` consumes exactly `N` scheduler frames.
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
- Batch tab enters individual-only mode when coordinated scope is unavailable.
- Batch tab changes apply only to the active instance in individual-only mode.
- Coordination scope renders a checkbox labeled `Apply to all open instances`
  only when coordinated scope is available.
- Coordination scope renders `Coordinated mode disabled; incompatible instances`
  without a checkbox glyph when architectures differ.
- Coordination scope is forced false when any instance is running or stopping.
- Coordination scope is forced false when fewer than two open instances are
  eligible.
- Coordination scope is forced false for incomplete instances.
- Coordination scope is forced false for bundled/read-only slots.
- Coordination scope is forced false for duplicate writer targets.
- Coordination scope is forced false for incompatible metadata.
- Coordination scope is forced false for mixed architectures.
- Coordination scope is forced false for mixed resolved devices.
- Coordination scope is forced false for non-CUDA resolved devices.
- Coordination scope is forced false when PyTorch/CUDA is unavailable.
- Coordination scope is forced false when display is on for any instance.
- Checked coordination scope copies match frame limit, rounds per batch, batch
  grouping, minibatch size, gradient steps, and learning rate to all open
  instances.
- Checking coordination scope confirms before changing existing open instances.
- `Start All` disabled while any instance is running.
- `Start All` disabled for incomplete instances.
- `Start All` disabled for duplicate writer targets.
- `Start All` disabled for bundled slots.
- `Start All` disabled for mixed architectures.
- `Start All` disabled for mixed resolved devices.
- `Start All` disabled for non-CUDA resolved devices.
- `Start All` disabled when GPU/CUDA is unavailable.
- `Start All` confirms before applying changed Batch tab values to included
  instances.
- `Start All` reserves every writer key before scheduler start.
- Failed scheduler startup releases all writer keys.
- During coordinated run, add/close/individual start/individual stop are
  disabled; Display and instance switching remain enabled.
- `Stop All` requests scheduler stop.
- Coordinated status snapshots populate active-instance console fields.
- Fixed frame windows continue after terminal reset.
- Fixed frame windows use scheduler-local frame counts rather than local
  `BattleSimulation.frame_id` after reset.
- `match_frame_limit=N` consumes exactly N frames, with no leading frame 0 and
  no extra frame N+1.
- Timeout/window boundary flushes pending reward samples exactly once.
- Permanent death resets the battle without consuming aftermath animation
  frames.
- Pending rebirth is non-terminal and consumes frame budget in the same match.
- Re-entry after rebirth consumes frame budget without resetting the match.
- Terminal reset starts a fresh reward pipeline.
- Rebirth/pending-rebirth frames keep the same reward pipeline.
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
- Should existing-AI opponent mode be allowed in the first coordinated
  implementation, or should Start All initially require simple-opponent mode?
- Should coordinated-mode metrics count fixed-frame windows, terminal episodes,
  or both in the user-facing batch summary?
- Should the independent regimen tab and shared Batch tab eventually merge?
