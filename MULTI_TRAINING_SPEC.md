# Multi-Instance AI Training Specification

## Goal

Add support for multiple concurrent AI training instances from the training UI.

The feature must let the user create, select, start, stop, and remove training
instances. Each instance owns its own training configuration and background
`TrainingSession`. Only one instance may be visualized at a time. All other
running instances continue training headlessly in the background.

The initial implementation should be comfortable for a few active instances,
but the design must leave room to test up to at least 25 active instances, one
per ship. The UI may warn about high instance counts, but it must not bake in
assumptions that only two or three instances can ever exist.

This document is the implementation source of truth for the feature.

## Current System Summary

Training is currently centered on one UI state and one optional session:

- `src/Menus/train_ai.py` owns one `TrainingUIState`.
- `src/Menus/train_ai.py` stores a single `training_session[0]`.
- `TrainingSession` already runs on a background worker thread.
- `TrainingSession.status`, `TrainingSession.history`, and
  `TrainingSession.log_lines` are already thread-safe read surfaces.
- The display toggle already gates whether `TrainingSession` keeps a frozen
  battle view.
- Existing-AI opponent mode already discovers and loads available stored AIs at
  batch boundaries.
- Training checkpoint writes are atomic, but there is no current ownership guard
  preventing two sessions from writing the same ship/slot.

The main implementation work is therefore not basic background execution. It is
multi-session ownership, UI state separation, writer safety, display routing,
and performance controls.

## User Experience Requirements

### Instance Strip

Add an instance control section at the top of the left control pane, above the
existing tabs.

The needed vertical space above the existing tabs has already been created. The
remaining implementation work should use that space for the controls below
without reintroducing overlap between the instance controls, tab row, and tab
content.

The instance section must include:

- An active-position indicator, starting from the left, in the form `[1/4]`.
  This means the active instance is first in the current instance list, out of
  four open instances.
- A dropdown selector for the active instance.
- A `Close Instance` button for the active instance.
- An `Add Instance` button.

The dropdown must list every open instance. Each row should:

- Be numbered by current instance-list order, starting at `01]`.
- Show the selected trainee ship and AI slot when both are selected.
- Show dashes for missing ship or slot information.
- Show the instance status.
- Render `Stopped` in red text and `Running` in green text.

Example rows:

```text
01]  Earthling-01 Stopped
02] Androsynth-01 Running
03]        Orz-01 Running
04] ------------- Stopped
```

Selecting a dropdown item must bring that instance into view and make it the
active instance.

The four controls should be arranged left-to-right in this order:

```text
[1/4]  [instance dropdown]  [Close Instance]  [Add Instance]
```

If the dropdown renderer cannot color only the status word, it should use the
closest available custom row rendering so that running and stopped states remain
visually distinguishable.

The first screen of the training menu should still be the usable training
interface, not a separate setup page.

### Active Instance

The active instance is the one whose configuration appears in the existing tabs.
Changing the active instance must swap the visible tab controls to that
instance's saved UI state.

Each instance must independently track:

- Active tab.
- Selected trainee ship.
- Selected AI slot.
- Slot labels as loaded into text fields.
- Loaded model conditions.
- Reward settings.
- Opponent mode and simple behavior settings.
- Regimen settings.
- Display-on preference.
- Current or most recent `TrainingSession`.
- Recent batch history and log lines.

Running one instance must not freeze the visible configuration of stopped
instances. A stopped active instance can still be edited even if other
instances are training in the background, subject to model-slot ownership rules.

### Adding Instances

The add control should create a new stopped instance with default
`TrainingUIState` values.

Suggested naming:

- New instances start as `Instance 1`, `Instance 2`, etc.
- Once a trainee ship and slot are selected, the selector label can become
  `Earthling-01`, `Chmmr-02`, etc.
- Duplicate visible labels should be disambiguated with the instance number.

There should be an implementation constant for the soft maximum active instance
count. The initial soft maximum can be small, such as 4, but the internal data
structures and layout code must support at least 25 instances.

When adding an instance beyond the soft maximum, the UI may show a confirmation
or warning. The spec requires room for experimentation, so the warning must not
be a hard architectural limit below 25.

### Removing Instances

Removing a stopped instance should remove it from the instance list.

Removing a running or stopping instance should not kill it abruptly. The UI
should request stop and mark the instance as pending removal. Once the session
has stopped, the instance can be removed automatically.

The final remaining instance should not be removable unless a replacement
instance is created immediately.

### Starting And Stopping

The existing Start/Stop button should apply only to the active instance.

Starting an instance should:

- Validate the active instance's selected trainee and slot.
- Persist a new user model if needed, using the existing behavior.
- Validate metadata compatibility.
- Enforce model-slot writer ownership.
- Create or resume a `TrainingSession` for that instance.
- Start the session.

Stopping an instance should:

- Request stop on that instance's session.
- Leave other instances running.
- Preserve the current behavior that the active batch may be abandoned.

Back navigation should:

- If any instance is running, request stop for all running instances and remain
  on the training screen until they stop.
- If no instance is running, return to the previous menu.

### Visualization

Only one instance can have live display enabled at a time.

When the user enables Display On for the active instance:

- All other instances must be forced to display off.
- Their stored battle views must be cleared.
- Their audio gates must remain off.
- The active instance becomes the visualized instance.

When the active instance changes:

- If the newly active instance has display enabled, all other instances remain
  display off.
- If it does not, the right arena should show that instance's text console when
  stopped or running headlessly.

The existing training battle and HUD renderers should continue to draw only the
active instance's `TrainingSessionStatus`.

### Background Status

The instance selector should expose enough progress information to make
background training understandable without switching constantly.

At minimum, each instance should be able to show:

- Running/stopped/stopping/error.
- Completed batch count.
- Current opponent or previous opponent.
- Trainee ship and slot when selected.

The detailed console and live battle view can remain active-instance-only.

## Functional Requirements

### Session Manager

Introduce a UI-facing session manager rather than storing a single
`training_session[0]`.

Suggested data structures:

```python
@dataclass
class TrainingInstance:
    instance_id: int
    label: str
    state: TrainingUIState
    session: TrainingSession | None = None
    pending_removal: bool = False
    last_running: bool = False


class TrainingInstanceManager:
    instances: list[TrainingInstance]
    active_instance_id: int
```

The exact names can differ, but the responsibilities should stay clear:

- Instance list management.
- Active instance lookup.
- Start/stop orchestration.
- Display ownership.
- Model-slot writer reservations.
- Cleanup of stopped pending-removal instances.

### Model-Slot Writer Ownership

Only one active training session may write a given user model slot at a time.

The reserved key should be:

```text
(ship, slot)
```

Starting an instance must fail with a user-facing notice if another running or
stopping instance owns the same key.

This prevents two sessions from racing to save the same `.pth`, `.json`, and
`.csv` files.

The first implementation can use an in-process reservation map because the UI
owns all instances. The design should leave room for a later filesystem lock if
multiple app processes ever train against the same model directory.

### Latest Opponent Models

In existing-AI opponent mode, each training batch should use the latest saved
versions available at the batch boundary.

The current `discover_existing_ai_opponents()` behavior already reloads stored
models at batch boundaries. The implementation must preserve this behavior when
multiple instances are running.

Important details:

- Checkpoint writes are atomic, so readers should see either the previous
  checkpoint or the next complete checkpoint.
- A session should be allowed to train against models written by other sessions.
- A session should not train against its own in-memory unsaved state. It should
  only appear as an opponent after its checkpoint has been saved.
- Empty placeholder checkpoints must continue to be skipped.

### Display Cost Control

Background instances should not pay full visualization cost.

Currently, training progress can include a battle view even when display is off,
and `TrainingSession` drops it. For multiple sessions, this should be improved
so headless sessions avoid constructing frozen or render-oriented battle view
data whenever possible.

Required behavior:

- Only the visualized instance should store `battle_view`.
- Only the visualized instance should throttle to visual frame rate.
- Headless sessions should run as fast as their simulation and training work
allow.

Preferred implementation:

- Add a cheap display/battle-view predicate that can be consulted before
  constructing expensive battle-view payloads.
- Keep current public behavior for tests and single-instance operation.

### GPU And Device Behavior

The default preferred device may remain CUDA when available.

The implementation must not assume that more instances always improve
throughput. It should allow experimentation while keeping failure modes
manageable.

Requirements:

- Do not hard-code a low instance limit into core training classes.
- Add a UI/config soft limit that can be raised to at least 25.
- Provide clear notices when starting an instance fails due to out-of-memory or
  PyTorch errors.
- Stopping one failed instance must not stop other instances.

Future optimization hooks should be considered but are not required in the
first implementation:

- CPU-loaded opponent models.
- Shared opponent model cache.
- Batch-level opponent model reuse across instances.
- Per-instance device selection.
- Global training scheduler.

### Thread Safety

The training UI must not read mutable session internals directly.

Allowed read surfaces:

- `TrainingSession.status`
- `TrainingSession.history`
- `TrainingSession.log_lines`

Any new shared state added for instance management must be read and written on
the UI thread, except for session status provided by `TrainingSession`.

### Audio

Only the visualized instance may start battle music.

When display is disabled for an instance:

- Its `DisplayGatedAudioService` must stop music if needed.
- It must not restart music while running headlessly.

Switching visualization between instances must not layer multiple battle music
streams.

## Non-Goals

The initial implementation should not attempt to solve all scaling issues.

Explicit non-goals for the first pass:

- Distributed training across processes or machines.
- A global reinforcement-learning scheduler.
- Merging replay buffers across instances.
- Sharing optimizer state.
- Simultaneously rendering multiple battles.
- Persisting the instance list across application restarts.
- Guaranteeing that 25 concurrent CUDA sessions will fit in VRAM.

The architecture should leave these possible, but not implement them.

## UI Layout Requirements

### Left Pane Layout

The left pane should be vertically reorganized:

```text
instance section
tabs
tab content
display toggle
start/back controls
HUD placeholders
```

The exact pixel values can change, but existing constraints still apply:

- Footer controls must not overlap HUD placeholders.
- Tab content must remain clipped to its content rect.
- Scrolling behavior must continue to work for trainee and rewards content.
- Text must fit within buttons and selector rows at the supported resolution.

### Instance Selector Scaling

The selector must be a dropdown so the instance section remains a compact
single row even when many instances are open.

The first implementation does not need to display 25 instances at once, but the
user must be able to select any of them without layout breakage. If 25 entries
do not fit in the expanded dropdown, the dropdown list should scroll or page
within the available menu area.

### Control Enablement

Control enablement must become instance-scoped.

Examples:

- If active instance A is stopped, its reward sliders are editable even if
  instance B is running.
- If active instance A is running, its training-defining controls are disabled.
- If instance B owns `Earthling-01`, active instance A cannot start training
  `Earthling-01`.
- Bundled model slots remain read-only for all instances.

## Implementation Phases

### Phase 1: Instance Data Model And Single-Instance Compatibility

Status: Planned.

Goals:

- Add `TrainingInstance` and a small manager structure.
- Replace direct single-session variables with manager calls while still
  creating exactly one instance.
- Preserve current visible behavior.

Work:

- Create the manager inside `train_ai.py` or a new UI helper module if the file
  becomes unwieldy.
- Move `TrainingUIState`, `TrainingSession`, `last_session_running`, and pending
  removal state under a `TrainingInstance`.
- Add helper functions for active instance lookup.
- Update start/stop/back/display paths to route through the active instance.
- Keep tests focused on behavior equivalence.

Verification:

- Existing `tests.test_train_ai_ui` focused tests pass.
- Existing `tests.test_training_session` focused tests pass.
- Manual smoke: one instance behaves exactly like the current training screen.

### Phase 2: Instance Selector UI

Status: Planned.

Goals:

- Add the top-of-left-pane instance section.
- Support adding, selecting, and removing stopped instances.
- Move tabs/content downward without overlap.

Work:

- Add layout constants for instance section height and spacing.
- Update `training_layout()` and dependent tests.
- Add instance selector drawing and event handling.
- Add instance add/remove controls.
- Update tab button positions to start below the instance section.
- Ensure text fitting for instance labels.

Verification:

- Layout tests cover arena, content, footer, and HUD non-overlap.
- Unit tests cover add/select/remove stopped instances.
- Manual smoke: switching instances swaps visible selected ship, slot, rewards,
  opponent settings, and regimen settings.

### Phase 3: Multi-Session Start/Stop And Writer Reservations

Status: Planned.

Goals:

- Allow more than one instance to train concurrently.
- Prevent two running sessions from writing the same model slot.
- Keep Start/Stop scoped to the active instance.

Work:

- Add reservation map keyed by `(ship, slot)`.
- Reserve before session start and release after session stops.
- Detect conflicts and show a user-facing notice.
- Ensure stopped instances can be edited while other instances run.
- Update back behavior to request stop on all running instances.
- Keep pending-removal instances until their sessions stop.

Verification:

- Unit tests cover concurrent distinct-slot starts using fake sessions.
- Unit tests cover same-slot conflict.
- Unit tests cover stop active instance while another keeps running.
- Unit tests cover back requesting stop for all running instances.

### Phase 4: Single Visualization Ownership

Status: Planned.

Goals:

- Enforce exactly zero or one visualized training instance.
- Keep background sessions headless.

Work:

- Route Display On through the manager.
- When enabling display on one instance, call `set_display_on(False)` for every
  other instance.
- Ensure display checkbox reflects the active instance.
- Draw live battle or console for only the active instance.
- Stop audio when visualization changes.

Verification:

- Unit tests cover display ownership transfer.
- Existing battle/HUD training render tests still pass.
- Manual smoke: enabling display for one running instance disables display for
  the previous visualized instance.

### Phase 5: Headless Battle-View Cost Reduction

Status: Planned.

Goals:

- Avoid unnecessary render-state construction for non-visualized background
  sessions.
- Preserve display behavior for the active visualized instance.

Work:

- Add an optional progress/display predicate or view-enabled callback to the
  training round path.
- Build and emit `battle_view` only when needed.
- Keep status counters and console metrics independent of battle-view emission.
- Update tests that currently assume battle-view payloads are always emitted.

Verification:

- Unit tests cover display-off sessions dropping or skipping battle views.
- Unit tests cover display-on sessions still producing frozen battle views.
- Manual observation: background instances run without live arena updates.

### Phase 6: Scaling Controls Up To 25 Instances

Status: Planned.

Goals:

- Make the UI and manager robust for at least 25 instances.
- Add warnings and diagnostic status without imposing a low hard limit.

Work:

- Add a configurable soft maximum and absolute supported maximum.
- Set the absolute supported maximum to at least 25.
- Add warning/confirmation when exceeding the soft maximum.
- Ensure selector remains usable with 25 instances.
- Ensure update loops iterate cleanly over all instances.
- Add aggregate status text such as `3 running / 8 total`.

Verification:

- Unit tests create 25 manager instances and select each one.
- Unit tests ensure labels remain unique enough for selection.
- Manual smoke with many stopped instances verifies layout and event handling.

### Phase 7: Performance Characterization

Status: Planned.

Goals:

- Measure whether multiple instances improve training throughput.
- Identify practical limits on CPU, GPU, and VRAM.

Work:

- Add lightweight instrumentation or a manual benchmark checklist.
- Compare batches/hour for 1, 2, 4, 8, and optionally 25 instances.
- Record GPU memory use where possible.
- Record whether existing-AI mode duplicates too much opponent-model memory.
- Identify whether minibatch size, replay updates, or session count is the best
  throughput lever.

Verification:

- Produce notes in a progress document or a follow-up section.
- No correctness tests required beyond smoke stability.

## Testing Strategy

### Unit Tests

Add or update tests for:

- `TrainingUIState` remains per-instance.
- Manager creates one default instance.
- Add/select/remove stopped instances.
- Pending removal of running instances.
- Same model slot cannot be trained by two running instances.
- Different model slots can train concurrently.
- Active instance controls reflect only active instance state.
- Display ownership is exclusive.
- Back requests stop on all running instances.
- 25 instances can be created and selected.

### Existing Tests To Preserve

Focused existing suites should continue to pass:

- `tests.test_train_ai_ui`
- `tests.test_training_session`
- `tests.test_training_orchestration`
- `tests.test_training_models`

Full discovery should be run when practical, but focused suites are acceptable
during early phases if unrelated pre-existing failures remain.

### Manual Smoke Checklist

Manual checks after implementation:

1. Start one instance and verify current behavior is unchanged.
2. Add a second instance with a different ship/slot and start it.
3. Switch between instances while both run.
4. Enable display on one, then on the other, and verify only one battle renders.
5. Stop one instance and verify the other keeps training.
6. Attempt to start two instances on the same ship/slot and verify the second is
   blocked.
7. Create at least 25 stopped instances and verify selection remains usable.
8. Optionally start many instances and observe GPU memory, CPU use, and training
   progress stability.

## Risks And Mitigations

### CPU Bottleneck

Risk:

The simulation, observation, and reward pipeline are Python-heavy. Multiple
threads may contend on the GIL, limiting scaling.

Mitigation:

Keep instance count configurable and characterize throughput before assuming
more sessions are better.

### GPU Memory Growth

Risk:

Each session owns a trainee model and optimizer state. Existing-AI mode can load
many opponent models, and multiple sessions can duplicate those models on CUDA.

Mitigation:

Start with warnings and clear failure notices. Later optimize with CPU opponent
models or shared caches if needed.

### Checkpoint Read/Write Races

Risk:

One session may read a checkpoint while another writes it.

Mitigation:

Checkpoint writes are already atomic. Preserve atomic writes and prevent
multiple writers for the same slot.

### UI Complexity

Risk:

`train_ai.py` is already large. Adding multi-instance behavior directly can make
it harder to maintain.

Mitigation:

Use small helper classes for instance management and keep rendering/event logic
separated where practical. Avoid broad unrelated refactors during each phase.

### Audio Leakage

Risk:

More than one session may attempt to start battle music if display ownership is
not centralized.

Mitigation:

All display toggles must go through the manager, which disables display for all
non-visualized instances.

## Open Design Questions

- Should the instance list persist across app restarts, or should every launch
  start with one fresh instance?
- Should an instance have an explicit user-editable name, or is ship/slot plus
  instance number enough?
- Should high instance counts require confirmation every time, or only once per
  training menu visit?
- Should opponent models be loaded on CPU for background instances by default?
- Should each instance eventually support explicit device selection?
- Should there be a global pause/resume-all control in addition to Back?

These questions do not block the first implementation. The first pass should
favor a simple in-memory manager and clear behavior over premature scheduling or
device-management abstractions.
