# Multi-Instance Training Progress

Last updated: 2026-07-12

## Phase 1: Instance Data Model And Single-Instance Compatibility

Status: Complete.

Completed:

- Added `TrainingInstance` to hold per-instance UI state, session ownership,
  pending removal state, and last-running state.
- Added `TrainingInstanceManager` with one default active instance to preserve
  current single-instance behavior.
- Routed the training UI's active state, start/stop/back/display paths, session
  continuity, and run-loop status tracking through the active instance.
- Removed the direct single-session closure cells from `train_ai.py`.
- Added focused unit coverage for default manager construction and active-session
  continuity clearing.
- Verified focused UI/session suites pass:
  `python -m unittest tests.test_train_ai_ui tests.test_training_session`.

Remaining:

- Manual smoke of the training screen remains pending.

## Phase 2: Instance Selector UI

Status: Complete.

Completed:

- Added the compact instance strip above the training tabs with active-position
  indicator, dropdown selector, `Close Instance`, and `Add Instance` controls.
- Added dropdown row formatting with ordered instance numbers, ship-slot labels,
  dashes for incomplete selections, and colored status text.
- Added stopped-instance add, select, and remove operations to
  `TrainingInstanceManager`.
- Wired active-instance switching through UI-to-state and state-to-UI sync so
  visible tabs and controls swap to the selected instance's saved state.
- Preserved unsaved stopped-instance slot-label text when switching instances.
- Kept the final remaining instance from being closed.
- Added dropdown scrolling for longer instance lists.
- Added focused unit coverage for manager add/select/remove behavior, instance
  row formatting, and instance-strip layout non-overlap.
- Verified focused UI/session suites pass:
  `python -m unittest tests.test_train_ai_ui tests.test_training_session`.

Remaining:

- Manual smoke of add/select/remove behavior remains pending.

## Phase 3: Multi-Session Start/Stop And Writer Reservations

Status: Complete.

Completed:

- Added UI-thread writer reservations keyed by `(ship, slot)` to prevent two
  running or stopping instances from writing the same user model slot.
- Routed active Start/Stop through instance-scoped manager helpers so stopping
  the active instance leaves other running instances alone.
- Updated the footer Back control to become `Stop All` when background
  instances are running, then confirm before requesting stops for every running
  instance.
- Added confirmation to the enabled `Back` path before leaving the training
  screen.
- Added running-instance close behavior: display is disabled first, stop is
  requested, and the instance remains pending removal until its session stops.
- Added automatic cleanup for stopped pending-removal instances and stopped
  writer reservations across all instances, not just the active one.
- Added a replacement instance when closing the only running instance so the UI
  never has zero open instances.
- Added focused unit coverage for distinct-slot concurrent reservations,
  same-slot conflicts, active-only stop, Back action selection, stop-all,
  pending-removal cleanup, and display-off-before-close behavior.
- Verified focused suites pass:
  `python -m unittest tests.test_train_ai_ui tests.test_training_session`.
- Verified related model/orchestration suites pass:
  `python -m unittest tests.test_training_orchestration tests.test_training_models`.

Remaining:

- Manual smoke of multi-instance start/stop, Stop All, and running close
  behavior remains pending.

## Phase 4: Single Visualization Ownership

Status: Complete.

Completed:

- Routed Display On changes through `TrainingInstanceManager` so enabling
  display for the active instance disables display for every other instance.
- Updated instance add/select/remove/close cleanup paths to force visualization
  off when active ownership changes.
- Replaced remaining active-session display toggles in the UI loop with manager
  calls.
- Preserved active-instance-only battle and HUD rendering.
- Added focused unit coverage for display ownership transfer and
  active-instance switching turning display off.
- Verified focused suites pass:
  `python -m unittest tests.test_train_ai_ui tests.test_training_session`.
- Verified related model/orchestration suites pass:
  `python -m unittest tests.test_training_orchestration tests.test_training_models`.

Remaining:

- Manual smoke of visualization switching and audio behavior remains pending.

## Phase 5: Headless Battle-View Cost Reduction

Status: Complete.

Completed:

- Preserved the training-round battle-view predicate path so battle-view payloads
  are only built and emitted when display is enabled for the visualized session.
- Ensured disabled battle-view progress still emits frame/status metrics for the
  console and session status.
- Updated the older display throttle path so an explicitly disabled battle-view
  predicate cannot sleep as a visualized round.
- Preserved display-on frozen battle-view storage and display-off dropped-view
  behavior in `TrainingSession`.
- Added focused unit coverage for disabled battle-view payload construction and
  disabled display-throttle behavior.
- Verified focused suites pass:
  `python -m unittest tests.test_training_orchestration tests.test_training_session`.
- Verified focused UI suite passes:
  `python -m unittest tests.test_train_ai_ui`.

Remaining:

- Manual observation of background instance speed without live arena updates
  remains pending.

## Later Phases

Status: Phase 6 not started.

- Phase 6: Scaling controls up to 25 instances.
- Phase 7: Performance characterization.
