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
- Running-instance close/pending-removal behavior is deferred to Phase 3.

## Later Phases

Status: Not started.

- Phase 3: Multi-session start/stop and writer reservations.
- Phase 4: Single visualization ownership.
- Phase 5: Headless battle-view cost reduction.
- Phase 6: Scaling controls up to 25 instances.
- Phase 7: Performance characterization.
