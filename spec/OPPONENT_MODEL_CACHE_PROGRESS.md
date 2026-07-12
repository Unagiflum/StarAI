# Opponent Model Cache Progress

## Phase 1: Cache Data Model And Initial Load

Status: Implemented and verified.

- Added `src/training/opponent_cache.py` with cache key, entry, diagnostics, and shared cache types.
- Implemented `OpponentModelCache.load_initial(repository)` for lazy initial repository loading.
- Implemented `OpponentModelCache.snapshot()` returning immutable tuples of shared opponent specs.
- Moved reusable opponent checkpoint loading into the cache module and kept `discover_existing_ai_opponents(repository)` behavior intact through a compatibility delegate.
- Added tests proving repeated loads and snapshots reuse the same model object per slot.
- Added tests proving earlier snapshot tuples remain stable when later initial loads add another model.
- Verified with `python -m unittest tests.test_training_models tests.test_training_orchestration tests.test_training_session`.

## Phase 2: Session Reads From Shared Cache

Status: Implemented and verified.

- Added optional `opponent_model_cache` dependency to `TrainingSession`.
- Updated existing-AI opponent discovery so sessions with a shared cache lazily call `load_initial(repository)` once, then use `opponent_model_cache.snapshot()` at each batch boundary.
- Kept the legacy session-local `discover_existing_ai_opponents(repository)` cadence unchanged for sessions without a cache.
- Added tests proving two sessions sharing one cache receive the same loaded opponent model object for a slot.
- Added tests proving the batch runner receives the immutable snapshot captured at batch start and later batches can see cache updates.
- Verified with `python -m unittest tests.test_training_session tests.test_training_models tests.test_training_orchestration`.

## Phase 3: Save Coordinator And Save Notifications

Status: Implemented and verified.

- Added `ModelSaveCoordinator` with counted per-key save tracking and context-manager support for `(ship, slot)` model keys.
- Added `OpponentModelCache.notify_model_saved(repository, ship, slot)` to refresh one saved user model slot without evicting the previous known-good cached model on load failure.
- Added blocked-refresh handling so notifications or initial loads received while a key is actively saving keep the old entry and record blocked diagnostics.
- Added optional `save_coordinator` dependency to `TrainingSession`.
- Wrapped `_save_state()` checkpoint and metadata writes in the coordinator when provided.
- Notified the shared opponent cache only after checkpoint save and metadata update completed successfully.
- Added tests proving saved models refresh the cache, failed refreshes keep the old model, blocked refreshes do not load or evict, and failed saves do not notify.
- Verified with `python -m unittest tests.test_training_models tests.test_training_session tests.test_training_orchestration`.

## Phase 4: Training UI Wiring

Status: Implemented and verified.

- Created one `ModelSaveCoordinator` and one coordinator-backed `OpponentModelCache` per `train_ai.run()` invocation.
- Passed the shared cache and coordinator into every `TrainingSession` created by the training UI.
- Kept writer reservations unchanged so same-slot writers remain blocked independently of shared opponent reads.
- Added a UI regression test proving the run path passes an `OpponentModelCache` and matching `ModelSaveCoordinator` into a started session.
- Verified with `python -m unittest tests.test_training_orchestration tests.test_training_session tests.test_training_models tests.test_train_ai_ui`.

## Pending Phases

- Phase 5: Diagnostics and regression tests.
