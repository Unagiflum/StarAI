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

## Pending Phases

- Phase 3: Save coordinator and save notifications.
- Phase 4: Training UI wiring.
- Phase 5: Diagnostics and regression tests.
