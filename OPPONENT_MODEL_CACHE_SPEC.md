# Shared Opponent Model Cache Specification

## Goal

Centralize existing-AI opponent model loading so concurrent training instances
share one loaded opponent model per ship/slot and refresh that shared model only
when a new checkpoint is saved by the running application.

The desired behavior is:

- Training instances in existing-AI opponent mode use a shared snapshot of
  loaded opponent models.
- A newly saved model becomes available to future training batches after the
  save completes and the cache successfully loads it.
- If a new checkpoint cannot be loaded, is blocked, or is still being saved,
  training continues with the previous known-good cached opponent model.
- If no known-good model exists for that slot, that slot is skipped and the
  existing opponent scheduler may fall back to simple behavior for that ship.
- Multiple training instances must not each load their own copy of the same
  `Earthling-01.pth` or equivalent checkpoint.

This document is the implementation source of truth for shared opponent model
caching and save-driven opponent refresh.

## Current System Summary

Existing-AI opponent mode currently discovers and loads stored AI opponents from
inside each `TrainingSession`:

- `TrainingSession._existing_ai_opponents_for_batch()` decides when the session
  should refresh its own cached opponent tuple.
- `discover_existing_ai_opponents()` scans the model repository and loads each
  available checkpoint into an `OpponentSpec`.
- `_load_opponent_model()` builds a new value network, loads a checkpoint, calls
  `eval()`, and returns that model.
- If a load fails, that slot is omitted from the new discovery result.
- The session replaces its previous cached opponent tuple with the new discovery
  result even if the new result lost models due to a transient load failure.
- Checkpoint and metadata writes use temp files and `os.replace()`, so partial
  target files are usually avoided, but there is no in-process save/load
  coordinator.
- UI-level writer reservations prevent two running instances from writing the
  same user model slot in the same training screen, but they do not coordinate
  opponent-model loads.

This is functional for one or a few sessions, but it duplicates model memory and
does not preserve the last known-good opponent model across reload failures.

## Requirements

### Shared Cache Ownership

Add a central opponent model cache owned above individual `TrainingSession`
instances.

The cache must:

- Be shared by all training instances in the current app process.
- Load at most one opponent model object per `(ship, slot)` key.
- Return immutable batch snapshots so active batches are not mutated by later
  cache updates.
- Be thread-safe for concurrent reads from training sessions and writes from
  save notifications.
- Treat loaded opponent models as read-only inference models.

### Initial Load

The cache must support an initial load from the repository.

Initial load is needed because previously saved models from earlier app runs
will not generate in-process save notifications.

Initial load should happen lazily when existing-AI opponent mode is first used
or explicitly when the training UI creates the shared cache. Either approach is
acceptable if tests verify that previously saved models are available without
requiring a new save.

### Save-Driven Refresh

Training checkpoint saves are relatively infrequent, so refresh should be
driven primarily by successful saves rather than repeated per-instance directory
scans.

After a training session finishes saving a checkpoint and metadata for a user
model slot, it must notify the central cache.

On notification, the cache should:

1. Resolve the updated slot from the repository.
2. Ignore empty, missing, bundled-read-only mismatch, or invalid slot state.
3. Try to load the checkpoint once.
4. If loading succeeds, replace the cached entry for that key.
5. If loading fails, keep the previous cached entry unchanged.
6. Record the load failure for diagnostics.

The notification must happen after the full save sequence completes. In the
current flow, this means after both `save_training_checkpoint(...)` and
`repository.create_or_update_user_model(...)` complete.

### Last Known-Good Behavior

The cache must preserve the last known-good model per key.

If `Earthling-01` batch 100 is cached and batch 125 is saved but fails to load,
training instances should continue using the cached batch 100 model. The failed
load must not remove or replace the cached model.

If there is no cached model for a failed key, the key should remain absent from
the opponent snapshot. The scheduler's existing simple fallback behavior can
then handle ships with no trained AI option.

### Save/Load Coordination

Add an in-process save coordinator for model keys.

The coordinator should allow code to mark a model key as currently saving:

```python
with save_coordinator.saving((ship, slot)):
    ...
```

Cache loads should not read a checkpoint while the same key is marked as saving.
The preferred behavior for training is non-blocking:

- If a cache refresh request targets a key currently being saved, defer or skip
  the load attempt for that notification.
- Keep the previous cached model.
- Record that the refresh was blocked by an active save.
- A later notification after save completion should perform the actual load.

If the implementation structure guarantees notifications are only emitted after
the save coordinator exits, then blocked refreshes should be rare. The cache
should still handle them defensively.

### Opponent Snapshot Semantics

Training sessions should request an opponent snapshot at batch boundaries.

A snapshot should be a tuple of `OpponentSpec` objects that point to the shared
loaded opponent models available at that moment.

The snapshot must remain stable for the duration of a batch:

- If a newer model is loaded mid-batch, the active batch continues using the
  model objects in its snapshot.
- The newer model becomes visible on the next batch boundary.

### Shared Inference Safety

Shared opponent models must not be trained or mutated by training instances.

The current generic inference helper calls `model.eval()` during prediction.
For shared opponent models, avoid repeated mode mutation from multiple threads.
The implementation should either:

- Ensure cached models are set to eval mode once at load time and use an
  opponent-specific inference helper that does not call `model.train()` or
  mutate model mode, or
- Wrap cached models in an adapter that owns read-only inference and enforces
  `torch.no_grad()` or `torch.inference_mode()`.

Training models owned by a `TrainingSession` may continue using the existing
training helpers.

### Diagnostics

The cache should expose enough diagnostic information for tests and future UI
notices:

- Loaded model keys.
- Last successful load identity per key, such as completed batch, checkpoint
  size, or modified timestamp.
- Last load error per key.
- Whether the latest attempted refresh was blocked by an active save.

The first implementation does not need to add a visible UI panel for these
diagnostics.

## Non-Goals

The first implementation does not need to support:

- Cross-process locking between multiple running StarAI processes.
- Automatic detection of externally copied model files while the app is running.
- Background file watchers.
- Per-instance device selection.
- Global GPU scheduling or stream management.
- Unloading least-recently-used opponent models.
- Live mutation of an already-running batch's opponent list.

External model changes can be handled later by a manual refresh button or a
low-frequency scan if needed.

## Proposed Architecture

### New Module

Create a new module such as `src/training/opponent_cache.py`.

Suggested types:

```python
@dataclass(frozen=True)
class OpponentModelKey:
    ship: str
    slot: int


@dataclass(frozen=True)
class OpponentCacheEntry:
    key: OpponentModelKey
    model: Any
    description: str
    completed_batches: int | None
    checkpoint_size: int | None
    checkpoint_mtime_ns: int | None


@dataclass(frozen=True)
class OpponentCacheDiagnostics:
    loaded_keys: tuple[OpponentModelKey, ...]
    last_errors: Mapping[OpponentModelKey, str]
    blocked_keys: tuple[OpponentModelKey, ...]
```

Suggested cache API:

```python
class OpponentModelCache:
    def load_initial(self, repository: TrainingModelRepository) -> None:
        ...

    def notify_model_saved(
        self,
        repository: TrainingModelRepository,
        ship: str,
        slot: int,
    ) -> None:
        ...

    def snapshot(self) -> tuple[OpponentSpec, ...]:
        ...

    def diagnostics(self) -> OpponentCacheDiagnostics:
        ...
```

The cache should use a `threading.RLock` or equivalent to protect its internal
entry and diagnostics dictionaries.

### Save Coordinator

Create a small in-process coordinator, either in the same module or a dedicated
module.

Suggested API:

```python
class ModelSaveCoordinator:
    def saving(self, key: OpponentModelKey):
        ...

    def is_saving(self, key: OpponentModelKey) -> bool:
        ...
```

The coordinator should support nested or repeated saves defensively by tracking
counts per key rather than only a boolean.

### Session Integration

`TrainingSession` should accept optional cache/coordinator dependencies:

```python
TrainingSession(
    ...,
    opponent_model_cache: OpponentModelCache | None = None,
    save_coordinator: ModelSaveCoordinator | None = None,
)
```

Behavior:

- Existing single-session tests and callers should continue to work when these
  dependencies are omitted.
- Existing-AI mode should use the shared cache snapshot when a cache is
  provided.
- Legacy `discover_existing_ai_opponents()` may remain for tests or fallback,
  but multi-instance UI should pass the shared cache to every session.
- `_save_state()` should wrap checkpoint and metadata writes in
  `save_coordinator.saving(key)` when a coordinator is provided.
- `_save_state()` should call `opponent_model_cache.notify_model_saved(...)`
  after a successful save when a cache is provided.

### UI Integration

`src/Menus/train_ai.py` should create one shared opponent cache and one save
coordinator per training screen run, then pass them into every `TrainingSession`
created by that UI.

This scope is intentionally per app/UI process. It avoids global mutable state
that can leak across tests while still sharing models across all instances in
one training screen.

## Implementation Phases

### Phase 1: Cache Data Model And Initial Load

Goal:

Introduce the shared cache without changing save behavior yet.

Tasks:

- Add `src/training/opponent_cache.py`.
- Move or reuse checkpoint-loading logic so the cache can load one slot into an
  eval-only opponent model.
- Implement `OpponentModelCache.load_initial(repository)`.
- Implement `OpponentModelCache.snapshot()`.
- Preserve the existing `discover_existing_ai_opponents(repository)` public
  behavior, either by leaving it unchanged or by having it use a temporary cache.
- Add unit tests proving one initial load creates one shared model per slot.
- Add unit tests proving `snapshot()` returns stable immutable tuples.

Acceptance criteria:

- Existing orchestration/session/model tests pass.
- Previously saved user models can appear in a cache snapshot after initial
  load.
- Repeated snapshots do not reload duplicate model objects.

### Phase 2: Session Reads From Shared Cache

Goal:

Allow training sessions to use the central cache at batch boundaries.

Tasks:

- Add optional `opponent_model_cache` to `TrainingSession`.
- Update `_existing_ai_opponents_for_batch()` so existing-AI mode uses
  `opponent_model_cache.snapshot()` when present.
- Keep the current session-local discovery fallback for callers that do not
  provide a cache.
- Remove or bypass per-session opponent reload cadence when a shared cache is
  present; the cache is already current based on initial load and save events.
- Add tests with two sessions sharing one cache and seeing the same model
  object for the same slot.
- Add tests proving a batch uses the snapshot captured at batch start even if
  the cache updates later.

Acceptance criteria:

- Two concurrent or sequential sessions using existing-AI mode do not load
  duplicate opponent objects for the same key.
- Sessions without a cache retain current behavior.

### Phase 3: Save Coordinator And Save Notifications

Goal:

Refresh shared opponent models when training saves a new checkpoint.

Tasks:

- Add `ModelSaveCoordinator`.
- Add optional `save_coordinator` to `TrainingSession`.
- Wrap `_save_state()` checkpoint and metadata writes with the save coordinator
  when provided.
- After a successful save, notify the cache using
  `opponent_model_cache.notify_model_saved(repository, ship, slot)`.
- Ensure notification happens after metadata has been written.
- If notification load fails, keep the old cache entry.
- If notification is blocked by an active save, keep the old cache entry and
  record diagnostics.

Acceptance criteria:

- Saving a newer model updates the shared opponent cache exactly once.
- If loading the newly saved model raises, the old cached model remains in the
  snapshot.
- If a key is marked saving, refresh does not read that key and does not evict
  its old model.
- A failed save does not notify the cache as though a new model is available.

### Phase 4: Training UI Wiring

Goal:

Make all instances in the training UI share the same cache and save
coordinator.

Tasks:

- Instantiate one `OpponentModelCache` and one `ModelSaveCoordinator` in
  `train_ai.run()`.
- Perform lazy or eager initial cache load before existing-AI snapshots are
  needed.
- Pass the cache and coordinator into every `TrainingSession`.
- Keep writer reservations unchanged; they still prevent same-slot training
  writers.
- Ensure closing/stopping one instance does not destroy the cache while other
  instances are still training.

Acceptance criteria:

- Multiple UI-created sessions share opponent models.
- Stopping one instance does not remove shared opponent entries used by other
  instances.
- Existing single-instance UI behavior remains unchanged from the user's point
  of view.

### Phase 5: Diagnostics And Regression Tests

Goal:

Make the behavior observable enough to maintain.

Tasks:

- Add tests for cache diagnostics after successful load, failed load, and
  blocked refresh.
- Add tests for keeping last known-good entries after failed refresh.
- Add tests for no-model fallback: if no cached model exists and load fails,
  the slot is absent from the snapshot and schedule can use simple behavior.
- Add tests for read-only opponent inference if an adapter/helper is introduced.
- Update progress documentation after implementation.

Acceptance criteria:

- Focused suites pass:
  `python -m unittest tests.test_training_orchestration tests.test_training_session tests.test_training_models tests.test_train_ai_ui`
- Tests explicitly cover the desired "newer if available, old if not" behavior.

## Open Design Questions

- Should initial load be eager when the training screen opens, or lazy on first
  existing-AI batch? Lazy loading keeps simple-opponent training startup cheaper;
  eager loading surfaces load errors earlier.
- Should cache entries include bundled default models if bundled models are ever
  valid existing-AI opponents, or only user models?
- Should the cache load opponent models on the same device as training
  instances, or should CPU inference be considered later to reduce VRAM
  duplication and contention?
- Should there be a manual "Refresh AI Opponents" UI action for externally
  modified model files?

## Suggested Test List

- `OpponentModelCache.load_initial()` skips empty slots and loads available user
  checkpoints.
- `OpponentModelCache.snapshot()` returns the same model object across repeated
  snapshots.
- Two `TrainingSession` instances using the same cache receive shared opponent
  model objects.
- `notify_model_saved()` promotes a newer checkpoint after save completion.
- `notify_model_saved()` keeps the previous entry when loading the new
  checkpoint fails.
- `notify_model_saved()` leaves a missing key absent when there is no previous
  entry and loading fails.
- Refresh blocked by `ModelSaveCoordinator.saving(key)` records blocked
  diagnostics and keeps the old entry.
- `_save_state()` notifies the cache only after successful checkpoint and
  metadata writes.
- Existing callers without a cache continue using current discovery behavior.
- Existing-AI schedule still falls back to simple behavior for ships without a
  cached trained opponent.
