# Multi-Process Coordinated Simulation Specification

## Goal

Add an optional coordinated training mode that uses multiple CPU worker
processes for battle simulation while keeping one parent scheduler responsible
for all model, GPU, replay, optimizer, checkpoint, and UI state.

The purpose is to use multi-core CPUs during coordinated training without
breaking the model-save and opponent-cache coordination already provided by the
single parent process.

The first implementation should preserve the existing coordinated training
semantics:

- `Start All` still launches one coordinated training session.
- The parent process still owns all trainable models and all read-only opponent
  models.
- The parent process still performs batched trainee inference, batched opponent
  inference, replay insertion, optimization, metrics, and saves.
- Worker processes own only CPU battle simulation window state.
- Workers do not read, write, cache, or reload model checkpoint files.
- The feature must be optional and fall back to the current in-process
  coordinated scheduler when workers are unavailable or disabled.

This document is the implementation source of truth for CPU worker process
parallelism in coordinated training.

## Motivation

The coordinated scheduler has improved GPU behavior by batching model work, but
the remaining cost is CPU-heavy Python work. Current coordinated training runs
inside one scheduler thread and advances active battle windows one frame at a
time. Because battle simulation is mostly Python code, adding more threads would
not meaningfully use multiple CPU cores under the Python GIL.

Launching multiple independent StarAI training processes is not the right first
solution. The current model-store safety mechanisms are process-local:

- `ModelSaveCoordinator` tracks saves only inside one Python process.
- `OpponentModelCache.notify_model_saved()` invalidates only the cache in the
  same process.
- Duplicate-writer reservation is owned by the running app instance.
- Checkpoint and metadata writes are atomic at the file level, but there is no
  cross-process stale-cache notification protocol.

Therefore, the safe design is one parent process with multiple CPU-only worker
processes. The parent remains the sole authority for model state and storage,
while workers supply parallel simulation throughput.

## Current System Summary

Relevant current surfaces:

- `src/training/coordinated.py`
  - `CoordinatedTrainingSession` owns one scheduler thread.
  - `_run_one_coordinated_batch()` creates active window runtimes, collects
    observations, runs batched inference, advances windows, optimizes, saves,
    and emits metrics.
  - `_CoordinatedWindowRuntime` owns one active battle window, including
    `BattleSimulation`, event ledger, reward pipeline, simple opponent
    controller, progress counters, and episode accounting.
  - `_advance_coordinated_window_frame()` performs one full CPU frame step:
    reward decision capture, simulation step, reward/outcome processing,
    replay insertion, progress emission, and terminal reset.
- `src/Battle/battle.py`
  - `BattleSimulation.step()` performs ship input handling, tracking updates,
    object updates, collision handling, and aftermath updates.
- `src/training/opponent_cache.py`
  - `OpponentModelCache` and `ModelSaveCoordinator` are currently in-process
    coordination tools.
- `src/training/replay.py`
  - Replay buffers and checkpoint saves are owned by training sessions.

The multi-process feature must not change independent single-instance training.
It should extend only the coordinated `Start All` runtime path.

## Architecture

### Parent Scheduler

The parent scheduler remains the authoritative training process. It owns:

- training records and statuses
- trainable value networks
- optimizers
- replay buffers
- opponent model cache
- model save coordinator
- batched trainee inference
- batched existing-AI opponent inference
- end-of-batch optimization
- checkpoint and metadata saves
- CSV metrics and timing stats
- UI-visible histories and log lines

The parent starts a persistent worker process for each active coordinated record
unless a configured worker limit is lower than the instance count. For the first
implementation, one worker per active instance is acceptable and preferred.
Windows should be left to schedule worker processes across available CPU cores;
manual CPU affinity is not required.

### Worker Processes

Each worker process owns CPU-only battle window state for one coordinated
record at a time:

- trainee ship and opponent ship instances
- `BattleSimulation`
- `BattleEventLedger`
- `RollingReturnPipeline`
- `SimpleOpponentController`
- frame and episode counters
- component totals
- current observation snapshots

Workers must not:

- load trainable model checkpoints
- load opponent model checkpoints
- hold PyTorch models
- write model files
- mutate parent replay buffers directly
- call optimizer steps
- emit final CSV rows directly

Workers may import modules that indirectly support PyTorch, but the worker
protocol must not require CUDA initialization or model construction in child
processes.

### Process Count

Initial default:

- spawn one worker process per coordinated record
- keep each worker alive for the whole coordinated run
- reuse the same worker across batches and rounds

Future extension:

- add a cap such as `min(instance_count, cpu_worker_count)`
- schedule multiple record windows onto a smaller worker pool

The first version should avoid worker-pool scheduling complexity unless process
startup or memory use becomes a measured problem.

## Worker Protocol

Use explicit command/result messages over `multiprocessing` connections or
queues. Messages must be plain picklable dataclasses or dictionaries containing
only simple Python data and project dataclasses that are already picklable.

The parent must treat worker messages as an internal protocol with versioned
message names. Invalid messages should fail the coordinated run cleanly.

### Commands

`START_RUN`

- Sent once after worker creation.
- Provides immutable process-level setup:
  - worker id
  - record id
  - base RNG seed
  - any reduced config needed by simulation

`START_WINDOW`

- Creates or resets worker window state for one round/opponent.
- Provides:
  - record id
  - round index
  - `TrainingOrchestrationConfig` fields needed for simulation and rewards
  - opponent ship
  - opponent mode metadata without model objects
  - frame limit
  - deterministic per-window RNG seed

`REQUEST_OBSERVATION`

- Asks the worker for the current trainee observation and any simple-opponent
  control that can be computed without model inference.
- Returns a `WINDOW_OBSERVATION` result.

`STEP_FRAME`

- Advances one worker window by one frame using parent-selected actions.
- Provides:
  - trainee action index
  - trainee exploratory flag
  - optional trainee action values for diagnostics
  - opponent direct controls selected by parent when opponent is model-backed
  - stop token or command sequence number
- Returns a `FRAME_STEPPED` result.

`FINISH_WINDOW`

- Flushes timeout reward state and returns the completed window result.
- Returns `WINDOW_FINISHED`.

`SHUTDOWN`

- Requests graceful worker exit.

### Results

`WORKER_READY`

- Worker process started and accepted setup.

`WINDOW_STARTED`

- Window state exists and is ready for observation requests.

`WINDOW_OBSERVATION`

- Contains:
  - record id
  - round index
  - frame count
  - trainee observation tuple
  - optional opponent observation tuple when opponent model inference is needed
  - simple opponent direct controls when no opponent model is needed
  - complete flag

`FRAME_STEPPED`

- Contains:
  - record id
  - frame count
  - complete flag
  - progress payload equivalent to `_emit_window_progress()`
  - mature replay samples produced by this frame
  - reward component deltas and timing details
  - terminal episode data, if a terminal reset occurred
  - next observation if the worker can cheaply include it

`WINDOW_FINISHED`

- Contains data needed to construct `CoordinatedFixedFrameWindowResult`.
- Contains any final matured replay samples produced by timeout flush.

`WORKER_ERROR`

- Contains:
  - record id
  - command name
  - exception type
  - exception message
  - formatted traceback

`WORKER_STOPPED`

- Worker acknowledged shutdown.

## Scheduling Flow

The parent coordinated batch loop should keep the same high-level rhythm as the
current in-process scheduler:

1. Build the per-record opponent schedules in the parent.
2. Start one worker window per active record for the current round.
3. Request observations from unfinished workers.
4. Batch trainee observations in the parent using `select_actions_for_records()`.
5. Batch model-backed opponent observations in the parent using the existing
   opponent batching helper.
6. Send one `STEP_FRAME` command to each unfinished worker with selected
   trainee action and opponent controls.
7. Receive frame results, append returned replay samples into the parent-owned
   replay buffers, update parent status, and merge timing counters.
8. Repeat until all active windows are complete.
9. Finish all windows, construct per-record batch results, and run the existing
   synchronized optimization/save logic in the parent.

Workers may step frames concurrently after actions are sent. The parent should
wait for all frame results for a scheduler frame before starting the next
batched inference cycle. This preserves the current lockstep coordinated
behavior and keeps GPU batching simple.

## Replay Ownership

Replay buffers stay in the parent process.

Workers return mature replay samples as values. The parent appends those samples
to the correct record's replay buffer after receiving `FRAME_STEPPED` or
`WINDOW_FINISHED`.

This avoids:

- shared mutable replay buffers
- cross-process replay serialization during saves
- child processes needing checkpoint save permissions

If per-frame sample transfer becomes expensive, the protocol may later batch
samples across multiple frames. The first version should prefer correctness and
clear timing over aggressive IPC batching.

## Opponent Models

Existing-AI opponent models stay in the parent process.

When a worker needs opponent model inference:

1. Worker returns an opponent observation in `WINDOW_OBSERVATION`.
2. Parent batches all opponent observations for that scheduler frame.
3. Parent selects direct opponent controls.
4. Parent sends controls in `STEP_FRAME`.

Simple opponents may remain worker-local because they do not require model
state. Workers may compute simple opponent controls directly and include them in
the observation result.

## Randomness And Determinism

The parent must generate deterministic seeds for worker windows. A recommended
seed tuple is:

- coordinated run seed
- record id
- batch number
- round index
- terminal episode reset count

Workers should not share a parent `random.Random` object. Each worker window
should construct local RNG state from parent-provided seeds.

The first version does not need bit-for-bit equivalence with the in-process
coordinated path, but repeated runs with the same seed and same worker count
should be reproducible where the current training path is reproducible.

## Stop And Error Behavior

`Stop All` should:

- set the parent stop event
- stop issuing new frame work
- send `SHUTDOWN` or cancellation messages to active workers
- join workers with a short timeout
- terminate unresponsive workers
- save only already completed unsaved batches, matching current coordinated
  behavior

Worker errors should:

- fail the whole coordinated run
- set every included record status to stopping/error
- include the worker traceback in parent logs or diagnostics
- shut down all workers
- avoid partial batch completion

If a worker process exits unexpectedly, treat it as `WORKER_ERROR`.

## Timing Instrumentation

Existing coordinated timing buckets should remain comparable. Add worker-aware
timing fields without removing current fields:

- worker IPC send seconds
- worker IPC receive/wait seconds
- worker simulation wall seconds
- worker reward wall seconds
- worker observation encode seconds
- max per-frame worker wait seconds
- sum of worker CPU-task seconds
- parent scheduler idle/wait seconds

The current `simulation`, `reward`, and `observation` totals should continue to
represent end-to-end scheduler time where useful. Additional worker sum metrics
should show parallel work performed across processes. For example, seven
workers can legitimately report worker simulation sum greater than wall-clock
batch time.

CSV rows should identify whether the batch used:

- in-process coordinated runtime
- multi-process coordinated runtime
- worker count

## Configuration

Add a conservative runtime switch before making this the default.

Suggested settings:

- `coordinated_cpu_workers_enabled`
- `coordinated_cpu_worker_count`

Initial behavior:

- default disabled until tests and manual smoke are stable
- when enabled with count `0` or `auto`, use one worker per active coordinated
  record
- if worker startup fails, either block start with a clear validation error or
  fall back to in-process mode with an explicit status/log message

Avoid exposing advanced process-affinity settings in the first implementation.

## Platform Notes

Windows uses spawn semantics for multiprocessing. Worker entry points must be
import-safe:

- no process creation at module import time
- no reliance on forked parent state
- all worker setup must be passed explicitly through commands
- worker functions must be defined at module top level

Packaged builds must include any worker entry module and resources needed by
headless battle simulation. PyInstaller verification is required before making
the feature default.

## Testing Requirements

### Unit Tests

Add focused tests for:

- command/result serialization
- worker lifecycle startup and shutdown
- starting a window and returning an observation
- stepping one frame and returning mature samples/progress
- terminal reset inside a fixed window
- timeout flush on `FINISH_WINDOW`
- worker error propagation
- parent replay insertion from worker-returned samples
- parent model ownership: workers receive no model objects
- fallback to in-process coordinated path

### Integration Tests

Add coordinated-session tests using a fake or lightweight worker client:

- parent batches trainee inference from multiple worker observations
- parent batches model-backed opponent inference from multiple worker
  observations
- one coordinated batch completes and records per-instance metrics
- stop mid-batch discards in-progress batch and saves only completed unsaved
  progress

Add at least one real multiprocessing smoke test guarded so it can be skipped in
restricted environments.

### Manual Verification

Manual performance runs should compare:

- current in-process coordinated runtime
- multi-process runtime with one worker per instance
- worker counts below instance count, if implemented

Capture:

- aggregate batches/hour
- per-instance batches/hour
- GPU utilization
- CPU utilization across cores
- timing CSV bucket shares
- worker crash/stop behavior

## Implementation Phases

### Phase 1: Protocol And Worker Skeleton

- Add worker command/result dataclasses.
- Add a top-level worker process entry point.
- Implement startup, shutdown, and error reporting.
- Add serialization and lifecycle tests.

### Phase 2: Single-Worker Window Parity

- Move enough `_CoordinatedWindowRuntime` behavior behind a worker-compatible
  boundary.
- Support `START_WINDOW`, `REQUEST_OBSERVATION`, `STEP_FRAME`, and
  `FINISH_WINDOW` for one worker.
- Keep parent replay insertion and status updates.
- Prove one worker can complete a fixed-frame window with the same semantic
  outputs as the in-process helper.

### Phase 3: Parent Scheduler Integration

- Add an optional worker-backed path in `CoordinatedTrainingSession`.
- Preserve the existing in-process path as fallback.
- Run one worker per active record.
- Keep parent-side batched trainee and opponent inference.
- Complete one coordinated batch through workers.

### Phase 4: Stop, Error, And Packaging Hardening

- Implement robust stop/terminate behavior.
- Handle worker exceptions and unexpected exits.
- Verify PyInstaller packaging.
- Add diagnostics for worker status and failures.

### Phase 5: Performance Characterization

- Add timing CSV fields for runtime mode and worker metrics.
- Run controlled timing comparisons.
- Decide whether the worker path should become the default for coordinated GPU
  training.

## Non-Goals

The first implementation should not:

- run multiple independent full training apps against the same model directory
- implement cross-process model save locks
- implement cross-process opponent cache invalidation
- move model inference into workers
- move replay buffers into workers
- pin workers to specific CPU cores
- redesign the UI beyond a minimal enable/disable setting and status messages

## Open Questions

- Should worker IPC use one `multiprocessing.Pipe` per worker or command/result
  queues? Pipes are simpler for one-parent/one-worker request-response flows;
  queues may be easier if a capped worker pool is added later.
- Should workers return the next observation as part of `FRAME_STEPPED` to
  remove one request/response cycle per frame?
- Should replay samples be returned every frame or batched over several frames
  to reduce IPC overhead?
- Should the first user-facing switch live in configuration metadata, the Batch
  tab, or a developer-only constant until performance is proven?
