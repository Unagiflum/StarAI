# Multi-Process Coordinated Simulation Progress

Last updated: 2026-07-14

## Phase 1: Protocol And Worker Skeleton

Status: complete

- Added `src/training/process_worker.py` with picklable command/result dataclasses for startup, window control, observation requests, frame stepping, finish, shutdown, and worker errors.
- Added a top-level `worker_process_main()` entry point that uses an explicit request/response connection loop and reports command failures as `WORKER_ERROR` results with traceback text.
- Added `start_worker_process()` for spawn-safe worker creation using one parent/child pipe.
- Enforced the model-ownership boundary by rejecting `START_WINDOW` commands that include opponent model objects.
- Added lifecycle tests for spawn-process startup and graceful shutdown.
- Added serialization tests for representative command/result messages.

## Phase 2: Single-Worker Window Parity

Status: complete

- Added `CoordinatedSimulationWorker`, which owns one CPU-only coordinated battle window at a time.
- Reused the existing coordinated battle construction, frame-advance, terminal reset, and finish helpers so worker semantics stay aligned with the in-process scheduler.
- Added a worker-local replay sample collector that returns matured `ReplaySample` values to the caller without mutating parent replay buffers.
- Implemented `START_WINDOW`, `REQUEST_OBSERVATION`, `STEP_FRAME`, and `FINISH_WINDOW` handling for a single worker.
- Supported worker-local simple-opponent direct controls and parent-supplied controls for model-backed opponents.
- Returned per-frame progress payloads, matured replay samples, terminal episode data, and optional next observations from `STEP_FRAME`.
- Added tests proving a worker can start a fixed-frame window, return observations, step frames, return samples for parent replay insertion, handle terminal reset inside a fixed window, flush timeout samples on finish, and report command errors.

## Phase 3: Parent Scheduler Integration

Status: complete

- Added an opt-in worker-backed coordinated scheduler path on `CoordinatedTrainingSession` via `coordinated_cpu_workers_enabled`, `coordinated_cpu_worker_count`, and an injectable worker client factory.
- Preserved the existing in-process coordinated scheduler as the default path when CPU workers are not enabled.
- Added a parent-owned `_ProcessWorkerClient` wrapper for persistent spawned simulation workers with explicit startup, request/response, unexpected-exit detection, graceful shutdown, and termination of unresponsive workers.
- Implemented one worker per active coordinated record for the first worker-backed scheduler version.
- Kept parent-side trainable model ownership, trainee batched inference, model-backed opponent batched inference, replay insertion, optimization, saving, metrics, and status updates.
- Stripped model objects from `START_WINDOW` commands and added parent-side opponent-observation batching for model-backed worker opponents.
- Returned worker mature replay samples into the parent replay buffers and adjusted progress payload replay sizes to reflect parent-owned buffers.
- Added tests proving worker-backed batches complete through the parent scheduler, parent action batching is preserved, worker samples are inserted into parent replay buffers, and per-instance batch status advances.

## Phase 4: Stop, Error, And Packaging Hardening

Status: complete

- Worker receive loops now honor coordinated stop requests and abort in-progress batches without recording partial completion.
- Worker errors and unexpected exits are surfaced as coordinated run failures, with worker tracebacks included when supplied by the worker protocol.
- Active workers are shut down in `finally` paths for completed, stopped, and failed worker-backed batches; real process clients terminate workers that do not exit after shutdown.
- Added a scheduler test proving worker protocol errors abort the batch, leave completed batch counts unchanged, and still shut down all started workers.
- Added `multiprocessing.freeze_support()` to the packaged entry point for Windows spawned workers.
- Added `src.training.process_worker` to all PyInstaller specs as an explicit hidden import.
- Extended the packaged smoke test to import and validate the worker process entry point.

## Verification

- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_coordinated_training.py`
- Result: 25 passed
- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_train_ai_ui.py tests\test_training_session.py`
- Result: 122 passed
- Passed: `.\.venv\Scripts\python.exe -m py_compile src\training\coordinated.py src\training\process_worker.py src\main.py`
- Passed: `.\build.cmd -SkipTests`
- Result: default PyInstaller build and packaged smoke test completed successfully; output at `dist\StarAI\StarAI.exe` and `dist\StarAI-windows-x64.zip`

## Deferred To Later Phases

- Eligible coordinated runs automatically attempt the worker-backed scheduler.
- If worker processes cannot be started, training falls back to the in-process
  coordinated scheduler and the UI displays a temporary notice.
- Worker pool scheduling with fewer workers than records remains deferred.
- Worker timing CSV fields remain for Phase 5.
