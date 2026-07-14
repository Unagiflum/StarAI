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

## Verification

- Passed: `.\.venv\Scripts\python.exe -m pytest tests\test_coordinated_training.py`
- Result: 23 passed

## Deferred To Later Phases

- Phase 3 parent scheduler integration is not implemented yet; coordinated `Start All` still uses the existing in-process scheduler path.
- Worker pool scheduling, stop/terminate hardening, runtime configuration switches, and worker timing CSV fields remain for Phases 3-5.
- Packaged-build verification remains pending for Phase 4.
