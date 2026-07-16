"""Spawn-picklable wire contracts for coordinated simulation workers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import src.const as const
from src.training.coordinated_contracts import CoordinatedFixedFrameWindowResult
from src.training.cpu_contracts import OpponentSpec, TrainingOrchestrationConfig
from src.training.observation_transfer import PackedObservation
from src.training.replay_contracts import ReplayTransferSample


COMMAND_START_RUN = "START_RUN"
COMMAND_START_WINDOW = "START_WINDOW"
COMMAND_REQUEST_OBSERVATION = "REQUEST_OBSERVATION"
COMMAND_STEP_FRAME = "STEP_FRAME"
COMMAND_FINISH_WINDOW = "FINISH_WINDOW"
COMMAND_SHUTDOWN = "SHUTDOWN"

RESULT_WORKER_READY = "WORKER_READY"
RESULT_WINDOW_STARTED = "WINDOW_STARTED"
RESULT_WINDOW_OBSERVATION = "WINDOW_OBSERVATION"
RESULT_FRAME_STEPPED = "FRAME_STEPPED"
RESULT_WINDOW_FINISHED = "WINDOW_FINISHED"
RESULT_WORKER_ERROR = "WORKER_ERROR"
RESULT_WORKER_STOPPED = "WORKER_STOPPED"

REPLAY_TRANSFER_CHUNK_SIZE = 256


@dataclass(frozen=True)
class StartRunCommand:
    worker_id: int
    record_id: int
    base_seed: int
    video_fps_multiplier: int = const.VIDEO_FPS_MULTIPLIER
    name: str = COMMAND_START_RUN


@dataclass(frozen=True)
class StartWindowCommand:
    record_id: int
    round_index: int
    config: TrainingOrchestrationConfig
    opponent: OpponentSpec
    rng_seed: int
    frame_limit: int | None = None
    name: str = COMMAND_START_WINDOW

    def __post_init__(self) -> None:
        if self.opponent.model is not None:
            raise ValueError("worker window commands must not include model objects")


@dataclass(frozen=True)
class RequestObservationCommand:
    record_id: int
    round_index: int
    name: str = COMMAND_REQUEST_OBSERVATION


@dataclass(frozen=True)
class StepFrameCommand:
    record_id: int
    round_index: int
    trainee_action_index: int
    trainee_exploratory: bool
    opponent_controls: Mapping[str, bool] | None = None
    sequence_number: int = 0
    capture_audio: bool = False
    display_buffer: "DisplayBufferSpec | None" = None
    include_progress: bool = True
    name: str = COMMAND_STEP_FRAME


@dataclass(frozen=True)
class DisplayBufferSpec:
    name: str
    width: int = const.SCREEN_WIDTH
    height: int = const.SCREEN_HEIGHT
    frame_count: int = const.VIDEO_FPS_MULTIPLIER


@dataclass(frozen=True)
class FinishWindowCommand:
    record_id: int
    round_index: int
    name: str = COMMAND_FINISH_WINDOW


@dataclass(frozen=True)
class ShutdownCommand:
    name: str = COMMAND_SHUTDOWN


@dataclass(frozen=True)
class WorkerReadyResult:
    worker_id: int
    record_id: int
    torch_imported: bool = False
    name: str = RESULT_WORKER_READY


@dataclass(frozen=True)
class WindowStartedResult:
    record_id: int
    round_index: int
    frame_limit: int
    trainee_observation: PackedObservation
    opponent_observation: PackedObservation | None = None
    simple_opponent_controls: Mapping[str, bool] | None = None
    torch_imported: bool = False
    name: str = RESULT_WINDOW_STARTED


@dataclass(frozen=True)
class WindowObservationResult:
    record_id: int
    round_index: int
    frame_count: int
    trainee_observation: PackedObservation | None
    opponent_observation: PackedObservation | None = None
    simple_opponent_controls: Mapping[str, bool] | None = None
    complete: bool = False
    name: str = RESULT_WINDOW_OBSERVATION


@dataclass(frozen=True)
class FrameSteppedResult:
    record_id: int
    round_index: int
    frame_count: int
    complete: bool
    progress_payload: Mapping[str, Any] | None = None
    mature_samples: tuple[ReplayTransferSample, ...] = ()
    timing_seconds: Mapping[str, float] = field(default_factory=dict)
    terminal_episode: Any | None = None
    next_trainee_observation: PackedObservation | None = None
    next_opponent_observation: PackedObservation | None = None
    next_simple_opponent_controls: Mapping[str, bool] | None = None
    audio_events: tuple[tuple[Any, ...], ...] = ()
    display_frames_ready: int = 0
    torch_imported: bool = False
    name: str = RESULT_FRAME_STEPPED


@dataclass(frozen=True)
class WindowFinishedResult:
    record_id: int
    round_index: int
    result: CoordinatedFixedFrameWindowResult
    mature_samples: tuple[ReplayTransferSample, ...] = ()
    mature_sample_chunks: tuple[tuple[ReplayTransferSample, ...], ...] = ()
    timing_seconds: Mapping[str, float] = field(default_factory=dict)
    name: str = RESULT_WINDOW_FINISHED


@dataclass(frozen=True)
class WorkerErrorResult:
    record_id: int | None
    command_name: str
    exception_type: str
    exception_message: str
    traceback_text: str
    name: str = RESULT_WORKER_ERROR


@dataclass(frozen=True)
class WorkerStoppedResult:
    worker_id: int | None
    record_id: int | None
    name: str = RESULT_WORKER_STOPPED
