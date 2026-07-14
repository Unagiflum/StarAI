"""Worker process protocol for coordinated CPU simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import multiprocessing
import traceback
from types import SimpleNamespace
from typing import Any
import random

from src.Battle.battle import BattleSimulation
from src.Objects.Ships.registry import create_ship
from src.audio import NullAudioService
from src.training.coordinated import (
    CoordinatedFixedFrameWindowResult,
    CoordinatedRuntimeComponents,
    TrainingEpisodeResult,
    _CoordinatedWindowRuntime,
    _advance_coordinated_window_frame,
    _finish_coordinated_window,
    _new_coordinated_battle,
)
from src.training.observation import encode_observation
from src.training.orchestration import (
    OPPONENT_MODE_EXISTING_AI,
    OpponentSpec,
    TrainingOrchestrationConfig,
)
from src.training.replay import ActionSelection, ReplaySample
from src.training.rewards import MatureTrainingSample, REWARD_COMPONENTS


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


@dataclass(frozen=True)
class StartRunCommand:
    worker_id: int
    record_id: int
    base_seed: int
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
    trainee_action_values: tuple[float, ...] | None = None
    opponent_controls: Mapping[str, bool] | None = None
    sequence_number: int = 0
    name: str = COMMAND_STEP_FRAME


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
    name: str = RESULT_WORKER_READY


@dataclass(frozen=True)
class WindowStartedResult:
    record_id: int
    round_index: int
    frame_limit: int
    name: str = RESULT_WINDOW_STARTED


@dataclass(frozen=True)
class WindowObservationResult:
    record_id: int
    round_index: int
    frame_count: int
    trainee_observation: tuple[float, ...]
    opponent_observation: tuple[float, ...] | None = None
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
    mature_samples: tuple[ReplaySample, ...] = ()
    timing_seconds: Mapping[str, float] = field(default_factory=dict)
    terminal_episode: TrainingEpisodeResult | None = None
    next_trainee_observation: tuple[float, ...] | None = None
    name: str = RESULT_FRAME_STEPPED


@dataclass(frozen=True)
class WindowFinishedResult:
    record_id: int
    round_index: int
    result: CoordinatedFixedFrameWindowResult
    mature_samples: tuple[ReplaySample, ...] = ()
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


class _NoModelPolicy:
    def reset_exploration_span(self) -> None:
        return None


class _ReplaySampleCollector:
    def __init__(self) -> None:
        self._samples: list[ReplaySample] = []
        self._pending: list[ReplaySample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def extend(self, samples: Sequence[ReplaySample | MatureTrainingSample]) -> None:
        for sample in samples:
            replay_sample = (
                ReplaySample.from_mature_sample(sample)
                if isinstance(sample, MatureTrainingSample)
                else sample
            )
            self._samples.append(replay_sample)
            self._pending.append(replay_sample)

    def drain_pending(self) -> tuple[ReplaySample, ...]:
        samples = tuple(self._pending)
        self._pending.clear()
        return samples


class CoordinatedSimulationWorker:
    """Own one CPU-only coordinated battle window for a child process."""

    def __init__(
        self,
        *,
        simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
        audio_service: Any | None = None,
        ship_factory: Callable[..., Any] = create_ship,
    ) -> None:
        self.worker_id: int | None = None
        self.record_id: int | None = None
        self._rng = random.Random()
        self._window_rng = random.Random()
        self._simulation_factory = simulation_factory
        self._audio_service = audio_service or NullAudioService()
        self._ship_factory = ship_factory
        self._runtime: _CoordinatedWindowRuntime | None = None
        self._collector: _ReplaySampleCollector | None = None
        self._round_index: int | None = None

    def handle(self, command: Any) -> Any:
        name = getattr(command, "name", "")
        if name == COMMAND_START_RUN:
            return self._handle_start_run(command)
        if name == COMMAND_START_WINDOW:
            return self._handle_start_window(command)
        if name == COMMAND_REQUEST_OBSERVATION:
            return self._handle_request_observation(command)
        if name == COMMAND_STEP_FRAME:
            return self._handle_step_frame(command)
        if name == COMMAND_FINISH_WINDOW:
            return self._handle_finish_window(command)
        if name == COMMAND_SHUTDOWN:
            return WorkerStoppedResult(self.worker_id, self.record_id)
        raise ValueError(f"unknown worker command: {name!r}")

    def _handle_start_run(self, command: StartRunCommand) -> WorkerReadyResult:
        self.worker_id = int(command.worker_id)
        self.record_id = int(command.record_id)
        self._rng = random.Random(int(command.base_seed))
        return WorkerReadyResult(
            worker_id=self.worker_id,
            record_id=self.record_id,
        )

    def _handle_start_window(
        self,
        command: StartWindowCommand,
    ) -> WindowStartedResult:
        self._ensure_record(command.record_id)
        self._round_index = int(command.round_index)
        config = command.config
        if command.frame_limit is not None:
            config = replace(config, match_time_limit=int(command.frame_limit))
        self._window_rng = random.Random(int(command.rng_seed))
        collector = _ReplaySampleCollector()
        record = SimpleNamespace(
            instance_id=int(command.record_id),
            config=config,
        )
        state = SimpleNamespace(
            record=record,
            components=CoordinatedRuntimeComponents(
                model=None,
                optimizer=None,
                replay_buffer=collector,
            ),
        )
        simulation, ledger, pipeline, simple_controller = _new_coordinated_battle(
            config,
            command.opponent,
            rng=self._window_rng,
            simulation_factory=self._simulation_factory,
            audio_service=self._audio_service,
            ship_factory=self._ship_factory,
        )
        self._collector = collector
        self._runtime = _CoordinatedWindowRuntime(
            state=state,
            opponent=command.opponent,
            policy=_NoModelPolicy(),
            simulation=simulation,
            ledger=ledger,
            pipeline=pipeline,
            simple_controller=simple_controller,
            component_sums={component: 0.0 for component in REWARD_COMPONENTS},
            episode_component_sums={
                component: 0.0
                for component in REWARD_COMPONENTS
            },
        )
        return WindowStartedResult(
            record_id=int(command.record_id),
            round_index=int(command.round_index),
            frame_limit=self._runtime.frame_limit,
        )

    def _handle_request_observation(
        self,
        command: RequestObservationCommand,
    ) -> WindowObservationResult:
        runtime = self._runtime_for(command.record_id, command.round_index)
        return self._observation_result(runtime)

    def _handle_step_frame(self, command: StepFrameCommand) -> FrameSteppedResult:
        runtime = self._runtime_for(command.record_id, command.round_index)
        collector = self._collector
        if collector is None:
            raise RuntimeError("worker replay collector is not initialized")
        progress_payloads: list[Mapping[str, Any]] = []
        episode_count = len(runtime.episode_results)
        selection = ActionSelection(
            action_index=int(command.trainee_action_index),
            exploratory=bool(command.trainee_exploratory),
            action_values=command.trainee_action_values,
        )
        opponent_controls = command.opponent_controls
        if opponent_controls is None and not _opponent_requires_parent_controls(
            runtime.opponent
        ):
            opponent_controls = runtime.simple_controller.direct_controls_for_frame(
                runtime.simulation,
            )
        if opponent_controls is None:
            raise RuntimeError("model-backed opponent controls were not provided")
        _advance_coordinated_window_frame(
            runtime,
            rng=self._window_rng,
            simulation_factory=self._simulation_factory,
            audio_service=self._audio_service,
            selection=selection,
            opponent_controls=dict(opponent_controls),
            progress_callback=progress_payloads.append,
            ship_factory=self._ship_factory,
        )
        terminal_episode = (
            runtime.episode_results[-1]
            if len(runtime.episode_results) > episode_count
            else None
        )
        next_observation = None
        if not runtime.complete:
            next_observation = _trainee_observation(runtime)
        return FrameSteppedResult(
            record_id=int(command.record_id),
            round_index=int(command.round_index),
            frame_count=runtime.frames_consumed,
            complete=runtime.complete,
            progress_payload=progress_payloads[-1] if progress_payloads else None,
            mature_samples=collector.drain_pending(),
            terminal_episode=terminal_episode,
            next_trainee_observation=next_observation,
        )

    def _handle_finish_window(
        self,
        command: FinishWindowCommand,
    ) -> WindowFinishedResult:
        runtime = self._runtime_for(command.record_id, command.round_index)
        collector = self._collector
        if collector is None:
            raise RuntimeError("worker replay collector is not initialized")
        result = _finish_coordinated_window(runtime)
        return WindowFinishedResult(
            record_id=int(command.record_id),
            round_index=int(command.round_index),
            result=result,
            mature_samples=collector.drain_pending(),
        )

    def _ensure_record(self, record_id: int) -> None:
        if self.record_id is None:
            self.record_id = int(record_id)
            return
        if int(record_id) != self.record_id:
            raise RuntimeError(
                f"worker {self.worker_id} owns record {self.record_id}, "
                f"not {record_id}"
            )

    def _runtime_for(
        self,
        record_id: int,
        round_index: int,
    ) -> _CoordinatedWindowRuntime:
        self._ensure_record(record_id)
        if self._runtime is None:
            raise RuntimeError("worker window has not been started")
        if int(round_index) != self._round_index:
            raise RuntimeError(
                f"worker window round is {self._round_index}, not {round_index}"
            )
        return self._runtime

    def _observation_result(
        self,
        runtime: _CoordinatedWindowRuntime,
    ) -> WindowObservationResult:
        opponent_observation = None
        simple_controls = None
        if not runtime.complete:
            if _opponent_requires_parent_controls(runtime.opponent):
                opponent_observation = _opponent_observation(runtime)
            else:
                simple_controls = runtime.simple_controller.direct_controls_for_frame(
                    runtime.simulation,
                )
        return WindowObservationResult(
            record_id=int(runtime.state.record.instance_id),
            round_index=int(self._round_index or 0),
            frame_count=runtime.frames_consumed,
            trainee_observation=(
                _trainee_observation(runtime) if not runtime.complete else ()
            ),
            opponent_observation=opponent_observation,
            simple_opponent_controls=simple_controls,
            complete=runtime.complete,
        )


def _trainee_observation(runtime: _CoordinatedWindowRuntime) -> tuple[float, ...]:
    simulation = runtime.simulation
    return tuple(
        encode_observation(
            simulation.player1,
            simulation.player2,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
    )


def _opponent_observation(runtime: _CoordinatedWindowRuntime) -> tuple[float, ...]:
    simulation = runtime.simulation
    return tuple(
        encode_observation(
            simulation.player2,
            simulation.player1,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
    )


def _opponent_requires_parent_controls(opponent: OpponentSpec) -> bool:
    return opponent.mode == OPPONENT_MODE_EXISTING_AI or opponent.slot is not None


def worker_process_main(connection) -> None:
    """Run the request/response worker loop for one multiprocessing connection."""

    worker = CoordinatedSimulationWorker()
    while True:
        try:
            command = connection.recv()
        except EOFError:
            break
        command_name = getattr(command, "name", "")
        try:
            result = worker.handle(command)
        except Exception as exc:
            result = WorkerErrorResult(
                record_id=getattr(command, "record_id", worker.record_id),
                command_name=command_name,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                traceback_text=traceback.format_exc(),
            )
        try:
            connection.send(result)
        except (BrokenPipeError, EOFError):
            break
        if getattr(result, "name", "") == RESULT_WORKER_STOPPED:
            break


def start_worker_process(
    *,
    context: multiprocessing.context.BaseContext | None = None,
):
    """Start a worker process and return ``(process, parent_connection)``."""

    ctx = context or multiprocessing.get_context("spawn")
    parent_connection, child_connection = ctx.Pipe()
    process = ctx.Process(
        target=worker_process_main,
        args=(child_connection,),
        daemon=True,
    )
    process.start()
    child_connection.close()
    return process, parent_connection
