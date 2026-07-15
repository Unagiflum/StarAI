"""Worker process protocol for coordinated CPU simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import ctypes
import multiprocessing
from multiprocessing import shared_memory
import sys
import traceback
from types import SimpleNamespace
from typing import Any
import random

import pygame

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.battle_draw import (
    BattleDrawController,
    BattleDrawOptions,
    DisplayStarField,
    create_play_battle_layout,
)
from src.Objects.Ships.registry import create_ship
from src.audio import RecordingAudioService
from src.resources import HeadlessAssetManager
from src.training.coordinated_simulation import (
    CoordinatedWindowRuntime,
    advance_coordinated_window_frame,
    finish_coordinated_window,
    new_coordinated_battle,
)
from src.training.observation import encode_observation
from src.training.cpu_contracts import (
    OPPONENT_MODE_EXISTING_AI,
    OpponentSpec,
)
from src.training.observation_transfer import PackedObservation, pack_observation
from src.training.replay_contracts import ActionSelection, ReplayTransferSample
from src.training.rewards import MatureTrainingSample, REWARD_COMPONENTS
from src.training.worker_protocol import (
    COMMAND_FINISH_WINDOW,
    COMMAND_REQUEST_OBSERVATION,
    COMMAND_SHUTDOWN,
    COMMAND_START_RUN,
    COMMAND_START_WINDOW,
    COMMAND_STEP_FRAME,
    RESULT_WORKER_STOPPED,
    DisplayBufferSpec,
    FinishWindowCommand,
    FrameSteppedResult,
    RequestObservationCommand,
    ShutdownCommand,
    StartRunCommand,
    StartWindowCommand,
    StepFrameCommand,
    WindowFinishedResult,
    WindowObservationResult,
    WindowStartedResult,
    WorkerErrorResult,
    WorkerReadyResult,
    WorkerStoppedResult,
)


_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _set_worker_process_below_normal_priority() -> bool:
    """Give Windows foreground work precedence over this simulation worker."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        set_priority_class = kernel32.SetPriorityClass
        set_priority_class.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        set_priority_class.restype = ctypes.c_int
        return bool(
            set_priority_class(
                get_current_process(),
                _BELOW_NORMAL_PRIORITY_CLASS,
            )
        )
    except (AttributeError, OSError):
        # Priority is a responsiveness optimization, not a startup requirement.
        return False


class _NoModelPolicy:
    def reset_exploration_span(self) -> None:
        return None


class _ReplaySampleCollector:
    def __init__(self) -> None:
        self._samples: list[ReplayTransferSample] = []
        self._pending: list[ReplayTransferSample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def extend(self, samples: Sequence[MatureTrainingSample]) -> None:
        for sample in samples:
            replay_sample = ReplayTransferSample.from_mature_sample(sample)
            self._samples.append(replay_sample)
            self._pending.append(replay_sample)

    def drain_pending(self) -> tuple[ReplayTransferSample, ...]:
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
        resources: Any | None = None,
        ship_factory: Callable[..., Any] = create_ship,
    ) -> None:
        self.worker_id: int | None = None
        self.record_id: int | None = None
        self._rng = random.Random()
        self._window_rng = random.Random()
        self._simulation_factory = simulation_factory
        self._audio_service = audio_service or RecordingAudioService()
        self._resources = resources or HeadlessAssetManager()
        self._ship_factory = ship_factory
        self._runtime: CoordinatedWindowRuntime | None = None
        self._collector: _ReplaySampleCollector | None = None
        self._round_index: int | None = None
        self._last_sequence_number = 0
        self._trainee_observation: list[float] | None = None
        self._packed_trainee_observation: PackedObservation | None = None
        self._packed_opponent_observation: PackedObservation | None = None
        self._simple_opponent_controls: dict[str, bool] | None = None
        self._display_memory: shared_memory.SharedMemory | None = None
        self._display_surface: pygame.Surface | None = None
        self._display_renderer: BattleDrawController | None = None
        self._display_star_field: DisplayStarField | None = None

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
            self.close()
            return WorkerStoppedResult(self.worker_id, self.record_id)
        raise ValueError(f"unknown worker command: {name!r}")

    def _handle_start_run(self, command: StartRunCommand) -> WorkerReadyResult:
        self.worker_id = int(command.worker_id)
        self.record_id = int(command.record_id)
        self._rng = random.Random(int(command.base_seed))
        const.VIDEO_FPS_MULTIPLIER = max(1, int(command.video_fps_multiplier))
        const.VIDEO_FPS = const.FPS * const.VIDEO_FPS_MULTIPLIER
        const._recompute_direction_constants()
        return WorkerReadyResult(
            worker_id=self.worker_id,
            record_id=self.record_id,
            torch_imported="torch" in sys.modules,
        )

    def _handle_start_window(
        self,
        command: StartWindowCommand,
    ) -> WindowStartedResult:
        self._ensure_record(command.record_id)
        self._round_index = int(command.round_index)
        self._last_sequence_number = 0
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
            components=SimpleNamespace(
                model=None,
                optimizer=None,
                replay_buffer=collector,
            ),
        )
        simulation, ledger, pipeline, simple_controller = new_coordinated_battle(
            config,
            command.opponent,
            rng=self._window_rng,
            simulation_factory=self._simulation_factory,
            audio_service=self._audio_service,
            resources=self._resources,
            ship_factory=self._ship_factory,
        )
        self._collector = collector
        self._runtime = CoordinatedWindowRuntime(
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
        self._drain_audio_events(include=False)
        self._cache_decision_state(self._runtime)
        if self._packed_trainee_observation is None:
            raise RuntimeError("worker did not produce an initial trainee observation")
        return WindowStartedResult(
            record_id=int(command.record_id),
            round_index=int(command.round_index),
            frame_limit=self._runtime.frame_limit,
            trainee_observation=self._packed_trainee_observation,
            opponent_observation=self._packed_opponent_observation,
            simple_opponent_controls=self._simple_opponent_controls,
            torch_imported="torch" in sys.modules,
        )

    def _handle_request_observation(
        self,
        command: RequestObservationCommand,
    ) -> WindowObservationResult:
        runtime = self._runtime_for(command.record_id, command.round_index)
        return self._observation_result(runtime)

    def _handle_step_frame(self, command: StepFrameCommand) -> FrameSteppedResult:
        runtime = self._runtime_for(command.record_id, command.round_index)
        sequence_number = int(command.sequence_number)
        if sequence_number > 0:
            if sequence_number <= self._last_sequence_number:
                raise RuntimeError(
                    f"worker frame sequence {sequence_number} is not newer than "
                    f"{self._last_sequence_number}"
                )
            self._last_sequence_number = sequence_number
        collector = self._collector
        if collector is None:
            raise RuntimeError("worker replay collector is not initialized")
        progress_payloads: list[Mapping[str, Any]] = []
        display_frames_ready = 0
        episode_count = len(runtime.episode_results)
        selection = ActionSelection(
            action_index=int(command.trainee_action_index),
            exploratory=bool(command.trainee_exploratory),
        )
        observation = self._trainee_observation
        if observation is None:
            raise RuntimeError("worker decision observation is not initialized")
        opponent_controls = command.opponent_controls
        if opponent_controls is None and not _opponent_requires_parent_controls(runtime.opponent):
            opponent_controls = self._simple_opponent_controls
        if opponent_controls is None:
            raise RuntimeError("model-backed opponent controls were not provided")

        def record_progress(payload):
            nonlocal display_frames_ready
            progress_payloads.append(payload)
            if command.display_buffer is not None:
                display_frames_ready = self._capture_display_frames(
                    runtime,
                    command.display_buffer,
                )

        advance_coordinated_window_frame(
            runtime,
            rng=self._window_rng,
            simulation_factory=self._simulation_factory,
            audio_service=self._audio_service,
            resources=self._resources,
            observation=observation,
            selection=selection,
            opponent_controls=dict(opponent_controls),
            progress_callback=record_progress,
            ship_factory=self._ship_factory,
        )
        terminal_episode = (
            runtime.episode_results[-1]
            if len(runtime.episode_results) > episode_count
            else None
        )
        self._clear_decision_state()
        if not runtime.complete:
            self._cache_decision_state(runtime)
        audio_events = self._drain_audio_events(include=bool(command.capture_audio))
        return FrameSteppedResult(
            record_id=int(command.record_id),
            round_index=int(command.round_index),
            frame_count=runtime.frames_consumed,
            complete=runtime.complete,
            progress_payload=progress_payloads[-1] if progress_payloads else None,
            mature_samples=collector.drain_pending(),
            terminal_episode=terminal_episode,
            next_trainee_observation=self._packed_trainee_observation,
            next_opponent_observation=self._packed_opponent_observation,
            next_simple_opponent_controls=self._simple_opponent_controls,
            audio_events=audio_events,
            display_frames_ready=display_frames_ready,
            torch_imported="torch" in sys.modules,
        )

    def _drain_audio_events(self, *, include: bool) -> tuple[tuple[Any, ...], ...]:
        operations = getattr(self._audio_service, "operations", None)
        if not isinstance(operations, list):
            return ()
        pending = tuple(operations)
        operations.clear()
        if not include:
            return ()
        events = []
        for operation in pending:
            if not operation:
                continue
            name, *args = operation
            if name == "play_effect" and len(args) >= 2:
                events.append((name, str(args[0]), float(args[1])))
            elif name == "play_victory_ditty" and args:
                events.append((name, getattr(args[0], "name", str(args[0]))))
        return tuple(events)

    def _capture_display_frames(
        self,
        runtime: CoordinatedWindowRuntime,
        spec: DisplayBufferSpec,
    ) -> int:
        width = int(spec.width)
        height = int(spec.height)
        frame_count = max(1, int(spec.frame_count))
        frame_bytes = width * height * 3
        required_bytes = frame_bytes * frame_count
        if self._display_memory is None or self._display_memory.name != spec.name:
            if self._display_memory is not None:
                self._display_memory.close()
            self._display_memory = shared_memory.SharedMemory(name=spec.name)
        if self._display_memory.size < required_bytes:
            raise ValueError("coordinated display buffer is too small")
        if (
            self._display_surface is None
            or self._display_surface.get_size() != (width, height)
        ):
            if not pygame.font.get_init():
                pygame.font.init()
            self._display_surface = pygame.Surface((width, height))
            self._display_renderer = BattleDrawController()
            self._display_star_field = DisplayStarField(resources=self._resources)

        surface = self._display_surface
        renderer = self._display_renderer
        star_field = self._display_star_field
        if renderer is None or star_field is None:
            raise RuntimeError("coordinated display renderer was not initialized")
        native_arena = pygame.Rect(
            const.SCREEN_LEFT,
            0,
            const.SCREEN_HEIGHT,
            const.SCREEN_HEIGHT,
        )
        layout = create_play_battle_layout(native_arena)
        simulation = runtime.simulation
        for index in range(frame_count):
            surface.fill((0, 0, 0))
            renderer.draw(
                surface,
                simulation.world.snapshot(),
                layout,
                simulation.border_color,
                star_field,
                camera_targets=(simulation.player1, simulation.player2),
                frame_id=simulation.frame_id,
                original_ships=(simulation.player1, simulation.player2),
                options=BattleDrawOptions(
                    draw_instructions=False,
                    interp_t=index / frame_count,
                ),
            )
            pixels = pygame.image.tobytes(surface, "RGB")
            offset = index * frame_bytes
            self._display_memory.buf[offset : offset + frame_bytes] = pixels
        return frame_count

    def close(self) -> None:
        self._runtime = None
        self._collector = None
        self._round_index = None
        self._last_sequence_number = 0
        self._clear_decision_state()
        self._detach_display_memory()

    def _detach_display_memory(self) -> None:
        if self._display_memory is not None:
            self._display_memory.close()
            self._display_memory = None

    def _handle_finish_window(
        self,
        command: FinishWindowCommand,
    ) -> WindowFinishedResult:
        runtime = self._runtime_for(command.record_id, command.round_index)
        collector = self._collector
        if collector is None:
            raise RuntimeError("worker replay collector is not initialized")
        result = finish_coordinated_window(runtime)
        mature_samples = collector.drain_pending()
        self._runtime = None
        self._collector = None
        self._round_index = None
        self._last_sequence_number = 0
        self._clear_decision_state()
        self._detach_display_memory()
        return WindowFinishedResult(
            record_id=int(command.record_id),
            round_index=int(command.round_index),
            result=result,
            mature_samples=mature_samples,
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
    ) -> CoordinatedWindowRuntime:
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
        runtime: CoordinatedWindowRuntime,
    ) -> WindowObservationResult:
        return WindowObservationResult(
            record_id=int(runtime.state.record.instance_id),
            round_index=int(self._round_index or 0),
            frame_count=runtime.frames_consumed,
            trainee_observation=self._packed_trainee_observation,
            opponent_observation=self._packed_opponent_observation,
            simple_opponent_controls=self._simple_opponent_controls,
            complete=runtime.complete,
        )

    def _cache_decision_state(self, runtime: CoordinatedWindowRuntime) -> None:
        if runtime.complete:
            self._clear_decision_state()
            return
        simulation = runtime.simulation
        observation = encode_observation(
            simulation.player1,
            simulation.player2,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
        self._trainee_observation = observation
        self._packed_trainee_observation = pack_observation(observation)
        if _opponent_requires_parent_controls(runtime.opponent):
            opponent_observation = encode_observation(
                simulation.player2,
                simulation.player1,
                frame_id=simulation.frame_id,
                game_objects=simulation.world,
            )
            self._packed_opponent_observation = pack_observation(opponent_observation)
            self._simple_opponent_controls = None
        else:
            controls = runtime.simple_controller.direct_controls_for_frame(simulation)
            self._packed_opponent_observation = None
            self._simple_opponent_controls = {
                "forward": bool(controls.thrust),
                "left": bool(controls.turn_left),
                "right": bool(controls.turn_right),
                "action1": bool(controls.a1),
                "action2": bool(controls.a2),
            }

    def _clear_decision_state(self) -> None:
        self._trainee_observation = None
        self._packed_trainee_observation = None
        self._packed_opponent_observation = None
        self._simple_opponent_controls = None


def _opponent_requires_parent_controls(opponent: OpponentSpec) -> bool:
    return opponent.mode == OPPONENT_MODE_EXISTING_AI or opponent.slot is not None


def worker_process_main(connection) -> None:
    """Run the request/response worker loop for one multiprocessing connection."""

    _set_worker_process_below_normal_priority()
    worker = CoordinatedSimulationWorker()
    try:
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
    finally:
        worker.close()


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
