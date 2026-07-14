"""Coordinated multi-instance training runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import csv
from pathlib import Path
import random
import threading
import time
from types import SimpleNamespace
from typing import Any

import pygame

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Objects.Ships.registry import create_ship
from src.audio import DisplayGatedAudioService, NullAudioService
from src.training import torch_backend
from src.training import event_ledger
from src.training.batched_value_network import (
    BatchedValueNetworkParameterCache,
    build_batched_value_network_parameters,
    can_batch_value_networks,
    predict_action_values_from_batched_parameters,
    train_selected_action_regression_batched,
)
from src.training.contracts import TrainingAction
from src.training.model_registry import (
    TrainingModelRepository,
    TrainingModelSlot,
    metadata_from_state,
    model_architecture_metadata,
    model_paths,
    normalize_architecture_metadata,
)
from src.training.observation import encode_observation
from src.training.orchestration import (
    OPPONENT_MODE_EXISTING_AI,
    OpponentSpec,
    SimpleOpponentController,
    TrainingBatchAborted,
    TrainingBatchResult,
    TrainingOrchestrationConfig,
    ValueNetworkPolicy,
    _accumulate_weighted_components,
    _average_components,
    _average_value,
    _classify_round_outcome,
    _initialize_training_simulation_ships,
    _opponent_direct_controls,
    _raise_if_stop_requested,
    _select_policy_action,
    controls_for_action_index,
    direct_controls_for_action_index,
    discover_existing_ai_opponents,
    existing_ai_opponent_schedule,
    _battle_view_from_simulation,
    simple_opponent_schedule,
)
from src.training.opponent_cache import OpponentModelKey
from src.training.render_view import freeze_battle_view
from src.training.replay import (
    TrainingReplayBuffer,
    load_training_checkpoint,
    optimize_from_replay,
    save_training_checkpoint,
    select_action_epsilon_greedy,
)
from src.training.replay import ActionSelection
from src.training.rewards import (
    REWARD_COMPONENTS,
    RollingReturnPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)
from src.training.session import (
    BatchMetrics,
    MAX_BATCH_LOG_LINES,
    RECENT_BATCH_METRICS_KEY,
    TrainingSessionStatus,
    TRAINING_CSV_OUTPUT_ENABLED,
    append_grouped_metrics_csv,
    batch_metrics_to_metadata,
    batch_metrics_history_from_metadata,
    format_batch_summary_line,
    metrics_from_batch_result,
    rolling_metrics,
    should_update_live_frame_status,
)
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
    predict_action_values_read_only,
    train_selected_action_regression,
)


@dataclass(frozen=True)
class CoordinatedTrainingRecord:
    instance_id: int
    repository: TrainingModelRepository
    slot: TrainingModelSlot
    metadata: Mapping[str, Any]
    config: TrainingOrchestrationConfig
    batch_grouping: int
    initial_history: tuple[BatchMetrics, ...] = ()
    initial_log_lines: tuple[str, ...] = ()


@dataclass
class CoordinatedRuntimeComponents:
    model: Any
    optimizer: Any
    replay_buffer: TrainingReplayBuffer


@dataclass(frozen=True)
class TrainingEpisodeResult:
    opponent: OpponentSpec
    frames: int
    terminal_reason: str
    mature_samples: int
    total_return: float
    win: bool
    loss: bool
    draw: bool
    component_totals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinatedFixedFrameWindowResult:
    opponent: OpponentSpec
    frames: int
    mature_samples: int
    episode_results: tuple[TrainingEpisodeResult, ...]
    total_return: float
    win: bool
    loss: bool
    draw: bool
    component_totals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinatedActionRequest:
    record_id: int
    policy: Any
    observation: Sequence[float]


@dataclass(frozen=True)
class CoordinatedActionBatchResult:
    selections: Mapping[int, ActionSelection]
    inference_mode: str
    request_count: int
    exploratory_count: int


@dataclass(frozen=True)
class CoordinatedOpponentActionRequest:
    request_id: int
    opponent: OpponentSpec
    observation: Sequence[float]


@dataclass(frozen=True)
class CoordinatedInferenceStats:
    last_mode: str = ""
    request_count: int = 0
    exploratory_count: int = 0
    mode_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinatedTimingStats:
    observation_seconds: float = 0.0
    observation_encode_seconds: float = 0.0
    trainee_inference_seconds: float = 0.0
    opponent_inference_seconds: float = 0.0
    simulation_seconds: float = 0.0
    simulation_ship_inputs_seconds: float = 0.0
    simulation_tracking_seconds: float = 0.0
    simulation_update_objects_seconds: float = 0.0
    simulation_collision_seconds: float = 0.0
    simulation_aftermath_seconds: float = 0.0
    reward_seconds: float = 0.0
    reward_decision_seconds: float = 0.0
    reward_terminal_seconds: float = 0.0
    reward_outcome_seconds: float = 0.0
    reward_pipeline_seconds: float = 0.0
    reward_replay_insert_seconds: float = 0.0
    reward_accumulate_seconds: float = 0.0
    reward_progress_seconds: float = 0.0
    reward_flush_seconds: float = 0.0
    optimization_seconds: float = 0.0
    save_seconds: float = 0.0
    collision_possible_physical_pairs: int = 0
    collision_candidate_pairs: int = 0
    collision_dispatched_pairs: int = 0
    collision_possible_laser_targets: int = 0
    collision_laser_candidates: int = 0
    collision_possible_area_targets: int = 0
    collision_area_candidates: int = 0
    collision_area_full_scan_fallbacks: int = 0
    collision_spatial_queries: int = 0
    collision_spatial_returned_candidates: int = 0
    completed_batches: int = 0
    frame_count: int = 0


_TIMING_TOTAL_BUCKETS = (
    "observation",
    "trainee_inference",
    "opponent_inference",
    "simulation",
    "reward",
    "optimization",
    "save",
)

_TIMING_DETAIL_BUCKETS = (
    "observation_encode",
    "simulation_ship_inputs",
    "simulation_tracking",
    "simulation_update_objects",
    "simulation_collision",
    "simulation_aftermath",
    "reward_decision",
    "reward_terminal",
    "reward_outcome",
    "reward_pipeline",
    "reward_replay_insert",
    "reward_accumulate",
    "reward_progress",
    "reward_flush",
)

_TIMING_COUNTER_BUCKETS = (
    "collision_possible_physical_pairs",
    "collision_candidate_pairs",
    "collision_dispatched_pairs",
    "collision_possible_laser_targets",
    "collision_laser_candidates",
    "collision_possible_area_targets",
    "collision_area_candidates",
    "collision_area_full_scan_fallbacks",
    "collision_spatial_queries",
    "collision_spatial_returned_candidates",
)

_TIMING_BUCKETS = (
    *_TIMING_TOTAL_BUCKETS,
    *_TIMING_DETAIL_BUCKETS,
    *_TIMING_COUNTER_BUCKETS,
)

COORDINATED_TIMING_METRICS_ENABLED = False

_COORDINATED_TIMING_CSV_HEADER = (
    "Batch",
    "Instance ID",
    "Ship",
    "Slot",
    "Instance Count",
    "Rounds",
    "Instance Frames",
    "Coordinated Record Frames",
    "Action Requests",
    "Exploratory Actions",
    "Inference Mode",
    "Batch Seconds",
    "Batches/Hour",
    "Win %",
    "Score",
    "Epsilon",
    "Learning Rate",
    "Loss",
    "Observation Seconds",
    "Trainee Inference Seconds",
    "Opponent Inference Seconds",
    "Simulation Seconds",
    "Simulation Ship Inputs Seconds",
    "Simulation Tracking Seconds",
    "Simulation Update Objects Seconds",
    "Simulation Collision Seconds",
    "Simulation Aftermath Seconds",
    "Reward Seconds",
    "Reward Decision Seconds",
    "Reward Terminal Seconds",
    "Reward Outcome Seconds",
    "Reward Pipeline Seconds",
    "Reward Replay Insert Seconds",
    "Reward Accumulate Seconds",
    "Reward Progress Seconds",
    "Reward Flush Seconds",
    "Optimization Seconds",
    "Save Seconds",
    "Timed Total Seconds",
    "Collision Possible Physical Pairs",
    "Collision Candidate Pairs",
    "Collision Dispatched Pairs",
    "Collision Possible Laser Targets",
    "Collision Laser Candidates",
    "Collision Possible Area Targets",
    "Collision Area Candidates",
    "Collision Area Full Scan Fallbacks",
    "Collision Spatial Queries",
    "Collision Spatial Returned Candidates",
)


@dataclass
class _CoordinatedRecordState:
    record: CoordinatedTrainingRecord
    status: TrainingSessionStatus
    history: list[BatchMetrics] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    components: CoordinatedRuntimeComponents | None = None
    current_epsilon: float = 0.0
    last_saved_completed_batches: int = 0


@dataclass
class _CoordinatedWindowRuntime:
    state: _CoordinatedRecordState
    opponent: OpponentSpec
    policy: Any
    simulation: Any
    ledger: Any
    pipeline: RollingReturnPipeline
    simple_controller: SimpleOpponentController
    frames_consumed: int = 0
    total_mature_count: int = 0
    return_sum: float = 0.0
    component_sums: dict[str, float] = field(default_factory=dict)
    episode_results: list[TrainingEpisodeResult] = field(default_factory=list)
    episode_start_frame: int = 0
    episode_mature_count: int = 0
    episode_return_sum: float = 0.0
    episode_component_sums: dict[str, float] = field(default_factory=dict)
    episode_needs_timeout: bool = True

    @property
    def frame_limit(self) -> int:
        return int(self.state.record.config.match_time_limit)

    @property
    def complete(self) -> bool:
        return self.frames_consumed >= self.frame_limit


@dataclass
class _WorkerWindowRuntime:
    state: _CoordinatedRecordState
    client: Any
    opponent: OpponentSpec
    policy: Any
    round_index: int
    complete: bool = False


class _WorkerRuntimeError(RuntimeError):
    """Raised when a coordinated simulation worker reports or hits a failure."""


class _CpuWorkerStartupUnavailable(RuntimeError):
    """Raised when coordinated worker processes cannot be started."""


CPU_WORKER_FALLBACK_NOTICE = (
    "Multiple CPU workers not available. Proceeding with single process."
)


class _ProcessWorkerClient:
    def __init__(
        self,
        *,
        worker_id: int,
        record_id: int,
        process_starter: Callable[..., Any] | None = None,
    ) -> None:
        self.worker_id = int(worker_id)
        self.record_id = int(record_id)
        self._process_starter = process_starter
        self.process = None
        self.connection = None

    def start(
        self,
        *,
        base_seed: int,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        from src.training.process_worker import StartRunCommand, start_worker_process

        starter = self._process_starter or start_worker_process
        self.process, self.connection = starter()
        self.send(
            StartRunCommand(
                worker_id=self.worker_id,
                record_id=self.record_id,
                base_seed=int(base_seed),
                video_fps_multiplier=const.VIDEO_FPS_MULTIPLIER,
            )
        )
        result = self.recv(stop_requested=stop_requested)
        _raise_for_worker_error(result)
        if getattr(result, "name", "") != "WORKER_READY":
            raise _WorkerRuntimeError(
                f"worker {self.worker_id} returned {getattr(result, 'name', '')!r} "
                "during startup"
            )

    def send(self, command: Any) -> None:
        if self.connection is None:
            raise _WorkerRuntimeError(f"worker {self.worker_id} is not started")
        self.connection.send(command)

    def recv(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
        timeout: float | None = None,
    ) -> Any:
        if self.connection is None:
            raise _WorkerRuntimeError(f"worker {self.worker_id} is not started")
        started_at = time.perf_counter()
        while True:
            if self.connection.poll(0.05):
                result = self.connection.recv()
                _raise_for_worker_error(result)
                return result
            if stop_requested is not None and stop_requested():
                raise TrainingBatchAborted("training stop requested")
            if self.process is not None and not self.process.is_alive():
                exitcode = getattr(self.process, "exitcode", None)
                raise _WorkerRuntimeError(
                    f"worker {self.worker_id} exited unexpectedly"
                    + (f" with code {exitcode}" if exitcode is not None else "")
                )
            if timeout is not None and time.perf_counter() - started_at >= timeout:
                raise _WorkerRuntimeError(
                    f"worker {self.worker_id} did not respond within {timeout:.1f}s"
                )

    def shutdown(self, *, timeout: float = 2.0) -> None:
        from src.training.process_worker import ShutdownCommand

        try:
            if self.connection is not None:
                self.send(ShutdownCommand())
                try:
                    self.recv(timeout=timeout)
                except Exception:
                    pass
        finally:
            self.close()

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
        if self.process is not None:
            self.process.join(0.1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2.0)
            self.process = None

    def abort_startup(self) -> None:
        """Immediately discard a worker that has not entered batch execution."""

        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
        if self.process is None:
            return
        process = self.process
        self.process = None
        try:
            if process.is_alive():
                process.terminate()
            process.join(0.5)
            if process.is_alive():
                kill = getattr(process, "kill", None)
                if callable(kill):
                    kill()
                    process.join(0.5)
        except Exception:
            pass


class CoordinatedTrainingStatusProxy:
    """Expose one record through the same narrow surface as TrainingSession."""

    def __init__(self, scheduler: "CoordinatedTrainingSession", instance_id: int):
        self._scheduler = scheduler
        self._instance_id = int(instance_id)
        self.slot = scheduler.slot_for_instance(instance_id)

    @property
    def status(self) -> TrainingSessionStatus:
        return self._scheduler.status_for_instance(self._instance_id)

    @property
    def history(self) -> tuple[BatchMetrics, ...]:
        return self._scheduler.history_for_instance(self._instance_id)

    @property
    def log_lines(self) -> tuple[str, ...]:
        return self._scheduler.log_lines_for_instance(self._instance_id)

    def start(self) -> None:
        self._scheduler.start()

    def request_stop(self) -> None:
        self._scheduler.request_stop()

    def join(self, timeout: float | None = None) -> None:
        self._scheduler.join(timeout)

    def set_display_on(self, enabled: bool) -> None:
        self._scheduler.set_display_on(self._instance_id, enabled)

    def set_starting_epsilon(self, value: float) -> None:
        self._scheduler.set_starting_epsilon(self._instance_id, value)


class CoordinatedTrainingSession:
    """Own one worker thread for a coordinated set of training records.

    The first runtime implementation runs fixed-frame battle windows in the
    central worker. Later phases add synchronized frame stepping across records,
    optimization, and save cadence.
    """

    def __init__(
        self,
        records: tuple[CoordinatedTrainingRecord, ...],
        *,
        component_builder: Callable[[CoordinatedTrainingRecord], CoordinatedRuntimeComponents] | None = None,
        simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
        audio_service: Any | None = None,
        rng: Any | None = None,
        run_batches: bool = True,
        idle_sleep_seconds: float = 0.01,
        opponent_model_cache: Any | None = None,
        save_coordinator: Any | None = None,
        coordinated_cpu_workers_enabled: bool = False,
        coordinated_cpu_worker_count: int | str | None = 0,
        worker_client_factory: Callable[..., Any] | None = None,
    ):
        if len(records) < 2:
            raise ValueError("Coordinated training requires at least two records")
        self._states = {
            int(record.instance_id): _CoordinatedRecordState(
                record=record,
                status=TrainingSessionStatus(
                    ship=str(record.slot.ship),
                    completed_batches=int(
                        record.metadata.get("progress", {}).get("completed_batches", 0)
                    ),
                    current_frame_limit=int(record.config.match_time_limit),
                    learning_rate=float(record.config.learning_rate),
                    current_epsilon=max(
                        float(record.config.epsilon_floor),
                        min(1.0, float(record.config.epsilon)),
                    ),
                    epsilon_decay=float(record.config.epsilon_decay),
                    gamma=float(record.config.gamma),
                ),
                history=list(
                    record.initial_history
                    or batch_metrics_history_from_metadata(record.metadata)
                ),
                log_lines=list(record.initial_log_lines),
                current_epsilon=max(
                    float(record.config.epsilon_floor),
                    min(1.0, float(record.config.epsilon)),
                ),
            )
            for record in records
        }
        self._component_builder = component_builder or build_coordinated_components
        self._simulation_factory = simulation_factory
        self._audio_service = audio_service or NullAudioService()
        self._record_audio_services = {
            instance_id: DisplayGatedAudioService(
                self._audio_service,
                lambda item=instance_id: self._display_enabled_for(item),
            )
            for instance_id in self._states
        }
        self._rng = rng or random.Random()
        self._run_batches = bool(run_batches)
        self._idle_sleep_seconds = max(0.0, float(idle_sleep_seconds))
        self.opponent_model_cache = opponent_model_cache
        self.save_coordinator = save_coordinator
        self.coordinated_cpu_workers_enabled = bool(coordinated_cpu_workers_enabled)
        self.coordinated_cpu_worker_count = coordinated_cpu_worker_count
        self._worker_client_factory = worker_client_factory
        self._notices: list[str] = []
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._display_changed = threading.Event()
        self._display_instance_id: int | None = None
        self._next_display_frame_time = time.perf_counter()
        self._display_memory = None
        self._thread: threading.Thread | None = None
        self._run_started_at: float | None = None
        self._run_stopped_at: float | None = None
        self._current_batch_started_at: float | None = None
        self._completed_batches_at_run_start = {
            instance_id: state.status.completed_batches
            for instance_id, state in self._states.items()
        }
        self._completed_batch_seconds = {
            instance_id: []
            for instance_id in self._states
        }
        self._inference_last_mode = ""
        self._inference_request_count = 0
        self._inference_exploratory_count = 0
        self._inference_mode_counts: dict[str, int] = {}
        self._timing_seconds: dict[str, float] = {
            bucket: 0.0
            for bucket in _TIMING_BUCKETS
        }
        self._timing_completed_batches = 0
        self._timing_frame_count = 0
        self._proxies = {
            instance_id: CoordinatedTrainingStatusProxy(self, instance_id)
            for instance_id in self._states
        }
        for state in self._states.values():
            state.last_saved_completed_batches = state.status.completed_batches

    @property
    def records(self) -> tuple[CoordinatedTrainingRecord, ...]:
        return tuple(state.record for state in self._states.values())

    @property
    def proxies(self) -> dict[int, CoordinatedTrainingStatusProxy]:
        return dict(self._proxies)

    @property
    def inference_stats(self) -> CoordinatedInferenceStats:
        with self._lock:
            return CoordinatedInferenceStats(
                last_mode=self._inference_last_mode,
                request_count=self._inference_request_count,
                exploratory_count=self._inference_exploratory_count,
                mode_counts=dict(self._inference_mode_counts),
            )

    @property
    def timing_stats(self) -> CoordinatedTimingStats:
        with self._lock:
            timing = dict(self._timing_seconds)
            return CoordinatedTimingStats(
                observation_seconds=timing.get("observation", 0.0),
                observation_encode_seconds=timing.get("observation_encode", 0.0),
                trainee_inference_seconds=timing.get("trainee_inference", 0.0),
                opponent_inference_seconds=timing.get("opponent_inference", 0.0),
                simulation_seconds=timing.get("simulation", 0.0),
                simulation_ship_inputs_seconds=timing.get("simulation_ship_inputs", 0.0),
                simulation_tracking_seconds=timing.get("simulation_tracking", 0.0),
                simulation_update_objects_seconds=timing.get(
                    "simulation_update_objects",
                    0.0,
                ),
                simulation_collision_seconds=timing.get("simulation_collision", 0.0),
                simulation_aftermath_seconds=timing.get("simulation_aftermath", 0.0),
                reward_seconds=timing.get("reward", 0.0),
                reward_decision_seconds=timing.get("reward_decision", 0.0),
                reward_terminal_seconds=timing.get("reward_terminal", 0.0),
                reward_outcome_seconds=timing.get("reward_outcome", 0.0),
                reward_pipeline_seconds=timing.get("reward_pipeline", 0.0),
                reward_replay_insert_seconds=timing.get("reward_replay_insert", 0.0),
                reward_accumulate_seconds=timing.get("reward_accumulate", 0.0),
                reward_progress_seconds=timing.get("reward_progress", 0.0),
                reward_flush_seconds=timing.get("reward_flush", 0.0),
                optimization_seconds=timing.get("optimization", 0.0),
                save_seconds=timing.get("save", 0.0),
                collision_possible_physical_pairs=int(
                    timing.get("collision_possible_physical_pairs", 0)
                ),
                collision_candidate_pairs=int(timing.get("collision_candidate_pairs", 0)),
                collision_dispatched_pairs=int(
                    timing.get("collision_dispatched_pairs", 0)
                ),
                collision_possible_laser_targets=int(
                    timing.get("collision_possible_laser_targets", 0)
                ),
                collision_laser_candidates=int(
                    timing.get("collision_laser_candidates", 0)
                ),
                collision_possible_area_targets=int(
                    timing.get("collision_possible_area_targets", 0)
                ),
                collision_area_candidates=int(timing.get("collision_area_candidates", 0)),
                collision_area_full_scan_fallbacks=int(
                    timing.get("collision_area_full_scan_fallbacks", 0)
                ),
                collision_spatial_queries=int(
                    timing.get("collision_spatial_queries", 0)
                ),
                collision_spatial_returned_candidates=int(
                    timing.get("collision_spatial_returned_candidates", 0)
                ),
                completed_batches=self._timing_completed_batches,
                frame_count=self._timing_frame_count,
            )

    @property
    def active(self) -> bool:
        with self._lock:
            return any(
                state.status.running or state.status.stopping
                for state in self._states.values()
            )

    def slot_for_instance(self, instance_id: int) -> TrainingModelSlot:
        return self._states[int(instance_id)].record.slot

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._stop_requested.clear()
            self._run_started_at = time.perf_counter()
            self._run_stopped_at = None
            self._current_batch_started_at = None
            self._completed_batches_at_run_start = {
                instance_id: state.status.completed_batches
                for instance_id, state in self._states.items()
            }
            self._completed_batch_seconds = {
                instance_id: []
                for instance_id in self._states
            }
            self._inference_last_mode = ""
            self._inference_request_count = 0
            self._inference_exploratory_count = 0
            self._inference_mode_counts = {}
            self._timing_seconds = {
                bucket: 0.0
                for bucket in _TIMING_BUCKETS
            }
            self._timing_completed_batches = 0
            self._timing_frame_count = 0
            for state in self._states.values():
                state.status.running = True
                state.status.stopping = False
                state.status.error = ""
                state.status.display_message = "Preparing coordinated run"
                state.status.battle_view = None
        self._thread = threading.Thread(
            target=self._run_worker,
            name="StarAICoordinatedTrainingSession",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_requested.set()
        with self._lock:
            for state in self._states.values():
                if state.status.running:
                    state.status.stopping = True
                    state.status.display_message = "Stopping coordinated run"

    def set_display_on(self, instance_id: int, enabled: bool) -> None:
        instance_id = int(instance_id)
        start_music = False
        stop_music = False
        with self._lock:
            if enabled:
                if instance_id not in self._states:
                    raise ValueError(f"unknown coordinated instance {instance_id}")
                if self._display_instance_id == instance_id:
                    return
                stop_music = self._display_instance_id is not None
                self._display_instance_id = instance_id
                self._next_display_frame_time = time.perf_counter()
                start_music = True
                for other_id, state in self._states.items():
                    if other_id != instance_id:
                        state.status.battle_view = None
            elif self._display_instance_id == instance_id:
                self._display_instance_id = None
                self._states[instance_id].status.battle_view = None
                stop_music = True
            else:
                return
        if stop_music:
            self._audio_service.stop_music()
        if start_music:
            self._audio_service.start_battle_music()
        self._display_changed.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status_for_instance(self, instance_id: int) -> TrainingSessionStatus:
        with self._lock:
            state = self._states[int(instance_id)]
            status = state.status
            elapsed = self._elapsed_seconds_locked()
            return TrainingSessionStatus(
                ship=status.ship,
                running=status.running,
                stopping=status.stopping,
                completed_batches=status.completed_batches,
                elapsed_training_seconds=elapsed,
                current_batch_seconds=elapsed if status.running else 0.0,
                last_batch_seconds=status.last_batch_seconds,
                average_batch_seconds=self._average_batch_seconds_locked(state),
                batches_per_hour=self._batches_per_hour_locked(state, elapsed),
                current_round=status.current_round,
                total_rounds=status.total_rounds,
                current_opponent=status.current_opponent,
                previous_opponent=status.previous_opponent,
                current_frame=status.current_frame,
                current_frame_limit=status.current_frame_limit,
                replay_size=status.replay_size,
                recent_loss=status.recent_loss,
                learning_rate=status.learning_rate,
                current_epsilon=status.current_epsilon,
                epsilon_decay=status.epsilon_decay,
                gamma=status.gamma,
                last_action_exploratory=status.last_action_exploratory,
                weighted_total_return=status.weighted_total_return,
                component_totals=dict(status.component_totals),
                batch_component_totals=dict(status.batch_component_totals),
                battle_view=status.battle_view,
                display_message=status.display_message,
                error=status.error,
            )

    def history_for_instance(self, instance_id: int) -> tuple[BatchMetrics, ...]:
        with self._lock:
            return tuple(self._states[int(instance_id)].history)

    def log_lines_for_instance(self, instance_id: int) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._states[int(instance_id)].log_lines)

    def consume_notice(self) -> str | None:
        """Return and remove the oldest scheduler notice for the UI."""
        with self._lock:
            return self._notices.pop(0) if self._notices else None

    def set_starting_epsilon(self, instance_id: int, value: float) -> None:
        epsilon = max(0.0, min(1.0, float(value)))
        with self._lock:
            state = self._states[int(instance_id)]
            current_epsilon = max(float(state.record.config.epsilon_floor), epsilon)
            state.current_epsilon = current_epsilon
            state.status.current_epsilon = current_epsilon

    def _run_worker(self) -> None:
        try:
            for state in self._states.values():
                if self._stop_requested.is_set():
                    break
                components = self._component_builder(state.record)
                with self._lock:
                    state.components = components
                    state.status.replay_size = len(components.replay_buffer)
                    state.status.display_message = "Preparing coordinated batch"
            if not self._run_batches:
                with self._lock:
                    for state in self._states.values():
                        state.status.display_message = "Coordinated scheduler idle"
                while not self._stop_requested.is_set():
                    time.sleep(self._idle_sleep_seconds)
                return
            while not self._stop_requested.is_set():
                ran_batch = self._run_one_coordinated_batch()
                if not ran_batch:
                    time.sleep(self._idle_sleep_seconds)
        except Exception as exc:
            self._stop_requested.set()
            with self._lock:
                for state in self._states.values():
                    state.status.error = str(exc)
                    state.status.stopping = True
        finally:
            try:
                self._save_unsaved_completed_progress()
            except Exception as exc:
                with self._lock:
                    for state in self._states.values():
                        if not state.status.error:
                            state.status.error = str(exc)
            self._mark_stopped()

    def _run_one_coordinated_batch(self) -> bool:
        if self._should_use_cpu_workers():
            try:
                return self._run_one_worker_backed_coordinated_batch()
            except _CpuWorkerStartupUnavailable:
                with self._lock:
                    self.coordinated_cpu_workers_enabled = False
                    self._notices.append(CPU_WORKER_FALLBACK_NOTICE)
                return self._run_one_in_process_coordinated_batch()

        return self._run_one_in_process_coordinated_batch()

    def _run_one_in_process_coordinated_batch(self) -> bool:
        timing_seconds = _new_timing_seconds()
        timing_frame_count = 0
        timing_completed_batch = False
        batch_action_requests = 0
        batch_exploratory_actions = 0
        batch_inference_mode_counts: dict[str, int] = {}
        trainee_parameter_cache = BatchedValueNetworkParameterCache()
        opponent_parameter_cache = BatchedValueNetworkParameterCache()
        schedules = {
            instance_id: self._opponents_for_batch(state)
            for instance_id, state in self._states.items()
        }
        if any(not schedule for schedule in schedules.values()):
            return False

        batch_started_at = time.perf_counter()
        with self._lock:
            self._current_batch_started_at = batch_started_at
            for instance_id, state in self._states.items():
                state.status.display_message = ""
                state.status.battle_view = None
                state.status.total_rounds = len(schedules[instance_id])
                state.status.current_round = 0
                state.status.current_frame = 0

        results: dict[int, list[CoordinatedFixedFrameWindowResult]] = {
            instance_id: []
            for instance_id in self._states
        }
        try:
            for round_index in range(1, max(len(s) for s in schedules.values()) + 1):
                active_windows: list[_CoordinatedWindowRuntime] = []
                for instance_id, state in self._states.items():
                    schedule = schedules[instance_id]
                    if round_index > len(schedule):
                        continue
                    _raise_if_stop_requested(self._stop_requested.is_set)
                    components = state.components
                    if components is None:
                        raise RuntimeError("coordinated components were not loaded")
                    opponent = schedule[round_index - 1]
                    with self._lock:
                        state.status.current_round = round_index
                        state.status.total_rounds = len(schedule)
                        state.status.current_opponent = opponent.ship
                        state.status.current_frame = 0
                    policy = ValueNetworkPolicy(
                        components.model,
                        epsilon=state.current_epsilon,
                        epsilon_frame_span=state.record.config.epsilon_frame_span,
                        rng=self._rng,
                    )
                    active_windows.append(
                        _create_coordinated_window_runtime(
                            state=state,
                            opponent=opponent,
                            policy=policy,
                            rng=self._rng,
                            simulation_factory=self._simulation_factory,
                            audio_service=self._audio_for_instance(instance_id),
                        )
                    )
                while any(not window.complete for window in active_windows):
                    _raise_if_stop_requested(self._stop_requested.is_set)
                    frame_requests = []
                    for window in active_windows:
                        if window.complete:
                            continue
                        observed_at = _timing_started_at(timing_seconds)
                        request = _action_request_for_window(
                            window,
                            timing_seconds=timing_seconds,
                        )
                        _add_timing_seconds(
                            timing_seconds,
                            "observation",
                            observed_at,
                        )
                        frame_requests.append((window, request))
                    inference_started_at = _timing_started_at(timing_seconds)
                    action_result = select_actions_for_records(
                        tuple(request for _window, request in frame_requests),
                        parameter_cache=trainee_parameter_cache,
                    )
                    _add_timing_seconds(
                        timing_seconds,
                        "trainee_inference",
                        inference_started_at,
                    )
                    batch_action_requests += int(action_result.request_count)
                    batch_exploratory_actions += int(action_result.exploratory_count)
                    batch_inference_mode_counts[action_result.inference_mode] = (
                        batch_inference_mode_counts.get(
                            action_result.inference_mode,
                            0,
                        )
                        + 1
                    )
                    self._record_inference_batch(action_result)
                    opponent_started_at = _timing_started_at(timing_seconds)
                    opponent_controls_by_window = select_opponent_controls_for_windows(
                        tuple(window for window, _request in frame_requests),
                        parameter_cache=opponent_parameter_cache,
                        direct=True,
                    )
                    _add_timing_seconds(
                        timing_seconds,
                        "opponent_inference",
                        opponent_started_at,
                    )
                    for window, request in frame_requests:
                        _advance_coordinated_window_frame(
                            window,
                            rng=self._rng,
                            simulation_factory=self._simulation_factory,
                            audio_service=self._audio_for_instance(
                                window.state.record.instance_id
                            ),
                            observation=request.observation,
                            selection=action_result.selections[request.record_id],
                            opponent_controls=opponent_controls_by_window[id(window)],
                            progress_callback=(
                                lambda payload, item=window:
                                    self._on_in_process_window_progress(item, payload)
                            ),
                            stop_requested=self._stop_requested.is_set,
                            timing_seconds=timing_seconds,
                        )
                        if timing_seconds is not None:
                            timing_frame_count += 1
                    self._throttle_display_frame()
                    _raise_if_stop_requested(self._stop_requested.is_set)
                for window in active_windows:
                    result = _finish_coordinated_window(
                        window,
                        timing_seconds=timing_seconds,
                    )
                    results[window.state.record.instance_id].append(result)
                    with self._lock:
                        window.state.status.previous_opponent = window.opponent.ship
                        window.state.status.component_totals = dict(
                            result.component_totals
                        )
            batch_finished_at = time.perf_counter()
        except TrainingBatchAborted:
            self._merge_timing_stats(
                timing_seconds,
                completed_batches=0,
                frame_count=timing_frame_count,
            )
            return False
        finally:
            with self._lock:
                self._current_batch_started_at = None

        with self._lock:
            for state in self._states.values():
                state.status.display_message = "Applying gradient descent"
                state.status.battle_view = None

        optimization_started_at = _timing_started_at(timing_seconds)
        optimization_losses = self._optimize_records()
        _add_timing_seconds(
            timing_seconds,
            "optimization",
            optimization_started_at,
        )

        completed_batch_numbers: dict[int, int] = {}
        for instance_id, state in self._states.items():
            components = state.components
            if components is None:
                continue
            batch_number = self._record_completed_batch(
                state,
                TrainingBatchResult(
                    completed_rounds=len(results[instance_id]),
                    replay_size=len(components.replay_buffer),
                    optimization_losses=optimization_losses[instance_id],
                    round_results=tuple(results[instance_id]),
                ),
                batch_seconds=max(0.0, batch_finished_at - batch_started_at),
            )
            completed_batch_numbers[instance_id] = batch_number
            if batch_number % state.record.batch_grouping == 0:
                save_started_at = _timing_started_at(timing_seconds)
                self._save_state(state, include_replay=False)
                _add_timing_seconds(timing_seconds, "save", save_started_at)
        inference_mode_summary = _format_inference_mode_counts(
            batch_inference_mode_counts
        )
        for instance_id, state in self._states.items():
            if instance_id not in completed_batch_numbers:
                continue
            if timing_seconds is not None:
                self._append_coordinated_batch_timing_row(
                    state,
                    batch_number=completed_batch_numbers[instance_id],
                    rounds=len(results[instance_id]),
                    instance_frames=sum(result.frames for result in results[instance_id]),
                    coordinated_record_frames=timing_frame_count,
                    action_requests=batch_action_requests,
                    exploratory_actions=batch_exploratory_actions,
                    inference_mode=inference_mode_summary,
                    timing_seconds=timing_seconds,
                )
        timing_completed_batch = True
        self._merge_timing_stats(
            timing_seconds,
            completed_batches=1 if timing_completed_batch else 0,
            frame_count=timing_frame_count,
        )
        return True

    def _run_one_worker_backed_coordinated_batch(self) -> bool:
        from src.training.process_worker import (
            DisplayBufferSpec,
            FinishWindowCommand,
            RequestObservationCommand,
            StartWindowCommand,
            StepFrameCommand,
        )

        timing_seconds = _new_timing_seconds()
        timing_frame_count = 0
        batch_action_requests = 0
        batch_exploratory_actions = 0
        batch_inference_mode_counts: dict[str, int] = {}
        trainee_parameter_cache = BatchedValueNetworkParameterCache()
        opponent_parameter_cache = BatchedValueNetworkParameterCache()
        workers: dict[int, Any] = {}
        results: dict[int, list[CoordinatedFixedFrameWindowResult]] = {
            instance_id: []
            for instance_id in self._states
        }
        try:
            try:
                workers = self._start_cpu_workers()
            except TrainingBatchAborted:
                raise
            except Exception as exc:
                raise _CpuWorkerStartupUnavailable(str(exc)) from exc
            schedules = {
                instance_id: self._opponents_for_batch(state)
                for instance_id, state in self._states.items()
            }
            if any(not schedule for schedule in schedules.values()):
                return False

            batch_started_at = time.perf_counter()
            with self._lock:
                self._current_batch_started_at = batch_started_at
                for instance_id, state in self._states.items():
                    state.status.display_message = "Running coordinated CPU workers"
                    state.status.battle_view = None
                    state.status.total_rounds = len(schedules[instance_id])
                    state.status.current_round = 0
                    state.status.current_frame = 0
            for round_index in range(1, max(len(s) for s in schedules.values()) + 1):
                active_windows: list[_WorkerWindowRuntime] = []
                for instance_id, state in self._states.items():
                    schedule = schedules[instance_id]
                    if round_index > len(schedule):
                        continue
                    _raise_if_stop_requested(self._stop_requested.is_set)
                    components = state.components
                    if components is None:
                        raise RuntimeError("coordinated components were not loaded")
                    opponent = schedule[round_index - 1]
                    with self._lock:
                        state.status.current_round = round_index
                        state.status.total_rounds = len(schedule)
                        state.status.current_opponent = opponent.ship
                        state.status.current_frame = 0
                    policy = ValueNetworkPolicy(
                        components.model,
                        epsilon=state.current_epsilon,
                        epsilon_frame_span=state.record.config.epsilon_frame_span,
                        rng=self._rng,
                    )
                    window = _WorkerWindowRuntime(
                        state=state,
                        client=workers[instance_id],
                        opponent=opponent,
                        policy=policy,
                        round_index=round_index,
                    )
                    window.client.send(
                        StartWindowCommand(
                            record_id=instance_id,
                            round_index=round_index,
                            config=state.record.config,
                            opponent=replace(opponent, model=None),
                            rng_seed=self._worker_window_seed(
                                instance_id,
                                round_index,
                            ),
                            frame_limit=state.record.config.match_time_limit,
                        )
                    )
                    start_result = self._recv_worker_result(window.client)
                    if getattr(start_result, "name", "") != "WINDOW_STARTED":
                        raise _WorkerRuntimeError(
                            f"worker {instance_id} returned "
                            f"{getattr(start_result, 'name', '')!r} for START_WINDOW"
                        )
                    active_windows.append(window)

                while any(not window.complete for window in active_windows):
                    _raise_if_stop_requested(self._stop_requested.is_set)
                    unfinished = tuple(
                        window for window in active_windows if not window.complete
                    )
                    observed_at = _timing_started_at(timing_seconds)
                    for window in unfinished:
                        window.client.send(
                            RequestObservationCommand(
                                record_id=window.state.record.instance_id,
                                round_index=window.round_index,
                            )
                        )
                    observations = {
                        window.state.record.instance_id: self._recv_worker_result(
                            window.client,
                        )
                        for window in unfinished
                    }
                    _add_timing_seconds(timing_seconds, "observation", observed_at)

                    frame_requests = []
                    for window in unfinished:
                        observation = observations[window.state.record.instance_id]
                        if bool(getattr(observation, "complete", False)):
                            window.complete = True
                            continue
                        frame_requests.append((window, observation))
                    if not frame_requests:
                        continue

                    inference_started_at = _timing_started_at(timing_seconds)
                    action_result = select_actions_for_records(
                        tuple(
                            CoordinatedActionRequest(
                                record_id=window.state.record.instance_id,
                                policy=window.policy,
                                observation=observation.trainee_observation,
                            )
                            for window, observation in frame_requests
                        ),
                        parameter_cache=trainee_parameter_cache,
                    )
                    _add_timing_seconds(
                        timing_seconds,
                        "trainee_inference",
                        inference_started_at,
                    )
                    batch_action_requests += int(action_result.request_count)
                    batch_exploratory_actions += int(action_result.exploratory_count)
                    batch_inference_mode_counts[action_result.inference_mode] = (
                        batch_inference_mode_counts.get(
                            action_result.inference_mode,
                            0,
                        )
                        + 1
                    )
                    self._record_inference_batch(action_result)

                    opponent_started_at = _timing_started_at(timing_seconds)
                    opponent_controls = select_opponent_controls_for_observations(
                        tuple(
                            CoordinatedOpponentActionRequest(
                                request_id=window.state.record.instance_id,
                                opponent=window.opponent,
                                observation=observation.opponent_observation,
                            )
                            for window, observation in frame_requests
                            if observation.opponent_observation is not None
                        ),
                        parameter_cache=opponent_parameter_cache,
                    )
                    _add_timing_seconds(
                        timing_seconds,
                        "opponent_inference",
                        opponent_started_at,
                    )

                    for window, observation in frame_requests:
                        record_id = window.state.record.instance_id
                        selection = action_result.selections[record_id]
                        direct_controls = opponent_controls.get(record_id)
                        if direct_controls is None:
                            direct_controls = observation.simple_opponent_controls
                        window.client.send(
                            StepFrameCommand(
                                record_id=record_id,
                                round_index=window.round_index,
                                trainee_action_index=selection.action_index,
                                trainee_exploratory=selection.exploratory,
                                trainee_action_values=selection.action_values,
                                opponent_controls=_worker_controls_mapping(
                                    direct_controls,
                                ),
                                sequence_number=timing_frame_count + 1,
                                capture_audio=self._display_enabled_for(record_id),
                                display_buffer=self._display_buffer_spec(
                                    record_id,
                                    DisplayBufferSpec,
                                ),
                            )
                        )
                    for window, _observation in frame_requests:
                        frame_result = self._recv_worker_result(window.client)
                        record_id = window.state.record.instance_id
                        components = window.state.components
                        if components is None:
                            raise RuntimeError("coordinated components were not loaded")
                        components.replay_buffer.extend(frame_result.mature_samples)
                        timing_frame_count += 1
                        window.complete = bool(frame_result.complete)
                        if frame_result.terminal_episode is not None:
                            window.policy.reset_exploration_span()
                        if frame_result.audio_events:
                            self._relay_audio_events(record_id, frame_result.audio_events)
                        progress_payload = frame_result.progress_payload
                        if progress_payload:
                            progress_payload = dict(progress_payload)
                            progress_payload["replay_size"] = len(
                                components.replay_buffer
                            )
                            if frame_result.display_frames_ready:
                                progress_payload["battle_view"] = (
                                    self._battle_view_from_display_buffer(
                                        frame_result.frame_count,
                                        frame_result.display_frames_ready,
                                    )
                                )
                            self._on_record_progress(window.state, progress_payload)
                    self._throttle_display_frame()
                    _raise_if_stop_requested(self._stop_requested.is_set)

                for window in active_windows:
                    window.client.send(
                        FinishWindowCommand(
                            record_id=window.state.record.instance_id,
                            round_index=window.round_index,
                        )
                    )
                for window in active_windows:
                    finish_result = self._recv_worker_result(window.client)
                    components = window.state.components
                    if components is None:
                        raise RuntimeError("coordinated components were not loaded")
                    components.replay_buffer.extend(finish_result.mature_samples)
                    window.policy.reset_exploration_span()
                    result = finish_result.result
                    results[window.state.record.instance_id].append(result)
                    with self._lock:
                        window.state.status.previous_opponent = window.opponent.ship
                        window.state.status.replay_size = len(components.replay_buffer)
                        window.state.status.component_totals = dict(
                            result.component_totals
                        )
            batch_finished_at = time.perf_counter()
        except TrainingBatchAborted:
            self._merge_timing_stats(
                timing_seconds,
                completed_batches=0,
                frame_count=timing_frame_count,
            )
            return False
        finally:
            self._shutdown_cpu_workers(workers.values())
            self._release_display_buffer()
            with self._lock:
                self._current_batch_started_at = None

        with self._lock:
            for state in self._states.values():
                state.status.display_message = "Applying gradient descent"
                state.status.battle_view = None

        optimization_started_at = _timing_started_at(timing_seconds)
        optimization_losses = self._optimize_records()
        _add_timing_seconds(
            timing_seconds,
            "optimization",
            optimization_started_at,
        )

        completed_batch_numbers: dict[int, int] = {}
        for instance_id, state in self._states.items():
            components = state.components
            if components is None:
                continue
            batch_number = self._record_completed_batch(
                state,
                TrainingBatchResult(
                    completed_rounds=len(results[instance_id]),
                    replay_size=len(components.replay_buffer),
                    optimization_losses=optimization_losses[instance_id],
                    round_results=tuple(results[instance_id]),
                ),
                batch_seconds=max(0.0, batch_finished_at - batch_started_at),
            )
            completed_batch_numbers[instance_id] = batch_number
            if batch_number % state.record.batch_grouping == 0:
                save_started_at = _timing_started_at(timing_seconds)
                self._save_state(state, include_replay=False)
                _add_timing_seconds(timing_seconds, "save", save_started_at)
        inference_mode_summary = _format_inference_mode_counts(
            batch_inference_mode_counts
        )
        for instance_id, state in self._states.items():
            if instance_id not in completed_batch_numbers:
                continue
            if timing_seconds is not None:
                self._append_coordinated_batch_timing_row(
                    state,
                    batch_number=completed_batch_numbers[instance_id],
                    rounds=len(results[instance_id]),
                    instance_frames=sum(result.frames for result in results[instance_id]),
                    coordinated_record_frames=timing_frame_count,
                    action_requests=batch_action_requests,
                    exploratory_actions=batch_exploratory_actions,
                    inference_mode=inference_mode_summary,
                    timing_seconds=timing_seconds,
                )
        self._merge_timing_stats(
            timing_seconds,
            completed_batches=1,
            frame_count=timing_frame_count,
        )
        return True

    def _should_use_cpu_workers(self) -> bool:
        if not self.coordinated_cpu_workers_enabled:
            return False
        worker_count = self.coordinated_cpu_worker_count
        if worker_count in (None, 0, "0", "auto", "AUTO"):
            return True
        try:
            return int(worker_count) >= len(self._states)
        except (TypeError, ValueError):
            return False

    def _start_cpu_workers(self) -> dict[int, Any]:
        workers: dict[int, Any] = {}
        factory = self._worker_client_factory
        try:
            for worker_id, (instance_id, _state) in enumerate(
                self._states.items(), start=1
            ):
                _raise_if_stop_requested(self._stop_requested.is_set)
                client = (
                    factory(worker_id=worker_id, record_id=instance_id)
                    if factory is not None
                    else _ProcessWorkerClient(worker_id=worker_id, record_id=instance_id)
                )
                # Register before startup so a process/pipe created by a
                # partially failed start is still closed.
                workers[instance_id] = client
                start = getattr(client, "start", None)
                if callable(start):
                    start(
                        base_seed=self._worker_base_seed(instance_id),
                        stop_requested=self._stop_requested.is_set,
                    )
                else:
                    raise _WorkerRuntimeError("worker client does not provide start()")
                _raise_if_stop_requested(self._stop_requested.is_set)
        except TrainingBatchAborted:
            self._abort_cpu_worker_startup(workers.values())
            raise
        except Exception:
            self._shutdown_cpu_workers(workers.values())
            raise
        return workers

    def _abort_cpu_worker_startup(self, workers: Sequence[Any]) -> None:
        for worker in tuple(workers):
            abort_startup = getattr(worker, "abort_startup", None)
            if callable(abort_startup):
                try:
                    abort_startup()
                except Exception:
                    pass
                continue
            close = getattr(worker, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _shutdown_cpu_workers(self, workers: Sequence[Any]) -> None:
        for worker in tuple(workers):
            shutdown = getattr(worker, "shutdown", None)
            if callable(shutdown):
                shutdown()
                continue
            close = getattr(worker, "close", None)
            if callable(close):
                close()

    def _recv_worker_result(self, client: Any) -> Any:
        recv = getattr(client, "recv", None)
        if not callable(recv):
            raise _WorkerRuntimeError("worker client does not provide recv()")
        try:
            result = recv(stop_requested=self._stop_requested.is_set)
        except TypeError:
            result = recv()
        _raise_for_worker_error(result)
        return result

    def _worker_base_seed(self, instance_id: int) -> int:
        return int(self._rng.randrange(0, 2**31)) ^ (int(instance_id) << 8)

    def _worker_window_seed(self, instance_id: int, round_index: int) -> int:
        return (
            int(self._rng.randrange(0, 2**31))
            ^ (int(instance_id) << 12)
            ^ int(round_index)
        )

    def _record_inference_batch(
        self,
        result: CoordinatedActionBatchResult,
    ) -> None:
        if result.request_count <= 0:
            return
        with self._lock:
            self._inference_last_mode = result.inference_mode
            self._inference_request_count += int(result.request_count)
            self._inference_exploratory_count += int(result.exploratory_count)
            self._inference_mode_counts[result.inference_mode] = (
                self._inference_mode_counts.get(result.inference_mode, 0) + 1
            )

    def _merge_timing_stats(
        self,
        timing_seconds: Mapping[str, float] | None,
        *,
        completed_batches: int,
        frame_count: int,
    ) -> None:
        if timing_seconds is None:
            return
        with self._lock:
            for bucket in _TIMING_BUCKETS:
                self._timing_seconds[bucket] = self._timing_seconds.get(
                    bucket,
                    0.0,
                ) + max(0.0, float(timing_seconds.get(bucket, 0.0)))
            self._timing_completed_batches += max(0, int(completed_batches))
            self._timing_frame_count += max(0, int(frame_count))

    def _opponents_for_batch(
        self,
        state: _CoordinatedRecordState,
    ) -> tuple[OpponentSpec, ...]:
        config = state.record.config
        if config.opponent_mode == OPPONENT_MODE_EXISTING_AI:
            if self.opponent_model_cache is not None:
                with self._lock:
                    state.status.display_message = "Loading AI opponents"
                    state.status.battle_view = None
                self.opponent_model_cache.load_initial(
                    state.record.repository,
                    device_choice=config.training_device,
                )
                opponents = self.opponent_model_cache.snapshot(
                    device_choice=config.training_device,
                )
                with self._lock:
                    state.status.display_message = ""
                return existing_ai_opponent_schedule(
                    config.rounds_per_batch,
                    opponents,
                    ai_opponent_chance=config.ai_opponent_chance,
                    rng=self._rng,
                )
            opponents = discover_existing_ai_opponents(
                state.record.repository,
                device_choice=config.training_device,
            ).opponents
            return existing_ai_opponent_schedule(
                config.rounds_per_batch,
                opponents,
                ai_opponent_chance=config.ai_opponent_chance,
                rng=self._rng,
            )
        return simple_opponent_schedule(config.rounds_per_batch)

    def _record_completed_batch(
        self,
        state: _CoordinatedRecordState,
        result: TrainingBatchResult,
        *,
        batch_seconds: float,
    ) -> int:
        with self._lock:
            state.status.completed_batches += 1
            batch_number = state.status.completed_batches
            self._completed_batch_seconds[state.record.instance_id].append(
                max(0.0, float(batch_seconds))
            )
            state.status.last_batch_seconds = self._completed_batch_seconds[
                state.record.instance_id
            ][-1]
            elapsed = self._elapsed_seconds_locked()
            state.status.elapsed_training_seconds = elapsed
            state.status.average_batch_seconds = self._average_batch_seconds_locked(state)
            state.status.batches_per_hour = self._batches_per_hour_locked(
                state,
                elapsed,
            )
            state.current_epsilon = max(
                float(state.record.config.epsilon_floor),
                min(
                    1.0,
                    state.current_epsilon * float(state.record.config.epsilon_decay),
                ),
            )
            state.status.current_epsilon = state.current_epsilon
            current_epsilon = state.current_epsilon

        metrics = metrics_from_batch_result(
            result,
            batch=batch_number,
            epsilon=current_epsilon,
            learning_rate=state.record.config.learning_rate,
        )
        with self._lock:
            state.history.append(metrics)
            rolling = rolling_metrics(tuple(state.history), state.record.batch_grouping)
            state.log_lines.append(format_batch_summary_line(metrics, rolling))
            if len(state.log_lines) > MAX_BATCH_LOG_LINES:
                del state.log_lines[: len(state.log_lines) - MAX_BATCH_LOG_LINES]
            state.status.recent_loss = metrics.average_loss
            state.status.replay_size = result.replay_size
            batch_components = {component: 0.0 for component in REWARD_COMPONENTS}
            for window_result in result.round_results:
                for component, value in window_result.component_totals.items():
                    if component in batch_components:
                        batch_components[component] += value
            if result.round_results:
                for component in batch_components:
                    batch_components[component] /= len(result.round_results)
            state.status.batch_component_totals = batch_components
            limit = max(1, int(state.record.batch_grouping))
            if len(state.history) > limit:
                del state.history[: len(state.history) - limit]
        if (
            TRAINING_CSV_OUTPUT_ENABLED
            and batch_number % state.record.batch_grouping == 0
        ):
            append_grouped_metrics_csv(self._csv_path(state), rolling)
        return batch_number

    def _optimize_record(self, state: _CoordinatedRecordState) -> tuple[float, ...]:
        components = state.components
        if components is None:
            raise RuntimeError("coordinated components were not loaded")
        config = state.record.config
        losses: list[float] = []
        for _ in range(config.replay_updates_per_batch):
            _raise_if_stop_requested(self._stop_requested.is_set)
            result = optimize_from_replay(
                components.model,
                components.optimizer,
                components.replay_buffer,
                batch_size=config.minibatch_size,
                rng=self._rng,
            )
            if result is not None:
                losses.append(result.loss)
        return tuple(losses)

    def _optimize_records(self) -> dict[int, tuple[float, ...]]:
        states = tuple(self._states.values())
        if not _can_batch_record_optimization(states):
            return {
                state.record.instance_id: self._optimize_record(state)
                for state in states
            }
        return _optimize_records_batched(
            states,
            rng=self._rng,
            stop_requested=self._stop_requested.is_set,
        )

    def _save_unsaved_completed_progress(self) -> None:
        for state in self._states.values():
            components = state.components
            if components is None:
                continue
            with self._lock:
                completed_batches = state.status.completed_batches
                last_saved = state.last_saved_completed_batches
            if completed_batches > last_saved:
                save_timing = _new_timing_seconds()
                save_started_at = _timing_started_at(save_timing)
                self._save_state(state, include_replay=True)
                _add_timing_seconds(save_timing, "save", save_started_at)
                self._merge_timing_stats(
                    save_timing,
                    completed_batches=0,
                    frame_count=0,
                )

    def _save_state(
        self,
        state: _CoordinatedRecordState,
        *,
        include_replay: bool,
    ) -> None:
        components = state.components
        if components is None:
            raise RuntimeError("coordinated components were not loaded")
        key = OpponentModelKey(state.record.slot.ship, state.record.slot.slot)
        context = (
            self.save_coordinator.saving(key)
            if self.save_coordinator is not None
            else nullcontext()
        )
        with self._lock:
            completed_batches = state.status.completed_batches
            history = tuple(state.history)
            metadata = dict(state.record.metadata)
            current_epsilon = float(state.current_epsilon)
        with context:
            pth_path, _ = model_paths(
                state.record.repository.user_dir,
                key.ship,
                key.slot,
            )
            save_training_checkpoint(
                pth_path,
                components.model,
                optimizer=components.optimizer,
                replay_buffer=components.replay_buffer if include_replay else None,
                extra_state={"completed_batches": completed_batches},
            )
            updated_metadata = metadata_from_state(
                ship=key.ship,
                slot=key.slot,
                description=str(metadata.get("description", state.record.slot.description)),
                architecture=metadata.get(
                    "architecture",
                    model_architecture_metadata(
                        state.record.config.hidden_layer_width,
                        state.record.config.hidden_layer_count,
                    ),
                ),
                training=self._training_metadata_for_save(
                    state,
                    current_epsilon=current_epsilon,
                ),
                progress={
                    "completed_batches": completed_batches,
                    RECENT_BATCH_METRICS_KEY: [
                        batch_metrics_to_metadata(metrics)
                        for metrics in history[-state.record.batch_grouping :]
                    ],
                },
            )
            updated_slot = state.record.repository.create_or_update_user_model(
                updated_metadata
            )
        with self._lock:
            state.record = replace(
                state.record,
                slot=updated_slot,
                metadata=updated_metadata,
            )
            state.last_saved_completed_batches = completed_batches
        if self.opponent_model_cache is not None:
            self.opponent_model_cache.notify_model_saved(
                state.record.repository,
                key.ship,
                key.slot,
                device_choice=state.record.config.training_device,
            )

    def _training_metadata_for_save(
        self,
        state: _CoordinatedRecordState,
        *,
        current_epsilon: float,
    ) -> dict[str, Any]:
        training = state.record.metadata.get("training", {})
        training = dict(training) if isinstance(training, Mapping) else {}
        regimen = training.get("regimen", {})
        regimen = dict(regimen) if isinstance(regimen, Mapping) else {}
        config = state.record.config
        regimen.update(
            {
                "starting_epsilon": float(config.starting_epsilon),
                "current_epsilon": float(current_epsilon),
                "epsilon": float(current_epsilon),
                "epsilon_floor": float(config.epsilon_floor),
                "epsilon_decay": float(config.epsilon_decay),
                "epsilon_frame_span": int(config.epsilon_frame_span),
            }
        )
        training["regimen"] = regimen
        return training

    def _csv_path(self, state: _CoordinatedRecordState):
        _, metadata_path = model_paths(
            state.record.repository.user_dir,
            state.record.slot.ship,
            state.record.slot.slot,
        )
        return metadata_path.with_suffix(".csv")

    def _coordinated_csv_path(self, state: _CoordinatedRecordState) -> Path:
        _, metadata_path = model_paths(
            state.record.repository.user_dir,
            state.record.slot.ship,
            state.record.slot.slot,
        )
        return metadata_path.with_suffix(".coordinated.csv")

    def _append_coordinated_batch_timing_row(
        self,
        state: _CoordinatedRecordState,
        *,
        batch_number: int,
        rounds: int,
        instance_frames: int,
        coordinated_record_frames: int,
        action_requests: int,
        exploratory_actions: int,
        inference_mode: str,
        timing_seconds: Mapping[str, float],
    ) -> None:
        if not TRAINING_CSV_OUTPUT_ENABLED:
            return
        with self._lock:
            metrics = state.history[-1] if state.history else None
            status = state.status
            batch_seconds = float(status.last_batch_seconds)
            batches_per_hour = float(status.batches_per_hour)
        if metrics is None:
            return
        append_coordinated_batch_timing_csv(
            self._coordinated_csv_path(state),
            batch_number=batch_number,
            instance_id=state.record.instance_id,
            ship=state.record.slot.ship,
            slot=state.record.slot.slot,
            instance_count=len(self._states),
            rounds=rounds,
            instance_frames=instance_frames,
            coordinated_record_frames=coordinated_record_frames,
            action_requests=action_requests,
            exploratory_actions=exploratory_actions,
            inference_mode=inference_mode,
            batch_seconds=batch_seconds,
            batches_per_hour=batches_per_hour,
            metrics=metrics,
            timing_seconds=timing_seconds,
        )

    def _on_record_progress(
        self,
        state: _CoordinatedRecordState,
        payload: Mapping[str, Any],
    ) -> None:
        if payload.get("event") != "frame":
            return
        frame = int(payload.get("frame", 0))
        display_on = self._display_enabled_for(state.record.instance_id)
        if not should_update_live_frame_status(frame, display_on=display_on):
            return
        opponent = payload.get("opponent")
        opponent_label = getattr(opponent, "ship", "") if opponent is not None else ""
        battle_view = payload.get("battle_view") if display_on else None
        if battle_view is not None and "rendered_frames" not in battle_view:
            battle_view = freeze_battle_view(battle_view)
        with self._lock:
            state.status.current_frame = frame
            state.status.current_opponent = opponent_label
            state.status.replay_size = int(payload.get("replay_size", 0))
            state.status.last_action_exploratory = bool(
                payload.get("exploratory", False)
            )
            state.status.weighted_total_return = float(
                payload.get("weighted_total_return", 0.0)
            )
            state.status.component_totals = dict(payload.get("component_totals", {}))
            if (
                battle_view is not None
                and self._display_instance_id == state.record.instance_id
            ):
                state.status.battle_view = battle_view

    def _on_in_process_window_progress(
        self,
        window: _CoordinatedWindowRuntime,
        payload: Mapping[str, Any],
    ) -> None:
        if self._display_enabled_for(window.state.record.instance_id):
            payload = dict(payload)
            battle_view = _battle_view_from_simulation(window.simulation)
            battle_view["frame_id"] = int(payload.get("frame", window.frames_consumed))
            payload["battle_view"] = battle_view
        self._on_record_progress(window.state, payload)

    def _display_enabled_for(self, instance_id: int) -> bool:
        with self._lock:
            return self._display_instance_id == int(instance_id)

    def _audio_for_instance(self, instance_id: int):
        return self._record_audio_services[int(instance_id)]

    def _relay_audio_events(
        self,
        instance_id: int,
        events: Sequence[tuple[Any, ...]],
    ) -> None:
        if not self._display_enabled_for(instance_id):
            return
        for event in events:
            if not event:
                continue
            name, *args = event
            if name == "play_effect" and len(args) >= 2:
                self._audio_service.play_effect(args[0], args[1])
            elif name == "play_victory_ditty" and args:
                self._audio_service.play_victory_ditty(
                    SimpleNamespace(name=str(args[0]))
                )

    def _throttle_display_frame(self) -> None:
        with self._lock:
            self._display_changed.clear()
            if self._display_instance_id is None:
                self._next_display_frame_time = time.perf_counter()
                return
            self._next_display_frame_time += 1.0 / const.FPS
            sleep_seconds = self._next_display_frame_time - time.perf_counter()
        if sleep_seconds > 0:
            if self._display_changed.wait(sleep_seconds):
                with self._lock:
                    self._next_display_frame_time = time.perf_counter()
        else:
            with self._lock:
                self._next_display_frame_time = time.perf_counter()

    def _display_buffer_spec(self, instance_id: int, spec_type):
        if not self._display_enabled_for(instance_id):
            return None
        if self._display_memory is None:
            from multiprocessing import shared_memory

            frame_bytes = const.SCREEN_WIDTH * const.SCREEN_HEIGHT * 3
            self._display_memory = shared_memory.SharedMemory(
                create=True,
                size=frame_bytes * max(1, const.VIDEO_FPS_MULTIPLIER),
            )
        return spec_type(
            name=self._display_memory.name,
            width=const.SCREEN_WIDTH,
            height=const.SCREEN_HEIGHT,
            frame_count=max(1, const.VIDEO_FPS_MULTIPLIER),
        )

    def _battle_view_from_display_buffer(
        self,
        frame_id: int,
        frame_count: int,
    ) -> dict[str, Any]:
        if self._display_memory is None:
            raise RuntimeError("coordinated display buffer is not available")
        width = const.SCREEN_WIDTH
        height = const.SCREEN_HEIGHT
        frame_bytes = width * height * 3
        frames = []
        for index in range(max(0, int(frame_count))):
            offset = index * frame_bytes
            pixels = bytes(self._display_memory.buf[offset : offset + frame_bytes])
            frames.append(pygame.image.frombytes(pixels, (width, height), "RGB"))
        return {
            "rendered_frames": tuple(frames),
            "frame_id": int(frame_id),
        }

    def _release_display_buffer(self) -> None:
        memory = self._display_memory
        self._display_memory = None
        if memory is None:
            return
        try:
            memory.close()
        finally:
            try:
                memory.unlink()
            except FileNotFoundError:
                pass

    def _mark_stopped(self) -> None:
        with self._lock:
            stop_music = self._display_instance_id is not None
            self._display_instance_id = None
            self._run_stopped_at = time.perf_counter()
            for state in self._states.values():
                state.status.running = False
                state.status.stopping = False
                if not state.status.error:
                    state.status.display_message = ""
        if stop_music:
            self._audio_service.stop_music()

    def _elapsed_seconds_locked(self) -> float:
        if self._run_started_at is None:
            return 0.0
        end = self._run_stopped_at or time.perf_counter()
        return max(0.0, end - self._run_started_at)

    def _average_batch_seconds_locked(self, state: _CoordinatedRecordState) -> float:
        durations = self._completed_batch_seconds.get(state.record.instance_id, ())
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    def _batches_per_hour_locked(
        self,
        state: _CoordinatedRecordState,
        elapsed_seconds: float,
    ) -> float:
        completed_this_run = (
            state.status.completed_batches
            - self._completed_batches_at_run_start.get(
                state.record.instance_id,
                state.status.completed_batches,
            )
        )
        if completed_this_run <= 0 or elapsed_seconds <= 0.0:
            return 0.0
        return completed_this_run * 3600.0 / elapsed_seconds


def run_coordinated_fixed_frame_window(
    *,
    opponent: OpponentSpec,
    trainee_policy,
    replay_buffer: TrainingReplayBuffer,
    config: TrainingOrchestrationConfig,
    rng: Any | None = None,
    simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
    audio_service: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> CoordinatedFixedFrameWindowResult:
    """Advance one opponent window for exactly ``match_time_limit`` frames.

    Real training simulations respawn dead ships in the same arena at episode
    boundaries. Compatibility simulations without that lifecycle still reset.
    """

    rng = rng or random.Random()
    audio = audio_service or NullAudioService()
    window_frame_limit = int(config.match_time_limit)
    window_frames_consumed = 0
    total_mature_count = 0
    return_sum = 0.0
    component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
    episode_results: list[TrainingEpisodeResult] = []

    def new_battle():
        trainee = ship_factory(config.trainee_ship, 1, audio_service=audio)
        opponent_ship = ship_factory(opponent.ship, 2, audio_service=audio)
        ledger = event_ledger.BattleEventLedger()
        simulation = simulation_factory(
            None,
            trainee,
            opponent_ship,
            audio_service=audio,
            rng=rng,
            include_stars=False,
            training_event_ledger=ledger,
        )
        _initialize_training_simulation_ships(simulation, rng)
        return simulation, ledger, RollingReturnPipeline(
            gamma=config.gamma,
            reward_weights=config.reward_weights,
        ), SimpleOpponentController(config, rng=rng)

    simulation, ledger, pipeline, simple_controller = new_battle()
    episode_start_window_frame = window_frames_consumed
    episode_mature_count = 0
    episode_return_sum = 0.0
    episode_component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
    episode_needs_timeout = True

    while window_frames_consumed < window_frame_limit:
        _raise_if_stop_requested(stop_requested)
        self_ship = simulation.player1
        enemy_ship = simulation.player2
        observation = encode_observation(
            self_ship,
            enemy_ship,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
        selection = _select_policy_action(trainee_policy, observation)
        decision = decision_frame_from_battle_state(
            frame_id=simulation.frame_id + 1,
            observation=observation,
            action_index=selection.action_index,
            self_ship=self_ship,
            enemy_ship=enemy_ship,
            world=simulation.world,
        )
        event_start = len(ledger.events)
        state = simulation.step(
            actions={
                1: direct_controls_for_action_index(selection.action_index),
                2: _opponent_direct_controls(
                    opponent,
                    simulation,
                    config,
                    simple_controller,
                ),
            }
        )
        window_frames_consumed += 1
        terminal, terminal_reason = _permanent_terminal_state(simulation)
        events = tuple(ledger.events[event_start:])
        outcome = frame_outcome_from_battle_state(
            frame_id=state["frame_id"],
            self_ship=self_ship,
            events=events,
            terminal=terminal,
        )
        mature_samples = pipeline.add_frame(decision, outcome)
        replay_buffer.extend(mature_samples)
        mature_count = len(mature_samples)
        total_mature_count += mature_count
        episode_mature_count += mature_count
        sample_return = sum(sample.return_value for sample in mature_samples)
        return_sum += sample_return
        episode_return_sum += sample_return
        _accumulate_weighted_components(component_sums, mature_samples)
        _accumulate_weighted_components(episode_component_sums, mature_samples)
        normalized_return = _average_value(return_sum, total_mature_count)
        _emit_window_progress(
            progress_callback,
            frame=window_frames_consumed,
            opponent=opponent,
            action_index=selection.action_index,
            exploratory=selection.exploratory,
            replay_size=len(replay_buffer),
            weighted_total_return=normalized_return,
            component_totals=_average_components(component_sums, total_mature_count),
        )
        _raise_if_stop_requested(stop_requested)

        if terminal:
            win, loss, draw = _classify_round_outcome(simulation, terminal_reason)
            episode_results.append(
                TrainingEpisodeResult(
                    opponent=opponent,
                    frames=window_frames_consumed - episode_start_window_frame,
                    terminal_reason=terminal_reason,
                    mature_samples=episode_mature_count,
                    total_return=_average_value(
                        episode_return_sum,
                        episode_mature_count,
                    ),
                    win=win,
                    loss=loss,
                    draw=draw,
                    component_totals=_average_components(
                        episode_component_sums,
                        episode_mature_count,
                    ),
                )
            )
            episode_needs_timeout = False
            reset_span = getattr(trainee_policy, "reset_exploration_span", None)
            if callable(reset_span):
                reset_span()
            if window_frames_consumed < window_frame_limit:
                if getattr(simulation, "training_episode_deaths", ()):
                    pipeline = RollingReturnPipeline(
                        gamma=config.gamma,
                        reward_weights=config.reward_weights,
                    )
                    simple_controller = SimpleOpponentController(config, rng=rng)
                else:
                    simulation, ledger, pipeline, simple_controller = new_battle()
                episode_start_window_frame = window_frames_consumed
                episode_mature_count = 0
                episode_return_sum = 0.0
                episode_component_sums = {
                    component: 0.0
                    for component in REWARD_COMPONENTS
                }
                episode_needs_timeout = True

    if episode_needs_timeout:
        mature_samples = tuple(
            pipeline.flush_pending(end_frame_id=simulation.frame_id)
        )
        replay_buffer.extend(mature_samples)
        total_mature_count += len(mature_samples)
        episode_mature_count += len(mature_samples)
        sample_return = sum(sample.return_value for sample in mature_samples)
        return_sum += sample_return
        episode_return_sum += sample_return
        _accumulate_weighted_components(component_sums, mature_samples)
        _accumulate_weighted_components(episode_component_sums, mature_samples)
        win, loss, draw = _classify_round_outcome(simulation, "timeout")
        episode_results.append(
            TrainingEpisodeResult(
                opponent=opponent,
                frames=window_frames_consumed - episode_start_window_frame,
                terminal_reason="timeout",
                mature_samples=episode_mature_count,
                total_return=_average_value(episode_return_sum, episode_mature_count),
                win=win,
                loss=loss,
                draw=draw,
                component_totals=_average_components(
                    episode_component_sums,
                    episode_mature_count,
                ),
            )
        )
        reset_span = getattr(trainee_policy, "reset_exploration_span", None)
        if callable(reset_span):
            reset_span()

    return CoordinatedFixedFrameWindowResult(
        opponent=opponent,
        frames=window_frames_consumed,
        mature_samples=total_mature_count,
        episode_results=tuple(episode_results),
        total_return=_average_value(return_sum, total_mature_count),
        win=any(result.win for result in episode_results),
        loss=any(result.loss for result in episode_results),
        draw=any(result.draw for result in episode_results),
        component_totals=_average_components(component_sums, total_mature_count),
    )


def _permanent_terminal_state(simulation) -> tuple[bool, str]:
    if getattr(simulation, "training_episode_deaths", ()):
        return True, "resolved"
    aftermath = getattr(simulation, "aftermath", None)
    if bool(getattr(aftermath, "pending_rebirths", None)):
        return False, "pending_rebirth"
    if aftermath is not None:
        return True, "resolved"
    if not _ship_alive(simulation.player1) or not _ship_alive(simulation.player2):
        return True, "resolved"
    return False, "running"


def _ship_alive(ship) -> bool:
    return (
        bool(getattr(ship, "currently_alive", True))
        and getattr(ship, "current_hp", 1) > 0
    )


def _add_timing_seconds(
    timing_seconds: dict[str, float] | None,
    bucket: str,
    started_at: float,
) -> None:
    if timing_seconds is None:
        return
    if bucket not in _TIMING_BUCKETS:
        raise ValueError(f"unknown coordinated timing bucket: {bucket}")
    timing_seconds[bucket] = timing_seconds.get(bucket, 0.0) + max(
        0.0,
        time.perf_counter() - float(started_at),
    )


def _new_timing_seconds() -> dict[str, float] | None:
    if not COORDINATED_TIMING_METRICS_ENABLED:
        return None
    return {bucket: 0.0 for bucket in _TIMING_BUCKETS}


def _timing_started_at(timing_seconds: Mapping[str, float] | None) -> float:
    return time.perf_counter() if timing_seconds is not None else 0.0


def _format_inference_mode_counts(mode_counts: Mapping[str, int]) -> str:
    return ";".join(
        f"{mode}:{count}"
        for mode, count in sorted(mode_counts.items())
        if int(count) > 0
    )


def append_coordinated_batch_timing_csv(
    path: Path,
    *,
    batch_number: int,
    instance_id: int,
    ship: str,
    slot: int,
    instance_count: int,
    rounds: int,
    instance_frames: int,
    coordinated_record_frames: int,
    action_requests: int,
    exploratory_actions: int,
    inference_mode: str,
    batch_seconds: float,
    batches_per_hour: float,
    metrics: BatchMetrics,
    timing_seconds: Mapping[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as existing_file:
            existing_header = next(csv.reader(existing_file), ())
        write_header = tuple(existing_header) != _COORDINATED_TIMING_CSV_HEADER
    timing_total = sum(
        max(0.0, float(timing_seconds.get(bucket, 0.0)))
        for bucket in _TIMING_TOTAL_BUCKETS
    )
    win_rate = (
        metrics.wins / metrics.match_count * 100.0
        if metrics.match_count
        else 0.0
    )
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(_COORDINATED_TIMING_CSV_HEADER)
        writer.writerow(
            (
                str(int(batch_number)),
                str(int(instance_id)),
                str(ship),
                str(int(slot)),
                str(int(instance_count)),
                str(int(rounds)),
                str(int(instance_frames)),
                str(int(coordinated_record_frames)),
                str(int(action_requests)),
                str(int(exploratory_actions)),
                str(inference_mode),
                f"{float(batch_seconds):.6f}",
                f"{float(batches_per_hour):.6f}",
                f"{win_rate:.1f}",
                f"{metrics.average_match_score:.6f}",
                f"{metrics.epsilon:.6f}",
                f"{metrics.learning_rate:.8f}",
                f"{metrics.average_loss:.6f}",
                f"{float(timing_seconds.get('observation', 0.0)):.6f}",
                f"{float(timing_seconds.get('trainee_inference', 0.0)):.6f}",
                f"{float(timing_seconds.get('opponent_inference', 0.0)):.6f}",
                f"{float(timing_seconds.get('simulation', 0.0)):.6f}",
                f"{float(timing_seconds.get('simulation_ship_inputs', 0.0)):.6f}",
                f"{float(timing_seconds.get('simulation_tracking', 0.0)):.6f}",
                f"{float(timing_seconds.get('simulation_update_objects', 0.0)):.6f}",
                f"{float(timing_seconds.get('simulation_collision', 0.0)):.6f}",
                f"{float(timing_seconds.get('simulation_aftermath', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_decision', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_terminal', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_outcome', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_pipeline', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_replay_insert', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_accumulate', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_progress', 0.0)):.6f}",
                f"{float(timing_seconds.get('reward_flush', 0.0)):.6f}",
                f"{float(timing_seconds.get('optimization', 0.0)):.6f}",
                f"{float(timing_seconds.get('save', 0.0)):.6f}",
                f"{timing_total:.6f}",
                str(int(timing_seconds.get("collision_possible_physical_pairs", 0))),
                str(int(timing_seconds.get("collision_candidate_pairs", 0))),
                str(int(timing_seconds.get("collision_dispatched_pairs", 0))),
                str(int(timing_seconds.get("collision_possible_laser_targets", 0))),
                str(int(timing_seconds.get("collision_laser_candidates", 0))),
                str(int(timing_seconds.get("collision_possible_area_targets", 0))),
                str(int(timing_seconds.get("collision_area_candidates", 0))),
                str(int(timing_seconds.get("collision_area_full_scan_fallbacks", 0))),
                str(int(timing_seconds.get("collision_spatial_queries", 0))),
                str(int(timing_seconds.get("collision_spatial_returned_candidates", 0))),
            )
        )


def select_actions_for_records(
    requests: Sequence[CoordinatedActionRequest],
    *,
    parameter_cache: BatchedValueNetworkParameterCache | None = None,
) -> CoordinatedActionBatchResult:
    """Select coordinated trainee actions behind one replaceable helper boundary."""

    if not requests:
        return CoordinatedActionBatchResult(
            selections={},
            inference_mode="none",
            request_count=0,
            exploratory_count=0,
        )

    _raise_for_duplicate_action_requests(requests)
    if not all(_policy_supports_batched_selection(request.policy) for request in requests):
        return _select_actions_for_records_sequential(requests)

    selections: dict[int, ActionSelection] = {}
    greedy_requests: list[CoordinatedActionRequest] = []
    for request in requests:
        record_id = int(request.record_id)
        prepared = request.policy.prepare_action_selection(request.observation)
        if prepared is None:
            greedy_requests.append(request)
        else:
            selections[record_id] = prepared

    inference_mode = "exploration_only"
    if greedy_requests:
        models = tuple(request.policy.model for request in greedy_requests)
        observations = tuple(tuple(request.observation) for request in greedy_requests)
        try:
            parameters = (
                parameter_cache.get(models, set_eval=True)
                if parameter_cache is not None
                else build_batched_value_network_parameters(models, set_eval=True)
            )
        except (TypeError, ValueError, StopIteration):
            parameters = None
        if parameters is not None:
            values = predict_action_values_from_batched_parameters(
                parameters,
                observations,
            )
            for request, row in zip(greedy_requests, values):
                cpu_row = row.detach().cpu()
                selection = request.policy.complete_greedy_selection(
                    int(cpu_row.argmax().item()),
                    tuple(float(value) for value in cpu_row.tolist()),
                )
                selections[int(request.record_id)] = selection
            inference_mode = "batched_value_network"
        else:
            for request in greedy_requests:
                selection = select_action_epsilon_greedy(
                    request.policy.model,
                    request.observation,
                    epsilon=0.0,
                    rng=request.policy.rng,
                )
                selections[int(request.record_id)] = (
                    request.policy.complete_greedy_selection(selection)
                )
            inference_mode = "sequential_fallback"

    return CoordinatedActionBatchResult(
        selections=selections,
        inference_mode=inference_mode,
        request_count=len(selections),
        exploratory_count=sum(
            1 for selection in selections.values() if selection.exploratory
        ),
    )


def _select_actions_for_records_sequential(
    requests: Sequence[CoordinatedActionRequest],
) -> CoordinatedActionBatchResult:
    selections = {
        int(request.record_id): _select_policy_action(
            request.policy,
            request.observation,
        )
        for request in requests
    }
    return CoordinatedActionBatchResult(
        selections=selections,
        inference_mode="sequential_fallback",
        request_count=len(selections),
        exploratory_count=sum(
            1 for selection in selections.values() if selection.exploratory
        ),
    )


def _raise_for_duplicate_action_requests(
    requests: Sequence[CoordinatedActionRequest],
) -> None:
    seen: set[int] = set()
    for request in requests:
        record_id = int(request.record_id)
        if record_id in seen:
            raise ValueError(f"duplicate coordinated action request for {record_id}")
        seen.add(record_id)


def _policy_supports_batched_selection(policy: Any) -> bool:
    return (
        callable(getattr(policy, "prepare_action_selection", None))
        and callable(getattr(policy, "complete_greedy_selection", None))
        and hasattr(policy, "model")
    )


def select_opponent_controls_for_windows(
    windows: Sequence[_CoordinatedWindowRuntime],
    *,
    parameter_cache: BatchedValueNetworkParameterCache | None = None,
    direct: bool = False,
) -> Mapping[int, Any]:
    controls: dict[int, Any] = {}
    ai_requests: list[tuple[_CoordinatedWindowRuntime, tuple[float, ...]]] = []
    for window in windows:
        simulation = window.simulation
        if window.opponent.model is None:
            controls[id(window)] = (
                window.simple_controller.direct_controls_for_frame(simulation)
                if direct
                else window.simple_controller.controls_for_frame(simulation)
            )
            continue
        observation = encode_observation(
            simulation.player2,
            simulation.player1,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
        ai_requests.append((window, tuple(observation)))

    if ai_requests:
        models = tuple(window.opponent.model for window, _observation in ai_requests)
        try:
            parameters = (
                parameter_cache.get(models)
                if parameter_cache is not None
                else build_batched_value_network_parameters(models)
            )
        except (TypeError, ValueError, StopIteration):
            parameters = None
        if parameters is not None:
            values = predict_action_values_from_batched_parameters(
                parameters,
                tuple(observation for _window, observation in ai_requests),
            )
            for (window, _observation), row in zip(ai_requests, values):
                action_index = int(row.detach().cpu().argmax().item())
                controls[id(window)] = (
                    direct_controls_for_action_index(action_index)
                    if direct
                    else controls_for_action_index(action_index)
                )
        else:
            for window, observation in ai_requests:
                selection = select_action_epsilon_greedy(
                    window.opponent.model,
                    observation,
                    epsilon=0.0,
                    value_predictor=predict_action_values_read_only,
                )
                controls[id(window)] = (
                    direct_controls_for_action_index(selection.action_index)
                    if direct
                    else controls_for_action_index(selection.action_index)
                )
    return controls


def select_opponent_controls_for_observations(
    requests: Sequence[CoordinatedOpponentActionRequest],
    *,
    parameter_cache: BatchedValueNetworkParameterCache | None = None,
) -> Mapping[int, TrainingAction]:
    controls: dict[int, TrainingAction] = {}
    if not requests:
        return controls

    ai_requests = tuple(
        request for request in requests if request.opponent.model is not None
    )
    missing_model = tuple(
        request for request in requests if request.opponent.model is None
    )
    if missing_model:
        ids = ", ".join(str(request.request_id) for request in missing_model)
        raise _WorkerRuntimeError(
            f"worker requested model-backed opponent controls without models: {ids}"
        )

    models = tuple(request.opponent.model for request in ai_requests)
    observations = tuple(tuple(request.observation) for request in ai_requests)
    try:
        parameters = (
            parameter_cache.get(models)
            if parameter_cache is not None
            else build_batched_value_network_parameters(models)
        )
    except (TypeError, ValueError, StopIteration):
        parameters = None
    if parameters is not None:
        values = predict_action_values_from_batched_parameters(
            parameters,
            observations,
        )
        for request, row in zip(ai_requests, values):
            action_index = int(row.detach().cpu().argmax().item())
            controls[int(request.request_id)] = direct_controls_for_action_index(
                action_index
            )
    else:
        for request in ai_requests:
            selection = select_action_epsilon_greedy(
                request.opponent.model,
                request.observation,
                epsilon=0.0,
                value_predictor=predict_action_values_read_only,
            )
            controls[int(request.request_id)] = direct_controls_for_action_index(
                selection.action_index
            )
    return controls


def _worker_controls_mapping(controls: Any) -> dict[str, bool]:
    if controls is None:
        raise _WorkerRuntimeError("worker frame step is missing opponent controls")
    if isinstance(controls, Mapping):
        return {
            "forward": bool(controls.get("forward", controls.get("thrust", False))),
            "left": bool(controls.get("left", controls.get("turn_left", False))),
            "right": bool(controls.get("right", controls.get("turn_right", False))),
            "action1": bool(controls.get("action1", controls.get("a1", False))),
            "action2": bool(controls.get("action2", controls.get("a2", False))),
        }
    return {
        "forward": bool(getattr(controls, "thrust", False)),
        "left": bool(getattr(controls, "turn_left", False)),
        "right": bool(getattr(controls, "turn_right", False)),
        "action1": bool(getattr(controls, "a1", False)),
        "action2": bool(getattr(controls, "a2", False)),
    }


def _raise_for_worker_error(result: Any) -> None:
    if getattr(result, "name", "") != "WORKER_ERROR":
        return
    message = getattr(result, "exception_message", "worker error")
    traceback_text = getattr(result, "traceback_text", "")
    if traceback_text:
        message = f"{message}\n{traceback_text}"
    raise _WorkerRuntimeError(message)


def _can_batch_record_optimization(
    states: Sequence[_CoordinatedRecordState],
) -> bool:
    if not states:
        return False
    first_config = states[0].record.config
    if int(first_config.replay_updates_per_batch) <= 0:
        return True
    for state in states:
        components = state.components
        if components is None:
            return False
        config = state.record.config
        if int(config.minibatch_size) != int(first_config.minibatch_size):
            return False
        if int(config.replay_updates_per_batch) != int(
            first_config.replay_updates_per_batch
        ):
            return False
    return can_batch_value_networks(
        tuple(state.components.model for state in states if state.components is not None)
    )


def _optimize_records_batched(
    states: Sequence[_CoordinatedRecordState],
    *,
    rng: Any,
    stop_requested: Callable[[], bool] | None,
) -> dict[int, tuple[float, ...]]:
    if not states:
        return {}
    update_count = int(states[0].record.config.replay_updates_per_batch)
    batch_size = int(states[0].record.config.minibatch_size)
    losses: dict[int, list[float]] = {
        state.record.instance_id: []
        for state in states
    }
    sampled_batches: list[list[tuple[Any, ...] | None]] = []
    for state in states:
        components = state.components
        if components is None:
            raise RuntimeError("coordinated components were not loaded")
        record_batches: list[tuple[Any, ...] | None] = []
        for _ in range(update_count):
            _raise_if_stop_requested(stop_requested)
            if len(components.replay_buffer) < batch_size:
                record_batches.append(None)
            else:
                record_batches.append(
                    components.replay_buffer.sample_minibatch(batch_size, rng=rng)
                )
        sampled_batches.append(record_batches)

    for update_index in range(update_count):
        _raise_if_stop_requested(stop_requested)
        active: list[tuple[_CoordinatedRecordState, tuple[Any, ...]]] = []
        for state, record_batches in zip(states, sampled_batches):
            batch = record_batches[update_index]
            if batch is not None:
                active.append((state, batch))
        if not active:
            continue
        batch_losses = _train_sampled_batches_batched(active)
        for (state, _batch), loss in zip(active, batch_losses):
            losses[state.record.instance_id].append(float(loss))
    return {
        instance_id: tuple(record_losses)
        for instance_id, record_losses in losses.items()
    }


def _train_sampled_batches_batched(
    active: Sequence[tuple[_CoordinatedRecordState, tuple[Any, ...]]],
) -> tuple[float, ...]:
    models = []
    optimizers = []
    observations_by_model = []
    action_indices_by_model = []
    returns_by_model = []
    for state, batch in active:
        components = state.components
        if components is None:
            raise RuntimeError("coordinated components were not loaded")
        models.append(components.model)
        optimizers.append(components.optimizer)
        observations_by_model.append([sample.observation for sample in batch])
        action_indices_by_model.append([sample.action_index for sample in batch])
        returns_by_model.append([sample.return_value for sample in batch])
    try:
        return train_selected_action_regression_batched(
            models,
            optimizers,
            observations_by_model,
            action_indices_by_model,
            returns_by_model,
        )
    except (TypeError, ValueError):
        losses = []
        for state, batch in active:
            components = state.components
            if components is None:
                raise RuntimeError("coordinated components were not loaded")
            losses.append(
                train_selected_action_regression(
                    components.model,
                    components.optimizer,
                    [sample.observation for sample in batch],
                    [sample.action_index for sample in batch],
                    [sample.return_value for sample in batch],
                )
            )
        return tuple(losses)


def _action_request_for_window(
    runtime: _CoordinatedWindowRuntime,
    *,
    timing_seconds: dict[str, float] | None = None,
) -> CoordinatedActionRequest:
    simulation = runtime.simulation
    encode_started_at = _timing_started_at(timing_seconds)
    observation = encode_observation(
        simulation.player1,
        simulation.player2,
        frame_id=simulation.frame_id,
        game_objects=simulation.world,
    )
    _add_timing_seconds(timing_seconds, "observation_encode", encode_started_at)
    return CoordinatedActionRequest(
        record_id=runtime.state.record.instance_id,
        policy=runtime.policy,
        observation=tuple(observation),
    )


def _create_coordinated_window_runtime(
    *,
    state: _CoordinatedRecordState,
    opponent: OpponentSpec,
    policy: Any,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    ship_factory: Callable[..., Any] = create_ship,
) -> _CoordinatedWindowRuntime:
    simulation, ledger, pipeline, simple_controller = _new_coordinated_battle(
        state.record.config,
        opponent,
        rng=rng,
        simulation_factory=simulation_factory,
        audio_service=audio_service,
        ship_factory=ship_factory,
    )
    return _CoordinatedWindowRuntime(
        state=state,
        opponent=opponent,
        policy=policy,
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


def _new_coordinated_battle(
    config: TrainingOrchestrationConfig,
    opponent: OpponentSpec,
    *,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    resources: Any | None = None,
    ship_factory: Callable[..., Any] = create_ship,
):
    trainee = ship_factory(
        config.trainee_ship,
        1,
        resources=resources,
        audio_service=audio_service,
    )
    opponent_ship = ship_factory(
        opponent.ship,
        2,
        resources=resources,
        audio_service=audio_service,
    )
    ledger = event_ledger.BattleEventLedger()
    simulation = simulation_factory(
        None,
        trainee,
        opponent_ship,
        audio_service=audio_service,
        rng=rng,
        resources=resources,
        include_stars=False,
        training_event_ledger=ledger,
    )
    _initialize_training_simulation_ships(simulation, rng)
    return (
        simulation,
        ledger,
        RollingReturnPipeline(
            gamma=config.gamma,
            reward_weights=config.reward_weights,
        ),
        SimpleOpponentController(config, rng=rng),
    )


def _advance_coordinated_window_frame(
    runtime: _CoordinatedWindowRuntime,
    *,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    resources: Any | None = None,
    observation: Sequence[float] | None = None,
    selection: ActionSelection | None = None,
    opponent_controls: Mapping[str, bool] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    timing_seconds: dict[str, float] | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> None:
    if runtime.complete:
        return
    _raise_if_stop_requested(stop_requested)
    state = runtime.state
    components = state.components
    if components is None:
        raise RuntimeError("coordinated components were not loaded")
    config = state.record.config
    simulation = runtime.simulation
    self_ship = simulation.player1
    enemy_ship = simulation.player2
    if observation is None:
        request = _action_request_for_window(
            runtime,
            timing_seconds=timing_seconds,
        )
        observation = request.observation
    else:
        observation = tuple(observation)
    if selection is None:
        selection = select_actions_for_records(
            (
                CoordinatedActionRequest(
                    record_id=state.record.instance_id,
                    policy=runtime.policy,
                    observation=observation,
                ),
            )
        ).selections[state.record.instance_id]
    reward_decision_started_at = _timing_started_at(timing_seconds)
    decision = decision_frame_from_battle_state(
        frame_id=simulation.frame_id + 1,
        observation=observation,
        action_index=selection.action_index,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        world=simulation.world,
    )
    _add_timing_seconds(
        timing_seconds,
        "reward_decision",
        reward_decision_started_at,
    )
    event_start = len(runtime.ledger.events)
    if opponent_controls is None:
        opponent_started_at = _timing_started_at(timing_seconds)
        opponent_controls = _opponent_direct_controls(
            runtime.opponent,
            simulation,
            config,
            runtime.simple_controller,
        )
        _add_timing_seconds(
            timing_seconds,
            "opponent_inference",
            opponent_started_at,
        )
    simulation_started_at = _timing_started_at(timing_seconds)
    step_state = _step_simulation_with_optional_timing(
        simulation,
        actions={
            1: direct_controls_for_action_index(selection.action_index),
            2: opponent_controls,
        },
        timing_seconds=timing_seconds,
    )
    _add_timing_seconds(timing_seconds, "simulation", simulation_started_at)
    runtime.frames_consumed += 1
    reward_started_at = _timing_started_at(timing_seconds)
    reward_terminal_started_at = _timing_started_at(timing_seconds)
    terminal, terminal_reason = _permanent_terminal_state(simulation)
    events = tuple(runtime.ledger.events[event_start:])
    _add_timing_seconds(
        timing_seconds,
        "reward_terminal",
        reward_terminal_started_at,
    )
    reward_outcome_started_at = _timing_started_at(timing_seconds)
    outcome = frame_outcome_from_battle_state(
        frame_id=step_state["frame_id"],
        self_ship=self_ship,
        events=events,
        terminal=terminal,
    )
    _add_timing_seconds(timing_seconds, "reward_outcome", reward_outcome_started_at)
    reward_pipeline_started_at = _timing_started_at(timing_seconds)
    mature_samples = runtime.pipeline.add_frame(decision, outcome)
    _add_timing_seconds(
        timing_seconds,
        "reward_pipeline",
        reward_pipeline_started_at,
    )
    reward_replay_started_at = _timing_started_at(timing_seconds)
    components.replay_buffer.extend(mature_samples)
    _add_timing_seconds(
        timing_seconds,
        "reward_replay_insert",
        reward_replay_started_at,
    )
    reward_accumulate_started_at = _timing_started_at(timing_seconds)
    mature_count = len(mature_samples)
    runtime.total_mature_count += mature_count
    runtime.episode_mature_count += mature_count
    sample_return = sum(sample.return_value for sample in mature_samples)
    runtime.return_sum += sample_return
    runtime.episode_return_sum += sample_return
    _accumulate_weighted_components(runtime.component_sums, mature_samples)
    _accumulate_weighted_components(runtime.episode_component_sums, mature_samples)
    _add_timing_seconds(
        timing_seconds,
        "reward_accumulate",
        reward_accumulate_started_at,
    )
    reward_progress_started_at = _timing_started_at(timing_seconds)
    _emit_window_progress(
        progress_callback,
        frame=runtime.frames_consumed,
        opponent=runtime.opponent,
        action_index=selection.action_index,
        exploratory=selection.exploratory,
        replay_size=len(components.replay_buffer),
        weighted_total_return=_average_value(
            runtime.return_sum,
            runtime.total_mature_count,
        ),
        component_totals=_average_components(
            runtime.component_sums,
            runtime.total_mature_count,
        ),
    )
    _add_timing_seconds(
        timing_seconds,
        "reward_progress",
        reward_progress_started_at,
    )
    _add_timing_seconds(timing_seconds, "reward", reward_started_at)
    _raise_if_stop_requested(stop_requested)
    if terminal:
        _record_coordinated_terminal_episode(
            runtime,
            terminal_reason=terminal_reason,
        )
        if runtime.frames_consumed < runtime.frame_limit:
            _reset_coordinated_window_battle(
                runtime,
                rng=rng,
                simulation_factory=simulation_factory,
                audio_service=audio_service,
                resources=resources,
                ship_factory=ship_factory,
            )


def _step_simulation_with_optional_timing(
    simulation: Any,
    *,
    actions: Mapping[int, Mapping[str, bool]],
    timing_seconds: dict[str, float] | None,
):
    if timing_seconds is None:
        return simulation.step(actions=actions)
    try:
        return simulation.step(actions=actions, timing_seconds=timing_seconds)
    except TypeError as exc:
        if "timing_seconds" not in str(exc):
            raise
        return simulation.step(actions=actions)


def _record_coordinated_terminal_episode(
    runtime: _CoordinatedWindowRuntime,
    *,
    terminal_reason: str,
) -> None:
    win, loss, draw = _classify_round_outcome(runtime.simulation, terminal_reason)
    runtime.episode_results.append(
        TrainingEpisodeResult(
            opponent=runtime.opponent,
            frames=runtime.frames_consumed - runtime.episode_start_frame,
            terminal_reason=terminal_reason,
            mature_samples=runtime.episode_mature_count,
            total_return=_average_value(
                runtime.episode_return_sum,
                runtime.episode_mature_count,
            ),
            win=win,
            loss=loss,
            draw=draw,
            component_totals=_average_components(
                runtime.episode_component_sums,
                runtime.episode_mature_count,
            ),
        )
    )
    runtime.episode_needs_timeout = False
    reset_span = getattr(runtime.policy, "reset_exploration_span", None)
    if callable(reset_span):
        reset_span()


def _reset_coordinated_window_battle(
    runtime: _CoordinatedWindowRuntime,
    *,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    resources: Any | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> None:
    config = runtime.state.record.config
    if getattr(runtime.simulation, "training_episode_deaths", ()):
        runtime.pipeline = RollingReturnPipeline(
            gamma=config.gamma,
            reward_weights=config.reward_weights,
        )
        runtime.simple_controller = SimpleOpponentController(config, rng=rng)
    else:
        (
            runtime.simulation,
            runtime.ledger,
            runtime.pipeline,
            runtime.simple_controller,
        ) = _new_coordinated_battle(
            config,
            runtime.opponent,
            rng=rng,
            simulation_factory=simulation_factory,
            audio_service=audio_service,
            resources=resources,
            ship_factory=ship_factory,
        )
    runtime.episode_start_frame = runtime.frames_consumed
    runtime.episode_mature_count = 0
    runtime.episode_return_sum = 0.0
    runtime.episode_component_sums = {
        component: 0.0
        for component in REWARD_COMPONENTS
    }
    runtime.episode_needs_timeout = True


def _finish_coordinated_window(
    runtime: _CoordinatedWindowRuntime,
    *,
    timing_seconds: dict[str, float] | None = None,
) -> CoordinatedFixedFrameWindowResult:
    components = runtime.state.components
    if components is None:
        raise RuntimeError("coordinated components were not loaded")
    if runtime.episode_needs_timeout:
        reward_started_at = _timing_started_at(timing_seconds)
        reward_flush_started_at = _timing_started_at(timing_seconds)
        mature_samples = tuple(
            runtime.pipeline.flush_pending(
                end_frame_id=runtime.simulation.frame_id,
            )
        )
        _add_timing_seconds(timing_seconds, "reward_flush", reward_flush_started_at)
        reward_replay_started_at = _timing_started_at(timing_seconds)
        components.replay_buffer.extend(mature_samples)
        _add_timing_seconds(
            timing_seconds,
            "reward_replay_insert",
            reward_replay_started_at,
        )
        reward_accumulate_started_at = _timing_started_at(timing_seconds)
        runtime.total_mature_count += len(mature_samples)
        runtime.episode_mature_count += len(mature_samples)
        sample_return = sum(sample.return_value for sample in mature_samples)
        runtime.return_sum += sample_return
        runtime.episode_return_sum += sample_return
        _accumulate_weighted_components(runtime.component_sums, mature_samples)
        _accumulate_weighted_components(
            runtime.episode_component_sums,
            mature_samples,
        )
        _add_timing_seconds(
            timing_seconds,
            "reward_accumulate",
            reward_accumulate_started_at,
        )
        reward_terminal_started_at = _timing_started_at(timing_seconds)
        win, loss, draw = _classify_round_outcome(runtime.simulation, "timeout")
        _add_timing_seconds(
            timing_seconds,
            "reward_terminal",
            reward_terminal_started_at,
        )
        runtime.episode_results.append(
            TrainingEpisodeResult(
                opponent=runtime.opponent,
                frames=runtime.frames_consumed - runtime.episode_start_frame,
                terminal_reason="timeout",
                mature_samples=runtime.episode_mature_count,
                total_return=_average_value(
                    runtime.episode_return_sum,
                    runtime.episode_mature_count,
                ),
                win=win,
                loss=loss,
                draw=draw,
                component_totals=_average_components(
                    runtime.episode_component_sums,
                    runtime.episode_mature_count,
                ),
            )
        )
        reset_span = getattr(runtime.policy, "reset_exploration_span", None)
        if callable(reset_span):
            reset_span()
        _add_timing_seconds(timing_seconds, "reward", reward_started_at)

    return CoordinatedFixedFrameWindowResult(
        opponent=runtime.opponent,
        frames=runtime.frames_consumed,
        mature_samples=runtime.total_mature_count,
        episode_results=tuple(runtime.episode_results),
        total_return=_average_value(
            runtime.return_sum,
            runtime.total_mature_count,
        ),
        win=any(result.win for result in runtime.episode_results),
        loss=any(result.loss for result in runtime.episode_results),
        draw=any(result.draw for result in runtime.episode_results),
        component_totals=_average_components(
            runtime.component_sums,
            runtime.total_mature_count,
        ),
    )


def _emit_window_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if callback is not None:
        callback({"event": "frame", **payload})


def build_coordinated_components(
    record: CoordinatedTrainingRecord,
) -> CoordinatedRuntimeComponents:
    architecture = normalize_architecture_metadata(
        record.metadata.get(
            "architecture",
            model_architecture_metadata(
                record.config.hidden_layer_width,
                record.config.hidden_layer_count,
            ),
        )
    )
    model = build_value_network(
        ValueNetworkConfig(
            hidden_layer_width=int(architecture["hidden_layer_width"]),
            hidden_layer_count=int(architecture["hidden_layer_count"]),
        ),
        device=torch_backend.training_device(record.config.training_device),
    )
    device = next(model.parameters()).device
    optimizer = build_optimizer(model, learning_rate=record.config.learning_rate)
    replay_buffer = TrainingReplayBuffer(record.config.replay_capacity)

    if record.slot.pth_path is not None and record.slot.pth_path.exists():
        if record.slot.pth_path.stat().st_size > 0:
            load_training_checkpoint(
                record.slot.pth_path,
                model,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                map_location=device,
            )
            torch_backend.move_optimizer_state_to_device(optimizer, device)

    return CoordinatedRuntimeComponents(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
    )
