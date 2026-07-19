"""Coordinated multi-instance training runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
import ctypes
from dataclasses import dataclass, field, replace
import csv
from pathlib import Path
import random
import sys
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
from src.training.causal_credit import REWARD_MODE_LEGACY
from src.training.batched_value_network import (
    BatchedValueNetworkParameterCache,
    build_batched_value_network_parameters,
    can_batch_value_networks,
    predict_action_values_from_batched_parameters,
    train_selected_action_regression_batched,
)
from src.training.contracts import (
    ACTION_OUTPUT_SIZE,
    OBSERVATION_INPUT_SIZE,
    SHIP_TYPE_CATALOG_ORDER,
    TrainingAction,
)
from src.training.coordinated_contracts import (
    CoordinatedFixedFrameWindowResult,
    TrainingEpisodeResult,
)
from src.training.coordinated_simulation import (
    CoordinatedWindowRuntime as _CoordinatedWindowRuntime,
    advance_coordinated_window_frame as _advance_coordinated_window_frame,
    create_coordinated_window_runtime as _create_coordinated_window_runtime,
    finish_coordinated_window as _finish_coordinated_window,
    new_coordinated_battle as _new_coordinated_battle,
)
from src.training.observation_transfer import unpack_observation_array
from src.training.episode_metrics import PendingCombatEpisode, finalize_pending_episodes
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
    _accumulate_weighted_components,
    _average_components,
    _average_value,
    _classify_kills_deaths,
    _classify_round_outcome,
    _initialize_training_simulation_ships,
    _opponent_direct_controls,
    _raise_if_stop_requested,
    _select_policy_action,
    controls_for_action_index,
    direct_controls_for_action_index,
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
    StagedTrajectoryPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)
from src.training.session import (
    BatchMetrics,
    MAX_BATCH_LOG_LINES,
    RECENT_BATCH_METRICS_KEY,
    SimulationSpeedTracker,
    TrainingSessionStatus,
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
from src.training.worker_transport import (
    TransportTimingAccumulator,
    receive_timed,
    send_timed,
)


TRAINEE_PARAMETER_CACHE_MAX_ENTRIES = 1
_THREAD_PRIORITY_BELOW_NORMAL = -1


def _set_current_thread_below_normal_priority() -> bool:
    """Lower the Windows coordinator thread without lowering the UI process."""

    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_thread = kernel32.GetCurrentThread
        get_current_thread.restype = ctypes.c_void_p
        set_thread_priority = kernel32.SetThreadPriority
        set_thread_priority.argtypes = (ctypes.c_void_p, ctypes.c_int)
        set_thread_priority.restype = ctypes.c_int
        return bool(
            set_thread_priority(
                get_current_thread(),
                _THREAD_PRIORITY_BELOW_NORMAL,
            )
        )
    except (AttributeError, OSError):
        # Priority is a responsiveness optimization, not a startup requirement.
        return False


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


class CoordinatedExplorationSchedule:
    """One shared explore/greedy gate with independent per-record actions."""

    def __init__(self, *, epsilon: float, frame_span: int, rng: Any):
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.frame_span = max(1, int(frame_span))
        self.rng = rng
        self._frames_remaining = 0
        self._exploring = False
        self._actions: dict[int, int] = {}

    def begin_round(self) -> None:
        self._frames_remaining = 0
        self._exploring = False
        self._actions.clear()

    def reset_record(self, record_id: int) -> None:
        """Do not carry one record's random action across an episode reset."""

        self._actions.pop(int(record_id), None)

    def prepare_selections(
        self,
        requests: Sequence[CoordinatedActionRequest],
    ) -> Mapping[int, ActionSelection] | None:
        if self._frames_remaining <= 0:
            self._exploring = self.epsilon >= 1.0 or (
                self.epsilon > 0.0 and self.rng.random() < self.epsilon
            )
            self._frames_remaining = self.frame_span
            self._actions.clear()
        self._frames_remaining -= 1
        if not self._exploring:
            return None
        selections = {}
        for request in requests:
            record_id = int(request.record_id)
            action_index = self._actions.get(record_id)
            if action_index is None:
                action_index = int(self.rng.randrange(ACTION_OUTPUT_SIZE))
                self._actions[record_id] = action_index
            selections[record_id] = ActionSelection(
                action_index=action_index,
                exploratory=True,
            )
        return selections


_PADDED_COORDINATED_OBSERVATION = (0.0,) * OBSERVATION_INPUT_SIZE


class CoordinatedValueNetworkPolicy:
    """Record-specific model view backed by a session-wide exploration gate."""

    def __init__(
        self,
        model: Any,
        *,
        record_id: int,
        exploration_schedule: CoordinatedExplorationSchedule,
    ):
        self.model = model
        self.record_id = int(record_id)
        self.coordinated_exploration_schedule = exploration_schedule
        self.rng = exploration_schedule.rng

    def complete_greedy_selection(self, action_index: int) -> ActionSelection:
        return ActionSelection(action_index=int(action_index), exploratory=False)

    def reset_exploration_span(self) -> None:
        self.coordinated_exploration_schedule.reset_record(self.record_id)


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
    ipc_seconds: float = 0.0
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
    "ipc",
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

_WORKER_TIMING_BUCKETS = (
    "window_start",
    "worker_step",
    "worker_finish",
    "observation_encode",
    "audio",
    "display",
    "simulation",
    "simulation_ship_inputs",
    "simulation_tracking",
    "simulation_update_objects",
    "simulation_collision",
    "simulation_aftermath",
    "reward",
    "reward_decision",
    "reward_terminal",
    "reward_outcome",
    "reward_pipeline",
    "reward_replay_insert",
    "reward_accumulate",
    "reward_progress",
    "reward_flush",
)

_WORKER_COUNTER_BUCKETS = _TIMING_COUNTER_BUCKETS
_WORKER_MAX_TIMING_BUCKETS = (
    "window_start",
    "worker_step",
    "worker_finish",
    "observation_encode",
    "audio",
    "display",
    "simulation",
    "reward",
)

_TRANSPORT_TIMING_BUCKETS = (
    "parent_send_serialize_seconds",
    "parent_send_transfer_seconds",
    "parent_recv_transfer_seconds",
    "parent_recv_deserialize_seconds",
    "parent_sent_bytes",
    "parent_received_bytes",
    "parent_sent_messages",
    "parent_received_messages",
    "worker_wait_seconds",
    "worker_send_serialize_seconds",
    "worker_send_transfer_seconds",
    "worker_recv_transfer_seconds",
    "worker_recv_deserialize_seconds",
    "worker_sent_bytes",
    "worker_received_bytes",
    "worker_sent_messages",
    "worker_received_messages",
)

_TIMING_BUCKETS = (
    *_TIMING_TOTAL_BUCKETS,
    *_TIMING_DETAIL_BUCKETS,
    *_TIMING_COUNTER_BUCKETS,
    "batch_wall_seconds",
    "coordinator_unattributed_seconds",
    "worker_startup_seconds",
    "worker_sum_active_seconds",
    "worker_max_active_seconds",
    "worker_sum_unattributed_seconds",
    "worker_max_unattributed_seconds",
    *tuple(f"worker_sum_{bucket}" for bucket in _WORKER_TIMING_BUCKETS),
    *tuple(f"worker_sum_{bucket}" for bucket in _WORKER_COUNTER_BUCKETS),
    *tuple(f"worker_max_{bucket}" for bucket in _WORKER_MAX_TIMING_BUCKETS),
    *_TRANSPORT_TIMING_BUCKETS,
)

_COORDINATED_TIMING_CSV_HEADER = (
    "Batch",
    "Instance Count",
    "Completed Rounds",
    "Coordinated Record Frames",
    "Action Requests",
    "Exploratory Actions",
    "Inference Mode",
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
    "IPC Endpoint Seconds",
    "Optimization Seconds",
    "Save Seconds",
    "Coordinator Timed Phase Total Seconds",
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

_ADDITIONAL_TIMING_CSV_FIELDS = (
    ("Batch Wall Seconds", "batch_wall_seconds", False),
    ("Worker Startup Seconds", "worker_startup_seconds", False),
    (
        "Coordinator Unattributed Seconds",
        "coordinator_unattributed_seconds",
        False,
    ),
    ("Worker Sum Active Seconds", "worker_sum_active_seconds", False),
    ("Worker Max Active Seconds", "worker_max_active_seconds", False),
    (
        "Worker Sum Unattributed Seconds",
        "worker_sum_unattributed_seconds",
        False,
    ),
    (
        "Worker Max Unattributed Seconds",
        "worker_max_unattributed_seconds",
        False,
    ),
    *tuple(
        (
            f"Worker Sum {bucket.replace('_', ' ').title()} Seconds",
            f"worker_sum_{bucket}",
            False,
        )
        for bucket in _WORKER_TIMING_BUCKETS
    ),
    *tuple(
        (
            f"Worker Sum {bucket.replace('_', ' ').title()}",
            f"worker_sum_{bucket}",
            True,
        )
        for bucket in _WORKER_COUNTER_BUCKETS
    ),
    *tuple(
        (
            f"Worker Max {bucket.replace('_', ' ').title()} Seconds",
            f"worker_max_{bucket}",
            False,
        )
        for bucket in _WORKER_MAX_TIMING_BUCKETS
    ),
    *tuple(
        (
            bucket.replace("_", " ").title(),
            bucket,
            bucket.endswith(("_bytes", "_messages")),
        )
        for bucket in _TRANSPORT_TIMING_BUCKETS
    ),
)

_COORDINATED_TIMING_CSV_HEADER += tuple(
    label for label, _bucket, _integer in _ADDITIONAL_TIMING_CSV_FIELDS
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
    simulation_speed_tracker: SimulationSpeedTracker = field(
        default_factory=SimulationSpeedTracker
    )


@dataclass
class _WorkerWindowRuntime:
    state: _CoordinatedRecordState
    client: Any
    opponent: OpponentSpec
    policy: Any
    round_index: int
    complete: bool = False
    frame_count: int = 0
    trainee_observation: Any | None = None
    opponent_observation: Any | None = None


class _WorkerRuntimeError(RuntimeError):
    """Raised when a coordinated simulation worker reports or hits a failure."""


class _CpuWorkerStartupUnavailable(RuntimeError):
    """Raised when coordinated worker processes cannot be started."""


CPU_WORKER_FALLBACK_NOTICE = (
    "Multiple CPU workers not available. Proceeding with single process."
)


def _coordinated_regimen_signature(
    record: CoordinatedTrainingRecord,
) -> tuple[Any, ...]:
    config = record.config
    return (
        int(config.match_time_limit),
        int(config.rounds_per_batch),
        int(record.batch_grouping),
        int(config.replay_capacity),
        float(config.starting_epsilon),
        float(config.epsilon_floor),
        float(config.epsilon_decay),
        int(config.epsilon_frame_span),
        float(config.gamma),
        int(config.minibatch_size),
        int(config.replay_updates_per_batch),
        float(config.learning_rate),
        int(config.hidden_layer_width),
        int(config.hidden_layer_count),
    )


def _validate_coordinated_record_contract(
    records: Sequence[CoordinatedTrainingRecord],
) -> None:
    first = records[0]
    signature = _coordinated_regimen_signature(first)
    if any(_coordinated_regimen_signature(record) != signature for record in records[1:]):
        raise ValueError("Coordinated training requires matching Regimen settings")
    slot_number = int(first.slot.slot)
    if any(int(record.slot.slot) != slot_number for record in records[1:]):
        raise ValueError("Coordinated training requires matching model slots")
    opponent_signature = (
        str(first.config.opponent_mode),
        float(first.config.ai_opponent_chance),
    )
    if any(
        (
            str(record.config.opponent_mode),
            float(record.config.ai_opponent_chance),
        )
        != opponent_signature
        for record in records[1:]
    ):
        raise ValueError(
            "Coordinated training requires matching opponent mode and AI frequency"
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
        self._ipc_seconds = 0.0
        self._timing_enabled = False
        self._transport_timing = TransportTimingAccumulator()
        self._worker_wait_seconds = 0.0

    @property
    def ipc_seconds(self) -> float:
        return self._ipc_seconds

    def drain_ipc_seconds(self) -> float:
        elapsed = self._ipc_seconds
        self._ipc_seconds = 0.0
        return elapsed

    def drain_timing_metrics(
        self,
        worker_transport_metrics: Mapping[str, float | int] | None = None,
    ) -> dict[str, float]:
        parent = self._transport_timing.snapshot()
        metrics = {
            f"parent_{key}": float(value)
            for key, value in parent.items()
        }
        metrics["worker_wait_seconds"] = float(self._worker_wait_seconds)
        for key, value in (worker_transport_metrics or {}).items():
            metrics[f"worker_{key}"] = float(value)
        self._transport_timing.reset()
        self._worker_wait_seconds = 0.0
        self._ipc_seconds = 0.0
        return metrics

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
                timing_enabled=bool(const.TRAINING_TIMING_ENABLED),
            ),
            record_timing=False,
        )
        result = self.recv(
            stop_requested=stop_requested,
            record_timing=False,
        )
        _raise_for_worker_error(result)
        if getattr(result, "name", "") != "WORKER_READY":
            raise _WorkerRuntimeError(
                f"worker {self.worker_id} returned {getattr(result, 'name', '')!r} "
                "during startup"
            )
        self._timing_enabled = bool(const.TRAINING_TIMING_ENABLED)

    def send(self, command: Any, *, record_timing: bool = True) -> None:
        if self.connection is None:
            raise _WorkerRuntimeError(f"worker {self.worker_id} is not started")
        if self._timing_enabled and record_timing:
            if getattr(command, "name", "") == "START_WINDOW":
                self._transport_timing.reset()
                self._worker_wait_seconds = 0.0
            measurement = send_timed(self.connection, command)
            self._transport_timing.record_send(measurement)
            self._ipc_seconds += (
                measurement.serialization_seconds + measurement.transfer_seconds
            )
            return
        self.connection.send(command)

    def recv(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
        timeout: float | None = None,
        record_timing: bool = True,
    ) -> Any:
        if self.connection is None:
            raise _WorkerRuntimeError(f"worker {self.worker_id} is not started")
        started_at = (
            time.perf_counter()
            if self._timing_enabled or timeout is not None
            else 0.0
        )
        while True:
            if self.connection.poll(0.05):
                if self._timing_enabled and record_timing:
                    self._worker_wait_seconds += max(
                        0.0,
                        time.perf_counter() - started_at,
                    )
                    result, measurement = receive_timed(self.connection)
                    self._transport_timing.record_receive(measurement)
                    self._ipc_seconds += (
                        measurement.transfer_seconds
                        + measurement.serialization_seconds
                    )
                else:
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

    def request_worker_timing(self) -> Mapping[str, float | int]:
        if not self._timing_enabled:
            return {}
        from src.training.worker_protocol import (
            RequestTimingCommand,
            WorkerTimingResult,
        )

        self.send(RequestTimingCommand(), record_timing=False)
        result = self.recv(record_timing=False)
        if not isinstance(result, WorkerTimingResult):
            raise _WorkerRuntimeError(
                f"worker {self.worker_id} returned "
                f"{getattr(result, 'name', '')!r} for REQUEST_TIMING"
            )
        return result.transport_metrics

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
        _validate_coordinated_record_contract(records)
        shared_floor = float(records[0].config.epsilon_floor)
        shared_current_epsilon = max(
            shared_floor,
            min(
                1.0,
                sum(float(record.config.epsilon) for record in records) / len(records),
            ),
        )
        self._shared_current_epsilon = shared_current_epsilon
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
                    current_epsilon=shared_current_epsilon,
                    epsilon_decay=float(record.config.epsilon_decay),
                    gamma=float(record.config.gamma),
                ),
                history=list(
                    record.initial_history
                    or batch_metrics_history_from_metadata(record.metadata)
                ),
                log_lines=list(record.initial_log_lines),
                current_epsilon=shared_current_epsilon,
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
        self._cpu_workers: dict[int, Any] = {}
        self._thread: threading.Thread | None = None
        self._run_started_at: float | None = None
        self._run_stopped_at: float | None = None
        self._current_batch_started_at: float | None = None
        self._batch_accounting_started_at: float | None = None
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
                ipc_seconds=timing.get("ipc", 0.0),
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
            run_started_at = time.perf_counter()
            self._run_started_at = run_started_at
            self._run_stopped_at = None
            self._current_batch_started_at = None
            self._batch_accounting_started_at = run_started_at
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
                state.status.simulation_speed_multiplier = 0.0
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
                simulation_speed_multiplier=status.simulation_speed_multiplier,
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
        del instance_id
        epsilon = max(0.0, min(1.0, float(value)))
        with self._lock:
            floor = float(next(iter(self._states.values())).record.config.epsilon_floor)
            current_epsilon = max(floor, epsilon)
            self._shared_current_epsilon = current_epsilon
            for state in self._states.values():
                state.current_epsilon = current_epsilon
                state.status.current_epsilon = current_epsilon

    def _run_worker(self) -> None:
        _set_current_thread_below_normal_priority()
        try:
            for state in self._states.values():
                if self._stop_requested.is_set():
                    break
                components = self._component_builder(state.record)
                with self._lock:
                    state.components = components
                    state.status.replay_size = len(components.replay_buffer)
                    state.status.display_message = "Preparing coordinated batch"
                    state.status.simulation_speed_multiplier = 0.0
            if not self._run_batches:
                with self._lock:
                    for state in self._states.values():
                        state.status.display_message = "Coordinated scheduler idle"
                        state.status.simulation_speed_multiplier = 0.0
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
                self._shutdown_persistent_cpu_workers()
            except Exception as exc:
                with self._lock:
                    for state in self._states.values():
                        if not state.status.error:
                            state.status.error = str(exc)
            self._release_display_buffer()
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
        batch_accounting_started_at = self._begin_batch_accounting()
        timing_seconds = _new_timing_seconds()
        measurement_started_at = _timing_started_at(timing_seconds)
        timing_frame_count = 0
        coordinated_frame_count = 0
        timing_completed_batch = False
        batch_action_requests = 0
        batch_exploratory_actions = 0
        batch_inference_mode_counts: dict[str, int] = {}
        trainee_parameter_cache = BatchedValueNetworkParameterCache(
            max_entries=TRAINEE_PARAMETER_CACHE_MAX_ENTRIES,
        )
        exploration_schedule = CoordinatedExplorationSchedule(
            epsilon=self._shared_current_epsilon,
            frame_span=next(iter(self._states.values())).record.config.epsilon_frame_span,
            rng=self._rng,
        )
        schedules = self._opponent_schedules_for_batch()
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
                state.status.simulation_speed_multiplier = 0.0

        results: dict[int, list[CoordinatedFixedFrameWindowResult]] = {
            instance_id: []
            for instance_id in self._states
        }
        try:
            for round_index in range(1, max(len(s) for s in schedules.values()) + 1):
                exploration_schedule.begin_round()
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
                        state.status.simulation_speed_multiplier = 0.0
                        state.simulation_speed_tracker.reset()
                    policy = CoordinatedValueNetworkPolicy(
                        components.model,
                        record_id=instance_id,
                        exploration_schedule=exploration_schedule,
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
                    requests_by_record: dict[int, CoordinatedActionRequest] = {}
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
                        requests_by_record[int(request.record_id)] = request
                    full_action_requests = tuple(
                        requests_by_record.get(
                            int(window.state.record.instance_id),
                            CoordinatedActionRequest(
                                record_id=window.state.record.instance_id,
                                policy=window.policy,
                                observation=_PADDED_COORDINATED_OBSERVATION,
                            ),
                        )
                        for window in active_windows
                    )
                    inference_started_at = _timing_started_at(timing_seconds)
                    action_result = select_actions_for_records(
                        full_action_requests,
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
                        tuple(active_windows),
                        direct=True,
                    )
                    _add_timing_seconds(
                        timing_seconds,
                        "opponent_inference",
                        opponent_started_at,
                    )
                    for window, request in frame_requests:
                        set_visual_effects = getattr(
                            window.simulation,
                            "set_visual_effects_enabled",
                            None,
                        )
                        if callable(set_visual_effects):
                            set_visual_effects(
                                self._display_enabled_for(request.record_id)
                            )
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
                    coordinated_frame_count += 1
                    self._yield_after_coordinated_frame(coordinated_frame_count)
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
                state.status.simulation_speed_multiplier = 0.0

        _synchronize_cuda_for_timing(timing_seconds)
        optimization_started_at = _timing_started_at(timing_seconds)
        optimization_losses = self._optimize_records()
        _synchronize_cuda_for_timing(timing_seconds)
        _add_timing_seconds(
            timing_seconds,
            "optimization",
            optimization_started_at,
        )

        self._advance_shared_epsilon()
        completed_batch_numbers: dict[int, int] = {}
        preliminary_batch_seconds = max(
            0.0,
            time.perf_counter() - batch_accounting_started_at,
        )
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
                batch_seconds=preliminary_batch_seconds,
                emit_outputs=False,
            )
            completed_batch_numbers[instance_id] = batch_number
            if batch_number % state.record.batch_grouping == 0:
                save_started_at = _timing_started_at(timing_seconds)
                self._save_state(state, include_replay=False)
                _add_timing_seconds(timing_seconds, "save", save_started_at)
        batch_accounting_finished_at = time.perf_counter()
        batch_seconds = max(
            0.0,
            batch_accounting_finished_at - batch_accounting_started_at,
        )
        for instance_id, batch_number in completed_batch_numbers.items():
            self._finalize_completed_batch_metrics(
                self._states[instance_id],
                batch_number=batch_number,
                batch_seconds=batch_seconds,
            )
        self._finish_batch_accounting(batch_accounting_finished_at)
        inference_mode_summary = _format_inference_mode_counts(
            batch_inference_mode_counts
        )
        if timing_seconds is not None:
            timing_seconds["batch_wall_seconds"] = max(
                0.0,
                time.perf_counter() - measurement_started_at,
            )
            accounted_coordinator = sum(
                max(0.0, float(timing_seconds.get(bucket, 0.0)))
                for bucket in _TIMING_TOTAL_BUCKETS
            )
            timing_seconds["coordinator_unattributed_seconds"] = max(
                0.0,
                timing_seconds["batch_wall_seconds"] - accounted_coordinator,
            )
        if timing_seconds is not None and completed_batch_numbers:
            self._append_coordinated_batch_timing_row(
                batch_number=max(completed_batch_numbers.values()),
                completed_rounds=sum(
                    len(results[instance_id])
                    for instance_id in completed_batch_numbers
                ),
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
            StartWindowCommand,
            StepFrameCommand,
        )

        batch_accounting_started_at = self._begin_batch_accounting()
        timing_seconds = _new_timing_seconds()
        measurement_started_at = _timing_started_at(timing_seconds)
        worker_timings: dict[int, dict[str, float]] = {}
        timing_frame_count = 0
        coordinated_frame_count = 0
        batch_action_requests = 0
        batch_exploratory_actions = 0
        batch_inference_mode_counts: dict[str, int] = {}
        trainee_parameter_cache = BatchedValueNetworkParameterCache(
            max_entries=TRAINEE_PARAMETER_CACHE_MAX_ENTRIES,
        )
        exploration_schedule = CoordinatedExplorationSchedule(
            epsilon=self._shared_current_epsilon,
            frame_span=next(iter(self._states.values())).record.config.epsilon_frame_span,
            rng=self._rng,
        )
        workers: dict[int, Any] = self._cpu_workers
        results: dict[int, list[CoordinatedFixedFrameWindowResult]] = {
            instance_id: []
            for instance_id in self._states
        }
        try:
            if not workers:
                try:
                    worker_startup_started_at = _timing_started_at(timing_seconds)
                    workers = self._start_cpu_workers()
                    self._cpu_workers = workers
                    _add_timing_seconds(
                        timing_seconds,
                        "worker_startup_seconds",
                        worker_startup_started_at,
                    )
                except TrainingBatchAborted:
                    raise
                except Exception as exc:
                    raise _CpuWorkerStartupUnavailable(str(exc)) from exc
            for client in workers.values():
                drain_ipc = getattr(client, "drain_ipc_seconds", None)
                if callable(drain_ipc):
                    drain_ipc()
            schedules = self._opponent_schedules_for_batch()
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
                    state.status.simulation_speed_multiplier = 0.0
            for round_index in range(1, max(len(s) for s in schedules.values()) + 1):
                exploration_schedule.begin_round()
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
                        state.status.simulation_speed_multiplier = 0.0
                        state.simulation_speed_tracker.reset()
                    policy = CoordinatedValueNetworkPolicy(
                        components.model,
                        record_id=instance_id,
                        exploration_schedule=exploration_schedule,
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
                    _retain_worker_decision_state(
                        window,
                        trainee_observation=start_result.trainee_observation,
                        opponent_observation=start_result.opponent_observation,
                    )
                    active_windows.append(window)

                while any(not window.complete for window in active_windows):
                    _raise_if_stop_requested(self._stop_requested.is_set)
                    unfinished = tuple(
                        window for window in active_windows if not window.complete
                    )
                    observed_at = _timing_started_at(timing_seconds)
                    frame_requests = []
                    for window in unfinished:
                        if window.trainee_observation is None:
                            raise _WorkerRuntimeError(
                                f"worker {window.state.record.instance_id} did not "
                                "return the next trainee observation"
                            )
                        frame_requests.append((window, window))
                    _add_timing_seconds(timing_seconds, "observation", observed_at)

                    inference_started_at = _timing_started_at(timing_seconds)
                    action_result = select_actions_for_records(
                        tuple(
                            CoordinatedActionRequest(
                                record_id=window.state.record.instance_id,
                                policy=window.policy,
                                observation=(
                                    window.trainee_observation
                                    if not window.complete
                                    else _PADDED_COORDINATED_OBSERVATION
                                ),
                            )
                            for window in active_windows
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
                                observation=(
                                    window.opponent_observation
                                    if not window.complete
                                    else _PADDED_COORDINATED_OBSERVATION
                                ),
                            )
                            for window in active_windows
                            if window.opponent.model is not None
                        ),
                    )
                    _add_timing_seconds(
                        timing_seconds,
                        "opponent_inference",
                        opponent_started_at,
                    )

                    for window, _observation in frame_requests:
                        record_id = window.state.record.instance_id
                        selection = action_result.selections[record_id]
                        direct_controls = opponent_controls.get(record_id)
                        display_enabled = self._display_enabled_for(record_id)
                        include_progress = should_update_live_frame_status(
                            window.frame_count + 1,
                            display_on=display_enabled,
                        )
                        window.client.send(
                            StepFrameCommand(
                                record_id=record_id,
                                round_index=window.round_index,
                                trainee_action_index=selection.action_index,
                                trainee_exploratory=selection.exploratory,
                                opponent_controls=(
                                    _worker_controls_mapping(direct_controls)
                                    if direct_controls is not None
                                    else None
                                ),
                                sequence_number=timing_frame_count + 1,
                                include_progress=include_progress,
                                capture_audio=display_enabled,
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
                        window.frame_count = int(frame_result.frame_count)
                        window.complete = bool(frame_result.complete)
                        observed_at = _timing_started_at(timing_seconds)
                        if window.complete:
                            if any(
                                value is not None
                                for value in (
                                    frame_result.next_trainee_observation,
                                    frame_result.next_opponent_observation,
                                    frame_result.next_simple_opponent_controls,
                                )
                            ):
                                raise _WorkerRuntimeError(
                                    f"worker {record_id} returned stale decision state "
                                    "for a completed window"
                                )
                            _clear_worker_decision_state(window)
                        else:
                            _retain_worker_decision_state(
                                window,
                                trainee_observation=(
                                    frame_result.next_trainee_observation
                                ),
                                opponent_observation=(
                                    frame_result.next_opponent_observation
                                ),
                            )
                        _add_timing_seconds(
                            timing_seconds,
                            "observation",
                            observed_at,
                        )
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
                    coordinated_frame_count += 1
                    self._yield_after_coordinated_frame(coordinated_frame_count)
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
                    if timing_seconds is not None:
                        record_id = window.state.record.instance_id
                        worker_timing = worker_timings.setdefault(record_id, {})
                        _accumulate_timing_mapping(
                            worker_timing,
                            finish_result.timing_seconds,
                        )
                        request_timing = getattr(
                            window.client,
                            "request_worker_timing",
                            None,
                        )
                        worker_transport = (
                            request_timing() if callable(request_timing) else {}
                        )
                        drain_timing = getattr(
                            window.client,
                            "drain_timing_metrics",
                            None,
                        )
                        if callable(drain_timing):
                            _accumulate_transport_timing(
                                timing_seconds,
                                drain_timing(worker_transport),
                            )
                    components = window.state.components
                    if components is None:
                        raise RuntimeError("coordinated components were not loaded")
                    components.replay_buffer.extend(finish_result.mature_samples)
                    for sample_chunk in finish_result.mature_sample_chunks:
                        components.replay_buffer.extend(sample_chunk)
                    window.policy.reset_exploration_span()
                    result = finish_result.result
                    results[window.state.record.instance_id].append(result)
                    with self._lock:
                        window.state.status.previous_opponent = window.opponent.ship
                        window.state.status.replay_size = len(components.replay_buffer)
                        window.state.status.component_totals = dict(
                            result.component_totals
                        )
        except TrainingBatchAborted:
            self._merge_timing_stats(
                timing_seconds,
                completed_batches=0,
                frame_count=timing_frame_count,
            )
            return False
        except Exception:
            self._shutdown_persistent_cpu_workers()
            raise
        finally:
            if timing_seconds is not None:
                for client in workers.values():
                    drain_timing = getattr(client, "drain_timing_metrics", None)
                    if callable(drain_timing):
                        _accumulate_transport_timing(
                            timing_seconds,
                            drain_timing(),
                        )
            self._release_display_buffer()
            with self._lock:
                self._current_batch_started_at = None

        with self._lock:
            for state in self._states.values():
                state.status.display_message = "Applying gradient descent"
                state.status.battle_view = None
                state.status.simulation_speed_multiplier = 0.0

        _synchronize_cuda_for_timing(timing_seconds)
        optimization_started_at = _timing_started_at(timing_seconds)
        optimization_losses = self._optimize_records()
        _synchronize_cuda_for_timing(timing_seconds)
        _add_timing_seconds(
            timing_seconds,
            "optimization",
            optimization_started_at,
        )

        self._advance_shared_epsilon()
        completed_batch_numbers: dict[int, int] = {}
        preliminary_batch_seconds = max(
            0.0,
            time.perf_counter() - batch_accounting_started_at,
        )
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
                batch_seconds=preliminary_batch_seconds,
                emit_outputs=False,
            )
            completed_batch_numbers[instance_id] = batch_number
            if batch_number % state.record.batch_grouping == 0:
                save_started_at = _timing_started_at(timing_seconds)
                self._save_state(state, include_replay=False)
                _add_timing_seconds(timing_seconds, "save", save_started_at)
        batch_accounting_finished_at = time.perf_counter()
        batch_seconds = max(
            0.0,
            batch_accounting_finished_at - batch_accounting_started_at,
        )
        for instance_id, batch_number in completed_batch_numbers.items():
            self._finalize_completed_batch_metrics(
                self._states[instance_id],
                batch_number=batch_number,
                batch_seconds=batch_seconds,
            )
        self._finish_batch_accounting(batch_accounting_finished_at)
        inference_mode_summary = _format_inference_mode_counts(
            batch_inference_mode_counts
        )
        if timing_seconds is not None:
            _merge_worker_timing_into_batch(timing_seconds, worker_timings)
            timing_seconds["batch_wall_seconds"] = max(
                0.0,
                time.perf_counter() - measurement_started_at,
            )
            accounted_coordinator = sum(
                max(0.0, float(timing_seconds.get(bucket, 0.0)))
                for bucket in _TIMING_TOTAL_BUCKETS
            ) + max(0.0, float(timing_seconds.get("worker_wait_seconds", 0.0)))
            accounted_coordinator += sum(
                max(0.0, float(timing_seconds.get(bucket, 0.0)))
                for bucket in (
                    "parent_send_serialize_seconds",
                    "parent_send_transfer_seconds",
                    "parent_recv_transfer_seconds",
                    "parent_recv_deserialize_seconds",
                )
            )
            accounted_coordinator += max(
                0.0,
                float(timing_seconds.get("worker_startup_seconds", 0.0)),
            )
            timing_seconds["coordinator_unattributed_seconds"] = max(
                0.0,
                timing_seconds["batch_wall_seconds"] - accounted_coordinator,
            )
        if timing_seconds is not None and completed_batch_numbers:
            self._append_coordinated_batch_timing_row(
                batch_number=max(completed_batch_numbers.values()),
                completed_rounds=sum(
                    len(results[instance_id])
                    for instance_id in completed_batch_numbers
                ),
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

    def _shutdown_persistent_cpu_workers(self) -> None:
        workers = self._cpu_workers
        self._cpu_workers = {}
        self._shutdown_cpu_workers(workers.values())

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

    def _opponent_schedules_for_batch(self) -> dict[int, tuple[OpponentSpec, ...]]:
        """Choose one live coordinated controller per ship for the whole batch."""

        first_state = next(iter(self._states.values()))
        config = first_state.record.config
        if config.opponent_mode != OPPONENT_MODE_EXISTING_AI:
            schedule = simple_opponent_schedule(config.rounds_per_batch)
            return {instance_id: schedule for instance_id in self._states}
        ai_probability = max(
            0.0,
            min(1.0, float(config.ai_opponent_chance) / 100.0),
        )
        trained_by_ship: dict[str, _CoordinatedRecordState] = {}
        for state in self._states.values():
            if state.components is None:
                raise RuntimeError("coordinated components were not loaded")
            set_eval = getattr(state.components.model, "eval", None)
            if callable(set_eval):
                set_eval()
            trained_by_ship[str(state.record.slot.ship)] = state

        selected_by_ship: dict[str, OpponentSpec] = {}
        for ship_name in SHIP_TYPE_CATALOG_ORDER:
            trained_state = trained_by_ship.get(ship_name)
            use_ai = (
                config.opponent_mode == OPPONENT_MODE_EXISTING_AI
                and trained_state is not None
                and (
                    ai_probability >= 1.0
                    or (
                        ai_probability > 0.0
                        and self._rng.random() < ai_probability
                    )
                )
            )
            if use_ai:
                selected_by_ship[ship_name] = OpponentSpec(
                    ship=ship_name,
                    mode=OPPONENT_MODE_EXISTING_AI,
                    slot=int(trained_state.record.slot.slot),
                    model=trained_state.components.model,
                    description=trained_state.record.slot.description,
                )
            else:
                selected_by_ship[ship_name] = OpponentSpec(ship=ship_name)

        schedule = tuple(
            selected_by_ship[ship_name]
            for _ in range(int(config.rounds_per_batch))
            for ship_name in SHIP_TYPE_CATALOG_ORDER
        )
        return {instance_id: schedule for instance_id in self._states}

    def _record_completed_batch(
        self,
        state: _CoordinatedRecordState,
        result: TrainingBatchResult,
        *,
        batch_seconds: float,
        emit_outputs: bool = True,
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
            state.status.current_epsilon = state.current_epsilon
            current_epsilon = state.current_epsilon

        metrics = metrics_from_batch_result(
            result,
            batch=batch_number,
            epsilon=current_epsilon,
            learning_rate=state.record.config.learning_rate,
            batch_seconds=batch_seconds,
        )
        with self._lock:
            state.history.append(metrics)
            rolling = rolling_metrics(tuple(state.history), state.record.batch_grouping)
            if emit_outputs:
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
        if emit_outputs and batch_number % state.record.batch_grouping == 0:
            append_grouped_metrics_csv(self._csv_path(state), metrics, rolling)
        return batch_number

    def _begin_batch_accounting(self) -> float:
        with self._lock:
            started_at = self._batch_accounting_started_at
            if started_at is None:
                started_at = self._run_started_at
            if started_at is None:
                started_at = time.perf_counter()
            self._batch_accounting_started_at = started_at
            return started_at

    def _finish_batch_accounting(self, finished_at: float) -> None:
        with self._lock:
            self._batch_accounting_started_at = float(finished_at)

    def _finalize_completed_batch_metrics(
        self,
        state: _CoordinatedRecordState,
        *,
        batch_number: int,
        batch_seconds: float,
    ) -> None:
        batch_seconds = max(0.0, float(batch_seconds))
        with self._lock:
            durations = self._completed_batch_seconds[state.record.instance_id]
            if durations:
                durations[-1] = batch_seconds
            metrics_index = next(
                (
                    index
                    for index in range(len(state.history) - 1, -1, -1)
                    if state.history[index].batch == int(batch_number)
                ),
                None,
            )
            if metrics_index is None:
                raise RuntimeError("completed coordinated batch metrics were not recorded")
            metrics = replace(
                state.history[metrics_index],
                batch_seconds=batch_seconds,
            )
            state.history[metrics_index] = metrics
            rolling = rolling_metrics(tuple(state.history), state.record.batch_grouping)
            state.status.last_batch_seconds = batch_seconds
            state.status.average_batch_seconds = self._average_batch_seconds_locked(state)
            state.log_lines.append(format_batch_summary_line(metrics, rolling))
            if len(state.log_lines) > MAX_BATCH_LOG_LINES:
                del state.log_lines[: len(state.log_lines) - MAX_BATCH_LOG_LINES]
        if batch_number % state.record.batch_grouping == 0:
            append_grouped_metrics_csv(self._csv_path(state), metrics, rolling)

    def _advance_shared_epsilon(self) -> None:
        with self._lock:
            first_state = next(iter(self._states.values()))
            config = first_state.record.config
            self._shared_current_epsilon = max(
                float(config.epsilon_floor),
                min(
                    1.0,
                    self._shared_current_epsilon * float(config.epsilon_decay),
                ),
            )
            for state in self._states.values():
                state.current_epsilon = self._shared_current_epsilon
                state.status.current_epsilon = self._shared_current_epsilon

    def _optimize_record(self, state: _CoordinatedRecordState) -> tuple[float, ...]:
        components = state.components
        if components is None:
            raise RuntimeError("coordinated components were not loaded")
        config = state.record.config
        losses: list[float] = []
        for _ in range(config.replay_updates_per_batch):
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
        # Treat optimization as an atomic commit phase across coordinated
        # records. Stop remains pending until every configured update finishes.
        states = tuple(self._states.values())
        if not _can_batch_record_optimization(states):
            return {
                state.record.instance_id: self._optimize_record(state)
                for state in states
            }
        return _optimize_records_batched(
            states,
            rng=self._rng,
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

    def _coordinated_timing_csv_path(self) -> Path:
        """Return the single timing report path for this coordinated session."""
        first_state = next(iter(self._states.values()))
        return Path(first_state.record.repository.user_dir) / "coordinated-timing.csv"

    def _append_coordinated_batch_timing_row(
        self,
        *,
        batch_number: int,
        completed_rounds: int,
        coordinated_record_frames: int,
        action_requests: int,
        exploratory_actions: int,
        inference_mode: str,
        timing_seconds: Mapping[str, float],
    ) -> None:
        if not const.TRAINING_TIMING_ENABLED:
            return
        append_coordinated_batch_timing_csv(
            self._coordinated_timing_csv_path(),
            batch_number=batch_number,
            instance_count=len(self._states),
            completed_rounds=completed_rounds,
            coordinated_record_frames=coordinated_record_frames,
            action_requests=action_requests,
            exploratory_actions=exploratory_actions,
            inference_mode=inference_mode,
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
        speed_multiplier = state.simulation_speed_tracker.sample(frame)
        opponent = payload.get("opponent")
        opponent_label = getattr(opponent, "ship", "") if opponent is not None else ""
        battle_view = payload.get("battle_view") if display_on else None
        if battle_view is not None and "rendered_frames" not in battle_view:
            battle_view = freeze_battle_view(battle_view)
        with self._lock:
            state.status.current_frame = frame
            if speed_multiplier is not None:
                state.status.simulation_speed_multiplier = speed_multiplier
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

    def _yield_after_coordinated_frame(self, frame_count: int) -> None:
        interval = int(const.COORDINATED_TRAINING_YIELD_INTERVAL_FRAMES)
        seconds = float(const.COORDINATED_TRAINING_YIELD_SECONDS)
        if (
            interval > 0
            and seconds > 0.0
            and int(frame_count) > 0
            and int(frame_count) % interval == 0
        ):
            time.sleep(seconds)

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
                state.status.simulation_speed_multiplier = 0.0
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
    pending_episodes: list[PendingCombatEpisode] = []
    causal_lifecycle = config.reward_mode != REWARD_MODE_LEGACY

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
        set_visual_effects = getattr(simulation, "set_visual_effects_enabled", None)
        if callable(set_visual_effects):
            set_visual_effects(bool(config.display_on))
        _initialize_training_simulation_ships(simulation, rng)
        return simulation, ledger, StagedTrajectoryPipeline(
            gamma=config.gamma,
            reward_weights=config.reward_weights,
            mode=config.reward_mode,
        ), SimpleOpponentController(config, rng=rng)

    simulation, ledger, pipeline, simple_controller = new_battle()
    episode_start_window_frame = window_frames_consumed
    episode_start_frame_id = simulation.frame_id
    episode_mature_count = 0
    episode_return_sum = 0.0
    episode_component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
    episode_needs_timeout = True

    while window_frames_consumed < window_frame_limit:
        if stop_requested is not None and stop_requested():
            if pipeline.pending_count:
                replay_buffer.extend(
                    pipeline.flush_pending(end_frame_id=simulation.frame_id)
                )
            raise TrainingBatchAborted("training batch stopped")
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
            enabled_components=pipeline.enabled_reward_components,
        )
        ledger.begin_decision(
            self_ship,
            decision.frame_id,
            selection.action_index,
            reward_mode=config.reward_mode,
        )
        staged_index = pipeline.stage_decision(
            decision,
            trajectory_id=ledger.active_trajectory_id,
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
        training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
        can_continue = bool(training_deaths) and window_frames_consumed < window_frame_limit
        reward_terminal = terminal and (
            not causal_lifecycle or 1 in training_deaths or not can_continue
        )
        events = tuple(ledger.events[event_start:])
        outcome = frame_outcome_from_battle_state(
            frame_id=state["frame_id"],
            self_ship=self_ship,
            events=events,
            terminal=reward_terminal,
            enabled_components=pipeline.enabled_reward_components,
        )
        mature_samples = pipeline.add_frame(
            decision,
            outcome,
            ledger=ledger,
            staged_index=staged_index,
        )
        replay_buffer.extend(mature_samples)
        mature_count = len(mature_samples)
        total_mature_count += mature_count
        episode_mature_count += mature_count
        sample_return = sum(sample.return_value for sample in mature_samples)
        return_sum += sample_return
        episode_return_sum += sample_return
        _accumulate_weighted_components(
            component_sums,
            mature_samples,
            episode_component_sums,
        )
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
            kills, deaths = _classify_kills_deaths(simulation)
            if causal_lifecycle:
                pending_episodes.append(
                    PendingCombatEpisode(
                        opponent=opponent,
                        start_frame_id=episode_start_frame_id,
                        end_frame_id=simulation.frame_id,
                        terminal_reason=terminal_reason,
                        win=win,
                        loss=loss,
                        draw=draw,
                        kills=kills,
                        deaths=deaths,
                    )
                )
                if reward_terminal:
                    episode_results.extend(
                        finalize_pending_episodes(pending_episodes, mature_samples)
                    )
                    pending_episodes.clear()
                    ledger.close_reward_trajectory()
            else:
                episode_results.append(TrainingEpisodeResult(
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
                    kills=kills,
                    deaths=deaths,
                    component_totals=_average_components(
                        episode_component_sums,
                        episode_mature_count,
                    ),
                ))
            episode_needs_timeout = False
            reset_span = getattr(trainee_policy, "reset_exploration_span", None)
            if callable(reset_span):
                reset_span()
            if window_frames_consumed < window_frame_limit:
                if training_deaths:
                    if reward_terminal:
                        pipeline = StagedTrajectoryPipeline(
                            gamma=config.gamma,
                            reward_weights=config.reward_weights,
                            mode=config.reward_mode,
                        )
                    simple_controller = SimpleOpponentController(config, rng=rng)
                else:
                    simulation, ledger, pipeline, simple_controller = new_battle()
                episode_start_window_frame = window_frames_consumed
                episode_start_frame_id = simulation.frame_id
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
        _accumulate_weighted_components(
            component_sums,
            mature_samples,
            episode_component_sums,
        )
        win, loss, draw = _classify_round_outcome(simulation, "timeout")
        kills, deaths = _classify_kills_deaths(simulation)
        if causal_lifecycle:
            pending_episodes.append(
                PendingCombatEpisode(
                    opponent=opponent,
                    start_frame_id=episode_start_frame_id,
                    end_frame_id=simulation.frame_id,
                    terminal_reason="timeout",
                    win=win,
                    loss=loss,
                    draw=draw,
                    kills=kills,
                    deaths=deaths,
                )
            )
            episode_results.extend(
                finalize_pending_episodes(pending_episodes, mature_samples)
            )
            pending_episodes.clear()
            ledger.close_reward_trajectory()
        else:
            episode_results.append(TrainingEpisodeResult(
                opponent=opponent,
                frames=window_frames_consumed - episode_start_window_frame,
                terminal_reason="timeout",
                mature_samples=episode_mature_count,
                total_return=_average_value(episode_return_sum, episode_mature_count),
                win=win,
                loss=loss,
                draw=draw,
                kills=kills,
                deaths=deaths,
                component_totals=_average_components(
                    episode_component_sums,
                    episode_mature_count,
                ),
            ))
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
        reward_diagnostics=ledger.diagnostics,
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
    if not const.TRAINING_TIMING_ENABLED:
        return None
    return {bucket: 0.0 for bucket in _TIMING_BUCKETS}


def _timing_started_at(timing_seconds: Mapping[str, float] | None) -> float:
    return time.perf_counter() if timing_seconds is not None else 0.0


def _synchronize_cuda_for_timing(
    timing_seconds: Mapping[str, float] | None,
) -> None:
    if timing_seconds is None:
        return
    torch = torch_backend.get_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


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
    instance_count: int,
    completed_rounds: int,
    coordinated_record_frames: int,
    action_requests: int,
    exploratory_actions: int,
    inference_mode: str,
    timing_seconds: Mapping[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    file_mode = "w"
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as existing_file:
            existing_header = next(csv.reader(existing_file), ())
        if tuple(existing_header) == _COORDINATED_TIMING_CSV_HEADER:
            write_header = False
            file_mode = "a"
    timing_total = sum(
        max(0.0, float(timing_seconds.get(bucket, 0.0)))
        for bucket in _TIMING_TOTAL_BUCKETS
    )
    with path.open(file_mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(_COORDINATED_TIMING_CSV_HEADER)
        writer.writerow(
            (
                str(int(batch_number)),
                str(int(instance_count)),
                str(int(completed_rounds)),
                str(int(coordinated_record_frames)),
                str(int(action_requests)),
                str(int(exploratory_actions)),
                str(inference_mode),
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
                f"{float(timing_seconds.get('ipc', 0.0)):.6f}",
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
            + tuple(
                str(int(timing_seconds.get(bucket, 0)))
                if integer
                else f"{float(timing_seconds.get(bucket, 0.0)):.6f}"
                for _label, bucket, integer in _ADDITIONAL_TIMING_CSV_FIELDS
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
    shared_schedule = getattr(
        requests[0].policy,
        "coordinated_exploration_schedule",
        None,
    )
    if isinstance(shared_schedule, CoordinatedExplorationSchedule) and all(
        getattr(request.policy, "coordinated_exploration_schedule", None)
        is shared_schedule
        for request in requests
    ):
        prepared = shared_schedule.prepare_selections(requests)
        if prepared is not None:
            return CoordinatedActionBatchResult(
                selections=prepared,
                inference_mode="exploration_only",
                request_count=len(prepared),
                exploratory_count=len(prepared),
            )
        greedy_requests = list(requests)
        selections: dict[int, ActionSelection] = {}
    else:
        greedy_requests = []
        selections = {}

    if not all(_policy_supports_batched_selection(request.policy) for request in requests):
        if not isinstance(shared_schedule, CoordinatedExplorationSchedule):
            return _select_actions_for_records_sequential(requests)
    if not isinstance(shared_schedule, CoordinatedExplorationSchedule):
        for request in requests:
            record_id = int(request.record_id)
            prepared = request.policy.prepare_action_selection(request.observation)
            if prepared is None:
                greedy_requests.append(request)
            else:
                selections[record_id] = prepared

    inference_mode = "exploration_only"
    if greedy_requests:
        # Independent policies infer only their greedy subset. A coordinated
        # exploration schedule reaches this path only when every record is
        # greedy, so its cached model tuple remains fixed for the whole batch.
        models = tuple(request.policy.model for request in greedy_requests)
        observations = tuple(request.observation for request in greedy_requests)
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
            action_indices = values.argmax(dim=1).detach().cpu().tolist()
            for request, action_index in zip(greedy_requests, action_indices):
                record_id = int(request.record_id)
                selections[record_id] = request.policy.complete_greedy_selection(
                    int(action_index),
                )
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
                    request.policy.complete_greedy_selection(selection.action_index)
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
        (
            callable(getattr(policy, "prepare_action_selection", None))
            and callable(getattr(policy, "complete_greedy_selection", None))
        )
        or hasattr(policy, "coordinated_exploration_schedule")
    ) and hasattr(policy, "model")


def select_opponent_controls_for_windows(
    windows: Sequence[_CoordinatedWindowRuntime],
    *,
    parameter_cache: BatchedValueNetworkParameterCache | None = None,
    direct: bool = False,
) -> Mapping[int, Any]:
    del parameter_cache
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

    for model, grouped in _group_opponent_requests_by_model(ai_requests):
        values = predict_action_values_read_only(
            model,
            tuple(observation for _window, observation in grouped),
        )
        action_indices = values.argmax(dim=1).detach().cpu().tolist()
        for (window, _observation), action_index in zip(grouped, action_indices):
            controls[id(window)] = (
                direct_controls_for_action_index(int(action_index))
                if direct
                else controls_for_action_index(int(action_index))
            )
    return controls


def select_opponent_controls_for_observations(
    requests: Sequence[CoordinatedOpponentActionRequest],
    *,
    parameter_cache: BatchedValueNetworkParameterCache | None = None,
) -> Mapping[int, TrainingAction]:
    del parameter_cache
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

    grouped_requests = tuple(
        (request, request.observation) for request in ai_requests
    )
    for model, grouped in _group_opponent_requests_by_model(grouped_requests):
        values = predict_action_values_read_only(
            model,
            tuple(observation for _request, observation in grouped),
        )
        action_indices = values.argmax(dim=1).detach().cpu().tolist()
        for (request, _observation), action_index in zip(grouped, action_indices):
            controls[int(request.request_id)] = direct_controls_for_action_index(
                int(action_index)
            )
    return controls


def _group_opponent_requests_by_model(requests):
    grouped: dict[int, tuple[Any, list[Any]]] = {}
    order = []
    for request, observation in requests:
        model = request.opponent.model
        key = id(model)
        if key not in grouped:
            grouped[key] = (model, [])
            order.append(key)
        grouped[key][1].append((request, observation))
    return tuple(grouped[key] for key in order)


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


def _retain_worker_decision_state(
    window: _WorkerWindowRuntime,
    *,
    trainee_observation: Any,
    opponent_observation: Any | None,
) -> None:
    if trainee_observation is None:
        raise _WorkerRuntimeError(
            f"worker {window.state.record.instance_id} omitted a trainee observation"
        )
    window.trainee_observation = unpack_observation_array(
        trainee_observation,
        validate_finite=False,
    )
    window.opponent_observation = (
        unpack_observation_array(opponent_observation, validate_finite=False)
        if opponent_observation is not None
        else None
    )
    if (
        window.opponent_observation is None
        and _worker_opponent_requires_parent_controls(window.opponent)
    ):
        raise _WorkerRuntimeError(
            f"worker {window.state.record.instance_id} omitted opponent decision state"
        )


def _accumulate_timing_mapping(
    target: dict[str, float],
    values: Mapping[str, float] | None,
) -> None:
    for key, value in (values or {}).items():
        target[str(key)] = target.get(str(key), 0.0) + max(0.0, float(value))


def _accumulate_transport_timing(
    timing_seconds: dict[str, float] | None,
    values: Mapping[str, float] | None,
) -> None:
    if timing_seconds is None:
        return
    for bucket in _TRANSPORT_TIMING_BUCKETS:
        timing_seconds[bucket] = timing_seconds.get(bucket, 0.0) + max(
            0.0,
            float((values or {}).get(bucket, 0.0)),
        )
    endpoint_seconds = sum(
        max(0.0, float((values or {}).get(bucket, 0.0)))
        for bucket in _TRANSPORT_TIMING_BUCKETS
        if bucket.endswith("_seconds") and bucket != "worker_wait_seconds"
    )
    timing_seconds["ipc"] = timing_seconds.get("ipc", 0.0) + endpoint_seconds


def _merge_worker_timing_into_batch(
    timing_seconds: dict[str, float] | None,
    worker_timings: Mapping[int, Mapping[str, float]],
) -> None:
    if timing_seconds is None:
        return
    worker_active = []
    worker_unattributed = []
    for values in worker_timings.values():
        active = sum(
            max(0.0, float(values.get(bucket, 0.0)))
            for bucket in ("window_start", "worker_step", "worker_finish")
        )
        attributed = sum(
            max(0.0, float(values.get(bucket, 0.0)))
            for bucket in (
                "simulation",
                "reward_decision",
                "reward",
                "observation_encode",
                "audio",
            )
        )
        worker_active.append(active)
        worker_unattributed.append(max(0.0, active - attributed))
    timing_seconds["worker_sum_active_seconds"] = sum(worker_active)
    timing_seconds["worker_max_active_seconds"] = max(worker_active, default=0.0)
    timing_seconds["worker_sum_unattributed_seconds"] = sum(worker_unattributed)
    timing_seconds["worker_max_unattributed_seconds"] = max(
        worker_unattributed,
        default=0.0,
    )
    for bucket in (*_WORKER_TIMING_BUCKETS, *_WORKER_COUNTER_BUCKETS):
        timing_seconds[f"worker_sum_{bucket}"] = sum(
            max(0.0, float(values.get(bucket, 0.0)))
            for values in worker_timings.values()
        )
    for bucket in _WORKER_MAX_TIMING_BUCKETS:
        timing_seconds[f"worker_max_{bucket}"] = max(
            (
                max(0.0, float(values.get(bucket, 0.0)))
                for values in worker_timings.values()
            ),
            default=0.0,
        )
def _clear_worker_decision_state(window: _WorkerWindowRuntime) -> None:
    window.trainee_observation = None
    window.opponent_observation = None


def _worker_opponent_requires_parent_controls(opponent: OpponentSpec) -> bool:
    return opponent.mode == OPPONENT_MODE_EXISTING_AI or opponent.slot is not None


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
) -> dict[int, tuple[float, ...]]:
    if not states:
        return {}
    update_count = int(states[0].record.config.replay_updates_per_batch)
    batch_size = int(states[0].record.config.minibatch_size)
    losses: dict[int, list[float]] = {
        state.record.instance_id: []
        for state in states
    }
    # Preserve the established deterministic RNG order (all updates for one
    # record before the next) while retaining only compact integer indices.
    sampled_indices: list[list[Any | None]] = []
    for state in states:
        components = state.components
        if components is None:
            raise RuntimeError("coordinated components were not loaded")
        record_indices = []
        for _ in range(update_count):
            record_indices.append(
                components.replay_buffer.sample_minibatch_indices(
                    batch_size,
                    rng=rng,
                )
                if len(components.replay_buffer) >= batch_size
                else None
            )
        sampled_indices.append(record_indices)

    for update_index in range(update_count):
        active: list[tuple[_CoordinatedRecordState, tuple[Any, Any, Any]]] = []
        for state, record_indices in zip(states, sampled_indices):
            components = state.components
            if components is None:
                raise RuntimeError("coordinated components were not loaded")
            indices = record_indices[update_index]
            if indices is not None:
                active.append(
                    (
                        state,
                        components.replay_buffer.minibatch_arrays_for_indices(
                            indices,
                        ),
                    )
                )
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
    active: Sequence[
        tuple[_CoordinatedRecordState, tuple[Any, Any, Any]]
    ],
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
        observations, action_indices, returns = batch
        observations_by_model.append(observations)
        action_indices_by_model.append(action_indices)
        returns_by_model.append(returns)
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
            observations, action_indices, returns = batch
            losses.append(
                train_selected_action_regression(
                    components.model,
                    components.optimizer,
                    observations,
                    action_indices,
                    returns,
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
    set_visual_effects = getattr(simulation, "set_visual_effects_enabled", None)
    if callable(set_visual_effects):
        set_visual_effects(bool(config.display_on))
    _initialize_training_simulation_ships(simulation, rng)
    return (
        simulation,
        ledger,
        StagedTrajectoryPipeline(
            gamma=config.gamma,
            reward_weights=config.reward_weights,
            mode=config.reward_mode,
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
        enabled_components=runtime.pipeline.enabled_reward_components,
    )
    runtime.ledger.begin_decision(
        self_ship,
        decision.frame_id,
        selection.action_index,
        reward_mode=config.reward_mode,
    )
    staged_index = runtime.pipeline.stage_decision(
        decision,
        trajectory_id=runtime.ledger.active_trajectory_id,
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
    training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
    causal_lifecycle = config.reward_mode != REWARD_MODE_LEGACY
    can_continue = bool(training_deaths) and runtime.frames_consumed < runtime.frame_limit
    reward_terminal = terminal and (
        not causal_lifecycle or 1 in training_deaths or not can_continue
    )
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
        terminal=reward_terminal,
        enabled_components=runtime.pipeline.enabled_reward_components,
    )
    _add_timing_seconds(timing_seconds, "reward_outcome", reward_outcome_started_at)
    reward_pipeline_started_at = _timing_started_at(timing_seconds)
    mature_samples = runtime.pipeline.add_frame(
        decision,
        outcome,
        ledger=runtime.ledger,
        staged_index=staged_index,
    )
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
    _accumulate_weighted_components(
        runtime.component_sums,
        mature_samples,
        runtime.episode_component_sums,
    )
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
            mature_samples=mature_samples,
            reward_terminal=reward_terminal,
        )
        if runtime.frames_consumed < runtime.frame_limit:
            _reset_coordinated_window_battle(
                runtime,
                reward_terminal=reward_terminal,
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
    mature_samples: Sequence,
    reward_terminal: bool,
) -> None:
    win, loss, draw = _classify_round_outcome(runtime.simulation, terminal_reason)
    kills, deaths = _classify_kills_deaths(runtime.simulation)
    if runtime.state.record.config.reward_mode != REWARD_MODE_LEGACY:
        runtime.pending_episodes.append(
            PendingCombatEpisode(
                opponent=runtime.opponent,
                start_frame_id=runtime.episode_start_frame,
                end_frame_id=runtime.simulation.frame_id,
                terminal_reason=terminal_reason,
                win=win,
                loss=loss,
                draw=draw,
                kills=kills,
                deaths=deaths,
            )
        )
        if reward_terminal:
            runtime.episode_results.extend(
                finalize_pending_episodes(runtime.pending_episodes, mature_samples)
            )
            runtime.pending_episodes.clear()
            runtime.ledger.close_reward_trajectory()
    else:
        runtime.episode_results.append(TrainingEpisodeResult(
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
            kills=kills,
            deaths=deaths,
            component_totals=_average_components(
                runtime.episode_component_sums,
                runtime.episode_mature_count,
            ),
        ))
    runtime.episode_needs_timeout = False
    reset_span = getattr(runtime.policy, "reset_exploration_span", None)
    if callable(reset_span):
        reset_span()


def _reset_coordinated_window_battle(
    runtime: _CoordinatedWindowRuntime,
    *,
    reward_terminal: bool,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    resources: Any | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> None:
    config = runtime.state.record.config
    if getattr(runtime.simulation, "training_episode_deaths", ()):
        if reward_terminal:
            runtime.pipeline = StagedTrajectoryPipeline(
                gamma=config.gamma,
                reward_weights=config.reward_weights,
                mode=config.reward_mode,
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
    runtime.episode_start_frame = runtime.simulation.frame_id
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
        _accumulate_weighted_components(
            runtime.component_sums,
            mature_samples,
            runtime.episode_component_sums,
        )
        _add_timing_seconds(
            timing_seconds,
            "reward_accumulate",
            reward_accumulate_started_at,
        )
        reward_terminal_started_at = _timing_started_at(timing_seconds)
        win, loss, draw = _classify_round_outcome(runtime.simulation, "timeout")
        kills, deaths = _classify_kills_deaths(runtime.simulation)
        _add_timing_seconds(
            timing_seconds,
            "reward_terminal",
            reward_terminal_started_at,
        )
        if runtime.state.record.config.reward_mode != REWARD_MODE_LEGACY:
            runtime.pending_episodes.append(
                PendingCombatEpisode(
                    opponent=runtime.opponent,
                    start_frame_id=runtime.episode_start_frame,
                    end_frame_id=runtime.simulation.frame_id,
                    terminal_reason="timeout",
                    win=win,
                    loss=loss,
                    draw=draw,
                    kills=kills,
                    deaths=deaths,
                )
            )
            runtime.episode_results.extend(
                finalize_pending_episodes(runtime.pending_episodes, mature_samples)
            )
            runtime.pending_episodes.clear()
            runtime.ledger.close_reward_trajectory()
        else:
            runtime.episode_results.append(TrainingEpisodeResult(
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
                kills=kills,
                deaths=deaths,
                component_totals=_average_components(
                    runtime.episode_component_sums,
                    runtime.episode_mature_count,
                ),
            ))
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
        reward_diagnostics=runtime.ledger.diagnostics,
    )


def _emit_window_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if callback is not None:
        callback({"event": "frame", **payload})


# Keep the long-standing private names available to callers and tests while the
# implementation itself lives below the CPU-only dependency boundary.
from src.training.coordinated_simulation import (
    advance_coordinated_window_frame as _advance_coordinated_window_frame,
    create_coordinated_window_runtime as _create_coordinated_window_runtime,
    finish_coordinated_window as _finish_coordinated_window,
    new_coordinated_battle as _new_coordinated_battle,
)


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
