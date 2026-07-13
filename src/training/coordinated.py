"""Coordinated multi-instance training runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import random
import threading
import time
from typing import Any

from src.Battle.battle import BattleSimulation
from src.Objects.Ships.registry import create_ship
from src.audio import NullAudioService
from src.training import torch_backend
from src.training import event_ledger
from src.training.model_registry import (
    TrainingModelRepository,
    TrainingModelSlot,
    model_architecture_metadata,
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
    _fully_arm_training_shofixti,
    _opponent_controls,
    _raise_if_stop_requested,
    _select_policy_action,
    controls_for_action_index,
    discover_existing_ai_opponents,
    existing_ai_opponent_schedule,
    simple_opponent_schedule,
)
from src.training.replay import TrainingReplayBuffer, load_training_checkpoint
from src.training.rewards import (
    REWARD_COMPONENTS,
    RollingReturnPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)
from src.training.session import (
    BatchMetrics,
    MAX_BATCH_LOG_LINES,
    TrainingSessionStatus,
    batch_metrics_history_from_metadata,
    format_batch_summary_line,
    metrics_from_batch_result,
    rolling_metrics,
)
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
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


@dataclass
class _CoordinatedRecordState:
    record: CoordinatedTrainingRecord
    status: TrainingSessionStatus
    history: list[BatchMetrics] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    components: CoordinatedRuntimeComponents | None = None
    current_epsilon: float = 0.0


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
        if enabled:
            raise RuntimeError("Coordinated training runs are headless")

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
    ):
        if len(records) < 2:
            raise ValueError("Coordinated training requires at least two records")
        self._states = {
            int(record.instance_id): _CoordinatedRecordState(
                record=record,
                status=TrainingSessionStatus(
                    completed_batches=int(
                        record.metadata.get("progress", {}).get("completed_batches", 0)
                    ),
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
        self._rng = rng or random.Random()
        self._run_batches = bool(run_batches)
        self._idle_sleep_seconds = max(0.0, float(idle_sleep_seconds))
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
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
        self._proxies = {
            instance_id: CoordinatedTrainingStatusProxy(self, instance_id)
            for instance_id in self._states
        }

    @property
    def records(self) -> tuple[CoordinatedTrainingRecord, ...]:
        return tuple(state.record for state in self._states.values())

    @property
    def proxies(self) -> dict[int, CoordinatedTrainingStatusProxy]:
        return dict(self._proxies)

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

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status_for_instance(self, instance_id: int) -> TrainingSessionStatus:
        with self._lock:
            state = self._states[int(instance_id)]
            status = state.status
            elapsed = self._elapsed_seconds_locked()
            return TrainingSessionStatus(
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
                replay_size=status.replay_size,
                recent_loss=status.recent_loss,
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
            self._mark_stopped()

    def _run_one_coordinated_batch(self) -> bool:
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
                            audio_service=self._audio_service,
                        )
                    )
                while any(not window.complete for window in active_windows):
                    _raise_if_stop_requested(self._stop_requested.is_set)
                    for window in active_windows:
                        if window.complete:
                            continue
                        _advance_coordinated_window_frame(
                            window,
                            rng=self._rng,
                            simulation_factory=self._simulation_factory,
                            audio_service=self._audio_service,
                            progress_callback=(
                                lambda payload, item=window.state: self._on_record_progress(
                                    item,
                                    payload,
                                )
                            ),
                            stop_requested=self._stop_requested.is_set,
                        )
                    _raise_if_stop_requested(self._stop_requested.is_set)
                for window in active_windows:
                    result = _finish_coordinated_window(window)
                    results[window.state.record.instance_id].append(result)
                    with self._lock:
                        window.state.status.previous_opponent = window.opponent.ship
                        window.state.status.component_totals = dict(
                            result.component_totals
                        )
            batch_finished_at = time.perf_counter()
        except TrainingBatchAborted:
            return False
        finally:
            with self._lock:
                self._current_batch_started_at = None

        for instance_id, state in self._states.items():
            components = state.components
            if components is None:
                continue
            self._record_completed_batch(
                state,
                TrainingBatchResult(
                    completed_rounds=len(results[instance_id]),
                    replay_size=len(components.replay_buffer),
                    optimization_losses=(),
                    round_results=tuple(results[instance_id]),
                ),
                batch_seconds=max(0.0, batch_finished_at - batch_started_at),
            )
        return True

    def _opponents_for_batch(
        self,
        state: _CoordinatedRecordState,
    ) -> tuple[OpponentSpec, ...]:
        config = state.record.config
        if config.opponent_mode == OPPONENT_MODE_EXISTING_AI:
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
    ) -> None:
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

    def _on_record_progress(
        self,
        state: _CoordinatedRecordState,
        payload: Mapping[str, Any],
    ) -> None:
        if payload.get("event") != "frame":
            return
        opponent = payload.get("opponent")
        opponent_label = getattr(opponent, "ship", "") if opponent is not None else ""
        with self._lock:
            state.status.current_frame = int(payload.get("frame", 0))
            state.status.current_opponent = opponent_label
            state.status.replay_size = int(payload.get("replay_size", 0))
            state.status.last_action_exploratory = bool(
                payload.get("exploratory", False)
            )
            state.status.weighted_total_return = float(
                payload.get("weighted_total_return", 0.0)
            )
            state.status.component_totals = dict(payload.get("component_totals", {}))

    def _mark_stopped(self) -> None:
        with self._lock:
            self._run_stopped_at = time.perf_counter()
            for state in self._states.values():
                state.status.running = False
                state.status.stopping = False
                if not state.status.error:
                    state.status.display_message = ""

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

    Terminal battles reset inside the same fixed window. The window frame count
    is scheduler-local, so a reset simulation with ``frame_id == 0`` cannot
    extend the configured frame budget.
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
        _fully_arm_training_shofixti(simulation.player1)
        _fully_arm_training_shofixti(simulation.player2)
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
                1: controls_for_action_index(selection.action_index),
                2: _opponent_controls(
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
    ship_factory: Callable[..., Any] = create_ship,
):
    trainee = ship_factory(config.trainee_ship, 1, audio_service=audio_service)
    opponent_ship = ship_factory(opponent.ship, 2, audio_service=audio_service)
    ledger = event_ledger.BattleEventLedger()
    simulation = simulation_factory(
        None,
        trainee,
        opponent_ship,
        audio_service=audio_service,
        rng=rng,
        include_stars=False,
        training_event_ledger=ledger,
    )
    _fully_arm_training_shofixti(simulation.player1)
    _fully_arm_training_shofixti(simulation.player2)
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
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
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
    observation = encode_observation(
        self_ship,
        enemy_ship,
        frame_id=simulation.frame_id,
        game_objects=simulation.world,
    )
    selection = _select_policy_action(runtime.policy, observation)
    decision = decision_frame_from_battle_state(
        frame_id=simulation.frame_id + 1,
        observation=observation,
        action_index=selection.action_index,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        world=simulation.world,
    )
    event_start = len(runtime.ledger.events)
    step_state = simulation.step(
        actions={
            1: controls_for_action_index(selection.action_index),
            2: _opponent_controls(
                runtime.opponent,
                simulation,
                config,
                runtime.simple_controller,
            ),
        }
    )
    runtime.frames_consumed += 1
    terminal, terminal_reason = _permanent_terminal_state(simulation)
    events = tuple(runtime.ledger.events[event_start:])
    outcome = frame_outcome_from_battle_state(
        frame_id=step_state["frame_id"],
        self_ship=self_ship,
        events=events,
        terminal=terminal,
    )
    mature_samples = runtime.pipeline.add_frame(decision, outcome)
    components.replay_buffer.extend(mature_samples)
    mature_count = len(mature_samples)
    runtime.total_mature_count += mature_count
    runtime.episode_mature_count += mature_count
    sample_return = sum(sample.return_value for sample in mature_samples)
    runtime.return_sum += sample_return
    runtime.episode_return_sum += sample_return
    _accumulate_weighted_components(runtime.component_sums, mature_samples)
    _accumulate_weighted_components(runtime.episode_component_sums, mature_samples)
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
                ship_factory=ship_factory,
            )


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
    ship_factory: Callable[..., Any] = create_ship,
) -> None:
    (
        runtime.simulation,
        runtime.ledger,
        runtime.pipeline,
        runtime.simple_controller,
    ) = _new_coordinated_battle(
        runtime.state.record.config,
        runtime.opponent,
        rng=rng,
        simulation_factory=simulation_factory,
        audio_service=audio_service,
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
) -> CoordinatedFixedFrameWindowResult:
    components = runtime.state.components
    if components is None:
        raise RuntimeError("coordinated components were not loaded")
    if runtime.episode_needs_timeout:
        mature_samples = tuple(
            runtime.pipeline.flush_pending(
                end_frame_id=runtime.simulation.frame_id,
            )
        )
        components.replay_buffer.extend(mature_samples)
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
        win, loss, draw = _classify_round_outcome(runtime.simulation, "timeout")
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
