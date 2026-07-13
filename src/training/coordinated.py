"""Coordinated multi-instance training runtime skeleton."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import threading
import time
from typing import Any

from src.training import torch_backend
from src.training.model_registry import (
    TrainingModelRepository,
    TrainingModelSlot,
    model_architecture_metadata,
    normalize_architecture_metadata,
)
from src.training.orchestration import TrainingOrchestrationConfig
from src.training.replay import TrainingReplayBuffer, load_training_checkpoint
from src.training.session import (
    BatchMetrics,
    TrainingSessionStatus,
    batch_metrics_history_from_metadata,
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


@dataclass
class _CoordinatedRecordState:
    record: CoordinatedTrainingRecord
    status: TrainingSessionStatus
    history: list[BatchMetrics] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    components: CoordinatedRuntimeComponents | None = None


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

    Phase 2 intentionally stops at lifecycle and component-loading behavior.
    Battle stepping, synchronized optimization, and saves are added by later
    phases.
    """

    def __init__(
        self,
        records: tuple[CoordinatedTrainingRecord, ...],
        *,
        component_builder: Callable[[CoordinatedTrainingRecord], CoordinatedRuntimeComponents] | None = None,
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
            )
            for record in records
        }
        self._component_builder = component_builder or build_coordinated_components
        self._idle_sleep_seconds = max(0.0, float(idle_sleep_seconds))
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_started_at: float | None = None
        self._run_stopped_at: float | None = None
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
                average_batch_seconds=status.average_batch_seconds,
                batches_per_hour=status.batches_per_hour,
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
                    state.status.display_message = "Coordinated scheduler idle"
            while not self._stop_requested.is_set():
                time.sleep(self._idle_sleep_seconds)
        except Exception as exc:
            self._stop_requested.set()
            with self._lock:
                for state in self._states.values():
                    state.status.error = str(exc)
                    state.status.stopping = True
        finally:
            self._mark_stopped()

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
