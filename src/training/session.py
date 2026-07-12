"""UI-facing training session state, metrics, and persistence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import csv
from pathlib import Path
import random
import threading
import time
from typing import Any

import src.const as const
from src.audio import DisplayGatedAudioService, NullAudioService
from src.training import torch_backend
from src.training.contracts import (
    ACTION_SCHEMA_METADATA,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_SCHEMA_VERSION,
)
from src.training.model_registry import (
    MODEL_METADATA_VERSION,
    TrainingModelRepository,
    TrainingModelSlot,
    current_game_settings_metadata,
    metadata_from_state,
    model_architecture_metadata,
    model_paths,
    normalize_architecture_metadata,
)
from src.training.orchestration import (
    TrainingBatchAborted,
    TrainingBatchResult,
    TrainingOrchestrationConfig,
    run_training_batch,
)
from src.training.replay import (
    TrainingReplayBuffer,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.render_view import freeze_battle_view
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
)


MAX_BATCH_LOG_LINES = 1000
RECENT_BATCH_METRICS_KEY = "recent_batch_metrics"


@dataclass(frozen=True)
class CompatibilityReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class BatchMetrics:
    batch: int
    match_count: int
    wins: int
    losses: int
    draws: int
    average_match_score: float
    epsilon: float
    learning_rate: float
    average_loss: float


def batch_metrics_to_metadata(metrics: BatchMetrics) -> dict[str, Any]:
    return {
        "batch": metrics.batch,
        "match_count": metrics.match_count,
        "wins": metrics.wins,
        "losses": metrics.losses,
        "draws": metrics.draws,
        "average_match_score": metrics.average_match_score,
        "epsilon": metrics.epsilon,
        "learning_rate": metrics.learning_rate,
        "average_loss": metrics.average_loss,
    }


def batch_metrics_from_metadata(value: Mapping[str, Any]) -> BatchMetrics:
    return BatchMetrics(
        batch=int(value["batch"]),
        match_count=int(value["match_count"]),
        wins=int(value["wins"]),
        losses=int(value["losses"]),
        draws=int(value["draws"]),
        average_match_score=float(value["average_match_score"]),
        epsilon=float(value["epsilon"]),
        learning_rate=float(value["learning_rate"]),
        average_loss=float(value["average_loss"]),
    )


def batch_metrics_history_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[BatchMetrics, ...]:
    progress = metadata.get("progress", {})
    if not isinstance(progress, Mapping):
        return ()
    raw_history = progress.get(RECENT_BATCH_METRICS_KEY, ())
    if not isinstance(raw_history, list | tuple):
        return ()

    history: list[BatchMetrics] = []
    for item in raw_history:
        if not isinstance(item, Mapping):
            continue
        try:
            history.append(batch_metrics_from_metadata(item))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(history)


@dataclass
class TrainingSessionStatus:
    running: bool = False
    stopping: bool = False
    completed_batches: int = 0
    elapsed_training_seconds: float = 0.0
    current_batch_seconds: float = 0.0
    last_batch_seconds: float = 0.0
    average_batch_seconds: float = 0.0
    batches_per_hour: float = 0.0
    current_round: int = 0
    total_rounds: int = 0
    current_opponent: str = ""
    previous_opponent: str = ""
    current_frame: int = 0
    replay_size: int = 0
    recent_loss: float | None = None
    current_epsilon: float = 0.0
    epsilon_decay: float = 0.0
    gamma: float = 0.0
    last_action_exploratory: bool | None = None
    weighted_total_return: float = 0.0
    component_totals: dict[str, float] = field(default_factory=dict)
    batch_component_totals: dict[str, float] = field(default_factory=dict)
    battle_view: Mapping[str, Any] | None = None
    display_message: str = ""
    error: str = ""


class TrainingSessionError(RuntimeError):
    """Raised for user-facing training-session startup failures."""


def validate_model_metadata(
    metadata: Mapping[str, Any],
    *,
    architecture: Mapping[str, Any] | None = None,
    game_settings: Mapping[str, Any] | None = None,
) -> CompatibilityReport:
    """Validate persisted model metadata before loading or continuing training."""

    errors: list[str] = []
    warnings: list[str] = []
    if not metadata:
        return CompatibilityReport()

    expected_action_ordering = [dict(action) for action in ACTION_SCHEMA_METADATA]
    checks = (
        ("schema_version", MODEL_METADATA_VERSION, "metadata schema"),
        (
            "observation_schema_version",
            OBSERVATION_SCHEMA_VERSION,
            "observation schema",
        ),
        ("observation_input_size", OBSERVATION_INPUT_SIZE, "observation input size"),
        ("action_schema_version", ACTION_SCHEMA_VERSION, "action schema"),
    )
    for key, expected, label in checks:
        if metadata.get(key) != expected:
            errors.append(f"{label} is incompatible")

    if metadata.get("action_ordering") != expected_action_ordering:
        errors.append("action ordering is incompatible")

    if architecture is not None:
        expected_architecture = normalize_architecture_metadata(architecture)
        actual_architecture = normalize_architecture_metadata(
            metadata.get("architecture", {})
        )
        if actual_architecture and actual_architecture != expected_architecture:
            errors.append("model architecture is incompatible")

    current_settings = dict(game_settings or current_game_settings_metadata())
    model_settings = metadata.get("game_settings", {})
    if isinstance(model_settings, Mapping):
        labels = {
            "ship_directions": "ship directions",
            "asteroid_count": "asteroid count",
            "repeat_key_delay": "repeat-key delay",
            "fps": "simulation FPS",
        }
        for key, label in labels.items():
            if key in model_settings and model_settings.get(key) != current_settings.get(key):
                warnings.append(
                    f"{label} differs: model {model_settings.get(key)}, "
                    f"current {current_settings.get(key)}"
                )

    return CompatibilityReport(tuple(errors), tuple(warnings))


def metrics_from_batch_result(
    result: TrainingBatchResult,
    *,
    batch: int,
    epsilon: float,
    learning_rate: float,
) -> BatchMetrics:
    match_count = len(result.round_results)
    score_total = sum(round_result.total_return for round_result in result.round_results)
    return BatchMetrics(
        batch=int(batch),
        match_count=match_count,
        wins=sum(1 for round_result in result.round_results if round_result.win),
        losses=sum(1 for round_result in result.round_results if round_result.loss),
        draws=sum(1 for round_result in result.round_results if round_result.draw),
        average_match_score=score_total / match_count if match_count else 0.0,
        epsilon=float(epsilon),
        learning_rate=float(learning_rate),
        average_loss=float(result.average_loss or 0.0),
    )


def rolling_metrics(history: tuple[BatchMetrics, ...], grouping: int) -> BatchMetrics:
    if not history:
        raise ValueError("history must contain at least one batch")
    window = history[-max(1, int(grouping)) :]
    count = len(window)
    latest = window[-1]
    return BatchMetrics(
        batch=latest.batch,
        match_count=sum(item.match_count for item in window),
        wins=sum(item.wins for item in window),
        losses=sum(item.losses for item in window),
        draws=sum(item.draws for item in window),
        average_match_score=sum(item.average_match_score for item in window) / count,
        epsilon=latest.epsilon,
        learning_rate=latest.learning_rate,
        average_loss=sum(item.average_loss for item in window) / count,
    )


def format_batch_summary_line(metrics: BatchMetrics, rolling: BatchMetrics) -> str:
    win_rate = (metrics.wins / metrics.match_count * 100.0) if metrics.match_count > 0 else 0.0
    loss_rate = (metrics.losses / metrics.match_count * 100.0) if metrics.match_count > 0 else 0.0
    draw_rate = (metrics.draws / metrics.match_count * 100.0) if metrics.match_count > 0 else 0.0
    rolling_win_rate = (rolling.wins / rolling.match_count * 100.0) if rolling.match_count > 0 else 0.0

    return (
        f"Batch {metrics.batch:6d} | "
        f"{win_rate:5.1f}% W, "
        f"{loss_rate:5.1f}% L, "
        f"{draw_rate:5.1f}% D | "
        f"({rolling_win_rate:6.2f}% W) | "
        f"Score: {metrics.average_match_score: =7.3f} ({rolling.average_match_score: =7.3f}) | "
        f"Epsilon: {metrics.epsilon:.5f} | "
        f"LR: {metrics.learning_rate:.5f} | "
        f"Loss: {metrics.average_loss:.4f} ({rolling.average_loss:.4f})"
    )


def append_grouped_metrics_csv(path: Path, metrics: BatchMetrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    win_rate = (metrics.wins / metrics.match_count * 100.0) if metrics.match_count else 0.0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(("Batch", "Win %", "Score", "Epsilon", "Learning Rate", "Loss"))
        writer.writerow(
            (
                str(metrics.batch),
                f"{win_rate:.1f}",
                f"{metrics.average_match_score:.1f}",
                f"{metrics.epsilon:.5f}",
                f"{metrics.learning_rate:.6f}",
                f"{metrics.average_loss:.4f}",
            )
        )


class TrainingSession:
    """Own a trainable model and run complete batches on a worker thread."""

    def __init__(
        self,
        *,
        repository: TrainingModelRepository,
        slot: TrainingModelSlot,
        metadata: Mapping[str, Any],
        config: TrainingOrchestrationConfig,
        batch_grouping: int,
        batch_runner: Callable[..., TrainingBatchResult] = run_training_batch,
        audio_service: Any | None = None,
        rng: Any | None = None,
        initial_history: tuple[BatchMetrics, ...] = (),
        initial_log_lines: tuple[str, ...] = (),
    ):
        if slot.is_bundled:
            raise TrainingSessionError("Bundled training models are read-only")
        self.repository = repository
        self.slot = slot
        self.metadata = dict(metadata)
        self.config = config
        self._current_epsilon = float(config.epsilon)
        self.batch_grouping = max(1, int(batch_grouping))
        self.batch_runner = batch_runner
        self.rng = rng or random.Random()
        self._status = TrainingSessionStatus(
            completed_batches=int(
                self.metadata.get("progress", {}).get("completed_batches", 0)
            ),
            current_epsilon=self._current_epsilon,
            epsilon_decay=float(config.epsilon_decay),
            gamma=float(config.gamma),
        )
        self._history: list[BatchMetrics] = list(
            initial_history or batch_metrics_history_from_metadata(self.metadata)
        )
        self._trim_history()
        self._log_lines: list[str] = list(initial_log_lines)[-MAX_BATCH_LOG_LINES:]
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._display_on = threading.Event()
        if config.display_on:
            self._display_on.set()
        self.audio_service = DisplayGatedAudioService(
            audio_service or NullAudioService(),
            self._display_on.is_set,
        )
        self._next_display_frame_time = time.perf_counter()
        self._run_started_at: float | None = None
        self._run_stopped_at: float | None = None
        self._current_batch_started_at: float | None = None
        self._completed_batches_at_run_start = self._status.completed_batches
        self._completed_batch_seconds: list[float] = []
        self._thread: threading.Thread | None = None

        report = validate_model_metadata(
            self.metadata,
            architecture=self.metadata.get(
                "architecture",
                model_architecture_metadata(
                    config.hidden_layer_width,
                    config.hidden_layer_count,
                ),
            ),
        )
        if report.errors:
            raise TrainingSessionError("; ".join(report.errors))

    @property
    def status(self) -> TrainingSessionStatus:
        with self._lock:
            elapsed_training_seconds = self._elapsed_training_seconds_locked()
            current_batch_seconds = self._current_batch_seconds_locked()
            average_batch_seconds = self._average_batch_seconds_locked()
            batches_per_hour = self._batches_per_hour_locked(
                elapsed_training_seconds
            )
            return TrainingSessionStatus(
                running=self._status.running,
                stopping=self._status.stopping,
                completed_batches=self._status.completed_batches,
                elapsed_training_seconds=elapsed_training_seconds,
                current_batch_seconds=current_batch_seconds,
                last_batch_seconds=self._status.last_batch_seconds,
                average_batch_seconds=average_batch_seconds,
                batches_per_hour=batches_per_hour,
                current_round=self._status.current_round,
                total_rounds=self._status.total_rounds,
                current_opponent=self._status.current_opponent,
                previous_opponent=self._status.previous_opponent,
                current_frame=self._status.current_frame,
                replay_size=self._status.replay_size,
                recent_loss=self._status.recent_loss,
                current_epsilon=self._status.current_epsilon,
                epsilon_decay=self._status.epsilon_decay,
                gamma=self._status.gamma,
                last_action_exploratory=self._status.last_action_exploratory,
                weighted_total_return=self._status.weighted_total_return,
                component_totals=dict(self._status.component_totals),
                batch_component_totals=dict(self._status.batch_component_totals),
                battle_view=self._status.battle_view,
                display_message=self._status.display_message,
                error=self._status.error,
            )

    @property
    def log_lines(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._log_lines)

    @property
    def history(self) -> tuple[BatchMetrics, ...]:
        with self._lock:
            return tuple(self._history)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._status.running = True
            self._status.stopping = False
            self._status.error = ""
        self._thread = threading.Thread(
            target=self._run_worker,
            name="StarAITrainingSession",
            daemon=True,
        )
        self._thread.start()

    def set_starting_epsilon(self, value: float) -> None:
        epsilon = max(0.0, min(1.0, float(value)))
        with self._lock:
            self.config = replace(
                self.config,
                starting_epsilon=epsilon,
                epsilon=epsilon,
            )
            self._current_epsilon = epsilon
            self._status.current_epsilon = epsilon

    def request_stop(self) -> None:
        self._stop_requested.set()
        with self._lock:
            self._status.stopping = True

    def set_display_on(self, enabled: bool) -> None:
        if enabled:
            if not self._display_on.is_set():
                self._next_display_frame_time = time.perf_counter()
                self._display_on.set()
                self.audio_service.start_battle_music()
            else:
                self._display_on.set()
        else:
            if self._display_on.is_set():
                self.audio_service.stop_music()
            self._display_on.clear()
            lock = getattr(self, "_lock", None)
            if lock is None:
                self._status.battle_view = None
            else:
                with lock:
                    self._status.battle_view = None

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def run_synchronously(self, *, max_batches: int = 1) -> None:
        try:
            self._run_loop(max_batches=max_batches)
        finally:
            self._mark_stopped()

    def _run_worker(self) -> None:
        try:
            self._run_loop()
        except Exception as exc:
            with self._lock:
                self._status.error = str(exc)
        finally:
            self._mark_stopped()

    def _run_loop(self, *, max_batches: int | None = None) -> None:
        model, optimizer, replay_buffer = self._build_components()
        batches_run = 0
        with self._lock:
            self._status.running = True
            self._status.error = ""
            self._run_started_at = time.perf_counter()
            self._run_stopped_at = None
            self._current_batch_started_at = None
            self._completed_batches_at_run_start = self._status.completed_batches
            self._completed_batch_seconds = []
            self._status.elapsed_training_seconds = 0.0
            self._status.current_batch_seconds = 0.0
            self._status.last_batch_seconds = 0.0
            self._status.average_batch_seconds = 0.0
            self._status.batches_per_hour = 0.0

        while not self._stop_requested.is_set():
            with self._lock:
                self._status.display_message = "Preparing new batch"
                self._status.battle_view = None
                batch_config = replace(
                    self.config,
                    epsilon=self._current_epsilon,
                    display_on=False,
                )
                batch_started_at = time.perf_counter()
                self._current_batch_started_at = batch_started_at
            try:
                result = self.batch_runner(
                    model=model,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    config=batch_config,
                    rng=self.rng,
                    model_repository=self.repository,
                    audio_service=self.audio_service,
                    progress_callback=self._on_progress,
                    stop_requested=self._stop_requested.is_set,
                    battle_view_enabled=self._display_on.is_set,
                )
            except TrainingBatchAborted:
                break
            finally:
                batch_finished_at = time.perf_counter()
                with self._lock:
                    self._current_batch_started_at = None
            if self._stop_requested.is_set():
                break
            self._record_completed_batch(
                result,
                batch_seconds=max(0.0, batch_finished_at - batch_started_at),
            )
            self._save_state(model, optimizer, replay_buffer)
            batches_run += 1
            if max_batches is not None and batches_run >= max_batches:
                break

    def _build_components(self):
        architecture = normalize_architecture_metadata(
            self.metadata.get(
                "architecture",
                model_architecture_metadata(
                    self.config.hidden_layer_width,
                    self.config.hidden_layer_count,
                ),
            )
        )
        model = build_value_network(
            ValueNetworkConfig(
                hidden_layer_width=int(architecture["hidden_layer_width"]),
                hidden_layer_count=int(architecture["hidden_layer_count"]),
            ),
            device=torch_backend.preferred_device(),
        )
        device = next(model.parameters()).device
        optimizer = build_optimizer(model, learning_rate=self.config.learning_rate)
        replay_buffer = TrainingReplayBuffer(self.config.replay_capacity)

        if self.slot.pth_path is not None and self.slot.pth_path.exists():
            if self.slot.pth_path.stat().st_size > 0:
                load_training_checkpoint(
                    self.slot.pth_path,
                    model,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    map_location=device,
                )
                torch_backend.move_optimizer_state_to_device(optimizer, device)
        return model, optimizer, replay_buffer

    def _record_completed_batch(
        self,
        result: TrainingBatchResult,
        *,
        batch_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            self._status.completed_batches += 1
            batch_number = self._status.completed_batches
            self._completed_batch_seconds.append(max(0.0, float(batch_seconds)))
            self._status.last_batch_seconds = self._completed_batch_seconds[-1]
            elapsed_training_seconds = self._elapsed_training_seconds_locked()
            self._status.elapsed_training_seconds = elapsed_training_seconds
            self._status.average_batch_seconds = self._average_batch_seconds_locked()
            self._status.batches_per_hour = self._batches_per_hour_locked(
                elapsed_training_seconds
            )
            self._current_epsilon = max(
                0.0,
                min(1.0, self._current_epsilon * float(self.config.epsilon_decay)),
            )
            self._status.current_epsilon = self._current_epsilon
            current_epsilon = self._current_epsilon
        metrics = metrics_from_batch_result(
            result,
            batch=batch_number,
            epsilon=current_epsilon,
            learning_rate=self.config.learning_rate,
        )
        with self._lock:
            self._history.append(metrics)
            rolling = rolling_metrics(tuple(self._history), self.batch_grouping)
            self._log_lines.append(format_batch_summary_line(metrics, rolling))
            
            from src.training.rewards import REWARD_COMPONENTS
            batch_components = {c: 0.0 for c in REWARD_COMPONENTS}
            for round_result in result.round_results:
                for comp, val in round_result.component_totals.items():
                    if comp in batch_components:
                        batch_components[comp] += val
                        
            num_rounds = len(result.round_results)
            if num_rounds > 0:
                for comp in batch_components:
                    batch_components[comp] /= num_rounds
            
            self._status.batch_component_totals = batch_components
            
            if len(self._log_lines) > MAX_BATCH_LOG_LINES:
                del self._log_lines[: len(self._log_lines) - MAX_BATCH_LOG_LINES]
            self._status.recent_loss = metrics.average_loss
            self._status.replay_size = result.replay_size

        if batch_number % self.batch_grouping == 0:
            append_grouped_metrics_csv(self._csv_path(), rolling)
        self._trim_history()

    def _save_state(self, model, optimizer, replay_buffer: TrainingReplayBuffer) -> None:
        pth_path, _ = model_paths(self.repository.user_dir, self.slot.ship, self.slot.slot)
        completed_batches = self.status.completed_batches
        save_training_checkpoint(
            pth_path,
            model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            extra_state={"completed_batches": completed_batches},
        )
        metadata = metadata_from_state(
            ship=self.slot.ship,
            slot=self.slot.slot,
            description=str(self.metadata.get("description", self.slot.description)),
            architecture=self.metadata.get(
                "architecture",
                model_architecture_metadata(
                    self.config.hidden_layer_width,
                    self.config.hidden_layer_count,
                ),
            ),
            training=self._training_metadata_for_save(),
            progress={
                "completed_batches": completed_batches,
                RECENT_BATCH_METRICS_KEY: [
                    batch_metrics_to_metadata(metrics)
                    for metrics in self.history[-self.batch_grouping :]
                ],
            },
        )
        self.metadata = metadata
        self.slot = self.repository.create_or_update_user_model(metadata)

    def _training_metadata_for_save(self) -> dict[str, Any]:
        training = self.metadata.get("training", {})
        training = dict(training) if isinstance(training, Mapping) else {}
        regimen = training.get("regimen", {})
        regimen = dict(regimen) if isinstance(regimen, Mapping) else {}
        with self._lock:
            starting_epsilon = float(self.config.starting_epsilon)
            current_epsilon = float(self._current_epsilon)
            epsilon_decay = float(self.config.epsilon_decay)
            epsilon_frame_span = int(self.config.epsilon_frame_span)
        regimen.update(
            {
                "starting_epsilon": starting_epsilon,
                "current_epsilon": current_epsilon,
                "epsilon": current_epsilon,
                "epsilon_decay": epsilon_decay,
                "epsilon_frame_span": epsilon_frame_span,
            }
        )
        training["regimen"] = regimen
        return training

    def _csv_path(self) -> Path:
        _, metadata_path = model_paths(self.repository.user_dir, self.slot.ship, self.slot.slot)
        return metadata_path.with_suffix(".csv")

    def _trim_history(self) -> None:
        limit = max(1, int(self.batch_grouping))
        if len(self._history) > limit:
            del self._history[: len(self._history) - limit]

    def _mark_stopped(self) -> None:
        stopped_at = time.perf_counter()
        with self._lock:
            self._status.running = False
            self._status.stopping = False
            self._run_stopped_at = stopped_at
            self._current_batch_started_at = None
            self._status.elapsed_training_seconds = self._elapsed_training_seconds_locked()
            self._status.current_batch_seconds = 0.0
            self._status.average_batch_seconds = self._average_batch_seconds_locked()
            self._status.batches_per_hour = self._batches_per_hour_locked(
                self._status.elapsed_training_seconds
            )

    def _elapsed_training_seconds_locked(self) -> float:
        started_at = getattr(self, "_run_started_at", None)
        if started_at is None:
            return float(getattr(self._status, "elapsed_training_seconds", 0.0))
        if self._status.running:
            ended_at = time.perf_counter()
        else:
            ended_at = getattr(self, "_run_stopped_at", None) or started_at
        return max(0.0, float(ended_at) - float(started_at))

    def _current_batch_seconds_locked(self) -> float:
        started_at = getattr(self, "_current_batch_started_at", None)
        if started_at is None or not self._status.running:
            return 0.0
        return max(0.0, time.perf_counter() - float(started_at))

    def _average_batch_seconds_locked(self) -> float:
        durations = getattr(self, "_completed_batch_seconds", ())
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    def _batches_per_hour_locked(self, elapsed_seconds: float) -> float:
        completed_this_run = (
            self._status.completed_batches
            - getattr(
                self,
                "_completed_batches_at_run_start",
                self._status.completed_batches,
            )
        )
        if completed_this_run <= 0 or elapsed_seconds <= 0.0:
            return 0.0
        return completed_this_run * 3600.0 / elapsed_seconds

    def _on_progress(self, payload: Mapping[str, Any]) -> None:
        event = payload.get("event")
        opponent = payload.get("opponent")
        opponent_label = getattr(opponent, "ship", "") if opponent is not None else ""
        with self._lock:
            if event == "round_start":
                self._status.display_message = ""
                self._status.battle_view = None
                self._status.current_round = int(payload.get("round_index", 0))
                self._status.total_rounds = int(payload.get("total_rounds", 0))
                self._status.current_opponent = opponent_label
                self._status.current_frame = 0
                self._status.weighted_total_return = 0.0
            if event == "batch_optimization_start":
                self._status.display_message = "Applying gradient descent"
                self._status.battle_view = None
            if event == "round_end":
                if "result" in payload:
                    self._status.component_totals = dict(
                        payload.get("result").component_totals
                    )
                self._status.previous_opponent = opponent_label
            if "battle_view" in payload:
                self._status.battle_view = (
                    freeze_battle_view(payload["battle_view"])
                    if self._display_on.is_set()
                    else None
                )
            if event == "frame":
                self._status.current_frame = int(payload.get("frame", 0))
                self._status.current_opponent = opponent_label
                self._status.replay_size = int(payload.get("replay_size", 0))
                self._status.last_action_exploratory = bool(
                    payload.get("exploratory", False)
                )
                self._status.weighted_total_return = float(
                    payload.get("weighted_total_return", 0.0)
                )
        if event == "frame" and self._display_on.is_set():
            self._throttle_display_frame()

    def _throttle_display_frame(self) -> None:
        self._next_display_frame_time += 1.0 / const.FPS
        sleep_seconds = self._next_display_frame_time - time.perf_counter()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        else:
            self._next_display_frame_time = time.perf_counter()
