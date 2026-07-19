"""Independent CPU training hosted in a spawn worker process."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import multiprocessing
from multiprocessing import shared_memory
from pathlib import Path
import queue
import struct
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

import pygame

import src.const as const
from src.Battle.battle_draw import (
    BattleDrawController,
    BattleDrawOptions,
    DisplayStarField,
    create_play_battle_layout,
)
from src.audio import NullAudioService, RecordingAudioService
from src.resources import HeadlessAssetManager
from src.training import torch_backend
from src.training.contracts import SHIP_TYPE_CATALOG_ORDER
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    TrainingModelRepository,
    TrainingModelSlot,
)
from src.training.opponent_cache import ModelSaveCoordinator, OpponentModelKey
from src.training.orchestration import (
    OPPONENT_MODE_EXISTING_AI,
    discover_existing_ai_opponents,
)
from src.training.process_worker import (
    _set_worker_process_below_normal_priority,
    _start_process_with_single_threaded_openblas,
)
from src.training.session import (
    LIVE_STATUS_FRAME_INTERVAL,
    BatchMetrics,
    TrainingSession,
    TrainingSessionStatus,
)


_DISPLAY_SEQUENCE_BYTES = 8
_MESSAGE_STATUS = "status"
_MESSAGE_SAVED = "saved"
_MESSAGE_AUDIO = "audio"
_CONTROL_DISPLAY_BUFFER = "display_buffer"
_CONTROL_STARTING_EPSILON = "starting_epsilon"


class CpuBatchPacingGroup:
    """Bound the batch lead among independent CPU training processes."""

    def __init__(
        self,
        participant_count: int,
        *,
        max_batch_lead: int = 0,
        context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        participant_count = int(participant_count)
        if participant_count < 2:
            raise ValueError("CPU pacing requires at least two participants")
        self.participant_count = participant_count
        self.max_batch_lead = max(0, int(max_batch_lead))
        process_context = context or multiprocessing.get_context("spawn")
        self._completed = process_context.Array(
            "Q", participant_count, lock=False
        )
        self._active = process_context.Array(
            "b", [1] * participant_count, lock=False
        )
        self._condition = process_context.Condition(process_context.RLock())

    def wait_after_batch(
        self,
        participant_index: int,
        completed_batches: int,
        *,
        stop_requested,
    ) -> None:
        """Record progress and wait while this participant is too far ahead."""

        index = self._validated_index(participant_index)
        completed_batches = max(0, int(completed_batches))
        with self._condition:
            if not self._active[index]:
                return
            self._completed[index] = completed_batches
            self._condition.notify_all()
            while self._active[index] and not stop_requested():
                active_progress = [
                    int(self._completed[other_index])
                    for other_index in range(self.participant_count)
                    if self._active[other_index]
                ]
                if (
                    not active_progress
                    or completed_batches
                    <= min(active_progress) + self.max_batch_lead
                ):
                    return
                self._condition.wait(timeout=0.1)
            if stop_requested():
                self._active[index] = 0
                self._condition.notify_all()

    def deactivate(self, participant_index: int) -> None:
        """Remove a stopped or failed participant without blocking its peers."""

        index = self._validated_index(participant_index)
        with self._condition:
            if self._active[index]:
                self._active[index] = 0
                self._condition.notify_all()

    def _validated_index(self, participant_index: int) -> int:
        index = int(participant_index)
        if not 0 <= index < self.participant_count:
            raise IndexError(f"unknown CPU pacing participant {index}")
        return index


class ProcessModelSaveCoordinator(ModelSaveCoordinator):
    """Share save-in-progress and committed generations with CPU workers."""

    def __init__(
        self,
        *,
        context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        entry_count = len(SHIP_TYPE_CATALOG_ORDER) * MODEL_SLOT_COUNT
        self._saving_counts = self._context.Array("i", entry_count, lock=False)
        self._generations = self._context.Array("Q", entry_count, lock=False)
        self._lock = self._context.RLock()

    @contextmanager
    def saving(
        self,
        key: OpponentModelKey | tuple[str, int],
    ):
        index = self._index_for(key)
        with self._lock:
            self._saving_counts[index] += 1
        committed = False
        try:
            yield
            committed = True
        finally:
            with self._lock:
                self._saving_counts[index] = max(
                    0,
                    self._saving_counts[index] - 1,
                )
                if committed:
                    self._generations[index] += 1

    def is_saving(self, key: OpponentModelKey | tuple[str, int]) -> bool:
        index = self._index_for(key)
        with self._lock:
            return self._saving_counts[index] > 0

    def snapshot(self) -> tuple[frozenset[tuple[str, int]], tuple[int, ...]]:
        with self._lock:
            saving = frozenset(
                self._key_for_index(index)
                for index, count in enumerate(self._saving_counts)
                if count > 0
            )
            generations = tuple(int(value) for value in self._generations)
        return saving, generations

    def _index_for(self, key: OpponentModelKey | tuple[str, int]) -> int:
        normalized = (
            key
            if isinstance(key, OpponentModelKey)
            else OpponentModelKey(str(key[0]), int(key[1]))
        )
        try:
            ship_index = SHIP_TYPE_CATALOG_ORDER.index(normalized.ship)
        except ValueError as exc:
            raise ValueError(f"unknown training ship: {normalized.ship}") from exc
        slot_index = int(normalized.slot) - 1
        if not 0 <= slot_index < MODEL_SLOT_COUNT:
            raise ValueError(f"invalid training model slot: {normalized.slot}")
        return ship_index * MODEL_SLOT_COUNT + slot_index

    @staticmethod
    def _key_for_index(index: int) -> tuple[str, int]:
        ship_index, slot_index = divmod(int(index), MODEL_SLOT_COUNT)
        return SHIP_TYPE_CATALOG_ORDER[ship_index], slot_index + 1


class _SavingAwareRepository:
    def __init__(self, repository: TrainingModelRepository, saving_keys) -> None:
        self._repository = repository
        self._saving_keys = frozenset(saving_keys)
        self.bundled_dir = repository.bundled_dir
        self.user_dir = repository.user_dir

    def slot_for(self, ship: str, slot: int) -> TrainingModelSlot:
        key = (str(ship), int(slot))
        if key in self._saving_keys:
            return TrainingModelSlot(key[0], key[1], SLOT_EMPTY)
        return self._repository.slot_for(*key)

    def slots_for_ship(self, ship: str) -> list[TrainingModelSlot]:
        return [self.slot_for(ship, slot) for slot in range(1, MODEL_SLOT_COUNT + 1)]


class _SharedDisplayRenderer:
    def __init__(self, frame_count: int) -> None:
        self._memory: shared_memory.SharedMemory | None = None
        self._memory_name: str | None = None
        self._sequence = 0
        self._surface: pygame.Surface | None = None
        self._renderer: BattleDrawController | None = None
        self._star_field: DisplayStarField | None = None
        self._resources = HeadlessAssetManager()
        self._frame_count = max(1, int(frame_count))

    def attach(self, name: str | None) -> None:
        normalized = str(name) if name else None
        if normalized == self._memory_name:
            return
        self.close()
        if normalized:
            self._memory = shared_memory.SharedMemory(name=normalized)
            self._memory_name = normalized

    def render(self, battle_view: Mapping[str, Any]) -> Mapping[str, int] | None:
        memory = self._memory
        if memory is None:
            return None
        width = const.SCREEN_WIDTH
        height = const.SCREEN_HEIGHT
        frame_count = self._frame_count
        frame_bytes = width * height * 3
        required = _DISPLAY_SEQUENCE_BYTES + frame_count * frame_bytes
        if memory.size < required:
            raise ValueError("independent display buffer is too small")
        if self._surface is None:
            if not pygame.font.get_init():
                pygame.font.init()
            self._surface = pygame.Surface((width, height))
            self._renderer = BattleDrawController()
            self._star_field = DisplayStarField(resources=self._resources)
        renderer = self._renderer
        star_field = self._star_field
        surface = self._surface
        if renderer is None or star_field is None:
            return None

        self._sequence += 2
        committed_sequence = self._sequence
        struct.pack_into("<Q", memory.buf, 0, committed_sequence - 1)
        layout = create_play_battle_layout(
            pygame.Rect(const.SCREEN_LEFT, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)
        )
        for index in range(frame_count):
            surface.fill((0, 0, 0))
            renderer.draw(
                surface,
                battle_view.get("game_objects", ()),
                layout,
                battle_view.get("border_color", (255, 255, 255)),
                star_field,
                camera_targets=battle_view.get("camera_targets"),
                entry_state=battle_view.get("entry_state"),
                frame_id=int(battle_view.get("frame_id", 0)),
                original_ships=battle_view.get("original_ships"),
                options=BattleDrawOptions(
                    draw_instructions=False,
                    interp_t=index / frame_count,
                ),
            )
            pixels = pygame.image.tobytes(surface, "RGB")
            offset = _DISPLAY_SEQUENCE_BYTES + index * frame_bytes
            memory.buf[offset : offset + frame_bytes] = pixels
        struct.pack_into("<Q", memory.buf, 0, committed_sequence)
        return {
            "frame_id": int(battle_view.get("frame_id", 0)),
            "shared_frame_count": frame_count,
            "shared_sequence": committed_sequence,
        }

    def close(self) -> None:
        if self._memory is not None:
            self._memory.close()
        self._memory = None
        self._memory_name = None


class _ProcessPublisher:
    def __init__(self, message_queue, control_queue, *, display_frame_count: int) -> None:
        self._messages = message_queue
        self._controls = control_queue
        self._renderer = _SharedDisplayRenderer(display_frame_count)
        self._recording_audio: RecordingAudioService | None = None

    def bind_audio(self, audio: RecordingAudioService) -> None:
        self._recording_audio = audio

    def apply_controls(self, session: TrainingSession) -> None:
        while True:
            try:
                name, value = self._controls.get_nowait()
            except queue.Empty:
                return
            if name == _CONTROL_DISPLAY_BUFFER:
                self._renderer.attach(value)
            elif name == _CONTROL_STARTING_EPSILON:
                session.set_starting_epsilon(float(value))

    def publish_status(
        self,
        session: TrainingSession,
        *,
        include_continuity: bool = False,
    ) -> None:
        status = session.status
        display_metadata = None
        if status.battle_view:
            display_metadata = self._renderer.render(status.battle_view)
        status = replace(status, battle_view=display_metadata)
        self._messages.put(
            (
                _MESSAGE_STATUS,
                {
                    "status": status,
                    "history": session.history if include_continuity else None,
                    "log_lines": session.log_lines if include_continuity else None,
                    "slot": session.slot,
                },
            )
        )
        self._publish_audio()

    def publish_saved(self, session: TrainingSession) -> None:
        self._messages.put((_MESSAGE_SAVED, session.slot))

    def _publish_audio(self) -> None:
        audio = self._recording_audio
        operations = getattr(audio, "operations", None)
        if not isinstance(operations, list) or not operations:
            return
        pending = tuple(operations)
        operations.clear()
        events = []
        for operation in pending:
            if not operation:
                continue
            name, *args = operation
            if name == "play_effect" and len(args) >= 2:
                events.append((name, str(args[0]), float(args[1])))
            elif name == "play_victory_ditty" and args:
                events.append((name, getattr(args[0], "name", str(args[0]))))
            elif name in {
                "start_battle_music",
                "stop_music",
                "pause",
                "unpause",
            }:
                events.append((name,))
        if events:
            self._messages.put((_MESSAGE_AUDIO, tuple(events)))

    def close(self) -> None:
        self._renderer.close()


class _ProcessTrainingEngine(TrainingSession):
    def __init__(
        self,
        *,
        publisher: _ProcessPublisher,
        stop_event,
        display_event,
        save_coordinator: ProcessModelSaveCoordinator,
        batch_pacing_group: CpuBatchPacingGroup | None = None,
        batch_pacing_index: int | None = None,
        batch_accounting_started_at: float | None = None,
        **kwargs,
    ) -> None:
        self._publisher = publisher
        self._process_save_coordinator = save_coordinator
        self._discovery_generations: tuple[int, ...] | None = None
        self._discovery_deferred = False
        self._last_frame_status_published_at = 0.0
        self._simulation_resources = HeadlessAssetManager()
        self._batch_pacing_group = batch_pacing_group
        self._batch_pacing_index = batch_pacing_index
        self._batch_accounting_started_at = (
            float(batch_accounting_started_at)
            if batch_accounting_started_at is not None
            else time.perf_counter()
        )
        self._pending_paced_batch_number: int | None = None
        super().__init__(save_coordinator=save_coordinator, **kwargs)
        self._stop_requested = stop_event
        self._display_on = display_event

    def _on_progress(self, payload: Mapping[str, Any]) -> None:
        self._publisher.apply_controls(self)
        super()._on_progress(payload)
        event = payload.get("event")
        frame = int(payload.get("frame", 0))
        display_on = self._display_on.is_set()
        now = time.perf_counter()
        frame_status_due = (
            event == "frame"
            and (
                display_on
                or (frame > 0 and frame % LIVE_STATUS_FRAME_INTERVAL == 0)
            )
            and (
                display_on
                or now - self._last_frame_status_published_at >= 0.05
            )
        )
        if event != "frame" or frame_status_due:
            self._publisher.publish_status(self)
            if event == "frame":
                self._last_frame_status_published_at = now

    def _record_completed_batch(self, *args, **kwargs) -> int:
        if self._batch_pacing_group is None or self._batch_pacing_index is None:
            batch_number = super()._record_completed_batch(*args, **kwargs)
            self._publisher.publish_status(self, include_continuity=True)
            return batch_number

        kwargs["batch_seconds"] = max(
            0.0,
            time.perf_counter() - self._batch_accounting_started_at,
        )
        kwargs["emit_outputs"] = False
        batch_number = super()._record_completed_batch(*args, **kwargs)
        if batch_number % self.batch_grouping == 0:
            self._pending_paced_batch_number = batch_number
        else:
            self._complete_paced_batch(batch_number)
        return batch_number

    def _save_state(self, *args, **kwargs) -> None:
        super()._save_state(*args, **kwargs)
        self._publisher.publish_saved(self)
        pending_batch_number = self._pending_paced_batch_number
        if pending_batch_number is not None:
            self._complete_paced_batch(pending_batch_number)

    def _complete_paced_batch(self, batch_number: int) -> None:
        completed_this_run = max(
            0,
            int(batch_number) - int(self._completed_batches_at_run_start),
        )
        with self._lock:
            self._status.display_message = "Waiting for synchronized CPU runs"
            self._status.battle_view = None
            self._status.simulation_speed_multiplier = 0.0
        self._publisher.publish_status(self, include_continuity=True)
        self._batch_pacing_group.wait_after_batch(
            self._batch_pacing_index,
            completed_this_run,
            stop_requested=self._stop_requested.is_set,
        )
        finished_at = time.perf_counter()
        with self._lock:
            self._status.display_message = ""
        self._finalize_completed_batch_metrics(
            batch_number=batch_number,
            batch_seconds=max(
                0.0,
                finished_at - self._batch_accounting_started_at,
            ),
        )
        self._batch_accounting_started_at = finished_at
        self._pending_paced_batch_number = None
        self._publisher.publish_status(self, include_continuity=True)

    def _run_timing_started_at(self) -> float:
        if self._batch_pacing_group is not None:
            return self._batch_accounting_started_at
        return super()._run_timing_started_at()

    def _run_batch(self, **kwargs) -> TrainingBatchResult:
        return self.batch_runner(resources=self._simulation_resources, **kwargs)

    def _existing_ai_opponents_for_batch(self):
        if self.config.opponent_mode != OPPONENT_MODE_EXISTING_AI:
            return None
        saving_keys, generations = self._process_save_coordinator.snapshot()
        with self._lock:
            completed_batches = self._status.completed_batches
            cached = self._cached_existing_ai_opponents
            cached_at = self._cached_existing_ai_opponents_at
            refresh = (
                cached is None
                or self._discovery_deferred
                or self._discovery_generations != generations
                or (
                    cached_at is not None
                    and completed_batches > cached_at
                    and completed_batches % self.batch_grouping == 0
                )
            )
            if not refresh:
                return cached
            self._status.display_message = "Loading AI opponents"
            self._status.battle_view = None

        visible_repository = _SavingAwareRepository(self.repository, saving_keys)
        discovered = discover_existing_ai_opponents(
            visible_repository,
            device_choice=self.config.training_device,
        ).opponents
        by_key = {
            (opponent.ship, int(opponent.slot)): opponent
            for opponent in discovered
            if opponent.slot is not None
        }
        for opponent in cached or ():
            key = (opponent.ship, int(opponent.slot))
            if key in saving_keys:
                by_key[key] = opponent
        ordered = tuple(
            by_key[key]
            for ship in SHIP_TYPE_CATALOG_ORDER
            for slot in range(1, MODEL_SLOT_COUNT + 1)
            if (key := (ship, slot)) in by_key
        )
        with self._lock:
            self._cached_existing_ai_opponents = ordered
            self._cached_existing_ai_opponents_at = completed_batches
            self._discovery_generations = generations
            self._discovery_deferred = bool(saving_keys)
            self._status.display_message = ""
        return ordered


def independent_training_process_main(
    *,
    bundled_dir: Path,
    user_dir: Path,
    slot: TrainingModelSlot,
    metadata: Mapping[str, Any],
    config,
    batch_grouping: int,
    stop_at_batch: int | None,
    stop_at_epsilon: float | None,
    initial_history: tuple[BatchMetrics, ...],
    initial_log_lines: tuple[str, ...],
    save_coordinator: ProcessModelSaveCoordinator,
    message_queue,
    control_queue,
    stop_event,
    display_event,
    display_frame_count: int,
    batch_pacing_group: CpuBatchPacingGroup | None = None,
    batch_pacing_index: int | None = None,
) -> None:
    process_started_at = time.perf_counter()
    _set_worker_process_below_normal_priority()
    torch = torch_backend.get_torch()
    if torch is not None:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    publisher = _ProcessPublisher(
        message_queue,
        control_queue,
        display_frame_count=display_frame_count,
    )
    recording_audio = RecordingAudioService()
    publisher.bind_audio(recording_audio)
    repository = TrainingModelRepository(bundled_dir, user_dir)
    session: _ProcessTrainingEngine | None = None
    try:
        session = _ProcessTrainingEngine(
            publisher=publisher,
            stop_event=stop_event,
            display_event=display_event,
            save_coordinator=save_coordinator,
            batch_pacing_group=batch_pacing_group,
            batch_pacing_index=batch_pacing_index,
            batch_accounting_started_at=process_started_at,
            repository=repository,
            slot=slot,
            metadata=metadata,
            config=config,
            batch_grouping=batch_grouping,
            stop_at_batch=stop_at_batch,
            stop_at_epsilon=stop_at_epsilon,
            audio_service=recording_audio,
            initial_history=initial_history,
            initial_log_lines=initial_log_lines,
            opponent_model_cache=None,
        )
        publisher.apply_controls(session)
        # ``TrainingSession.run_synchronously`` defaults to one batch for
        # deterministic callers and tests.  An independent process is a
        # long-lived training owner, so only its stop event should end it.
        session.run_synchronously(max_batches=None)
    except Exception as exc:
        if session is None:
            status = TrainingSessionStatus(
                ship=slot.ship,
                running=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
            message_queue.put(
                (
                    _MESSAGE_STATUS,
                    {
                        "status": status,
                        "history": tuple(initial_history),
                        "log_lines": tuple(initial_log_lines),
                        "slot": slot,
                    },
                )
            )
        else:
            with session._lock:
                session._status.error = (
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
    finally:
        if batch_pacing_group is not None and batch_pacing_index is not None:
            batch_pacing_group.deactivate(batch_pacing_index)
        if session is not None:
            publisher.publish_status(session, include_continuity=True)
        publisher.close()


class ProcessTrainingSession:
    """UI-facing facade for one complete independent CPU process."""

    def __init__(
        self,
        *,
        repository: TrainingModelRepository,
        slot: TrainingModelSlot,
        metadata: Mapping[str, Any],
        config,
        batch_grouping: int,
        audio_service: Any | None = None,
        initial_history: tuple[BatchMetrics, ...] = (),
        initial_log_lines: tuple[str, ...] = (),
        opponent_model_cache: Any | None = None,
        save_coordinator: ProcessModelSaveCoordinator | None = None,
        context: multiprocessing.context.BaseContext | None = None,
        display_frame_count: int | None = None,
        batch_pacing_group: CpuBatchPacingGroup | None = None,
        batch_pacing_index: int | None = None,
        stop_at_batch: int | None = None,
        stop_at_epsilon: float | None = None,
    ) -> None:
        self.repository = repository
        self.slot = slot
        self.metadata = dict(metadata)
        self.config = config
        self.batch_grouping = max(1, int(batch_grouping))
        self.stop_at_batch = (
            max(1, int(stop_at_batch)) if stop_at_batch is not None else None
        )
        epsilon_target = (
            float(stop_at_epsilon) if stop_at_epsilon is not None else 0.0
        )
        self.stop_at_epsilon = (
            epsilon_target if 0.0 < epsilon_target < 1.0 else None
        )
        self.audio_service = audio_service or NullAudioService()
        self.opponent_model_cache = opponent_model_cache
        self.save_coordinator = save_coordinator or ProcessModelSaveCoordinator()
        self._context = context or multiprocessing.get_context("spawn")
        self._batch_pacing_group = batch_pacing_group
        self._batch_pacing_index = batch_pacing_index
        if (batch_pacing_group is None) != (batch_pacing_index is None):
            raise ValueError(
                "CPU pacing group and participant index must be provided together"
            )
        self._display_frame_count = max(
            1,
            int(
                const.VIDEO_FPS_MULTIPLIER
                if display_frame_count is None
                else display_frame_count
            ),
        )
        self._messages = self._context.Queue()
        self._controls = self._context.Queue()
        self._stop_event = self._context.Event()
        self._display_event = self._context.Event()
        if config.display_on:
            self._display_event.set()
        self._status = TrainingSessionStatus(
            ship=slot.ship,
            completed_batches=int(metadata.get("progress", {}).get("completed_batches", 0)),
            current_frame_limit=int(config.match_time_limit),
            learning_rate=float(config.learning_rate),
            current_epsilon=float(config.epsilon),
            epsilon_decay=float(config.epsilon_decay),
            gamma=float(config.gamma),
        )
        self._history = tuple(initial_history)
        self._log_lines = tuple(initial_log_lines)
        self._process = None
        self._lock = threading.RLock()
        self._display_memory: shared_memory.SharedMemory | None = None

    @property
    def status(self) -> TrainingSessionStatus:
        self._drain_messages()
        with self._lock:
            status = replace(self._status)
        process = self._process
        if (
            process is not None
            and not process.is_alive()
        ):
            self._drain_messages()
            with self._lock:
                if self._status.running:
                    error = self._status.error
                    if not error and process.exitcode not in (None, 0):
                        error = f"CPU training worker exited with code {process.exitcode}"
                    self._status = replace(
                        self._status,
                        running=False,
                        stopping=False,
                        error=error,
                    )
                status = replace(self._status)
            self._deactivate_batch_pacing()
            try:
                process.join(0)
                process.close()
            finally:
                self._process = None
                self._release_display_memory()
        return status

    @property
    def history(self) -> tuple[BatchMetrics, ...]:
        self._drain_messages()
        with self._lock:
            return tuple(self._history)

    @property
    def log_lines(self) -> tuple[str, ...]:
        self._drain_messages()
        with self._lock:
            return tuple(self._log_lines)

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        if self._display_event.is_set():
            self._ensure_display_memory()
        with self._lock:
            self._status = replace(
                self._status,
                running=True,
                stopping=False,
                error="",
            )
        process = self._context.Process(
            target=independent_training_process_main,
            kwargs={
                "bundled_dir": self.repository.bundled_dir,
                "user_dir": self.repository.user_dir,
                "slot": self.slot,
                "metadata": self.metadata,
                "config": self.config,
                "batch_grouping": self.batch_grouping,
                "stop_at_batch": self.stop_at_batch,
                "stop_at_epsilon": self.stop_at_epsilon,
                "initial_history": self._history,
                "initial_log_lines": self._log_lines,
                "save_coordinator": self.save_coordinator,
                "message_queue": self._messages,
                "control_queue": self._controls,
                "stop_event": self._stop_event,
                "display_event": self._display_event,
                "display_frame_count": self._display_frame_count,
                "batch_pacing_group": self._batch_pacing_group,
                "batch_pacing_index": self._batch_pacing_index,
            },
            daemon=True,
            name="StarAIIndependentCPUTraining",
        )
        self._process = process
        try:
            _start_process_with_single_threaded_openblas(process)
        except Exception:
            self._deactivate_batch_pacing()
            self._process = None
            with self._lock:
                self._status = replace(self._status, running=False)
            self._release_display_memory()
            raise

    def request_stop(self) -> None:
        self._deactivate_batch_pacing()
        self._stop_event.set()
        with self._lock:
            self._status = replace(self._status, stopping=True)

    def set_display_on(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled:
            self._ensure_display_memory()
            self._display_event.set()
            self.audio_service.start_battle_music()
        else:
            self._display_event.clear()
            self.audio_service.stop_music()
            with self._lock:
                self._status = replace(self._status, battle_view=None)

    def set_starting_epsilon(self, value: float) -> None:
        epsilon = max(0.0, min(1.0, float(value)))
        self.config = replace(
            self.config,
            starting_epsilon=epsilon,
            epsilon=max(float(self.config.epsilon_floor), epsilon),
        )
        self._controls.put((_CONTROL_STARTING_EPSILON, epsilon))

    def join(self, timeout: float | None = None) -> None:
        process = self._process
        if process is not None:
            process.join(timeout)
            self._drain_messages()
            if not process.is_alive():
                process.close()
                self._process = None
                self._release_display_memory()

    def _ensure_display_memory(self) -> None:
        if self._display_memory is None:
            frame_bytes = const.SCREEN_WIDTH * const.SCREEN_HEIGHT * 3
            size = (
                _DISPLAY_SEQUENCE_BYTES
                + frame_bytes * self._display_frame_count
            )
            self._display_memory = shared_memory.SharedMemory(create=True, size=size)
            struct.pack_into("<Q", self._display_memory.buf, 0, 0)
            self._controls.put(
                (_CONTROL_DISPLAY_BUFFER, self._display_memory.name)
            )

    def _drain_messages(self) -> None:
        while True:
            try:
                name, payload = self._messages.get_nowait()
            except queue.Empty:
                return
            if name == _MESSAGE_STATUS:
                self._accept_status(payload)
            elif name == _MESSAGE_SAVED:
                self._accept_saved(payload)
            elif name == _MESSAGE_AUDIO:
                self._relay_audio(payload)

    def _accept_status(self, payload: Mapping[str, Any]) -> None:
        status = payload["status"]
        battle_view = status.battle_view
        if battle_view and "shared_frame_count" in battle_view:
            rendered = self._read_display_frames(battle_view)
            if rendered is not None:
                battle_view = {
                    "frame_id": int(battle_view.get("frame_id", 0)),
                    "rendered_frames": rendered,
                }
            else:
                with self._lock:
                    battle_view = self._status.battle_view
            status = replace(status, battle_view=battle_view)
        with self._lock:
            self._status = status
            self.slot = payload.get("slot") or self.slot
            if payload.get("history") is not None:
                self._history = tuple(payload["history"])
            if payload.get("log_lines") is not None:
                self._log_lines = tuple(payload["log_lines"])
        if not status.running:
            self._deactivate_batch_pacing()

    def _deactivate_batch_pacing(self) -> None:
        if (
            self._batch_pacing_group is not None
            and self._batch_pacing_index is not None
        ):
            self._batch_pacing_group.deactivate(self._batch_pacing_index)

    def _read_display_frames(self, metadata: Mapping[str, Any]):
        memory = self._display_memory
        if memory is None:
            return None
        expected = int(metadata["shared_sequence"])
        sequence_before = struct.unpack_from("<Q", memory.buf, 0)[0]
        if sequence_before != expected or sequence_before % 2:
            return None
        width = const.SCREEN_WIDTH
        height = const.SCREEN_HEIGHT
        frame_bytes = width * height * 3
        frames = []
        for index in range(int(metadata["shared_frame_count"])):
            offset = _DISPLAY_SEQUENCE_BYTES + index * frame_bytes
            pixels = bytes(memory.buf[offset : offset + frame_bytes])
            frames.append(pygame.image.frombytes(pixels, (width, height), "RGB"))
        sequence_after = struct.unpack_from("<Q", memory.buf, 0)[0]
        if sequence_after != expected:
            return None
        return tuple(frames)

    def _accept_saved(self, slot: TrainingModelSlot) -> None:
        self.slot = slot
        if self.opponent_model_cache is not None:
            self.opponent_model_cache.notify_model_saved(
                self.repository,
                slot.ship,
                slot.slot,
                device_choice=self.config.training_device,
            )

    def _relay_audio(self, events) -> None:
        if not self._display_event.is_set():
            return
        for event in events:
            if not event:
                continue
            name, *args = event
            if name == "play_effect" and len(args) >= 2:
                self.audio_service.play_effect(Path(args[0]), float(args[1]))
            elif name == "play_victory_ditty" and args:
                self.audio_service.play_victory_ditty(
                    SimpleNamespace(name=str(args[0]))
                )
            elif name == "start_battle_music":
                self.audio_service.start_battle_music()
            elif name == "stop_music":
                self.audio_service.stop_music()
            elif name == "pause":
                self.audio_service.pause()
            elif name == "unpause":
                self.audio_service.unpause()

    def _release_display_memory(self) -> None:
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
