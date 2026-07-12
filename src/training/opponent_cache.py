"""Shared cache for read-only existing-AI opponent models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
from collections.abc import Mapping
from typing import Any, Iterator

from src.training.contracts import SHIP_TYPE_CATALOG_ORDER
from src.training.model_loader import load_inference_model
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    model_slot_has_checkpoint,
)


@dataclass(frozen=True)
class OpponentModelKey:
    ship: str
    slot: int


@dataclass(frozen=True)
class OpponentCacheEntry:
    key: OpponentModelKey
    model: Any
    description: str
    completed_batches: int | None
    checkpoint_size: int | None
    checkpoint_mtime_ns: int | None


@dataclass(frozen=True)
class OpponentCacheDiagnostics:
    loaded_keys: tuple[OpponentModelKey, ...]
    last_errors: Mapping[OpponentModelKey, str]
    blocked_keys: tuple[OpponentModelKey, ...]
    entries: Mapping[OpponentModelKey, OpponentCacheEntry] = field(default_factory=dict)


class ModelSaveCoordinator:
    """Track in-process model saves by opponent cache key."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._saving_counts: dict[OpponentModelKey, int] = {}

    @contextmanager
    def saving(
        self,
        key: OpponentModelKey | tuple[str, int],
    ) -> Iterator[None]:
        normalized = _coerce_key(key)
        with self._lock:
            self._saving_counts[normalized] = self._saving_counts.get(normalized, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                count = self._saving_counts.get(normalized, 0) - 1
                if count > 0:
                    self._saving_counts[normalized] = count
                else:
                    self._saving_counts.pop(normalized, None)

    def is_saving(self, key: OpponentModelKey | tuple[str, int]) -> bool:
        normalized = _coerce_key(key)
        with self._lock:
            return self._saving_counts.get(normalized, 0) > 0


class OpponentModelCache:
    """Process-local cache of loaded existing-AI opponent models."""

    def __init__(self, save_coordinator: ModelSaveCoordinator | None = None) -> None:
        self._lock = threading.RLock()
        self._entries: dict[OpponentModelKey, OpponentCacheEntry] = {}
        self._last_errors: dict[OpponentModelKey, str] = {}
        self._blocked_keys: set[OpponentModelKey] = set()
        self._save_coordinator = save_coordinator

    def load_initial(self, repository: TrainingModelRepository) -> None:
        """Load available model slots that are not already cached."""

        for ship_name in SHIP_TYPE_CATALOG_ORDER:
            for slot_number in range(1, MODEL_SLOT_COUNT + 1):
                key = OpponentModelKey(ship_name, slot_number)
                with self._lock:
                    if key in self._entries:
                        continue
                if (
                    self._save_coordinator is not None
                    and self._save_coordinator.is_saving(key)
                ):
                    with self._lock:
                        self._blocked_keys.add(key)
                    continue
                slot = repository.slot_for(ship_name, slot_number)
                if slot.source == SLOT_EMPTY or not model_slot_has_checkpoint(slot):
                    continue
                with self._lock:
                    if key in self._entries:
                        continue
                    self._load_slot(key, slot)

    def notify_model_saved(
        self,
        repository: TrainingModelRepository,
        ship: str,
        slot: int,
    ) -> None:
        """Refresh one cached model after a successful save."""

        key = OpponentModelKey(str(ship), int(slot))
        if (
            self._save_coordinator is not None
            and self._save_coordinator.is_saving(key)
        ):
            with self._lock:
                self._blocked_keys.add(key)
            return

        with self._lock:
            self._blocked_keys.discard(key)

        model_slot = repository.slot_for(key.ship, key.slot)
        if model_slot.source != SLOT_USER or not model_slot_has_checkpoint(model_slot):
            return

        loaded = _entry_from_slot(key, model_slot)
        with self._lock:
            if isinstance(loaded, str):
                self._last_errors[key] = loaded
                return
            self._entries[key] = loaded
            self._last_errors.pop(key, None)
            self._blocked_keys.discard(key)

    def snapshot(self) -> tuple[Any, ...]:
        """Return an immutable batch snapshot of cached opponent specs."""

        from src.training.orchestration import OPPONENT_MODE_EXISTING_AI, OpponentSpec

        with self._lock:
            entries = tuple(
                self._entries[key]
                for key in sorted(self._entries, key=lambda item: (item.ship, item.slot))
            )
        return tuple(
            OpponentSpec(
                ship=entry.key.ship,
                mode=OPPONENT_MODE_EXISTING_AI,
                slot=entry.key.slot,
                model=entry.model,
                description=entry.description,
            )
            for entry in entries
        )

    def diagnostics(self) -> OpponentCacheDiagnostics:
        with self._lock:
            return OpponentCacheDiagnostics(
                loaded_keys=tuple(
                    sorted(self._entries, key=lambda item: (item.ship, item.slot))
                ),
                last_errors=dict(self._last_errors),
                blocked_keys=tuple(
                    sorted(self._blocked_keys, key=lambda item: (item.ship, item.slot))
                ),
                entries=dict(self._entries),
            )

    def _load_slot(self, key: OpponentModelKey, slot: TrainingModelSlot) -> None:
        loaded = _entry_from_slot(key, slot)
        with self._lock:
            if isinstance(loaded, str):
                self._last_errors[key] = loaded
                return
            self._entries[key] = loaded
            self._last_errors.pop(key, None)


def _entry_from_slot(
    key: OpponentModelKey,
    slot: TrainingModelSlot,
) -> OpponentCacheEntry | str:
    try:
        loaded = load_inference_model(slot)
    except Exception as exc:
        return str(exc)
    return OpponentCacheEntry(
        key=key,
        model=loaded.model,
        description=slot.description,
        completed_batches=loaded.completed_batches,
        checkpoint_size=loaded.checkpoint_size,
        checkpoint_mtime_ns=loaded.checkpoint_mtime_ns,
    )


def _coerce_key(key: OpponentModelKey | tuple[str, int]) -> OpponentModelKey:
    if isinstance(key, OpponentModelKey):
        return key
    ship, slot = key
    return OpponentModelKey(str(ship), int(slot))


def load_opponent_model(slot: TrainingModelSlot):
    """Build, load, and freeze one opponent model for inference."""

    try:
        return load_inference_model(slot).model
    except Exception as exc:
        return str(exc)
