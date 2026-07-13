"""Shared cache for read-only existing-AI opponent models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
from collections.abc import Mapping
from typing import Any, Iterator

from src.training.contracts import SHIP_TYPE_CATALOG_ORDER
from src.training import torch_backend
from src.training.model_loader import load_inference_model
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    TrainingModelRepository,
    TrainingModelSlot,
    model_slot_has_checkpoint,
)


@dataclass(frozen=True)
class OpponentModelKey:
    ship: str
    slot: int
    device: str = torch_backend.DEVICE_AUTO


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

    def load_initial(
        self,
        repository: TrainingModelRepository,
        *,
        device_choice: str | None = torch_backend.DEVICE_AUTO,
    ) -> None:
        """Load available model slots that are not already cached."""

        for ship_name in SHIP_TYPE_CATALOG_ORDER:
            for slot_number in range(1, MODEL_SLOT_COUNT + 1):
                key = _cache_key(ship_name, slot_number, device_choice)
                with self._lock:
                    if key in self._entries:
                        continue
                if (
                    self._save_coordinator is not None
                    and self._save_coordinator.is_saving((key.ship, key.slot))
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
        *,
        device_choice: str | None = None,
    ) -> None:
        """Mark cached copies stale after a successful save.

        Reloading is deliberately deferred to the next batch boundary. Building
        replacement CPU/CUDA models synchronously on the saving training thread
        can briefly duplicate model memory while other instances are active.
        """

        save_key = OpponentModelKey(str(ship), int(slot))
        with self._lock:
            existing_target_keys = tuple(
                key
                for key in self._entries
                if key.ship == str(ship) and key.slot == int(slot)
            )
            if device_choice is None:
                target_keys = existing_target_keys
            else:
                requested_key = _cache_key(ship, slot, device_choice)
                target_keys = tuple(
                    dict.fromkeys((*existing_target_keys, requested_key))
                )
            if not target_keys:
                target_keys = (_cache_key(ship, slot, device_choice),)

        if (
            self._save_coordinator is not None
            and self._save_coordinator.is_saving(save_key)
        ):
            with self._lock:
                self._blocked_keys.update(target_keys)
            return

        with self._lock:
            self._blocked_keys.discard(save_key)
            for key in target_keys:
                self._blocked_keys.discard(key)
                self._last_errors.pop(key, None)
                self._entries.pop(key, None)

    def snapshot(
        self,
        *,
        device_choice: str | None = torch_backend.DEVICE_AUTO,
    ) -> tuple[Any, ...]:
        """Return an immutable batch snapshot of cached opponent specs."""

        from src.training.orchestration import OPPONENT_MODE_EXISTING_AI, OpponentSpec

        device = torch_backend.training_device_key(device_choice)
        with self._lock:
            entries = tuple(
                self._entries[key]
                for key in sorted(
                    self._entries,
                    key=lambda item: (item.ship, item.slot, item.device),
                )
                if key.device == device
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
        loaded = load_inference_model(slot, device_choice=key.device)
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


def _cache_key(
    ship: str,
    slot: int,
    device_choice: str | None,
) -> OpponentModelKey:
    return OpponentModelKey(
        str(ship),
        int(slot),
        torch_backend.training_device_key(device_choice),
    )


def _coerce_key(key: OpponentModelKey | tuple[str, int]) -> OpponentModelKey:
    if isinstance(key, OpponentModelKey):
        return OpponentModelKey(key.ship, key.slot)
    ship, slot = key
    return OpponentModelKey(str(ship), int(slot))


def load_opponent_model(
    slot: TrainingModelSlot,
    *,
    device_choice: str | None = torch_backend.DEVICE_AUTO,
):
    """Build, load, and freeze one opponent model for inference."""

    try:
        return load_inference_model(slot, device_choice=device_choice).model
    except Exception as exc:
        return str(exc)
