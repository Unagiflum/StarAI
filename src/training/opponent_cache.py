"""Shared cache for read-only existing-AI opponent models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import threading
from collections.abc import Mapping
from typing import Any, Iterator

from src.persistence import EXPECTED_READ_ERRORS, read_json
from src.training import torch_backend
from src.training.contracts import SHIP_TYPE_CATALOG_ORDER
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    model_slot_has_checkpoint,
    normalize_architecture_metadata,
)
from src.training.replay import load_training_checkpoint
from src.training.value_network import ValueNetworkConfig, build_value_network


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
    loaded = load_opponent_model(slot)
    if isinstance(loaded, str):
        return loaded
    return OpponentCacheEntry(
        key=key,
        model=loaded,
        description=slot.description,
        completed_batches=_completed_batches_for_slot(slot),
        checkpoint_size=_checkpoint_size(slot.pth_path),
        checkpoint_mtime_ns=_checkpoint_mtime_ns(slot.pth_path),
    )


def _coerce_key(key: OpponentModelKey | tuple[str, int]) -> OpponentModelKey:
    if isinstance(key, OpponentModelKey):
        return key
    ship, slot = key
    return OpponentModelKey(str(ship), int(slot))


def load_opponent_model(slot: TrainingModelSlot):
    """Build, load, and freeze one opponent model for inference."""

    if slot.pth_path is None or not slot.pth_path.exists():
        return "missing weights"
    if slot.pth_path.stat().st_size <= 0:
        return "empty weights"
    metadata = _metadata_for_slot(slot)
    architecture = normalize_architecture_metadata(metadata.get("architecture", {}))
    try:
        config = ValueNetworkConfig(
            hidden_layer_width=int(architecture["hidden_layer_width"]),
            hidden_layer_count=int(architecture["hidden_layer_count"]),
        )
        device = torch_backend.preferred_device()
        model = build_value_network(config, device=device)
        load_training_checkpoint(slot.pth_path, model, map_location=device)
        model.eval()
        return model
    except Exception as exc:
        return str(exc)


def _completed_batches_for_slot(slot: TrainingModelSlot) -> int | None:
    metadata = _metadata_for_slot(slot)
    progress = metadata.get("progress", {})
    if not isinstance(progress, Mapping):
        return None
    completed = progress.get("completed_batches")
    return int(completed) if completed is not None else None


def _checkpoint_size(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _checkpoint_mtime_ns(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _metadata_for_slot(slot: TrainingModelSlot) -> Mapping[str, Any]:
    if isinstance(slot.metadata, Mapping):
        return slot.metadata
    if slot.metadata_path is None or not slot.metadata_path.exists():
        return {}
    try:
        metadata = read_json(slot.metadata_path)
    except EXPECTED_READ_ERRORS:
        return {}
    return metadata if isinstance(metadata, Mapping) else {}
