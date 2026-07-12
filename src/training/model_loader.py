"""Shared read-only inference model loading for trained ship models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from src.persistence import EXPECTED_READ_ERRORS, read_json
from src.training import torch_backend
from src.training.contracts import (
    ACTION_SCHEMA_METADATA,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    SHIP_TYPE_CATALOG_ORDER,
)
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    TrainingModelRepository,
    TrainingModelSlot,
    model_architecture_metadata,
    model_slot_has_checkpoint,
    normalize_architecture_metadata,
)
from src.training.replay import load_training_checkpoint
from src.training.value_network import ValueNetworkConfig, build_value_network


@dataclass(frozen=True)
class InferenceModelKey:
    ship: str
    slot: int


@dataclass(frozen=True)
class LoadedInferenceModel:
    key: InferenceModelKey
    model: Any
    slot: TrainingModelSlot
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    completed_batches: int | None = None
    checkpoint_size: int | None = None
    checkpoint_mtime_ns: int | None = None


@dataclass(frozen=True)
class InferenceModelCacheDiagnostics:
    loaded_keys: tuple[InferenceModelKey, ...]
    last_errors: Mapping[InferenceModelKey, str]
    entries: Mapping[InferenceModelKey, LoadedInferenceModel] = field(default_factory=dict)


class InferenceModelLoadError(RuntimeError):
    """Raised when a trained model cannot be loaded for read-only inference."""


class InferenceModelCache:
    """Process-local cache of loaded read-only inference models."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[InferenceModelKey, LoadedInferenceModel] = {}
        self._last_errors: dict[InferenceModelKey, str] = {}

    def load_initial(
        self,
        repository: TrainingModelRepository,
        *,
        ships: Iterable[str] = SHIP_TYPE_CATALOG_ORDER,
    ) -> None:
        """Load all available model slots that are not already cached."""

        for ship_name in ships:
            for slot_number in range(1, MODEL_SLOT_COUNT + 1):
                key = InferenceModelKey(str(ship_name), int(slot_number))
                with self._lock:
                    if key in self._entries:
                        continue
                slot = repository.slot_for(key.ship, key.slot)
                if slot.source == SLOT_EMPTY or not model_slot_has_checkpoint(slot):
                    continue
                self.load_slot(slot)

    def load_slot(self, slot: TrainingModelSlot) -> LoadedInferenceModel | None:
        key = InferenceModelKey(str(slot.ship), int(slot.slot))
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached

        try:
            loaded = load_inference_model(slot)
        except Exception as exc:
            with self._lock:
                self._last_errors[key] = str(exc)
            return None

        with self._lock:
            self._entries[key] = loaded
            self._last_errors.pop(key, None)
        return loaded

    def replace_slot(self, slot: TrainingModelSlot) -> LoadedInferenceModel | None:
        key = InferenceModelKey(str(slot.ship), int(slot.slot))
        try:
            loaded = load_inference_model(slot)
        except Exception as exc:
            with self._lock:
                self._last_errors[key] = str(exc)
            return None

        with self._lock:
            self._entries[key] = loaded
            self._last_errors.pop(key, None)
        return loaded

    def entry_for(self, ship: str, slot: int) -> LoadedInferenceModel | None:
        key = InferenceModelKey(str(ship), int(slot))
        with self._lock:
            return self._entries.get(key)

    def entries_for_ship(self, ship: str) -> tuple[LoadedInferenceModel, ...]:
        ship_name = str(ship)
        with self._lock:
            return tuple(
                self._entries[key]
                for key in sorted(self._entries, key=lambda item: item.slot)
                if key.ship == ship_name
            )

    def error_for(self, ship: str, slot: int) -> str | None:
        key = InferenceModelKey(str(ship), int(slot))
        with self._lock:
            return self._last_errors.get(key)

    def diagnostics(self) -> InferenceModelCacheDiagnostics:
        with self._lock:
            return InferenceModelCacheDiagnostics(
                loaded_keys=tuple(
                    sorted(self._entries, key=lambda item: (item.ship, item.slot))
                ),
                last_errors=dict(self._last_errors),
                entries=dict(self._entries),
            )


def load_inference_model(slot: TrainingModelSlot) -> LoadedInferenceModel:
    """Build, load, and freeze one trained model for read-only inference."""

    if slot.pth_path is None or not slot.pth_path.exists():
        raise InferenceModelLoadError("missing weights")
    if slot.pth_path.stat().st_size <= 0:
        raise InferenceModelLoadError("empty weights")
    if torch_backend.get_torch() is None:
        raise InferenceModelLoadError("PyTorch unavailable")

    metadata = metadata_for_slot(slot)
    validate_schema_metadata(metadata)
    architecture = normalize_architecture_metadata(
        metadata.get("architecture", model_architecture_metadata(128, 2))
    )
    try:
        config = ValueNetworkConfig(
            hidden_layer_width=int(architecture["hidden_layer_width"]),
            hidden_layer_count=int(architecture["hidden_layer_count"]),
        )
        device = torch_backend.preferred_device()
        model = build_value_network(config, device=device)
        load_training_checkpoint(slot.pth_path, model, map_location=device)
        model.eval()
    except Exception as exc:
        raise InferenceModelLoadError(str(exc)) from exc

    return LoadedInferenceModel(
        key=InferenceModelKey(str(slot.ship), int(slot.slot)),
        model=model,
        slot=slot,
        description=slot.description,
        metadata=metadata,
        completed_batches=completed_batches_from_metadata(metadata),
        checkpoint_size=checkpoint_size(slot.pth_path),
        checkpoint_mtime_ns=checkpoint_mtime_ns(slot.pth_path),
    )


def metadata_for_slot(slot: TrainingModelSlot) -> Mapping[str, Any]:
    if isinstance(slot.metadata, Mapping):
        return slot.metadata
    if slot.metadata_path is None or not slot.metadata_path.exists():
        return {}
    try:
        metadata = read_json(slot.metadata_path)
    except EXPECTED_READ_ERRORS:
        return {}
    return metadata if isinstance(metadata, Mapping) else {}


def validate_schema_metadata(metadata: Mapping[str, Any]) -> None:
    if not metadata:
        return
    expected = {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_input_size": OBSERVATION_INPUT_SIZE,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if key in metadata and int(metadata[key]) != int(value):
            raise InferenceModelLoadError(f"incompatible {key}")
    ordering = metadata.get("action_ordering")
    if ordering is not None and list(ordering) != [
        dict(action) for action in ACTION_SCHEMA_METADATA
    ]:
        raise InferenceModelLoadError("incompatible action_ordering")


def completed_batches_from_metadata(metadata: Mapping[str, Any]) -> int | None:
    progress = metadata.get("progress", {})
    if not isinstance(progress, Mapping):
        return None
    completed = progress.get("completed_batches")
    return int(completed) if completed is not None else None


def checkpoint_size(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def checkpoint_mtime_ns(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None
