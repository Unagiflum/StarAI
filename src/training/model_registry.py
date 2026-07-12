"""Persistence for training model placeholders and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import src.const as const
from src.persistence import EXPECTED_READ_ERRORS, atomic_write_json, read_json
from src.training.contracts import (
    ACTION_OUTPUT_SIZE,
    ACTION_SCHEMA_METADATA,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_INPUT_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    SHIP_TYPE_CATALOG_ORDER,
)


MODEL_SLOT_COUNT = 4
MODEL_METADATA_VERSION = 2
SLOT_EMPTY = "empty"
SLOT_BUNDLED = "bundled"
SLOT_USER = "user"


@dataclass(frozen=True)
class TrainingModelSlot:
    ship: str
    slot: int
    source: str
    description: str = ""
    pth_path: Path | None = None
    metadata_path: Path | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def exists(self) -> bool:
        return self.source in (SLOT_BUNDLED, SLOT_USER)

    @property
    def is_bundled(self) -> bool:
        return self.source == SLOT_BUNDLED

    @property
    def is_user(self) -> bool:
        return self.source == SLOT_USER


def model_basename(ship: str, slot: int) -> str:
    """Return the user-visible model basename, such as ``Arilou-01``."""
    return f"{ship.replace(' ', '')}-{slot:02d}"


def model_paths(directory: Path, ship: str, slot: int) -> tuple[Path, Path]:
    base = Path(directory) / model_basename(ship, slot)
    return base.with_suffix(".pth"), base.with_suffix(".json")


def replay_checkpoint_path(model_path: Path) -> Path:
    return Path(model_path).with_suffix(".replay.pth")


def model_architecture_metadata(
    hidden_layer_width: int,
    hidden_layer_count: int,
    *,
    optimizer: str = "adam",
    loss: str = "huber",
) -> dict[str, Any]:
    return {
        "input_size": OBSERVATION_INPUT_SIZE,
        "hidden_layer_width": int(hidden_layer_width),
        "hidden_layer_count": int(hidden_layer_count),
        "output_count": ACTION_OUTPUT_SIZE,
        "activation": "relu",
        "output_activation": None,
        "optimizer": optimizer,
        "loss": loss,
    }


def normalize_architecture_metadata(architecture: Mapping[str, Any]) -> dict[str, Any]:
    width = architecture.get("hidden_layer_width", architecture.get("hidden_layer_size"))
    count = architecture.get("hidden_layer_count")
    if width is None or count is None:
        return dict(architecture)
    return model_architecture_metadata(
        int(width),
        int(count),
        optimizer=str(architecture.get("optimizer", "adam")),
        loss=str(architecture.get("loss", "huber")),
    )


def current_game_settings_metadata() -> dict[str, int]:
    return {
        "ship_directions": const.SHIP_DIRECTIONS,
        "asteroid_count": const.ASTEROID_COUNT,
        "repeat_key_delay": const.INPUT_REPEAT_DELAY_FRAMES,
        "fps": const.FPS,
    }


def _description_from_metadata(path: Path) -> tuple[str, Mapping[str, Any] | None]:
    if not path.exists():
        return "", None
    try:
        metadata = read_json(path)
    except EXPECTED_READ_ERRORS:
        return "", None
    if not isinstance(metadata, Mapping):
        return "", None
    description = metadata.get("description", "")
    return (description if isinstance(description, str) else ""), metadata


def metadata_from_state(
    *,
    ship: str,
    slot: int,
    description: str,
    architecture: Mapping[str, Any],
    training: Mapping[str, Any],
    game_settings: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    game_settings_metadata = (
        dict(game_settings) if game_settings is not None else current_game_settings_metadata()
    )
    return {
        "schema_version": MODEL_METADATA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_input_size": OBSERVATION_INPUT_SIZE,
        "ship_type_catalog_order": list(SHIP_TYPE_CATALOG_ORDER),
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_ordering": [dict(action) for action in ACTION_SCHEMA_METADATA],
        "ship": ship,
        "slot": slot,
        "description": description,
        "architecture": normalize_architecture_metadata(architecture),
        "training": dict(training),
        "game_settings": game_settings_metadata,
        "progress": {"completed_batches": 0, **dict(progress or {})},
    }


class TrainingModelRepository:
    """Two-tier lookup for read-only bundled models and writable user models."""

    def __init__(self, bundled_dir: Path, user_dir: Path):
        self.bundled_dir = Path(bundled_dir)
        self.user_dir = Path(user_dir)

    def slot_for(self, ship: str, slot: int) -> TrainingModelSlot:
        bundled_pth, bundled_json = model_paths(self.bundled_dir, ship, slot)
        if bundled_pth.exists():
            return TrainingModelSlot(
                ship=ship,
                slot=slot,
                source=SLOT_BUNDLED,
                description="Default",
                pth_path=bundled_pth,
                metadata_path=bundled_json if bundled_json.exists() else None,
            )

        user_pth, user_json = model_paths(self.user_dir, ship, slot)
        if user_pth.exists():
            description, metadata = _description_from_metadata(user_json)
            return TrainingModelSlot(
                ship=ship,
                slot=slot,
                source=SLOT_USER,
                description=description,
                pth_path=user_pth,
                metadata_path=user_json if user_json.exists() else None,
                metadata=metadata,
            )

        return TrainingModelSlot(ship=ship, slot=slot, source=SLOT_EMPTY)

    def slots_for_ship(self, ship: str) -> list[TrainingModelSlot]:
        return [self.slot_for(ship, slot) for slot in range(1, MODEL_SLOT_COUNT + 1)]

    def create_or_update_user_model(self, metadata: Mapping[str, Any]) -> TrainingModelSlot:
        ship = str(metadata["ship"])
        slot = int(metadata["slot"])
        current_slot = self.slot_for(ship, slot)
        if current_slot.is_bundled:
            raise PermissionError("Bundled training models are read-only")
        description = str(metadata.get("description", "")).strip()
        if not current_slot.exists and not description:
            raise ValueError("New training models require a description")

        pth_path, metadata_path = model_paths(self.user_dir, ship, slot)
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        if not pth_path.exists():
            pth_path.touch()
        atomic_write_json(metadata_path, dict(metadata))
        return self.slot_for(ship, slot)

    def delete_user_model(self, ship: str, slot: int) -> None:
        model_slot = self.slot_for(ship, slot)
        if not model_slot.is_user:
            return
        pth_path, metadata_path = model_paths(self.user_dir, ship, slot)
        for path in (pth_path, metadata_path, replay_checkpoint_path(pth_path)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def model_slot_has_checkpoint(slot: TrainingModelSlot) -> bool:
    """Return whether a model slot has usable checkpoint bytes."""
    if (
        not slot.exists
        or slot.pth_path is None
        or not 1 <= int(slot.slot) <= MODEL_SLOT_COUNT
    ):
        return False
    try:
        return slot.pth_path.exists() and slot.pth_path.stat().st_size > 0
    except OSError:
        return False


def trained_model_count_for_ship(
    repository: TrainingModelRepository, ship: str
) -> int:
    return sum(
        1 for slot in repository.slots_for_ship(ship) if model_slot_has_checkpoint(slot)
    )


def trained_model_counts_for_ships(
    repository: TrainingModelRepository, ships
) -> dict[str, int]:
    return {
        str(ship): trained_model_count_for_ship(repository, str(ship))
        for ship in ships
    }
