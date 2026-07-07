"""Persistence for training model placeholders and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.persistence import EXPECTED_READ_ERRORS, atomic_write_json, read_json


MODEL_SLOT_COUNT = 4
MODEL_METADATA_VERSION = 1
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
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_METADATA_VERSION,
        "ship": ship,
        "slot": slot,
        "description": description,
        "architecture": dict(architecture),
        "training": dict(training),
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
        if self.slot_for(ship, slot).is_bundled:
            raise PermissionError("Bundled training models are read-only")

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
        for path in (pth_path, metadata_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
