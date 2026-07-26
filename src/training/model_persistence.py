"""Paired model checkpoint and metadata backup handling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.persistence import EXPECTED_READ_ERRORS, atomic_copy_file, read_json
from src.training import torch_backend
from src.training.model_registry import (
    SLOT_USER,
    TrainingModelSlot,
    model_paths,
    previous_model_paths,
    previous_save_path,
)
from src.training.replay import (
    TrainingCheckpointError,
    training_checkpoint_extra_state,
)


def rotate_previous_model_save(
    checkpoint_path: Path,
    metadata_path: Path,
) -> bool:
    """Atomically preserve a valid current pair before its next replacement."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(metadata_path)
    if _load_model_save_pair(checkpoint_path, metadata_path) is None:
        return False

    previous_checkpoint = previous_save_path(checkpoint_path)
    previous_metadata = previous_save_path(metadata_path)
    atomic_copy_file(checkpoint_path, previous_checkpoint)
    atomic_copy_file(metadata_path, previous_metadata)
    return True


def resolve_model_save(slot: TrainingModelSlot) -> TrainingModelSlot:
    """Use the previous save when the current checkpoint/metadata pair is bad."""
    if (
        slot.source != SLOT_USER
        or slot.pth_path is None
        or torch_backend.get_torch() is None
    ):
        return slot

    directory = slot.pth_path.parent
    current_checkpoint, current_metadata = model_paths(
        directory,
        slot.ship,
        slot.slot,
    )
    previous_checkpoint, previous_metadata = previous_model_paths(
        directory,
        slot.ship,
        slot.slot,
    )
    if not previous_checkpoint.exists() or not previous_metadata.exists():
        return slot

    current = _load_model_save_pair(current_checkpoint, current_metadata)
    if current is not None:
        return _slot_for_pair(slot, current_checkpoint, current_metadata, current)

    previous = _load_model_save_pair(previous_checkpoint, previous_metadata)
    if previous is not None:
        return _slot_for_pair(
            slot,
            previous_checkpoint,
            previous_metadata,
            previous,
        )
    return slot


def _slot_for_pair(
    slot: TrainingModelSlot,
    checkpoint_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
) -> TrainingModelSlot:
    description = metadata.get("description", "")
    return replace(
        slot,
        description=description if isinstance(description, str) else "",
        pth_path=checkpoint_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def _load_model_save_pair(
    checkpoint_path: Path,
    metadata_path: Path,
) -> Mapping[str, Any] | None:
    try:
        if checkpoint_path.stat().st_size <= 0:
            return None
        metadata = read_json(metadata_path)
        if not isinstance(metadata, Mapping):
            return None
        extra_state = training_checkpoint_extra_state(
            checkpoint_path,
            map_location="cpu",
        )
        if not _completed_batches_match(extra_state, metadata):
            return None
        return metadata
    except EXPECTED_READ_ERRORS + (TrainingCheckpointError,):
        return None


def _completed_batches_match(
    extra_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    if "completed_batches" not in extra_state:
        # Checkpoints predating embedded progress remain compatible. Every new
        # application save supplies the value in both files.
        return True
    try:
        checkpoint_batches = int(extra_state["completed_batches"])
    except (TypeError, ValueError):
        return False

    progress = metadata.get("progress", {})
    if not isinstance(progress, Mapping) or "completed_batches" not in progress:
        return False
    try:
        metadata_batches = int(progress["completed_batches"])
    except (TypeError, ValueError):
        return False
    return checkpoint_batches == metadata_batches
