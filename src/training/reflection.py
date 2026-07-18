"""Left/right reflection augmentation for replay training samples."""

from __future__ import annotations

import numpy as np

from src.training.contracts import (
    ACTION_INDEX_TABLE,
    ACTION_OUTPUT_SIZE,
    OBSERVATION_FIELD_NAMES,
    OBSERVATION_INPUT_SIZE,
)


_FIELD_INDEX = {
    field_name: index
    for index, field_name in enumerate(OBSERVATION_FIELD_NAMES)
}
_ANGLE_INDICES = tuple(
    _FIELD_INDEX[f"{ship_role}.absolute_angle"]
    for ship_role in ("self", "enemy")
)
_NEGATED_INDICES = tuple(
    [
        *(
            _FIELD_INDEX[f"{ship_role}.{field_name}"]
            for ship_role in ("self", "enemy")
            for field_name in (
                "absolute_x_velocity",
                "orz_turret_relative_sine",
            )
        ),
        *(
            index
            for index, field_name in enumerate(OBSERVATION_FIELD_NAMES)
            if field_name.startswith("object.")
            and field_name.endswith(
                (".relative_bearing_sine", ".relative_velocity_sine")
            )
        ),
    ]
)
_SWAPPED_INDEX_PAIRS = tuple(
    (
        _FIELD_INDEX[f"{ship_role}.{left_name}"],
        _FIELD_INDEX[f"{ship_role}.{right_name}"],
    )
    for ship_role in ("self", "enemy")
    for left_name, right_name in (
        ("left_repeat_countdown", "right_repeat_countdown"),
        ("left_held", "right_held"),
    )
)

_ACTION_INDEX_FOR_MASK = {
    action.mask: index
    for index, action in enumerate(ACTION_INDEX_TABLE)
}


def _reflected_action_mask(mask: int) -> int:
    left = mask & 2
    right = mask & 4
    return (mask & ~6) | (4 if left else 0) | (2 if right else 0)


REFLECTED_ACTION_INDEX_TABLE = tuple(
    _ACTION_INDEX_FOR_MASK[_reflected_action_mask(action.mask)]
    for action in ACTION_INDEX_TABLE
)


def _copy_values(values):
    clone = getattr(values, "clone", None)
    return clone() if callable(clone) else values.copy()


def reflect_observations(observations):
    """Return observations reflected across the arena's vertical axis.

    The function accepts either NumPy arrays or PyTorch tensors and preserves
    all leading dimensions. The last dimension must be the observation width.
    """

    values = observations
    if not hasattr(values, "shape"):
        values = np.asarray(values, dtype=np.float32)
    if not values.shape or int(values.shape[-1]) != OBSERVATION_INPUT_SIZE:
        raise ValueError(
            f"observations must end with {OBSERVATION_INPUT_SIZE} values"
        )

    mirrored = _copy_values(values)
    mirrored[..., _ANGLE_INDICES] = (-values[..., _ANGLE_INDICES]) % 1.0
    mirrored[..., _NEGATED_INDICES] = -values[..., _NEGATED_INDICES]
    for left_index, right_index in _SWAPPED_INDEX_PAIRS:
        mirrored[..., left_index] = values[..., right_index]
        mirrored[..., right_index] = values[..., left_index]
    return mirrored


def reflect_action_indices(action_indices):
    """Return action indices with the left and right control bits exchanged."""

    values = action_indices
    if hasattr(values, "new_tensor"):
        if bool(((values < 0) | (values >= ACTION_OUTPUT_SIZE)).any().item()):
            raise ValueError(f"action indices must be in [0, {ACTION_OUTPUT_SIZE})")
        mapping = values.new_tensor(REFLECTED_ACTION_INDEX_TABLE).long()
        return mapping[values.long()]

    values = np.asarray(values)
    if np.any(values < 0) or np.any(values >= ACTION_OUTPUT_SIZE):
        raise ValueError(f"action indices must be in [0, {ACTION_OUTPUT_SIZE})")
    mapping = np.asarray(REFLECTED_ACTION_INDEX_TABLE, dtype=np.uint8)
    return mapping[values.astype(np.int64, copy=False)]
