"""Versioned training contracts shared by encoders, models, and metadata."""

from __future__ import annotations

from dataclasses import dataclass

from src.Objects.Ships.catalog import SHIP_DEFINITIONS


OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_INPUT_SIZE = 533

ACTION_SCHEMA_VERSION = 1
ACTION_OUTPUT_SIZE = 24

CONTROL_ORDER = ("thrust", "turn_left", "turn_right", "a1", "a2")
SHIP_TYPE_CATALOG_ORDER = tuple(SHIP_DEFINITIONS.keys())
SHIP_TYPE_COUNT = 25


@dataclass(frozen=True)
class TrainingAction:
    """One frame of held training controls."""

    mask: int
    thrust: bool
    turn_left: bool
    turn_right: bool
    a1: bool
    a2: bool

    @classmethod
    def from_mask(cls, mask: int) -> "TrainingAction":
        return cls(
            mask=mask,
            thrust=bool(mask & 1),
            turn_left=bool(mask & 2),
            turn_right=bool(mask & 4),
            a1=bool(mask & 8),
            a2=bool(mask & 16),
        )

    @property
    def held_controls(self) -> tuple[str, ...]:
        return tuple(control for control in CONTROL_ORDER if getattr(self, control))

    def to_metadata(self, index: int) -> dict[str, object]:
        return {
            "index": index,
            "mask": self.mask,
            "held_controls": list(self.held_controls),
            "thrust": self.thrust,
            "turn_left": self.turn_left,
            "turn_right": self.turn_right,
            "a1": self.a1,
            "a2": self.a2,
        }


# Stable action-index table. Masks are ordinary five-control bit masks using
# CONTROL_ORDER bit positions, with simultaneous left+right masks excluded.
_VALID_ACTION_MASKS = (
    0,
    1,
    2,
    3,
    4,
    5,
    8,
    9,
    10,
    11,
    12,
    13,
    16,
    17,
    18,
    19,
    20,
    21,
    24,
    25,
    26,
    27,
    28,
    29,
)

ACTION_INDEX_TABLE = tuple(
    TrainingAction.from_mask(mask) for mask in _VALID_ACTION_MASKS
)
ACTION_SCHEMA_METADATA = tuple(
    action.to_metadata(index) for index, action in enumerate(ACTION_INDEX_TABLE)
)


def action_for_index(index: int) -> TrainingAction:
    return ACTION_INDEX_TABLE[index]
