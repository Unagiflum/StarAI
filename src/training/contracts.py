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
SHIP_BLOCK_SIZE = 45
OBJECT_SLOT_SIZE = 11
OBJECT_SLOT_COUNT = 38
ENEMY_SHIP_TYPE_OFFSET = 0
SELF_SHIP_BLOCK_OFFSET = SHIP_TYPE_COUNT
ENEMY_SHIP_BLOCK_OFFSET = SELF_SHIP_BLOCK_OFFSET + SHIP_BLOCK_SIZE
OBJECT_SLOT_OFFSET = ENEMY_SHIP_BLOCK_OFFSET + SHIP_BLOCK_SIZE

SHIP_BLOCK_FIELDS = (
    "maximum_crew",
    "maximum_battery",
    "current_crew",
    "current_battery",
    "thrust_wait",
    "thrust_timer",
    "turn_wait",
    "turn_timer",
    "thrust_increment",
    "a1_wait",
    "a1_timer",
    "a2_wait",
    "a2_timer",
    "a3_wait",
    "a3_timer",
    "energy_wait_timer",
    "absolute_angle",
    "absolute_speed",
    "absolute_x_velocity",
    "absolute_y_velocity",
    "thrust_repeat_countdown",
    "left_repeat_countdown",
    "right_repeat_countdown",
    "a1_repeat_countdown",
    "a2_repeat_countdown",
    "trackable",
    "thrust_held",
    "left_held",
    "right_held",
    "a1_held",
    "a2_held",
    "androsynth_blazer_form",
    "mmrnmrhm_alternate_form",
    "limpet_count",
    "damage_shield_active",
    "ilwrath_cloak_transition",
    "cloak_transition_direction",
    "orz_turret_relative_sine",
    "orz_turret_relative_cosine",
    "orz_marines_floating",
    "orz_marines_boarded_on_enemy",
    "ur_quan_fighters",
    "chmmr_satellites",
    "chenjesu_dogis",
    "kohr_ah_saws",
)

OBJECT_SLOT_FIELDS = (
    "present",
    "expires",
    "remaining_timer",
    "relative_bearing_sine",
    "relative_bearing_cosine",
    "inverse_distance",
    "relative_velocity_sine",
    "relative_velocity_cosine",
    "relative_speed",
    "expected_crew_effect",
    "expected_battery_effect",
)

OBJECT_SLOT_GROUPS = (
    ("enemy_ship", 1),
    ("planet", 1),
    ("enemy_a1", 8),
    ("enemy_non_a1", 8),
    ("friendly_a1", 5),
    ("friendly_non_a1", 5),
    ("asteroid", 5),
    ("syreen_crew", 5),
)


def _observation_field_names() -> tuple[str, ...]:
    names: list[str] = [
        f"enemy_ship_type.{ship_name}" for ship_name in SHIP_TYPE_CATALOG_ORDER
    ]
    names.extend(f"self.{field_name}" for field_name in SHIP_BLOCK_FIELDS)
    names.extend(f"enemy.{field_name}" for field_name in SHIP_BLOCK_FIELDS)
    for group_name, count in OBJECT_SLOT_GROUPS:
        for slot_index in range(count):
            names.extend(
                f"object.{group_name}.{slot_index}.{field_name}"
                for field_name in OBJECT_SLOT_FIELDS
            )
    return tuple(names)


OBSERVATION_FIELD_NAMES = _observation_field_names()

if len(SHIP_BLOCK_FIELDS) != SHIP_BLOCK_SIZE:
    raise RuntimeError("SHIP_BLOCK_FIELDS must define exactly 45 fields")
if len(OBJECT_SLOT_FIELDS) != OBJECT_SLOT_SIZE:
    raise RuntimeError("OBJECT_SLOT_FIELDS must define exactly 11 fields")
if sum(count for _, count in OBJECT_SLOT_GROUPS) != OBJECT_SLOT_COUNT:
    raise RuntimeError("OBJECT_SLOT_GROUPS must define exactly 38 slots")
if len(OBSERVATION_FIELD_NAMES) != OBSERVATION_INPUT_SIZE:
    raise RuntimeError("OBSERVATION_FIELD_NAMES must define exactly 533 fields")


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
