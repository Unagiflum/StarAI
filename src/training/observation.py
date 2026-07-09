"""Read-only training observation encoder.

Phase 2 encodes the enemy ship type and both 45-value ship blocks. The object
slot section is deliberately zero-filled until Phase 3 adds object adapters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import src.const as const
from src.training.contracts import (
    OBJECT_SLOT_OFFSET,
    OBSERVATION_INPUT_SIZE,
    SHIP_BLOCK_SIZE,
    SHIP_TYPE_CATALOG_ORDER,
)


_CONTROL_ATTRIBUTES = {
    "thrust": "thrust_active",
    "turn_left": "turn_left_active",
    "turn_right": "turn_right_active",
    "action1": "action1_active",
    "action2": "action2_active",
}

_CONTROL_REPEAT_FIELDS = (
    "thrust",
    "turn_left",
    "turn_right",
    "action1",
    "action2",
)


def encode_observation(self_ship, enemy_ship, *, frame_id: int | None = None) -> list[float]:
    """Return the canonical 533-value observation for ``self_ship``.

    The function only reads ship attributes and never calls gameplay methods
    that can update timers, RNG, physics, or controls.
    """
    values = [0.0] * OBSERVATION_INPUT_SIZE
    try:
        enemy_type_index = SHIP_TYPE_CATALOG_ORDER.index(enemy_ship.name)
    except ValueError:
        enemy_type_index = None
    if enemy_type_index is not None:
        values[enemy_type_index] = 1.0

    values[len(SHIP_TYPE_CATALOG_ORDER):OBJECT_SLOT_OFFSET] = (
        _ship_block(self_ship, enemy_ship, frame_id=frame_id)
        + _ship_block(enemy_ship, self_ship, frame_id=frame_id)
    )

    _validate_observation(values)
    return values


def _ship_block(ship, opponent, *, frame_id: int | None) -> list[float]:
    velocity = _vector(ship, "velocity")
    speed = math.hypot(velocity[0], velocity[1])
    rotation = _rotation_degrees(ship)
    repeat_countdowns = [
        _repeat_countdown(ship, control_name, frame_id=frame_id)
        for control_name in _CONTROL_REPEAT_FIELDS
    ]

    block = [
        _number(ship, "max_hp") / 50.0,
        _number(ship, "max_energy") / 50.0,
        _number(ship, "current_hp") / 50.0,
        _number(ship, "current_energy") / 50.0,
        _number(ship, "thrust_wait") / const.FPS,
        _number(ship, "thrust_timer") / const.FPS,
        _number(ship, "turn_wait") / const.FPS,
        _number(ship, "turn_timer") / const.FPS,
        _number(ship, "thrust_increment") / 10.0,
        _number(ship, "a1_wait") / const.FPS,
        _number(ship, "action1_timer") / const.FPS,
        _number(ship, "a2_wait") / const.FPS,
        _number(ship, "action2_timer") / const.FPS,
        _number(ship, "a3_wait") / const.FPS,
        _number(ship, "action3_timer") / const.FPS,
        _number(ship, "energy_timer") / const.FPS,
        (rotation % 360.0) / 360.0,
        speed / 100.0,
        velocity[0] / 100.0,
        velocity[1] / 100.0,
        *repeat_countdowns,
        _flag(getattr(ship, "trackable", True)),
        *(_flag(_control_held(ship, control_name)) for control_name in _CONTROL_REPEAT_FIELDS),
        _flag(_is_androsynth_blazer(ship)),
        _flag(_is_mmrnmrhm_alternate_form(ship)),
        _number(ship, "limpets_attached") / 64.0,
        _flag(_damage_shield_active(ship)),
        _flag(_cloak_transition_direction(ship) != 0),
        float(_cloak_transition_direction(ship)),
        *_orz_turret_relative_angle(ship),
        0.0,
        _boarded_marines_on_opponent(ship, opponent) / 8.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    if len(block) != SHIP_BLOCK_SIZE:
        raise RuntimeError(f"ship block must contain {SHIP_BLOCK_SIZE} values")
    return [float(value) for value in block]


def _number(obj, attribute: str, default: float = 0.0) -> float:
    value = getattr(obj, attribute, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return float(default)


def _vector(obj, attribute: str) -> tuple[float, float]:
    value = getattr(obj, attribute, (0.0, 0.0))
    if not isinstance(value, Sequence) or len(value) < 2:
        return 0.0, 0.0
    x, y = value[0], value[1]
    return _finite_float(x), _finite_float(y)


def _finite_float(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return 0.0


def _rotation_degrees(ship) -> float:
    rotation = getattr(ship, "rotation", None)
    if isinstance(rotation, (int, float)) and math.isfinite(rotation):
        return float(rotation)
    return _number(ship, "heading") * const.TURN_ANGLE


def _flag(value) -> float:
    return 1.0 if value else 0.0


def _control_held(ship, control_name: str) -> bool:
    return bool(getattr(ship, _CONTROL_ATTRIBUTES[control_name], False))


def _repeat_countdown(ship, control_name: str, *, frame_id: int | None) -> float:
    if not _control_held(ship, control_name):
        return 0.0
    newly_pressed = getattr(ship, "newly_pressed_controls", set())
    if control_name in newly_pressed:
        return 0.0
    pressed_frames = getattr(ship, "input_pressed_frames", {})
    pressed_frame = pressed_frames.get(control_name) if pressed_frames else None
    if frame_id is None or pressed_frame is None:
        return 0.0
    elapsed = max(0, int(frame_id) - int(pressed_frame))
    return float(max(0, const.INPUT_REPEAT_DELAY_FRAMES - elapsed))


def _is_androsynth_blazer(ship) -> bool:
    return getattr(ship, "name", None) == "Androsynth" and getattr(ship, "form", None) == "A2"


def _is_mmrnmrhm_alternate_form(ship) -> bool:
    return getattr(ship, "name", None) == "Mmrnmrhm" and getattr(ship, "form", None) == "YWing"


def _damage_shield_active(ship) -> bool:
    shield = getattr(ship, "_active_damage_shield", None)
    return bool(
        shield is not None
        and getattr(shield, "currently_alive", False)
        and getattr(shield, "blocks_damage", False)
    )


def _cloak_transition_direction(ship) -> int:
    direction = getattr(ship, "fade_direction", 0)
    if not isinstance(direction, (int, float)) or not math.isfinite(direction):
        return 0
    if direction < 0:
        return -1
    if direction > 0:
        return 1
    return 0


def _orz_turret_relative_angle(ship) -> tuple[float, float]:
    if getattr(ship, "name", None) != "Orz":
        return 0.0, 0.0
    turret_heading = getattr(ship, "turret_heading", None)
    if not isinstance(turret_heading, (int, float)) or not math.isfinite(turret_heading):
        turret = getattr(ship, "turret", None)
        turret_heading = getattr(turret, "absolute_heading", None)
    if not isinstance(turret_heading, (int, float)) or not math.isfinite(turret_heading):
        return 0.0, 0.0
    relative = math.radians((float(turret_heading) - _rotation_degrees(ship)) % 360.0)
    return math.sin(relative), math.cos(relative)


def _boarded_marines_on_opponent(ship, opponent) -> float:
    marines = getattr(opponent, "boarded_marines", ())
    if marines is None:
        return 0.0
    count = 0
    for marine in marines:
        if getattr(marine, "parent", None) is ship and getattr(marine, "currently_alive", True):
            count += 1
    return float(count)


def _validate_observation(values: list[float]) -> None:
    if len(values) != OBSERVATION_INPUT_SIZE:
        raise RuntimeError(f"observation must contain {OBSERVATION_INPUT_SIZE} values")
    non_finite = [value for value in values if not math.isfinite(value)]
    if non_finite:
        raise RuntimeError("observation contains non-finite values")
