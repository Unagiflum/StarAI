"""Read-only ability pointing and range predicates for training rewards."""

from __future__ import annotations

import math
from collections.abc import Iterable

import src.const as const
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.toroidal import wrapped_delta, wrapped_distance


def is_pointing_at_enemy(ship, enemy, world=None, action_number: int = 1) -> bool:
    definition = ability_definition_for_action(ship, action_number)
    if definition is None or not _target_is_live(enemy):
        return False
    if _is_omnidirectional_effect(definition):
        return _automatic_target_is_valid(definition, enemy)
    if definition.tracking and _automatic_target_is_valid(definition, enemy):
        return True

    target_angle = _angle_to_target(ship, enemy)
    for weapon_angle in _weapon_angles(ship, definition, action_number):
        if abs(_signed_angle(target_angle - weapon_angle)) <= const.TURN_ANGLE / 2 + 1e-9:
            return True
    return False


def is_enemy_in_effective_range(ship, enemy, world=None, action_number: int = 1) -> bool:
    definition = ability_definition_for_action(ship, action_number)
    if definition is None or not _target_is_live(enemy):
        return False
    distance = wrapped_distance(_position(ship), _position(enemy))

    if definition.range is not None:
        return distance <= float(definition.range)
    if definition.ability_type == "area":
        return False
    if definition.ability_type == "laser":
        return False
    if definition.ability_type not in {"projectile", "special_object"}:
        return _special_effective_range(ship, enemy, definition, action_number)

    if definition.tracking and _automatic_target_is_valid(definition, enemy):
        return distance <= _projectile_travel_distance(definition)

    return any(
        _projectile_can_intercept(ship, enemy, definition, weapon_angle)
        for weapon_angle in _weapon_angles(ship, definition, action_number)
    )


def is_a1_pointing_at_enemy(ship, enemy, world=None) -> bool:
    return is_pointing_at_enemy(ship, enemy, world, action_number=1)


def is_a2_pointing_at_enemy(ship, enemy, world=None) -> bool:
    return is_pointing_at_enemy(ship, enemy, world, action_number=2)


def is_enemy_in_a1_effective_range(ship, enemy, world=None) -> bool:
    return is_enemy_in_effective_range(ship, enemy, world, action_number=1)


def is_enemy_in_a2_effective_range(ship, enemy, world=None) -> bool:
    return is_enemy_in_effective_range(ship, enemy, world, action_number=2)


def ability_definition_for_action(ship, action_number: int):
    if action_number not in (1, 2):
        return None
    for ability_name in _ability_names_for_action(ship, action_number):
        definition = ABILITY_DEFINITIONS.get(ability_name)
        if definition is not None:
            return definition
    return None


def _ability_names_for_action(ship, action_number: int) -> Iterable[str]:
    ship_name = getattr(ship, "name", "")
    if ship_name == "Mmrnmrhm" and action_number == 1:
        if getattr(ship, "form", None) == "YWing":
            yield "MmrnmrhmYWingA1"
        else:
            yield "MmrnmrhmXFormA1"
    yield f"{ship_name}A{action_number}"


def _is_omnidirectional_effect(definition) -> bool:
    return bool(definition.omnidirectional) and definition.ability_type in {
        "area",
        "laser",
        "other",
        "shield",
    }


def _automatic_target_is_valid(definition, enemy) -> bool:
    if definition.tracking and not getattr(enemy, "trackable", True):
        return False
    return _target_is_live(enemy)


def _target_is_live(enemy) -> bool:
    return bool(
        enemy is not None
        and getattr(enemy, "currently_alive", True)
        and getattr(enemy, "current_hp", 1) > 0
    )


def _special_effective_range(ship, enemy, definition, action_number: int) -> bool:
    name = definition.ship_name
    distance = wrapped_distance(_position(ship), _position(enemy))
    if name == "Chmmr" and action_number == 2:
        base_speed = definition.base_speed or 0.0
        return base_speed > 0 and distance <= base_speed * max(1.0, definition.life_time)
    if name == "Androsynth" and action_number == 2:
        return distance <= max(_radius(ship), _radius(enemy)) + 80
    return False


def _weapon_angles(ship, definition, action_number: int) -> tuple[float, ...]:
    if getattr(ship, "name", None) == "Orz" and action_number == 1:
        turret_heading = getattr(ship, "turret_heading", None)
        if _finite(turret_heading):
            return (float(turret_heading) % 360.0,)
        turret = getattr(ship, "turret", None)
        turret_heading = getattr(turret, "absolute_heading", None)
        if _finite(turret_heading):
            return (float(turret_heading) % 360.0,)

    directions = definition.gun_directions or (0.0,)
    heading = _rotation(ship)
    return tuple((heading + float(direction)) % 360.0 for direction in directions)


def _projectile_can_intercept(ship, enemy, definition, weapon_angle: float) -> bool:
    lifetime = max(0.0, float(definition.life_time))
    if lifetime <= 0:
        return False
    speed = max(0.0, float(definition.speed))
    if speed <= 0:
        return False

    angle = math.radians(weapon_angle)
    projectile_velocity = (
        math.sin(angle) * speed + _velocity(ship)[0] * float(definition.parent_vel),
        -math.cos(angle) * speed + _velocity(ship)[1] * float(definition.parent_vel),
    )
    rel_position = wrapped_delta(_position(ship), _position(enemy))
    rel_velocity = (
        _velocity(enemy)[0] - projectile_velocity[0],
        _velocity(enemy)[1] - projectile_velocity[1],
    )
    vv = rel_velocity[0] * rel_velocity[0] + rel_velocity[1] * rel_velocity[1]
    if vv <= 0:
        closest_time = 0.0
    else:
        closest_time = -(
            rel_position[0] * rel_velocity[0]
            + rel_position[1] * rel_velocity[1]
        ) / vv
        closest_time = min(lifetime, max(0.0, closest_time))
    closest_x = rel_position[0] + rel_velocity[0] * closest_time
    closest_y = rel_position[1] + rel_velocity[1] * closest_time
    hit_radius = _radius(ship) + _radius(enemy) + max(1.0, max(definition.damage or (1,)))
    return math.hypot(closest_x, closest_y) <= hit_radius


def _projectile_travel_distance(definition) -> float:
    return max(0.0, float(definition.speed)) * max(0.0, float(definition.life_time))


def _angle_to_target(ship, enemy) -> float:
    dx, dy = wrapped_delta(_position(ship), _position(enemy))
    if dx == 0 and dy == 0:
        return _rotation(ship)
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _signed_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _position(obj) -> tuple[float, float]:
    value = getattr(obj, "position", (0.0, 0.0))
    return float(value[0]), float(value[1])


def _velocity(obj) -> tuple[float, float]:
    value = getattr(obj, "velocity", (0.0, 0.0))
    return float(value[0]), float(value[1])


def _rotation(ship) -> float:
    rotation = getattr(ship, "rotation", None)
    if _finite(rotation):
        return float(rotation) % 360.0
    return (float(getattr(ship, "heading", 0)) * const.TURN_ANGLE) % 360.0


def _radius(obj) -> float:
    size = getattr(obj, "size", (0.0, 0.0))
    if not size:
        return 0.0
    return max(float(size[0]), float(size[1])) / 2.0


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
