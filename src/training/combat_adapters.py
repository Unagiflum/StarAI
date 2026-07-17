"""Read-only ability pointing and range predicates for training rewards."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import src.const as const
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.launch_geometry import gun_world_position
from src.toroidal import wrapped_delta, wrapped_distance

_INDEFINITE_RANGE_ABILITIES = frozenset(
    {
        "ChenjesuA1",
        "ChenjesuA2",
        "KohrAhA1",
        "OrzA2",
        "OrzA3",
    }
)


@dataclass(frozen=True, slots=True)
class RewardCombatFlags:
    """All read-only weapon geometry needed by one reward decision."""

    a1_pointing: bool
    a1_in_range: bool
    a2_pointing: bool
    a2_in_range: bool


def reward_combat_flags(ship, enemy, world=None) -> RewardCombatFlags:
    """Compute both actions' reward predicates from one geometry snapshot."""

    if not _target_is_live(enemy):
        return RewardCombatFlags(False, False, False, False)

    distance = None
    action_flags = []
    for action_number in (1, 2):
        primary, specifications = _reward_action_specifications(
            str(getattr(ship, "name", "")),
            getattr(ship, "form", None) == "YWing",
            action_number,
        )
        pointing = False
        primary_mounts = None
        if primary is not None:
            _, definition = primary
            if _is_omnidirectional_effect(definition):
                pointing = _automatic_target_is_valid(definition, enemy)
            else:
                primary_mounts = _weapon_mounts(
                    ship,
                    definition,
                    action_number,
                )
                fallback_angle = _rotation(ship)
                pointing = any(
                    abs(
                        _signed_angle(
                            _angle_to_target(
                                origin,
                                enemy,
                                fallback_angle=fallback_angle,
                            )
                            - weapon_angle
                        )
                    )
                    <= _pointing_tolerance(definition)
                    for origin, weapon_angle in primary_mounts
                )

        in_range = False
        if specifications:
            if distance is None:
                distance = wrapped_distance(_position(ship), _position(enemy))
            for ability_name, definition in specifications:
                mounts = None
                if definition.ability_type in {"projectile", "special_object"}:
                    mounts = (
                        primary_mounts
                        if primary is not None and definition is primary[1]
                        else _weapon_mounts(ship, definition, action_number)
                    )
                if _is_enemy_in_definition_range(
                    ship,
                    enemy,
                    ability_name,
                    definition,
                    action_number,
                    distance=distance,
                    mounts=mounts,
                ):
                    in_range = True
                    break
        action_flags.extend((pointing, in_range))

    return RewardCombatFlags(*action_flags)


@lru_cache(maxsize=None)
def _reward_action_specifications(ship_name, alternate_form, action_number):
    base_names = []
    if ship_name == "Mmrnmrhm" and action_number == 1:
        base_names.append(
            "MmrnmrhmYWingA1" if alternate_form else "MmrnmrhmXFormA1"
        )
    base_names.append(f"{ship_name}A{action_number}")
    linked_names = (
        ("OrzA3",) if ship_name == "Orz" and action_number == 2 else ()
    )
    yielded = set()
    base = []
    linked = []
    for destination, names in ((base, base_names), (linked, linked_names)):
        for ability_name in names:
            if ability_name in yielded:
                continue
            definition = ABILITY_DEFINITIONS.get(ability_name)
            if definition is not None:
                yielded.add(ability_name)
                destination.append((ability_name, definition))
    specifications = tuple((*base, *linked))
    return (base[0] if base else None), specifications


def is_pointing_at_enemy(ship, enemy, world=None, action_number: int = 1) -> bool:
    definition = ability_definition_for_action(ship, action_number)
    if definition is None or not _target_is_live(enemy):
        return False
    if _is_omnidirectional_effect(definition):
        return _automatic_target_is_valid(definition, enemy)

    for origin, weapon_angle in _weapon_mounts(ship, definition, action_number):
        target_angle = _angle_to_target(origin, enemy, fallback_angle=_rotation(ship))
        if abs(_signed_angle(target_angle - weapon_angle)) <= _pointing_tolerance(definition):
            return True
    return False


def is_enemy_in_effective_range(ship, enemy, world=None, action_number: int = 1) -> bool:
    if not _target_is_live(enemy):
        return False
    return any(
        _is_enemy_in_definition_range(ship, enemy, ability_name, definition, action_number)
        for ability_name, definition in _ability_specs_for_action(
            ship,
            action_number,
            include_linked=True,
        )
    )


def _is_enemy_in_definition_range(
    ship,
    enemy,
    ability_name: str,
    definition,
    action_number: int,
    *,
    distance: float | None = None,
    mounts=None,
) -> bool:
    if distance is None:
        distance = wrapped_distance(_position(ship), _position(enemy))

    effective_range = _fixed_effective_range(ship, definition)
    if effective_range is not None:
        return distance <= effective_range
    if ability_name in _INDEFINITE_RANGE_ABILITIES:
        return True
    if definition.ability_type in {"projectile", "special_object"}:
        return any(
            _projectile_can_reach_perfectly_aimed(ship, enemy, definition, origin)
            for origin, _ in (
                mounts
                if mounts is not None
                else _weapon_mounts(ship, definition, action_number)
            )
        )
    if definition.ability_type in {"area", "laser"}:
        return False
    return _special_effective_range(
        ship,
        enemy,
        definition,
        action_number,
        distance=distance,
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
    for _, definition in _ability_specs_for_action(ship, action_number):
        return definition
    return None


def _ability_specs_for_action(
    ship,
    action_number: int,
    *,
    include_linked: bool = False,
):
    if action_number not in (1, 2):
        return
    yielded = set()
    ability_names = list(_ability_names_for_action(ship, action_number))
    if include_linked:
        ability_names.extend(_linked_ability_names_for_action(ship, action_number))
    for ability_name in ability_names:
        if ability_name in yielded:
            continue
        definition = ABILITY_DEFINITIONS.get(ability_name)
        if definition is not None:
            yielded.add(ability_name)
            yield ability_name, definition


def _ability_names_for_action(ship, action_number: int) -> Iterable[str]:
    ship_name = getattr(ship, "name", "")
    if ship_name == "Mmrnmrhm" and action_number == 1:
        if getattr(ship, "form", None) == "YWing":
            yield "MmrnmrhmYWingA1"
        else:
            yield "MmrnmrhmXFormA1"
    yield f"{ship_name}A{action_number}"


def _linked_ability_names_for_action(ship, action_number: int) -> Iterable[str]:
    if getattr(ship, "name", "") == "Orz" and action_number == 2:
        yield "OrzA3"


def _is_omnidirectional_effect(definition) -> bool:
    return bool(definition.omnidirectional) and definition.ability_type in {
        "area",
        "other",
        "shield",
    }


def _pointing_tolerance(definition) -> float:
    if definition.ability_type == "laser":
        return const.TURN_ANGLE + 1e-9
    return const.TURN_ANGLE / 2 + 1e-9


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


def _special_effective_range(
    ship,
    enemy,
    definition,
    action_number: int,
    *,
    distance: float | None = None,
) -> bool:
    name = definition.ship_name
    if distance is None:
        distance = wrapped_distance(_position(ship), _position(enemy))
    if name == "Chmmr" and action_number == 2:
        base_speed = definition.base_speed or 0.0
        return base_speed > 0 and distance <= base_speed * max(1.0, definition.life_time)
    if name == "Androsynth" and action_number == 2:
        return distance <= max(_radius(ship), _radius(enemy)) + 80
    return False


def _fixed_effective_range(ship, definition) -> float | None:
    if definition.range is not None:
        return float(definition.range)
    if definition.ability_type == "area" and definition.area_length is not None:
        return float(definition.area_length)
    if definition.ship_name == "Slylandro" and definition.action == "A1":
        return _slylandro_lightning_range(ship, definition)
    return None


def _slylandro_lightning_range(ship, definition) -> float | None:
    if definition.segment_length_max is None:
        return None
    weapon_wait = getattr(ship, "a1_wait", None)
    if not _finite(weapon_wait):
        ship_definition = SHIP_DEFINITIONS.get(definition.ship_name)
        weapon_wait = ship_definition.a1_wait if ship_definition else None
    if not _finite(weapon_wait) or weapon_wait <= 0:
        return None
    max_segment_count = int(math.floor(float(weapon_wait) / 2.0)) + 1
    return max_segment_count * float(definition.segment_length_max)


def _weapon_mounts(
    ship,
    definition,
    action_number: int,
) -> tuple[tuple[tuple[float, float], float], ...]:
    directions = definition.gun_directions or (0.0,)
    locations = definition.gun_locations or ()
    mount_rotation = _weapon_mount_rotation(ship, action_number)
    launch_base = mount_rotation if mount_rotation is not None else _rotation(ship)

    mounts = []
    for index, direction in enumerate(directions):
        origin = _position(ship)
        if index < len(locations):
            origin = _gun_position(ship, locations[index], rotation=launch_base)
        mounts.append((origin, (launch_base + float(direction)) % 360.0))
    return tuple(mounts) or ((_position(ship), launch_base % 360.0),)


def _weapon_mount_rotation(ship, action_number: int) -> float | None:
    if getattr(ship, "name", None) != "Orz" or action_number != 1:
        return None
    turret_heading = getattr(ship, "turret_heading", None)
    if _finite(turret_heading):
        return (float(turret_heading) * const.TURN_ANGLE) % 360.0
    turret = getattr(ship, "turret", None)
    turret_heading = getattr(turret, "absolute_heading", None)
    if _finite(turret_heading):
        return (float(turret_heading) * const.TURN_ANGLE) % 360.0
    return None


def _projectile_can_reach_perfectly_aimed(ship, enemy, definition, origin) -> bool:
    lifetime = max(0.0, float(definition.life_time))
    if lifetime <= 0:
        return False
    speed = max(0.0, float(definition.speed))

    parent_velocity = _velocity(ship)
    parent_x = parent_velocity[0] * float(definition.parent_vel)
    parent_y = parent_velocity[1] * float(definition.parent_vel)
    rel_position = wrapped_delta(origin, _position(enemy))
    rel_velocity = (
        _velocity(enemy)[0] - parent_x,
        _velocity(enemy)[1] - parent_y,
    )
    hit_radius = _radius(ship) + _radius(enemy) + max(1.0, max(definition.damage or (1,)))
    return _moving_target_is_reachable(
        rel_position,
        rel_velocity,
        speed,
        lifetime,
        hit_radius,
    )


def _moving_target_is_reachable(
    rel_position: tuple[float, float],
    rel_velocity: tuple[float, float],
    projectile_speed: float,
    lifetime: float,
    hit_radius: float,
) -> bool:
    rx, ry = rel_position
    vx, vy = rel_velocity
    speed = max(0.0, projectile_speed)
    radius = max(0.0, hit_radius)

    def miss_margin_squared(time: float) -> float:
        return (
            (rx + vx * time) ** 2
            + (ry + vy * time) ** 2
            - (speed * time + radius) ** 2
        )

    if miss_margin_squared(0.0) <= 1e-9 or miss_margin_squared(lifetime) <= 1e-9:
        return True

    a = vx * vx + vy * vy - speed * speed
    if a <= 1e-9:
        return False
    b = 2.0 * (rx * vx + ry * vy - speed * radius)
    closest_time = -b / (2.0 * a)
    if closest_time < 0.0 or closest_time > lifetime:
        return False
    return miss_margin_squared(closest_time) <= 1e-9


def _angle_to_target(origin, enemy, *, fallback_angle: float) -> float:
    dx, dy = wrapped_delta(origin, _position(enemy))
    if dx == 0 and dy == 0:
        return fallback_angle
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _signed_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _position(obj) -> tuple[float, float]:
    value = getattr(obj, "position", (0.0, 0.0))
    return float(value[0]), float(value[1])


def _gun_position(ship, gun_location, *, rotation: float) -> tuple[float, float]:
    if not hasattr(ship, "sprites"):
        return _position(ship)
    try:
        position = gun_world_position(ship, gun_location, rotation=rotation)
    except (AttributeError, IndexError, TypeError, ValueError):
        return _position(ship)
    return float(position[0]), float(position[1])


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
