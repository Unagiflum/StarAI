"""Read-only training observation encoder."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import src.const as const
from src.toroidal import wrapped_delta
from src.training.contracts import (
    OBJECT_SLOT_GROUPS,
    OBJECT_SLOT_OFFSET,
    OBJECT_SLOT_SIZE,
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

_SHIP_TYPE_INDEX = {
    ship_name: index for index, ship_name in enumerate(SHIP_TYPE_CATALOG_ORDER)
}
_FLOAT32_MAX = 3.4028234663852886e38


@dataclass(frozen=True, slots=True)
class _ObservedObject:
    obj: object
    name: str
    position: tuple[float, float]
    velocity: tuple[float, float]
    has_position: bool
    alive: bool
    is_planet: bool
    is_asteroid: bool
    is_syreen_crew: bool
    is_positional_ability: bool
    owner: object | None
    player_sort: float
    action_group: str


@dataclass(frozen=True, slots=True)
class ObservationFrameContext:
    """Immutable derived values shared by both observation perspectives."""

    objects: tuple
    observed_objects: tuple[_ObservedObject, ...]
    owned_live_counts: dict[tuple[int, str], int]
    floating_orz_counts: dict[int, int]

    def owned_live_count(self, ship, name: str, *, floating_orz: bool = False) -> float:
        counts = self.floating_orz_counts if floating_orz else self.owned_live_counts
        key = id(ship) if floating_orz else (id(ship), name)
        return float(counts.get(key, 0))


def build_observation_context(self_ship, enemy_ship, game_objects=None) -> ObservationFrameContext:
    objects = _object_snapshot(self_ship, enemy_ship, game_objects)
    observed = []
    owned_live_counts: dict[tuple[int, str], int] = {}
    floating_orz_counts: dict[int, int] = {}
    for obj in objects:
        name = str(getattr(obj, "name", ""))
        has_position = _has_position(obj)
        alive = _object_alive(obj)
        owner = _root_owner(obj)
        if alive and owner is not None:
            count_key = (id(owner), name)
            owned_live_counts[count_key] = owned_live_counts.get(count_key, 0) + 1
            if name == "OrzA3" and getattr(obj, "mode", None) != "boarded":
                owner_id = id(owner)
                floating_orz_counts[owner_id] = floating_orz_counts.get(owner_id, 0) + 1
        observed.append(
            _ObservedObject(
                obj=obj,
                name=name,
                position=_vector(obj, "position") if has_position else (0.0, 0.0),
                velocity=_observed_velocity(obj) if has_position else (0.0, 0.0),
                has_position=has_position,
                alive=alive,
                is_planet=_is_planet(obj),
                is_asteroid=_is_asteroid(obj),
                is_syreen_crew=_is_syreen_crew(obj),
                is_positional_ability=_is_positional_ability(obj),
                owner=owner,
                player_sort=_number(obj, "player", 0.0),
                action_group="a1" if _is_a1_object(obj) else "non_a1",
            )
        )
    return ObservationFrameContext(
        objects=objects,
        observed_objects=tuple(observed),
        owned_live_counts=owned_live_counts,
        floating_orz_counts=floating_orz_counts,
    )

def encode_observation(
    self_ship,
    enemy_ship,
    *,
    frame_id: int | None = None,
    game_objects=None,
    context: ObservationFrameContext | None = None,
) -> list[float]:
    """Return the canonical observation for ``self_ship``.

    The function only reads ship/object attributes and never calls gameplay
    methods that can update timers, RNG, physics, controls, or collisions.
    """
    values = [0.0] * OBSERVATION_INPUT_SIZE
    context = context or build_observation_context(
        self_ship,
        enemy_ship,
        game_objects,
    )
    enemy_type_index = _SHIP_TYPE_INDEX.get(getattr(enemy_ship, "name", None))
    if enemy_type_index is not None:
        values[enemy_type_index] = 1.0

    values[len(SHIP_TYPE_CATALOG_ORDER):OBJECT_SLOT_OFFSET] = (
        _ship_block(self_ship, enemy_ship, context=context, frame_id=frame_id)
        + _ship_block(enemy_ship, self_ship, context=context, frame_id=frame_id)
    )
    values[OBJECT_SLOT_OFFSET:] = _object_slots(self_ship, enemy_ship, context)

    _validate_observation(values)
    return values


def _ship_block(
    ship,
    opponent,
    *,
    context: ObservationFrameContext,
    frame_id: int | None,
) -> list[float]:
    velocity = _vector(ship, "velocity")
    speed = math.hypot(velocity[0], velocity[1])
    rotation = _rotation_degrees(ship)
    repeat_countdowns = [
        _repeat_countdown(ship, control_name, frame_id=frame_id)
        for control_name in _CONTROL_REPEAT_FIELDS
    ]
    cloak_transition_direction = _cloak_transition_direction(ship)

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
        _flag(cloak_transition_direction != 0),
        float(cloak_transition_direction),
        *_orz_turret_relative_angle(ship),
        context.owned_live_count(ship, "OrzA3", floating_orz=True) / 8.0,
        _boarded_marines_on_opponent(ship, opponent) / 8.0,
        context.owned_live_count(ship, "KzerZaA2") / 25.0,
        context.owned_live_count(ship, "ChmmrSatellite") / 3.0,
        context.owned_live_count(ship, "ChenjesuA2") / 4.0,
        context.owned_live_count(ship, "KohrAhA1") / 8.0,
    ]
    if len(block) != SHIP_BLOCK_SIZE:
        raise RuntimeError(f"ship block must contain {SHIP_BLOCK_SIZE} values")
    return [float(value) for value in block]


def _object_snapshot(self_ship, enemy_ship, game_objects) -> tuple:
    candidates = []
    if game_objects is not None:
        if hasattr(game_objects, "objects"):
            candidates.extend(list(game_objects.objects))
        else:
            candidates.extend(list(game_objects))
    else:
        for ship in (self_ship, enemy_ship):
            candidates.extend(_optional_iterable(getattr(ship, "friendly_objects", ())))
            candidates.extend(_optional_iterable(getattr(ship, "enemy_objects", ())))
            candidates.extend(_optional_iterable(getattr(ship, "asteroids", ())))

    candidates.extend([self_ship, enemy_ship])
    for ship in (self_ship, enemy_ship):
        planet = getattr(ship, "planet", None)
        if planet is not None:
            candidates.append(planet)

    seen = set()
    objects = []
    for obj in candidates:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        objects.append(obj)
    return tuple(objects)


def _optional_iterable(value) -> tuple:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _object_slots(
    self_ship,
    enemy_ship,
    context: ObservationFrameContext,
) -> list[float]:
    groups = _classify_object_slots(self_ship, enemy_ship, context.observed_objects)
    slots: list[float] = []
    self_position = _vector(self_ship, "position")
    self_velocity = _vector(self_ship, "velocity")
    self_rotation = _rotation_degrees(self_ship)
    for group_name, count in OBJECT_SLOT_GROUPS:
        selected = _nearest_objects(self_position, groups[group_name], count)
        for observed in selected:
            slots.extend(
                _encode_object_slot(
                    self_ship,
                    enemy_ship,
                    observed,
                    self_position=self_position,
                    self_velocity=self_velocity,
                    self_rotation=self_rotation,
                )
            )
        for _ in range(count - len(selected)):
            slots.extend([0.0] * OBJECT_SLOT_SIZE)
    return slots


def _classify_object_slots(
    self_ship,
    enemy_ship,
    objects: tuple[_ObservedObject, ...],
) -> dict[str, list]:
    groups = {group_name: [] for group_name, _ in OBJECT_SLOT_GROUPS}
    enemy_fact = next((item for item in objects if item.obj is enemy_ship), None)
    if enemy_fact is not None and enemy_fact.has_position:
        groups["enemy_ship"].append(enemy_fact)

    for observed in objects:
        obj = observed.obj
        if obj is self_ship or obj is enemy_ship:
            continue
        if not observed.has_position or not observed.alive:
            continue
        if observed.is_planet:
            groups["planet"].append(observed)
        elif observed.is_asteroid:
            groups["asteroid"].append(observed)
        elif observed.is_syreen_crew:
            groups["syreen_crew"].append(observed)
        elif observed.is_positional_ability:
            if _same_owner(observed.owner, self_ship):
                group_prefix = "friendly"
            elif _same_owner(observed.owner, enemy_ship):
                group_prefix = "enemy"
            else:
                group_prefix = "friendly" if _same_player(obj, self_ship) else "enemy"
            groups[f"{group_prefix}_{observed.action_group}"].append(observed)
    return groups


def _nearest_objects(
    self_position: tuple[float, float],
    objects: list[_ObservedObject],
    count: int,
) -> list[_ObservedObject]:
    indexed = list(enumerate(objects))
    indexed.sort(
        key=lambda item: (
            _distance_squared_positions(self_position, item[1].position),
            item[1].name or type(item[1].obj).__name__,
            item[1].player_sort,
            item[1].position[0],
            item[1].position[1],
            item[0],
        )
    )
    return [observed for _, observed in indexed[:count]]


def _encode_object_slot(
    self_ship,
    enemy_ship,
    observed: _ObservedObject,
    *,
    self_position: tuple[float, float],
    self_velocity: tuple[float, float],
    self_rotation: float,
) -> list[float]:
    obj = observed.obj
    dx, dy = wrapped_delta(self_position, observed.position)
    distance = math.hypot(dx, dy)
    relative_bearing = math.radians(
        (_vector_angle(dx, dy) - self_rotation) % 360.0
    )
    object_velocity = observed.velocity
    relative_velocity = (
        object_velocity[0] - self_velocity[0],
        object_velocity[1] - self_velocity[1],
    )
    relative_speed = math.hypot(relative_velocity[0], relative_velocity[1])
    if relative_speed > 0:
        velocity_angle = math.radians(_vector_angle(*relative_velocity))
        velocity_sin = math.sin(velocity_angle)
        velocity_cos = math.cos(velocity_angle)
    else:
        velocity_sin = 0.0
        velocity_cos = 0.0

    expires = _object_expires(obj)
    crew_effect, battery_effect = _expected_contact_effects(
        obj,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
    )
    return [
        1.0,
        _flag(expires),
        _remaining_timer_value(obj) if expires else 5.0,
        math.sin(relative_bearing),
        math.cos(relative_bearing),
        5.0 if distance <= 0 else min(5.0, 100.0 / distance),
        velocity_sin,
        velocity_cos,
        relative_speed / 100.0,
        crew_effect / 10.0,
        battery_effect / 10.0,
        _number(obj, "current_hp", 0.0) / 50.0,
    ]


def _has_position(obj) -> bool:
    position = getattr(obj, "position", None)
    return isinstance(position, Sequence) and len(position) >= 2


def _object_alive(obj) -> bool:
    return bool(getattr(obj, "currently_alive", True)) and _number(obj, "current_hp", 1.0) > 0


def _is_planet(obj) -> bool:
    return hasattr(obj, "gravity") and hasattr(obj, "diameter") and not _is_asteroid(obj)


def _is_asteroid(obj) -> bool:
    return getattr(obj, "name", None) == "Asteroid"


def _is_syreen_crew(obj) -> bool:
    return getattr(obj, "name", None) == "SyreenCrew"


def _is_positional_ability(obj) -> bool:
    return getattr(obj, "parent", None) is not None or getattr(obj, "type", None) in {
        "projectile",
        "special_object",
        "laser",
        "area",
    }


def _observed_velocity(obj) -> tuple[float, float]:
    """Return frame displacement for satellites and runtime velocity otherwise."""
    if getattr(obj, "name", None) != "ChmmrSatellite":
        return _vector(obj, "velocity")

    position = getattr(obj, "position", None)
    previous_position = getattr(obj, "previous_position", None)
    if not (
        isinstance(position, Sequence)
        and len(position) >= 2
        and isinstance(previous_position, Sequence)
        and len(previous_position) >= 2
    ):
        return 0.0, 0.0

    displacement = wrapped_delta(
        (_finite_float(previous_position[0]), _finite_float(previous_position[1])),
        (_finite_float(position[0]), _finite_float(position[1])),
    )
    speed_scale = _finite_float(const.SPEED_SCALE)
    if speed_scale == 0.0:
        return 0.0, 0.0
    return displacement[0] / speed_scale, displacement[1] / speed_scale


def _is_a1_object(obj) -> bool:
    action_number = getattr(obj, "action_number", None)
    if action_number is not None:
        return action_number == 1
    name = str(getattr(obj, "name", ""))
    return name.endswith("A1")


def _root_owner(obj):
    parent = getattr(obj, "parent", None)
    seen = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        grandparent = getattr(parent, "parent", None)
        if grandparent is None:
            return parent
        parent = grandparent
    return parent


def _same_owner(owner, ship) -> bool:
    return owner is ship


def _same_player(obj, ship) -> bool:
    obj_player = getattr(obj, "player", None)
    ship_player = getattr(ship, "player", None)
    return obj_player is not None and obj_player == ship_player


def _distance_squared(from_obj, to_obj) -> float:
    dx, dy = wrapped_delta(_vector(from_obj, "position"), _vector(to_obj, "position"))
    return dx * dx + dy * dy


def _distance_squared_positions(
    from_position: tuple[float, float],
    to_position: tuple[float, float],
) -> float:
    dx, dy = wrapped_delta(from_position, to_position)
    return dx * dx + dy * dy


def _vector_angle(dx: float, dy: float) -> float:
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _object_expires(obj) -> bool:
    timer = getattr(obj, "expiration_timer", None)
    return bool(getattr(obj, "can_expire", False)) and _is_finite_number(timer)


def _remaining_timer_value(obj) -> float:
    return max(0.0, _finite_float(getattr(obj, "expiration_timer", 0.0))) / const.FPS


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _expected_contact_effects(obj, *, self_ship, enemy_ship) -> tuple[float, float]:
    if _is_syreen_crew(obj):
        return 1.0, 0.0
    crew_effect = -max(0.0, _number(obj, "current_damage"))
    battery_effect = 0.0
    drain = getattr(obj, "drain", None)
    if _is_finite_number(drain):
        battery_effect -= max(0.0, float(drain))
    battery_gain = getattr(obj, "battery_gain", getattr(obj, "energy_gain", None))
    if _is_finite_number(battery_gain):
        battery_effect += float(battery_gain)
    return crew_effect, battery_effect


def _owned_live_count(objects: tuple, ship, name: str, predicate=None) -> float:
    count = 0
    for obj in objects:
        if getattr(obj, "name", None) != name or not _object_alive(obj):
            continue
        if not _same_owner(_root_owner(obj), ship):
            continue
        if predicate is not None and not predicate(obj):
            continue
        count += 1
    return float(count)


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
    for value in values:
        if not math.isfinite(value) or value < -_FLOAT32_MAX or value > _FLOAT32_MAX:
            raise RuntimeError("observation contains values not representable as float32")
