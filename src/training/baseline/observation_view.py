"""Named, unscaled access to the canonical training observation vector.

The baseline strategies read only the observation, never the simulation, so
they run identically in the parent process, in CPU simulation workers, and in
battle mode. Values are converted back to game units:

* distances in world units, speeds in world units per frame;
* positions and velocities of objects in the observing ship's frame, where
  ``forward`` points out of the nose and ``right`` to the starboard side;
* bearings in degrees, clockwise from the nose, in ``(-180, 180]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import src.const as const
from src.training.contracts import (
    ENEMY_SHIP_BLOCK_OFFSET,
    OBJECT_SLOT_FIELDS,
    OBJECT_SLOT_GROUPS,
    OBJECT_SLOT_OFFSET,
    OBJECT_SLOT_SIZE,
    SELF_SHIP_BLOCK_OFFSET,
    SHIP_BLOCK_FIELDS,
    SHIP_TYPE_CATALOG_ORDER,
)

_SHIP_FIELD = {name: index for index, name in enumerate(SHIP_BLOCK_FIELDS)}
_SLOT_FIELD = {name: index for index, name in enumerate(OBJECT_SLOT_FIELDS)}

# Scales applied by ``src.training.observation``.
_CREW_SCALE = 50.0
_CONTACT_EFFECT_SCALE = 10.0
_INVERSE_DISTANCE_NUMERATOR = 100.0
_INVERSE_DISTANCE_CAP = 5.0


def _velocity_scale() -> float:
    scale = float(const.SPEED_LIMIT)
    return scale if scale > 0.0 else 1.0


def wrap_degrees(angle: float) -> float:
    """Return ``angle`` wrapped into ``(-180, 180]``."""

    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def bearing_of(forward: float, right: float) -> float:
    """Return the clockwise bearing of a ship-frame vector from the nose."""

    if forward == 0.0 and right == 0.0:
        return 0.0
    return math.degrees(math.atan2(right, forward))


def _group_offsets() -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    offset = OBJECT_SLOT_OFFSET
    for group_name, count in OBJECT_SLOT_GROUPS:
        offsets[group_name] = (offset, count)
        offset += count * OBJECT_SLOT_SIZE
    return offsets


_GROUP_OFFSETS = _group_offsets()


@dataclass(frozen=True, slots=True)
class ObservedObject:
    """One observed object, expressed in the observing ship's frame."""

    group: str
    distance: float
    bearing: float
    forward: float
    right: float
    velocity_forward: float
    velocity_right: float
    closing_speed: float
    lateral_speed: float
    crew_effect: float
    battery_effect: float
    hit_points: float
    expires: bool
    remaining_seconds: float

    @property
    def relative_speed(self) -> float:
        return math.hypot(self.velocity_forward, self.velocity_right)

    def closing_seconds(self, distance: float | None = None) -> float:
        """Seconds until contact at the current closing speed (inf if opening)."""

        if self.closing_speed <= 0.0:
            return math.inf
        span = self.distance if distance is None else max(0.0, float(distance))
        return span / (self.closing_speed * const.FPS)

    def closest_approach(self) -> tuple[float, float]:
        """Return ``(frames, miss_distance)`` of the closest approach to me."""

        speed_squared = (
            self.velocity_forward * self.velocity_forward
            + self.velocity_right * self.velocity_right
        )
        if speed_squared <= 1e-9:
            return math.inf, self.distance
        frames = -(
            self.forward * self.velocity_forward + self.right * self.velocity_right
        ) / speed_squared
        if frames <= 0.0:
            return frames, self.distance
        miss_forward = self.forward + self.velocity_forward * frames
        miss_right = self.right + self.velocity_right * frames
        return frames, math.hypot(miss_forward, miss_right)


class ShipBlock:
    """Unscaled view of one 51-value ship block."""

    __slots__ = ("_values", "_offset")

    def __init__(self, values: Sequence[float], offset: int):
        self._values = values
        self._offset = int(offset)

    def raw(self, field_name: str) -> float:
        return float(self._values[self._offset + _SHIP_FIELD[field_name]])

    def flag(self, field_name: str) -> bool:
        return self.raw(field_name) > 0.5

    @property
    def max_crew(self) -> float:
        return self.raw("maximum_crew") * _CREW_SCALE

    @property
    def crew(self) -> float:
        return self.raw("current_crew") * _CREW_SCALE

    @property
    def max_battery(self) -> float:
        return self.raw("maximum_battery") * _CREW_SCALE

    @property
    def battery(self) -> float:
        return self.raw("current_battery") * _CREW_SCALE

    @property
    def battery_fraction(self) -> float:
        maximum = self.max_battery
        return self.battery / maximum if maximum > 0 else 0.0

    @property
    def max_thrust(self) -> float:
        return self.raw("maximum_thrust") * _velocity_scale()

    @property
    def turn_wait(self) -> float:
        """Frames between heading steps, including any limpets' penalty."""

        return self.raw("turn_wait") * const.FPS

    @property
    def limpets(self) -> int:
        return int(round(self.raw("limpet_count") * 64.0))

    @property
    def speed(self) -> float:
        return self.raw("absolute_speed") * _velocity_scale()

    @property
    def velocity_forward(self) -> float:
        return self.raw("local_forward_velocity") * _velocity_scale()

    @property
    def velocity_right(self) -> float:
        return self.raw("local_right_velocity") * _velocity_scale()

    @property
    def heading(self) -> float:
        """Absolute heading in degrees, clockwise from screen-up."""

        return math.degrees(
            math.atan2(
                self.raw("absolute_heading_sine"),
                self.raw("absolute_heading_cosine"),
            )
        ) % 360.0

    @property
    def opponent_bearing(self) -> float | None:
        """Bearing of this ship's opponent from this ship's nose."""

        sine = self.raw("opponent_bearing_sine")
        cosine = self.raw("opponent_bearing_cosine")
        if sine == 0.0 and cosine == 0.0:
            return None
        return math.degrees(math.atan2(sine, cosine))

    def timer_seconds(self, field_name: str) -> float:
        """Timers are encoded in seconds already."""

        return self.raw(field_name)

    @property
    def a1_ready(self) -> bool:
        return self.raw("a1_timer") <= 0.0

    @property
    def a2_ready(self) -> bool:
        return self.raw("a2_timer") <= 0.0

    def held(self, control: str) -> bool:
        return self.flag(f"{control}_held")

    @property
    def trackable(self) -> bool:
        return self.flag("trackable")

    @property
    def androsynth_blazer(self) -> bool:
        return self.flag("androsynth_blazer_form")

    @property
    def mmrnmrhm_ywing(self) -> bool:
        return self.flag("mmrnmrhm_alternate_form")

    @property
    def damage_shield_active(self) -> bool:
        return self.flag("damage_shield_active")

    @property
    def cloak_transition_direction(self) -> int:
        value = self.raw("cloak_transition_direction")
        return 1 if value > 0.5 else (-1 if value < -0.5 else 0)

    @property
    def orz_turret_bearing(self) -> float:
        sine = self.raw("orz_turret_relative_sine")
        cosine = self.raw("orz_turret_relative_cosine")
        if sine == 0.0 and cosine == 0.0:
            return 0.0
        return math.degrees(math.atan2(sine, cosine))

    @property
    def orz_marines(self) -> int:
        floating = self.raw("orz_marines_floating") * 8.0
        boarded = self.raw("orz_marines_boarded_on_enemy") * 8.0
        return int(round(floating + boarded))

    @property
    def kzer_za_fighters(self) -> int:
        return int(round(self.raw("ur_quan_fighters") * 25.0))

    @property
    def chmmr_satellites(self) -> int:
        return int(round(self.raw("chmmr_satellites") * 3.0))

    @property
    def chenjesu_dogis(self) -> int:
        return int(round(self.raw("chenjesu_dogis") * 4.0))


class ObservationView:
    """Decoded observation for one observing ship."""

    def __init__(self, values: Sequence[float]):
        self.values = values
        self.me = ShipBlock(values, SELF_SHIP_BLOCK_OFFSET)
        self.enemy_block = ShipBlock(values, ENEMY_SHIP_BLOCK_OFFSET)
        self._groups: dict[str, tuple[ObservedObject, ...]] = {}

    @property
    def enemy_ship_type(self) -> str | None:
        for index, ship_name in enumerate(SHIP_TYPE_CATALOG_ORDER):
            if self.values[index] > 0.5:
                return ship_name
        return None

    def objects(self, group_name: str) -> tuple[ObservedObject, ...]:
        cached = self._groups.get(group_name)
        if cached is not None:
            return cached
        offset, count = _GROUP_OFFSETS[group_name]
        decoded: list[ObservedObject] = []
        for slot_index in range(count):
            base = offset + slot_index * OBJECT_SLOT_SIZE
            if self.values[base + _SLOT_FIELD["present"]] <= 0.5:
                continue
            decoded.append(_decode_slot(group_name, self.values, base))
        result = tuple(decoded)
        self._groups[group_name] = result
        return result

    @property
    def enemy(self) -> ObservedObject | None:
        found = self.objects("enemy_ship")
        return found[0] if found else None

    @property
    def planet(self) -> ObservedObject | None:
        found = self.objects("planet")
        return found[0] if found else None

    @property
    def enemy_objects(self) -> tuple[ObservedObject, ...]:
        return self.objects("enemy_a1") + self.objects("enemy_non_a1")

    @property
    def friendly_a1_objects(self) -> tuple[ObservedObject, ...]:
        return self.objects("friendly_a1")

    @property
    def asteroids(self) -> tuple[ObservedObject, ...]:
        return self.objects("asteroid")

    @property
    def crew_pods(self) -> tuple[ObservedObject, ...]:
        return self.objects("syreen_crew")


def _decode_slot(group_name: str, values: Sequence[float], base: int) -> ObservedObject:
    def field(name: str) -> float:
        return float(values[base + _SLOT_FIELD[name]])

    sine = field("relative_bearing_sine")
    cosine = field("relative_bearing_cosine")
    inverse_distance = field("inverse_distance")
    if inverse_distance >= _INVERSE_DISTANCE_CAP or inverse_distance <= 0.0:
        distance = (
            _INVERSE_DISTANCE_NUMERATOR / _INVERSE_DISTANCE_CAP
            if inverse_distance > 0.0
            else math.inf
        )
    else:
        distance = _INVERSE_DISTANCE_NUMERATOR / inverse_distance
    scale = _velocity_scale()
    finite_distance = distance if math.isfinite(distance) else 0.0
    return ObservedObject(
        group=group_name,
        distance=distance,
        bearing=math.degrees(math.atan2(sine, cosine)),
        forward=finite_distance * cosine,
        right=finite_distance * sine,
        velocity_forward=field("relative_forward_velocity") * scale,
        velocity_right=field("relative_right_velocity") * scale,
        closing_speed=field("relative_closing_speed") * scale,
        lateral_speed=field("relative_lateral_velocity") * scale,
        crew_effect=field("expected_crew_effect") * _CONTACT_EFFECT_SCALE,
        battery_effect=field("expected_battery_effect") * _CONTACT_EFFECT_SCALE,
        hit_points=field("current_hit_points") * _CREW_SCALE,
        expires=field("expires") > 0.5,
        remaining_seconds=field("remaining_timer"),
    )
