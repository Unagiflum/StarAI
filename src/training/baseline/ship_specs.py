"""Static ship and weapon facts used by the baseline strategies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import src.const as const
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS

# Projectile reach is speed x lifetime; the margin keeps the baseline from
# firing at targets the shot would only just reach.
PROJECTILE_RANGE_MARGIN = 0.9
# All planets share one diameter and one gravity, a pull of this many
# world units per frame per frame on inertial ships inside the well.
PLANET_RADIUS = 150.0
PLANET_GRAVITY = 1.0
# How far each hull reaches toward the planet at first contact, beyond the
# planet's radius: the largest over headings and approach directions,
# measured on the real collision masks of the base form.
PLANET_HULL_REACH = {
    "Androsynth": 50.0,
    "Arilou": 34.0,
    "Chenjesu": 72.0,
    "Chmmr": 86.0,
    "Druuge": 72.0,
    "Earthling": 70.0,
    "Ilwrath": 82.0,
    "KohrAh": 78.0,
    "KzerZa": 66.0,
    "Mmrnmrhm": 52.0,
    "Melnorme": 64.0,
    "Mycon": 54.0,
    "Orz": 68.0,
    "Pkunk": 54.0,
    "Shofixti": 28.0,
    "Slylandro": 86.0,
    "Spathi": 64.0,
    "Supox": 58.0,
    "Syreen": 74.0,
    "Thraddash": 48.0,
    "Umgah": 30.0,
    "Utwig": 78.0,
    "Vux": 46.0,
    "Yehat": 38.0,
    "ZoqFotPik": 54.0,
}
# Miss distance at which a small shot passing a hull grazes it: the hull's
# shadow half-width, averaged over the directions a shot can pass, measured
# on the real collision masks of the base form.
SHOT_HULL_REACH = {
    "Androsynth": 38.0,
    "Arilou": 33.0,
    "Chenjesu": 67.0,
    "Chmmr": 69.0,
    "Druuge": 48.0,
    "Earthling": 52.0,
    "Ilwrath": 59.0,
    "KohrAh": 65.0,
    "KzerZa": 54.0,
    "Mmrnmrhm": 43.0,
    "Melnorme": 61.0,
    "Mycon": 48.0,
    "Orz": 54.0,
    "Pkunk": 46.0,
    "Shofixti": 22.0,
    "Slylandro": 67.0,
    "Spathi": 55.0,
    "Supox": 44.0,
    "Syreen": 47.0,
    "Thraddash": 43.0,
    "Umgah": 27.0,
    "Utwig": 62.0,
    "Vux": 33.0,
    "Yehat": 33.0,
    "ZoqFotPik": 47.0,
}
# Contact margin for "collision imminent" checks: two typical hull radii.
COLLISION_MARGIN = 60.0
# A hull's typical reach from its centre: the mean over all ships of half
# way between the half-width and half-length of the collision mask (from 23
# for the Shofixti to 69 for the Chmmr).
TYPICAL_HULL_RADIUS = 45.0
# The Umgah cone is mask-shaped and has no configured length.
UMGAH_CONE_REACH = 180.0
# Ships that always fly at top speed along the nose, where a fresh thrust
# press turns them about instead of pushing.
CONSTANT_SPEED_SHIPS = frozenset({"Slylandro"})

FORM_BASE = "base"
FORM_BLAZER = "blazer"
FORM_XFORM = "xform"
FORM_YWING = "ywing"


@dataclass(frozen=True, slots=True)
class WeaponSpec:
    ability_name: str
    kind: str
    range: float
    speed: float
    parent_velocity: float
    tracking: bool
    gun_directions: tuple[float, ...]
    damage: float = 0.0
    # Whether the shot feels the planet's pull, its lifetime in frames, and
    # for homing shots the frames between heading steps less one.
    inertial: bool = False
    life_frames: float = 0.0
    turn_wait: int = 0

    @property
    def auto_aims(self) -> bool:
        """Lasers and areas that pick their own target need no aiming."""

        return self.tracking and self.kind in {"laser", "area"}

    @property
    def leads_target(self) -> bool:
        return self.kind == "projectile" and not self.tracking and self.speed > 0

    @property
    def pointing_tolerance(self) -> float:
        if self.kind == "laser":
            return float(const.TURN_ANGLE)
        return float(const.TURN_ANGLE) / 2.0


@dataclass(frozen=True, slots=True)
class ShipSpec:
    name: str
    max_thrust: float
    a1_cost: float
    a2_cost: float
    hp_gain: float
    inertial: bool
    thrust_increment: float = 0.0
    thrust_wait: int = 0
    turn_wait: int = 0
    planet_reach: float = 0.0
    shot_reach: float = 0.0
    constant_speed: bool = False


_NO_WEAPON = WeaponSpec("", "none", 0.0, 0.0, 0.0, False, (0.0,))


def _ability_name(ship_name: str, action: int, form: str) -> str:
    if ship_name == "Mmrnmrhm" and action == 1:
        return "MmrnmrhmYWingA1" if form == FORM_YWING else "MmrnmrhmXFormA1"
    return f"{ship_name}A{action}"


def _weapon_range(ship_name: str, definition) -> float:
    if definition.range is not None:
        return float(definition.range)
    if definition.area_length is not None:
        return float(definition.area_length)
    if ship_name == "Umgah" and definition.action == "A1":
        return UMGAH_CONE_REACH
    if ship_name == "Slylandro" and definition.action == "A1":
        ship = SHIP_DEFINITIONS[ship_name]
        segment = float(definition.segment_length_max or 0.0)
        count = int(math.floor(float(ship.a1_wait) / 2.0)) + 1
        return segment * count
    speed = float(definition.speed or definition.base_speed or 0.0)
    life_time = float(definition.life_time or 0.0)
    if speed > 0 and life_time > 0:
        return speed * life_time * PROJECTILE_RANGE_MARGIN
    return 0.0


@lru_cache(maxsize=None)
def weapon_spec(ship_name: str, action: int, form: str = FORM_BASE) -> WeaponSpec:
    return ability_spec(_ability_name(ship_name, action, form))


@lru_cache(maxsize=None)
def ability_spec(ability_name: str) -> WeaponSpec:
    """Facts for one ability by its catalog name, such as ``"OrzA3"``."""

    definition = ABILITY_DEFINITIONS.get(ability_name)
    if definition is None:
        return _NO_WEAPON
    directions = tuple(float(value) for value in (definition.gun_directions or (0.0,)))
    return WeaponSpec(
        ability_name=ability_name,
        kind=str(definition.ability_type),
        range=_weapon_range(str(definition.ship_name), definition),
        speed=float(definition.speed or 0.0),
        parent_velocity=float(definition.parent_vel or 0.0),
        tracking=bool(definition.tracking),
        gun_directions=directions or (0.0,),
        damage=float(definition.damage[0]) if definition.damage else 0.0,
        inertial=bool(definition.inertia),
        life_frames=float(definition.life_time or 0.0),
        turn_wait=int(definition.turn_wait or 0),
    )


def shofixti_blast_damage(distance: float) -> int:
    """Crew the Shofixti glory device removes at a centre-to-centre distance."""

    definition = ABILITY_DEFINITIONS["ShofixtiA2"]
    reach = float(definition.range)
    distance = max(0.0, float(distance))
    if distance > reach:
        return 0
    return 1 + math.floor(float(definition.damage[0]) * (1.0 - distance / reach))


def shofixti_kill_distance(crew: float) -> float:
    """Farthest distance at which the glory device removes ``crew`` crew.

    Returns minus infinity when even a blast at contact falls short.
    """

    definition = ABILITY_DEFINITIONS["ShofixtiA2"]
    reach = float(definition.range)
    needed = math.ceil(float(crew) - 1e-6)
    if needed <= 1:
        return reach
    fraction = 1.0 - (needed - 1) / float(definition.damage[0])
    if fraction < 0.0:
        return -math.inf
    return reach * fraction


@lru_cache(maxsize=None)
def ship_spec(ship_name: str, form: str = FORM_BASE) -> ShipSpec:
    ship = SHIP_DEFINITIONS[ship_name]
    max_thrust = float(ship.max_thrust)
    a1_cost = float(ship.a1_cost)
    a2_cost = float(ship.a2_cost)
    inertial = bool(ship.inertia)
    thrust_increment = float(ship.thrust_increment)
    thrust_wait = int(ship.thrust_wait)
    turn_wait = int(ship.turn_wait)
    forms = getattr(ship, "forms", None) or {}
    form_key = {
        FORM_BLAZER: "A2",
        FORM_XFORM: "XForm",
        FORM_YWING: "YWing",
    }.get(form)
    form_definition = forms.get(form_key) if form_key else None
    if form_definition is not None:
        if form_definition.max_thrust is not None:
            max_thrust = float(form_definition.max_thrust)
        if form_definition.a1_cost is not None:
            a1_cost = float(form_definition.a1_cost)
        if form_definition.inertia is not None:
            inertial = bool(form_definition.inertia)
        if form_definition.thrust_increment is not None:
            thrust_increment = float(form_definition.thrust_increment)
        if form_definition.thrust_wait is not None:
            thrust_wait = int(form_definition.thrust_wait)
        if form_definition.turn_wait is not None:
            turn_wait = int(form_definition.turn_wait)
    a2_definition = ABILITY_DEFINITIONS.get(f"{ship_name}A2")
    hp_gain = float(getattr(a2_definition, "hp_gain", 0.0) or 0.0)
    return ShipSpec(
        name=ship_name,
        max_thrust=max_thrust,
        a1_cost=a1_cost,
        a2_cost=a2_cost,
        hp_gain=hp_gain,
        inertial=inertial,
        thrust_increment=thrust_increment,
        thrust_wait=thrust_wait,
        turn_wait=turn_wait,
        planet_reach=PLANET_HULL_REACH.get(ship_name, TYPICAL_HULL_RADIUS),
        shot_reach=SHOT_HULL_REACH.get(ship_name, TYPICAL_HULL_RADIUS),
        constant_speed=ship_name in CONSTANT_SPEED_SHIPS,
    )
