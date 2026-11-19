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
# All planets share one diameter.
PLANET_RADIUS = 150.0
# Contact margin for "collision imminent" checks: two typical hull radii.
COLLISION_MARGIN = 60.0
# A hull's typical reach from its centre: the mean over all ships of half
# way between the half-width and half-length of the collision mask (from 23
# for the Shofixti to 69 for the Chmmr).
TYPICAL_HULL_RADIUS = 45.0
# The Umgah cone is mask-shaped and has no configured length.
UMGAH_CONE_REACH = 180.0

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
    definition = ABILITY_DEFINITIONS.get(_ability_name(ship_name, action, form))
    if definition is None:
        return _NO_WEAPON
    directions = tuple(float(value) for value in (definition.gun_directions or (0.0,)))
    return WeaponSpec(
        ability_name=_ability_name(ship_name, action, form),
        kind=str(definition.ability_type),
        range=_weapon_range(ship_name, definition),
        speed=float(definition.speed or 0.0),
        parent_velocity=float(definition.parent_vel or 0.0),
        tracking=bool(definition.tracking),
        gun_directions=directions or (0.0,),
        damage=float(definition.damage[0]) if definition.damage else 0.0,
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
    a2_definition = ABILITY_DEFINITIONS.get(f"{ship_name}A2")
    hp_gain = float(getattr(a2_definition, "hp_gain", 0.0) or 0.0)
    return ShipSpec(
        name=ship_name,
        max_thrust=max_thrust,
        a1_cost=a1_cost,
        a2_cost=a2_cost,
        hp_gain=hp_gain,
        inertial=inertial,
    )
