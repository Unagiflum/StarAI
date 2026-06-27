"""Explicit collision identity shared by battle objects and collision dispatch."""

from dataclasses import dataclass
from enum import Enum, auto


class CollisionRole(Enum):
    NONE = auto()
    SHIP = auto()
    PROJECTILE = auto()
    SPECIAL_OBJECT = auto()
    LASER = auto()
    ASTEROID = auto()
    PLANET = auto()


@dataclass(frozen=True)
class CollisionCapabilities:
    role: CollisionRole = CollisionRole.NONE


@dataclass(frozen=True)
class LaserTargetCapabilities:
    targetable: bool = True
    vulnerable: bool = True


@dataclass(frozen=True)
class AreaDamageCapabilities:
    emits: bool = False
    targetable: bool = False
    vulnerable: bool = True
    persistent: bool = False
    plays_impact_sound: bool = False


@dataclass(frozen=True)
class SpecialObjectCollisionCapabilities:
    collides_with_planets: bool = True
    collides_with_asteroids: bool = True
    damages_asteroids: bool = True
    collides_with_projectiles: bool = True
    damages_projectiles: bool = True
    collides_with_enemy_ships: bool = True
    collides_with_friendly_ships: bool = False
    collides_with_fighters: bool = True


@dataclass(frozen=True)
class ShipImpactContext:
    normal: tuple[float, float]
    distance: float
    overlap: float
    closing_speed: float


@dataclass(frozen=True)
class ShipImpactResult:
    damage_to_other: float = 0.0


@dataclass(frozen=True)
class PhysicalCollisionCapabilities:
    is_solid: bool = True
    is_intangible: bool = False
    is_immovable: bool = False
    is_fragile: bool = False
    fragile_to_immovable: bool = False
    bounces_on_immovable: bool = False
    is_projectile: bool = False


@dataclass(frozen=True)
class DurabilityCapabilities:
    is_invulnerable: bool = False
    immune_to_psychic: bool = False


@dataclass(frozen=True)
class ImpactCapabilities:
    impact_damage_percent: float = 0.0
    ramming_damage: float = 0.0
