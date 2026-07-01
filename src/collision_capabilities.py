"""Explicit collision identity shared by battle objects and collision dispatch."""

from dataclasses import dataclass
from enum import Enum, auto


class CollisionRole(Enum):
    NONE = auto()
    SHIP = auto()
    PROJECTILE = auto()
    SPECIAL_OBJECT = auto()
    LASER = auto()
    AREA = auto()
    ASTEROID = auto()
    PLANET = auto()


class ProjectileContactPolicy(Enum):
    """How a special object responds after an eligible projectile contact."""

    DEFAULT = auto()
    FRAGILE = auto()
    TAKE_DAMAGE = auto()
    TAKE_DAMAGE_AND_DESTROY_PROJECTILE = auto()


class SameTypeContactPolicy(Enum):
    """How special objects of the same gameplay type contact one another."""

    DEFAULT = auto()
    IGNORE = auto()
    BOUNCE = auto()


class SpecialObjectPairOutcome(Enum):
    """Resolved response for one ordered pair of special objects."""

    DEFAULT = auto()
    IGNORE = auto()
    DESTROY_FIRST = auto()
    DESTROY_SECOND = auto()
    BOUNCE_BOTH = auto()
    BOUNCE_FIRST = auto()
    BOUNCE_SECOND = auto()


@dataclass(frozen=True)
class CollisionCapabilities:
    role: CollisionRole = CollisionRole.NONE


@dataclass(frozen=True)
class LaserTargetCapabilities:
    targetable: bool = True
    vulnerable: bool = True
    blocks_lasers: bool = True


@dataclass(frozen=True)
class AreaDamageCapabilities:
    emits: bool = False
    targetable: bool = False
    vulnerable: bool = True
    persistent: bool = False
    plays_impact_sound: bool = False
    immune_to_sources: frozenset[str] = frozenset()


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
    bounces_off_same_type: bool = False
    bounces_off_ships_without_damage: bool = False
    destroys_fragile: bool = False
    projectile_contact_policy: ProjectileContactPolicy = (
        ProjectileContactPolicy.DEFAULT
    )
    same_type_contact_policy: SameTypeContactPolicy = SameTypeContactPolicy.DEFAULT


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
