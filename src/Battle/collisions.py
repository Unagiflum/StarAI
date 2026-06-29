"""Ordered collision pipeline."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from src.Battle import collision_responses as responses
from src.Battle.area_dispatch import AreaTargetRegistry
from src.Battle.collision_contract import (
    CollisionContext,
    CollisionEnvironment,
    CollisionOutcome,
    collision_context,
)
from src.Battle.collision_dispatch import CollisionPairRegistry
from src.Battle.collision_geometry import laser_hit_info
from src.Battle.laser_dispatch import LaserTargetRegistry

# BattleEffect remains exposed here for existing callers and test patches.
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.collision_capabilities import CollisionRole
from src.Objects.Space.space_obj import Asteroid
from src.toroidal import (
    wrapped_delta as _wrapped_delta,
)


def _dispatch_collision_pairs(
    first_objects,
    second_objects,
    effects,
    *,
    stop_after_handled=True,
    environment=None,
):
    """Dispatch ordered object pairs by their explicit collision roles."""
    if environment is None:
        environment = CollisionEnvironment()
    context = CollisionContext(effects, environment)
    for first in first_objects:
        for second in second_objects:
            outcome = _dispatch_collision_pair(first, second, context)
            if outcome.handled and stop_after_handled:
                break


def _dispatch_unique_collision_pairs(
    objects,
    effects,
    first_is_active,
    environment=None,
):
    """Dispatch each unordered pair while preserving outer-loop activity rules."""
    if environment is None:
        environment = CollisionEnvironment()
    context = CollisionContext(effects, environment)
    for index, first in enumerate(objects):
        if not first_is_active(first):
            continue
        for second in objects[index + 1 :]:
            _dispatch_collision_pair(first, second, context)


def _dispatch_collision_pair(first, second, context_or_effects, environment=None):
    context = collision_context(context_or_effects, environment)

    phys_first = getattr(first, "physical_collision_capabilities", None)
    phys_second = getattr(second, "physical_collision_capabilities", None)

    if (phys_first and phys_first.is_intangible) or (phys_second and phys_second.is_intangible):
        # Preserve the existing stop-after-handled behavior until pair policies
        # explicitly define how intangible objects affect candidate scanning.
        return CollisionOutcome.RESOLVED

    return COLLISION_PAIR_REGISTRY.dispatch(first, second, context)


def _create_collision_pair_registry():
    registry = CollisionPairRegistry()
    registry.register(
        CollisionRole.SHIP,
        CollisionRole.SHIP,
        responses.resolve_ship_ship_collision,
    )
    registry.register(
        CollisionRole.ASTEROID,
        CollisionRole.ASTEROID,
        responses.resolve_asteroid_asteroid_collision,
    )
    registry.register(
        CollisionRole.SHIP,
        CollisionRole.ASTEROID,
        responses.resolve_ship_asteroid_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.ASTEROID,
        CollisionRole.PLANET,
        responses.resolve_asteroid_planet_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SHIP,
        CollisionRole.PLANET,
        responses.resolve_ship_planet_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.PROJECTILE,
        responses.resolve_projectile_projectile_collision,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.SHIP,
        responses.resolve_projectile_ship_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.ASTEROID,
        responses.resolve_projectile_asteroid_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.PLANET,
        responses.resolve_projectile_planet_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.SPECIAL_OBJECT,
        responses.resolve_projectile_projectile_collision,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.PROJECTILE,
        responses.resolve_projectile_projectile_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.SHIP,
        responses.resolve_projectile_ship_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.ASTEROID,
        responses.resolve_projectile_asteroid_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.PLANET,
        responses.resolve_projectile_planet_collision,
        canonical_order=True,
    )
    return registry


COLLISION_PAIR_REGISTRY = _create_collision_pair_registry()


def _create_area_target_registry():
    registry = AreaTargetRegistry()
    registry.register(
        CollisionRole.SHIP,
        is_eligible=responses.ship_is_area_target,
        apply_damage=responses.apply_ship_area_damage,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        is_eligible=responses.projectile_is_area_target,
        apply_damage=responses.apply_projectile_area_damage,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        is_eligible=responses.special_object_is_area_target,
        apply_damage=responses.apply_special_object_area_damage,
    )
    registry.register(
        CollisionRole.ASTEROID,
        is_eligible=responses.asteroid_is_area_target,
        apply_damage=responses.apply_asteroid_area_damage,
    )
    registry.register(
        CollisionRole.PLANET,
        is_eligible=responses.planet_is_area_target,
        apply_damage=responses.apply_planet_area_damage,
    )
    return registry


AREA_TARGET_REGISTRY = _create_area_target_registry()


def _create_laser_target_registry():
    registry = LaserTargetRegistry()
    registry.register(
        CollisionRole.SHIP,
        is_eligible=responses.ship_is_laser_target,
        apply_impact=responses.apply_ship_laser_impact,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        is_eligible=responses.projectile_is_laser_target,
        apply_impact=responses.apply_projectile_laser_impact,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        is_eligible=responses.special_object_is_laser_target,
        apply_impact=responses.apply_special_object_laser_impact,
    )
    registry.register(
        CollisionRole.ASTEROID,
        is_eligible=responses.asteroid_is_laser_target,
        apply_impact=responses.apply_asteroid_laser_impact,
    )
    registry.register(
        CollisionRole.PLANET,
        is_eligible=responses.planet_is_laser_target,
        apply_impact=responses.apply_planet_laser_impact,
    )
    return registry


LASER_TARGET_REGISTRY = _create_laser_target_registry()


@dataclass(frozen=True)
class _CollisionPhase:
    first_group: str
    second_group: str | None = None
    unique_pairs: bool = False
    first_is_active: Callable[[object], bool] | None = None
    stop_after_handled: bool = True
    collects_effects: bool = True
    includes_ship_environment: bool = False


def _always_active(obj):
    return True


def _currently_alive(obj):
    return obj.currently_alive


def _is_intangible(obj):
    physics = getattr(obj, "physical_collision_capabilities", None)
    return bool(physics and physics.is_intangible)


COLLISION_PHASES = (
    _CollisionPhase("ships", unique_pairs=True),
    _CollisionPhase("ships", "asteroids", stop_after_handled=False),
    _CollisionPhase(
        "asteroids",
        unique_pairs=True,
        first_is_active=_currently_alive,
    ),
    _CollisionPhase(
        "ships",
        "planets",
        stop_after_handled=False,
        collects_effects=False,
    ),
    _CollisionPhase(
        "asteroids",
        "planets",
        includes_ship_environment=True,
    ),
    _CollisionPhase(
        "projectiles",
        unique_pairs=True,
        first_is_active=responses.is_live_projectile_like,
    ),
    _CollisionPhase("projectiles", "ships"),
    _CollisionPhase("projectiles", "asteroids"),
    _CollisionPhase("projectiles", "planets"),
    _CollisionPhase(
        "special_objects",
        unique_pairs=True,
        first_is_active=responses.is_live_special_object,
    ),
    _CollisionPhase("special_objects", "projectiles"),
    _CollisionPhase("special_objects", "ships"),
    _CollisionPhase("special_objects", "asteroids"),
    _CollisionPhase(
        "special_objects",
        "planets",
        collects_effects=False,
    ),
)


def handle_collisions(
    game_objects,
    *,
    rng=None,
    resources=None,
    excluded_objects=(),
):
    world = World.coerce(game_objects)
    excluded_ids = {id(obj) for obj in excluded_objects}
    effects = []
    all_asteroids = world.asteroids
    _handle_area_damage(world, effects, excluded_ids)

    ships = [
        ship
        for ship in world.live_ships
        if (
            id(ship) not in excluded_ids
            and not _is_intangible(ship)
        )
    ]
    projectiles = world.colliding_projectiles
    special_objects = world.colliding_special_objects
    lasers = world.colliding_lasers
    asteroids = world.live_asteroids
    planets = world.planets

    _handle_laser_collisions(
        lasers,
        ships,
        projectiles,
        special_objects,
        asteroids,
        planets,
        effects,
        excluded_ids,
    )
    _run_collision_phases(
        {
            "ships": ships,
            "asteroids": asteroids,
            "projectiles": projectiles,
            "special_objects": special_objects,
            "planets": planets,
        },
        effects,
    )
    _spawn_replacement_asteroids(
        world,
        all_asteroids,
        ships,
        planets,
        rng=rng,
        resources=resources,
    )

    world.add_all(effects)
    world.remove_dead_collision_objects()


def _handle_area_damage(game_objects, effects, excluded_ids=frozenset()):
    world = World.coerce(game_objects)
    area_abilities = world.pending_area_damage

    for ability in area_abilities:
        ability.area_damage_pending = False
        for target in world:
            if id(target) in excluded_ids:
                continue
            if not AREA_TARGET_REGISTRY.is_eligible(ability, target):
                continue

            delta = _wrapped_delta(ability.position, target.position)
            distance = math.hypot(delta[0], delta[1])
            damage = ability.area_damage_for_target(target, distance)
            if damage <= 0:
                continue

            applied_damage = AREA_TARGET_REGISTRY.apply_damage(
                ability,
                target,
                effects,
                delta,
                distance,
                damage,
            )
            if ability.area_damage_capabilities.plays_impact_sound:
                BattleEffect.play_boom(damage)
            ability.on_area_damage_hit(target, applied_damage)

        ability.area_damage_pending = bool(
            ability.area_damage_capabilities.persistent and ability.currently_alive
        )


def _run_collision_phases(groups, effects):
    """Run physical contact phases in their gameplay-significant order."""
    for phase in COLLISION_PHASES:
        phase_effects = effects if phase.collects_effects else []
        environment = (
            CollisionEnvironment(ships=tuple(groups["ships"]))
            if phase.includes_ship_environment
            else None
        )
        first_objects = groups[phase.first_group]

        if phase.unique_pairs:
            _dispatch_unique_collision_pairs(
                first_objects,
                phase_effects,
                phase.first_is_active or _always_active,
                environment,
            )
            continue

        _dispatch_collision_pairs(
            first_objects,
            groups[phase.second_group],
            phase_effects,
            stop_after_handled=phase.stop_after_handled,
            environment=environment,
        )


def _handle_laser_collisions(
    lasers,
    ships,
    projectiles,
    special_objects,
    asteroids,
    planets,
    effects,
    excluded_ids=frozenset(),
):
    for laser in lasers:
        if not responses.is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        laser.calculate_end_position()
        if laser.target is not None:
            target_delta = _wrapped_delta(laser.position, laser.target.position)
            laser.end_position = [
                laser.position[0] + target_delta[0],
                laser.position[1] + target_delta[1],
            ]

        targets = _laser_targets(
            laser,
            ships,
            projectiles,
            special_objects,
            asteroids,
            planets,
            excluded_ids,
        )
        hit_infos = [
            hit_info
            for hit_info in (laser_hit_info(laser, target) for target in targets)
            if hit_info is not None
        ]
        if not hit_infos:
            continue

        for hit_info in sorted(hit_infos, key=lambda info: info["distance"]):
            target = hit_info["target"]
            responses.resolve_laser_hit(
                laser,
                target,
                effects,
                hit_info["normal"],
                hit_info["contact"],
                _apply_laser_impact,
                segment_index=hit_info.get("segment_index"),
            )
            target_capabilities = getattr(
                target, "laser_target_capabilities", None
            )
            if (
                target_capabilities is None
                or getattr(target_capabilities, "blocks_lasers", True)
            ):
                break


def _apply_laser_impact(target, effects, normal, damage, contact):
    LASER_TARGET_REGISTRY.apply_impact(
        target,
        effects,
        normal,
        damage,
        contact,
    )


def _laser_targets(
    laser,
    ships,
    projectiles,
    special_objects,
    asteroids,
    planets,
    excluded_ids=frozenset(),
):
    explicit_target = laser.target
    if explicit_target is not None:
        targets = (
            [explicit_target]
            if (
                id(explicit_target) not in excluded_ids
                and _laser_target_is_eligible(laser, explicit_target, explicit=True)
            )
            else []
        )
        targets.extend(
            special_object
            for special_object in special_objects
            if (
                special_object is not explicit_target
                and _laser_target_is_eligible(laser, special_object)
            )
        )
        return targets

    targets = [
        target
        for target in (*ships, *projectiles, *special_objects, *asteroids, *planets)
        if (id(target) not in excluded_ids and _laser_target_is_eligible(laser, target))
    ]
    return targets


def _laser_target_is_eligible(laser, target, explicit=False):
    target_filter = getattr(laser, "should_consider_laser_target", None)
    if target_filter is not None and not target_filter(target):
        return False
    return LASER_TARGET_REGISTRY.is_eligible(
        laser,
        target,
        explicit=explicit,
    )


def _spawn_replacement_asteroids(
    game_objects,
    asteroids,
    ships,
    planets,
    *,
    rng=None,
    resources=None,
):
    world = World.coerce(game_objects)
    if not planets:
        return

    dead_count = sum(1 for asteroid in asteroids if not asteroid.currently_alive)
    if dead_count <= 0:
        return

    planet = planets[0]
    avoid_bodies = world.asteroid_spawn_avoid_bodies

    for _ in range(dead_count):
        if resources is None and rng is None:
            asteroid = Asteroid()
        else:
            asteroid = Asteroid(resources=resources, rng=rng)
        asteroid.set_planet(planet)
        asteroid.position = asteroid.get_respawn_position(planet, ships, avoid_bodies)
        asteroid.previous_position = asteroid.position.copy()
        avoid_bodies.append(asteroid)
        world.add(asteroid)
