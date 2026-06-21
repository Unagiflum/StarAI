"""Ordered collision pipeline."""

import math
from dataclasses import dataclass

from src.Battle import collision_responses as responses
from src.Battle.collision_geometry import (
    distance_between,
    laser_hit_info,
)
# BattleEffect remains exposed here for existing callers and test patches.
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.collision_capabilities import CollisionRole
from src.Objects.Space.space_obj import Asteroid
from src.toroidal import (
    wrapped_delta as _wrapped_delta,
)


@dataclass(frozen=True)
class CollisionEnvironment:
    ships: tuple = ()


def _object_on_screen(obj, ships):
    return responses.object_on_screen(obj, ships)


def _asteroid_impacts_planet(asteroid, planet, effects, environment):
    """Supply the collision pipeline's visibility policy to the response."""
    return responses.asteroid_impacts_planet(
        asteroid,
        planet,
        effects,
        environment,
        object_on_screen_policy=_object_on_screen,
    )


_PAIR_COLLISION_HANDLERS = {
    (CollisionRole.SHIP, CollisionRole.SHIP): responses.ship_impacts_ship,
    (CollisionRole.SHIP, CollisionRole.ASTEROID): responses.ship_impacts_asteroid,
    (CollisionRole.SHIP, CollisionRole.PLANET): responses.ship_impacts_planet,
    (CollisionRole.ASTEROID, CollisionRole.PLANET): _asteroid_impacts_planet,
    (CollisionRole.PROJECTILE, CollisionRole.PROJECTILE): responses.projectile_impacts_projectile,
    (CollisionRole.PROJECTILE, CollisionRole.SHIP): responses.projectile_impacts_ship,
    (CollisionRole.PROJECTILE, CollisionRole.ASTEROID): responses.projectile_impacts_asteroid,
    (CollisionRole.PROJECTILE, CollisionRole.PLANET): responses.projectile_impacts_planet,
    (CollisionRole.FIGHTER, CollisionRole.FIGHTER): responses.fighter_impacts_fighter,
    (CollisionRole.FIGHTER, CollisionRole.PROJECTILE): responses.fighter_impacts_projectile,
    (CollisionRole.FIGHTER, CollisionRole.SHIP): responses.fighter_impacts_ship,
    (CollisionRole.FIGHTER, CollisionRole.ASTEROID): responses.fighter_impacts_asteroid,
    (CollisionRole.FIGHTER, CollisionRole.PLANET): responses.fighter_impacts_planet,
}

_LASER_TARGET_POLICIES = {
    CollisionRole.SHIP: responses.ship_is_laser_target,
    CollisionRole.PROJECTILE: responses.projectile_is_laser_target,
    CollisionRole.FIGHTER: responses.fighter_is_laser_target,
    CollisionRole.ASTEROID: responses.asteroid_is_laser_target,
    CollisionRole.PLANET: responses.planet_is_laser_target,
    CollisionRole.NONE: responses.generic_is_laser_target,
}

_LASER_IMPACT_POLICIES = {
    CollisionRole.SHIP: responses.laser_impacts_ship,
    CollisionRole.PROJECTILE: responses.laser_impacts_ability,
    CollisionRole.FIGHTER: responses.laser_impacts_ability,
    CollisionRole.ASTEROID: responses.laser_impacts_asteroid,
    CollisionRole.PLANET: responses.laser_impacts_planet,
}

_AREA_DAMAGE_IMPACT_POLICIES = {
    CollisionRole.SHIP: responses.area_damage_impacts_ship,
    CollisionRole.NONE: responses.area_damage_impacts_ability,
    CollisionRole.PROJECTILE: responses.area_damage_impacts_ability,
    CollisionRole.FIGHTER: responses.area_damage_impacts_ability,
    CollisionRole.ASTEROID: responses.area_damage_impacts_asteroid,
}


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
    for first in first_objects:
        for second in second_objects:
            handled = _dispatch_collision_pair(first, second, effects, environment)
            if handled and stop_after_handled:
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
    for index, first in enumerate(objects):
        if not first_is_active(first):
            continue
        for second in objects[index + 1:]:
            _dispatch_collision_pair(first, second, effects, environment)


def _dispatch_collision_pair(first, second, effects, environment=None):
    if environment is None:
        environment = CollisionEnvironment()
    pair = (
        first.collision_capabilities.role,
        second.collision_capabilities.role,
    )
    handler = _PAIR_COLLISION_HANDLERS.get(pair)
    if handler is None:
        return False
    return handler(first, second, effects, environment)


def handle_collisions(game_objects, *, rng=None, resources=None):
    world = World.coerce(game_objects)
    effects = []
    all_asteroids = world.asteroids
    _handle_area_damage(world, effects)

    ships = world.live_ships
    projectiles = world.colliding_projectiles
    fighters = world.colliding_fighters
    lasers = world.colliding_lasers
    asteroids = world.live_asteroids
    planets = world.planets

    _handle_laser_collisions(lasers, ships, projectiles, fighters, asteroids, planets, effects)
    _handle_ship_ship_collisions(ships, effects)
    _handle_ship_asteroid_collisions(ships, asteroids)
    _handle_ship_planet_collisions(ships, planets)
    _handle_asteroid_planet_collisions(asteroids, planets, ships, effects)
    _handle_projectile_projectile_collisions(projectiles, effects)
    _handle_projectile_ship_collisions(projectiles, ships, effects)
    _handle_projectile_asteroid_collisions(projectiles, asteroids, effects)
    _handle_projectile_planet_collisions(projectiles, planets, effects)
    _handle_fighter_fighter_collisions(fighters, effects)
    _handle_fighter_projectile_collisions(fighters, projectiles, effects)
    _handle_fighter_ship_collisions(fighters, ships, effects)
    _handle_fighter_asteroid_collisions(fighters, asteroids, effects)
    _handle_fighter_planet_collisions(fighters, planets)
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


def _handle_area_damage(game_objects, effects):
    world = World.coerce(game_objects)
    area_abilities = world.pending_area_damage

    for ability in area_abilities:
        ability.area_damage_pending = False
        for target in world:
            if not responses.area_damage_target_is_eligible(
                ability, target, _AREA_DAMAGE_IMPACT_POLICIES
            ):
                continue

            delta = _wrapped_delta(ability.position, target.position)
            distance = math.hypot(delta[0], delta[1])
            damage = ability.damage_at_distance(distance)
            if damage <= 0:
                continue

            policy = _AREA_DAMAGE_IMPACT_POLICIES[
                target.collision_capabilities.role
            ]
            policy(target, effects, delta, distance, damage)

def _handle_ship_ship_collisions(ships, effects=None):
    if effects is None:
        effects = []
    _dispatch_unique_collision_pairs(ships, effects, lambda ship: True)


def _handle_ship_asteroid_collisions(ships, asteroids):
    _dispatch_collision_pairs(
        ships,
        asteroids,
        [],
        stop_after_handled=False,
    )


def _handle_ship_planet_collisions(ships, planets):
    _dispatch_collision_pairs(
        ships,
        planets,
        [],
        stop_after_handled=False,
    )


def _handle_asteroid_planet_collisions(asteroids, planets, ships, effects):
    _dispatch_collision_pairs(
        asteroids,
        planets,
        effects,
        environment=CollisionEnvironment(ships=tuple(ships)),
    )


def _handle_projectile_projectile_collisions(projectiles, effects):
    _dispatch_unique_collision_pairs(
        projectiles, effects, responses.is_live_projectile
    )


def _handle_projectile_ship_collisions(projectiles, ships, effects):
    _dispatch_collision_pairs(projectiles, ships, effects)


def _handle_projectile_asteroid_collisions(projectiles, asteroids, effects):
    _dispatch_collision_pairs(projectiles, asteroids, effects)


def _handle_projectile_planet_collisions(projectiles, planets, effects):
    _dispatch_collision_pairs(projectiles, planets, effects)


def _handle_fighter_fighter_collisions(fighters, effects):
    _dispatch_unique_collision_pairs(fighters, effects, responses.is_live_fighter)


def _handle_fighter_projectile_collisions(fighters, projectiles, effects):
    _dispatch_collision_pairs(fighters, projectiles, effects)


def _handle_fighter_ship_collisions(fighters, ships, effects):
    _dispatch_collision_pairs(fighters, ships, effects)


def _handle_fighter_asteroid_collisions(fighters, asteroids, effects):
    _dispatch_collision_pairs(fighters, asteroids, effects)


def _handle_fighter_planet_collisions(fighters, planets):
    _dispatch_collision_pairs(fighters, planets, [])


def _handle_laser_collisions(lasers, ships, projectiles, fighters, asteroids, planets, effects):
    for laser in lasers:
        if not responses.is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        laser.calculate_end_position()

        targets = _laser_targets(laser, ships, projectiles, fighters, asteroids, planets)
        hit_infos = [
            hit_info for hit_info in (laser_hit_info(laser, target) for target in targets)
            if hit_info is not None
        ]
        if not hit_infos:
            continue

        hit_info = min(hit_infos, key=lambda info: info["distance"])
        target = hit_info["target"]
        contact = hit_info["contact"]
        normal = hit_info["normal"]
        responses.resolve_laser_hit(
            laser,
            target,
            effects,
            normal,
            contact,
            _apply_laser_impact,
        )


def _apply_laser_impact(target, effects, normal, damage, contact):
    policy = _LASER_IMPACT_POLICIES.get(target.collision_capabilities.role)
    if policy is not None:
        policy(target, effects, normal, damage, contact)


def _laser_targets(laser, ships, projectiles, fighters, asteroids, planets):
    explicit_target = laser.target
    if explicit_target is not None:
        targets = (
            [explicit_target]
            if _laser_target_is_eligible(laser, explicit_target, explicit=True)
            else []
        )
        targets.extend(
            fighter for fighter in fighters
            if (
                fighter is not explicit_target and
                _laser_target_is_eligible(laser, fighter)
            )
        )
        return targets

    targets = [
        target
        for target in (*ships, *projectiles, *fighters, *asteroids, *planets)
        if _laser_target_is_eligible(laser, target)
    ]
    return sorted(targets, key=lambda target: distance_between(laser.parent, target))


def _laser_target_is_eligible(laser, target, explicit=False):
    if not target.laser_target_capabilities.targetable:
        return False
    policy = _LASER_TARGET_POLICIES.get(target.collision_capabilities.role)
    if policy is None:
        return False
    return policy(laser, target, explicit)


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
        avoid_bodies.append(asteroid)
        world.add(asteroid)
