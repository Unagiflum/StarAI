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
        for second in objects[index + 1 :]:
            _dispatch_collision_pair(first, second, effects, environment)


def _dispatch_collision_pair(first, second, effects, environment=None):
    if environment is None:
        environment = CollisionEnvironment()

    phys_first = getattr(first, "physical_collision_capabilities", None)
    phys_second = getattr(second, "physical_collision_capabilities", None)

    if (phys_first and phys_first.is_intangible) or (phys_second and phys_second.is_intangible):
        return True

    return _resolve_generic_collision(first, second, effects, environment)


def _resolve_generic_collision(first, second, effects, environment):
    return responses.resolve_generic_collision(
        first,
        second,
        effects,
        environment,
        object_on_screen_policy=_object_on_screen,
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

    ships = [ship for ship in world.live_ships if id(ship) not in excluded_ids]
    projectiles = world.colliding_projectiles
    special_objects = world.colliding_fighters
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
    _handle_ship_ship_collisions(ships, effects)
    _handle_ship_asteroid_collisions(ships, asteroids, effects)
    _handle_asteroid_asteroid_collisions(asteroids, effects)
    _handle_ship_planet_collisions(ships, planets)
    _handle_asteroid_planet_collisions(asteroids, planets, ships, effects)
    _handle_projectile_projectile_collisions(projectiles, effects)
    _handle_projectile_ship_collisions(projectiles, ships, effects)
    _handle_projectile_asteroid_collisions(projectiles, asteroids, effects)
    _handle_projectile_planet_collisions(projectiles, planets, effects)
    _handle_fighter_fighter_collisions(special_objects, effects)
    _handle_fighter_projectile_collisions(special_objects, projectiles, effects)
    _handle_fighter_ship_collisions(special_objects, ships, effects)
    _handle_fighter_asteroid_collisions(special_objects, asteroids, effects)
    _handle_fighter_planet_collisions(special_objects, planets)
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
            if not responses.generic_area_damage_target_is_eligible(
                ability, target
            ):
                continue

            delta = _wrapped_delta(ability.position, target.position)
            distance = math.hypot(delta[0], delta[1])
            damage = ability.area_damage_for_target(target, distance)
            if damage <= 0:
                continue

            applied_damage = responses.apply_generic_area_damage(
                ability, target, effects, delta, distance, damage
            )
            if ability.area_damage_capabilities.plays_impact_sound:
                BattleEffect.play_boom(damage)
            ability.on_area_damage_hit(target, applied_damage)

        ability.area_damage_pending = bool(
            ability.area_damage_capabilities.persistent and ability.currently_alive
        )


def _handle_ship_ship_collisions(ships, effects=None):
    if effects is None:
        effects = []
    _dispatch_unique_collision_pairs(ships, effects, lambda ship: True)


def _handle_ship_asteroid_collisions(ships, asteroids, effects=None):
    _dispatch_collision_pairs(
        ships,
        asteroids,
        effects if effects is not None else [],
        stop_after_handled=False,
    )


def _handle_asteroid_asteroid_collisions(asteroids, effects):
    _dispatch_unique_collision_pairs(
        asteroids,
        effects,
        lambda asteroid: asteroid.currently_alive,
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
    _dispatch_unique_collision_pairs(projectiles, effects, responses.is_live_projectile)


def _handle_projectile_ship_collisions(projectiles, ships, effects):
    _dispatch_collision_pairs(projectiles, ships, effects)


def _handle_projectile_asteroid_collisions(projectiles, asteroids, effects):
    _dispatch_collision_pairs(projectiles, asteroids, effects)


def _handle_projectile_planet_collisions(projectiles, planets, effects):
    _dispatch_collision_pairs(projectiles, planets, effects)


def _handle_fighter_fighter_collisions(special_objects, effects):
    _dispatch_unique_collision_pairs(special_objects, effects, responses.is_live_fighter)


def _handle_fighter_projectile_collisions(special_objects, projectiles, effects):
    _dispatch_collision_pairs(special_objects, projectiles, effects)


def _handle_fighter_ship_collisions(special_objects, ships, effects):
    _dispatch_collision_pairs(special_objects, ships, effects)


def _handle_fighter_asteroid_collisions(special_objects, asteroids, effects):
    _dispatch_collision_pairs(special_objects, asteroids, effects)


def _handle_fighter_planet_collisions(special_objects, planets):
    _dispatch_collision_pairs(special_objects, planets, [])


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
    responses.apply_generic_laser_impact(target, effects, normal, damage, contact)


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
    if not getattr(target.collision_capabilities, "role", None) == CollisionRole.SPECIAL_OBJECT:
        if not target.laser_target_capabilities.targetable:
            return False
    return responses.generic_is_laser_target(laser, target, explicit)


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
