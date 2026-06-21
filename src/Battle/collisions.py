"""Ordered collision pipeline and its legacy compatibility surface."""

import math
from dataclasses import dataclass

from src.Battle import collision_responses as responses
# Legacy geometry helper names remain importable from this module.
from src.Battle.collision_geometry import (
    collision_info as _collision_info,
    collision_size as _collision_size,
    contact_point as _contact_point,
    distance_between as _distance_between,
    distance_from_segment_to_point as _distance_from_segment_to_point,
    estimated_impact as _estimated_impact,
    estimated_impact_at_positions as _estimated_impact_at_positions,
    get_collision_mask as _get_collision_mask,
    laser_hit_info as _laser_hit_info,
    mask_broadphase_overlap as _mask_broadphase_overlap,
    mask_broadphase_overlap_at_positions as _mask_broadphase_overlap_at_positions,
    mask_radius as _mask_radius,
    normal_from_target as _normal_from_target,
    objects_overlap as _objects_overlap,
    objects_overlap_at_positions as _objects_overlap_at_positions,
    projectile_impact as _projectile_impact,
    radius as _radius,
    sample_laser_mask_hit as _sample_laser_mask_hit,
    segment_circle_intercept as _segment_circle_intercept,
    segment_direction as _segment_direction,
    ship_rotation_blocked,
    sweep_previous_position as _sweep_previous_position,
    sweep_step_size as _sweep_step_size,
    swept_impact as _swept_impact,
    wrapped_segment as _wrapped_segment,
)
# BattleEffect remains exposed here for existing callers and test patches.
from src.Battle.effects import BattleEffect
# Legacy physics helper names remain importable from this module.
from src.Battle.collision_physics import (
    bounce_off_static_body as _bounce_off_static_body,
    dot as _dot,
    elastic_bounce as _elastic_bounce,
    mass as _mass,
    separate_dynamic_bodies as _separate_dynamic_bodies,
    separate_from_static_body as _separate_from_static_body,
    stop_at_static_body as _stop_at_static_body,
)
from src.Battle.world import World
from src.collision_capabilities import CollisionRole
from src.Objects.Space.space_obj import Asteroid
from src.toroidal import (
    wrapped_delta as _wrapped_delta,
)


PLANET_CONTACT_EXIT_MARGIN = 4.0
_PAIR_COLLISION_HANDLERS = {}
_LASER_TARGET_POLICIES = {}
_LASER_IMPACT_POLICIES = {}
_AREA_DAMAGE_IMPACT_POLICIES = {}


@dataclass(frozen=True)
class CollisionEnvironment:
    ships: tuple = ()


def _pair_collision_handler(first_role, second_role):
    """Register one ordered collision-pair policy."""
    def register(handler):
        pair = (first_role, second_role)
        if pair in _PAIR_COLLISION_HANDLERS:
            raise ValueError(f"Collision handler already registered for {pair}")
        _PAIR_COLLISION_HANDLERS[pair] = handler
        return handler

    return register


def _laser_target_policy(role):
    """Register one laser target-eligibility policy by collision role."""
    def register(policy):
        if role in _LASER_TARGET_POLICIES:
            raise ValueError(f"Laser target policy already registered for {role}")
        _LASER_TARGET_POLICIES[role] = policy
        return policy

    return register


def _laser_impact_policy(role):
    """Register one laser impact policy by collision role."""
    def register(policy):
        if role in _LASER_IMPACT_POLICIES:
            raise ValueError(f"Laser impact policy already registered for {role}")
        _LASER_IMPACT_POLICIES[role] = policy
        return policy

    return register


def _area_damage_impact_policy(role):
    """Register one area-damage impact policy by collision role."""
    def register(policy):
        if role in _AREA_DAMAGE_IMPACT_POLICIES:
            raise ValueError(f"Area-damage policy already registered for {role}")
        _AREA_DAMAGE_IMPACT_POLICIES[role] = policy
        return policy

    return register


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


def handle_collisions(game_objects):
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
    _spawn_replacement_asteroids(world, all_asteroids, ships, planets)

    world.add_all(effects)
    world.remove_dead_collision_objects()


def _handle_area_damage(game_objects, effects):
    world = World.coerce(game_objects)
    area_abilities = world.pending_area_damage

    for ability in area_abilities:
        ability.area_damage_pending = False
        for target in world:
            if not _area_damage_target_is_eligible(ability, target):
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


def _area_damage_target_is_eligible(source, target):
    return responses.area_damage_target_is_eligible(
        source, target, _AREA_DAMAGE_IMPACT_POLICIES
    )


@_area_damage_impact_policy(CollisionRole.SHIP)
def _area_damage_impacts_ship(target, effects, delta, distance, damage):
    return responses.area_damage_impacts_ship(
        target, effects, delta, distance, damage
    )


@_area_damage_impact_policy(CollisionRole.NONE)
@_area_damage_impact_policy(CollisionRole.PROJECTILE)
@_area_damage_impact_policy(CollisionRole.FIGHTER)
def _area_damage_impacts_ability(target, effects, delta, distance, damage):
    return responses.area_damage_impacts_ability(
        target, effects, delta, distance, damage
    )


@_area_damage_impact_policy(CollisionRole.ASTEROID)
def _area_damage_impacts_asteroid(target, effects, delta, distance, damage):
    return responses.area_damage_impacts_asteroid(
        target, effects, delta, distance, damage
    )


def _handle_ship_ship_collisions(ships, effects=None):
    if effects is None:
        effects = []
    _dispatch_unique_collision_pairs(ships, effects, lambda ship: True)


@_pair_collision_handler(CollisionRole.SHIP, CollisionRole.SHIP)
def _ship_impacts_ship(ship, other, effects, environment):
    return responses.ship_impacts_ship(ship, other, effects, environment)


def _apply_ship_impact_damage(ship, damage):
    return responses.apply_ship_impact_damage(ship, damage)


def _handle_ship_asteroid_collisions(ships, asteroids):
    _dispatch_collision_pairs(
        ships,
        asteroids,
        [],
        stop_after_handled=False,
    )


@_pair_collision_handler(CollisionRole.SHIP, CollisionRole.ASTEROID)
def _ship_impacts_asteroid(ship, asteroid, effects, environment):
    return responses.ship_impacts_asteroid(
        ship, asteroid, effects, environment
    )


def _handle_ship_planet_collisions(ships, planets):
    _dispatch_collision_pairs(
        ships,
        planets,
        [],
        stop_after_handled=False,
    )


@_pair_collision_handler(CollisionRole.SHIP, CollisionRole.PLANET)
def _ship_impacts_planet(ship, planet, effects, environment):
    return responses.ship_impacts_planet(ship, planet, effects, environment)


def _planet_contact_has_ended(ship, planet, distance):
    return responses.planet_contact_has_ended(
        ship, planet, distance, PLANET_CONTACT_EXIT_MARGIN
    )


def _handle_asteroid_planet_collisions(asteroids, planets, ships, effects):
    _dispatch_collision_pairs(
        asteroids,
        planets,
        effects,
        environment=CollisionEnvironment(ships=tuple(ships)),
    )


@_pair_collision_handler(CollisionRole.ASTEROID, CollisionRole.PLANET)
def _asteroid_impacts_planet(asteroid, planet, effects, environment):
    return responses.asteroid_impacts_planet(
        asteroid,
        planet,
        effects,
        environment,
        object_on_screen_policy=_object_on_screen,
    )


def _handle_projectile_projectile_collisions(projectiles, effects):
    _dispatch_unique_collision_pairs(projectiles, effects, _is_live_projectile)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.PROJECTILE)
def _projectile_impacts_projectile(projectile, other, effects, environment):
    return responses.projectile_impacts_projectile(
        projectile, other, effects, environment
    )


def _projectiles_can_hit_each_other(projectile, other):
    return responses.projectiles_can_hit_each_other(
        projectile, other
    )


def _handle_projectile_ship_collisions(projectiles, ships, effects):
    _dispatch_collision_pairs(projectiles, ships, effects)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.SHIP)
def _projectile_impacts_ship(projectile, ship, effects, environment):
    return responses.projectile_impacts_ship(
        projectile, ship, effects, environment
    )


def _handle_projectile_asteroid_collisions(projectiles, asteroids, effects):
    _dispatch_collision_pairs(projectiles, asteroids, effects)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.ASTEROID)
def _projectile_impacts_asteroid(projectile, asteroid, effects, environment):
    return responses.projectile_impacts_asteroid(
        projectile, asteroid, effects, environment
    )


def _handle_projectile_planet_collisions(projectiles, planets, effects):
    _dispatch_collision_pairs(projectiles, planets, effects)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.PLANET)
def _projectile_impacts_planet(projectile, planet, effects, environment):
    return responses.projectile_impacts_planet(
        projectile, planet, effects, environment
    )


def _handle_fighter_fighter_collisions(fighters, effects):
    _dispatch_unique_collision_pairs(fighters, effects, _is_live_fighter)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.FIGHTER)
def _fighter_impacts_fighter(fighter, other, effects, environment):
    return responses.fighter_impacts_fighter(
        fighter, other, effects, environment
    )


def _handle_fighter_projectile_collisions(fighters, projectiles, effects):
    _dispatch_collision_pairs(fighters, projectiles, effects)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.PROJECTILE)
def _fighter_impacts_projectile(fighter, projectile, effects, environment):
    return responses.fighter_impacts_projectile(
        fighter, projectile, effects, environment
    )


def _handle_fighter_ship_collisions(fighters, ships, effects):
    _dispatch_collision_pairs(fighters, ships, effects)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.SHIP)
def _fighter_impacts_ship(fighter, ship, effects, environment):
    return responses.fighter_impacts_ship(
        fighter, ship, effects, environment
    )


def _handle_fighter_asteroid_collisions(fighters, asteroids, effects):
    _dispatch_collision_pairs(fighters, asteroids, effects)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.ASTEROID)
def _fighter_impacts_asteroid(fighter, asteroid, effects, environment):
    return responses.fighter_impacts_asteroid(
        fighter, asteroid, effects, environment
    )


def _handle_fighter_planet_collisions(fighters, planets):
    _dispatch_collision_pairs(fighters, planets, [])


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.PLANET)
def _fighter_impacts_planet(fighter, planet, effects, environment):
    return responses.fighter_impacts_planet(
        fighter, planet, effects, environment
    )


def _is_live_projectile(obj):
    return responses.is_live_projectile(obj)


def _is_live_fighter(obj):
    return responses.is_live_fighter(obj)


def _is_live_laser(obj):
    return responses.is_live_laser(obj)


def _projectile_can_hit_ship(projectile, ship):
    return responses.projectile_can_hit_ship(projectile, ship)


def _handle_laser_collisions(lasers, ships, projectiles, fighters, asteroids, planets, effects):
    for laser in lasers:
        if not _is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        laser.calculate_end_position()

        targets = _laser_targets(laser, ships, projectiles, fighters, asteroids, planets)
        hit_infos = [
            hit_info for hit_info in (_laser_hit_info(laser, target) for target in targets)
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


@_laser_impact_policy(CollisionRole.SHIP)
def _laser_impacts_ship(target, effects, normal, damage, contact):
    return responses.laser_impacts_ship(
        target, effects, normal, damage, contact
    )


@_laser_impact_policy(CollisionRole.PROJECTILE)
@_laser_impact_policy(CollisionRole.FIGHTER)
def _laser_impacts_ability(target, effects, normal, damage, contact):
    return responses.laser_impacts_ability(
        target, effects, normal, damage, contact
    )


@_laser_impact_policy(CollisionRole.ASTEROID)
def _laser_impacts_asteroid(target, effects, normal, damage, contact):
    return responses.laser_impacts_asteroid(
        target, effects, normal, damage, contact
    )


@_laser_impact_policy(CollisionRole.PLANET)
def _laser_impacts_planet(target, effects, normal, damage, contact):
    return responses.laser_impacts_planet(
        target, effects, normal, damage, contact
    )


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
    return sorted(targets, key=lambda target: _distance_between(laser.parent, target))


def _laser_target_is_eligible(laser, target, explicit=False):
    if not target.laser_target_capabilities.targetable:
        return False
    policy = _LASER_TARGET_POLICIES.get(target.collision_capabilities.role)
    if policy is None:
        return False
    return policy(laser, target, explicit)


@_laser_target_policy(CollisionRole.SHIP)
def _ship_is_laser_target(laser, target, explicit):
    return responses.ship_is_laser_target(laser, target, explicit)


@_laser_target_policy(CollisionRole.PROJECTILE)
def _projectile_is_laser_target(laser, target, explicit):
    return responses.projectile_is_laser_target(laser, target, explicit)


@_laser_target_policy(CollisionRole.FIGHTER)
def _fighter_is_laser_target(laser, target, explicit):
    return responses.fighter_is_laser_target(laser, target, explicit)


@_laser_target_policy(CollisionRole.ASTEROID)
def _asteroid_is_laser_target(laser, target, explicit):
    return responses.asteroid_is_laser_target(laser, target, explicit)


@_laser_target_policy(CollisionRole.PLANET)
def _planet_is_laser_target(laser, target, explicit):
    return responses.planet_is_laser_target(laser, target, explicit)


@_laser_target_policy(CollisionRole.NONE)
def _generic_is_laser_target(laser, target, explicit):
    return responses.generic_is_laser_target(laser, target, explicit)


def _remove_dead_collision_objects(game_objects):
    World.coerce(game_objects).remove_dead_collision_objects()


def _spawn_replacement_asteroids(game_objects, asteroids, ships, planets):
    world = World.coerce(game_objects)
    if not planets:
        return

    dead_count = sum(1 for asteroid in asteroids if not asteroid.currently_alive)
    if dead_count <= 0:
        return

    planet = planets[0]
    avoid_bodies = world.asteroid_spawn_avoid_bodies

    for _ in range(dead_count):
        asteroid = Asteroid()
        asteroid.set_planet(planet)
        asteroid.position = asteroid.get_respawn_position(planet, ships, avoid_bodies)
        avoid_bodies.append(asteroid)
        world.add(asteroid)


def _destroy_projectile(projectile, effects, direction, damage, contact_position=None):
    return responses.destroy_projectile(
        projectile, effects, direction, damage, contact_position
    )


def _destroy_asteroid(asteroid, effects):
    return responses.destroy_asteroid(asteroid, effects)


def _object_on_screen(obj, ships):
    return responses.object_on_screen(obj, ships)


def _set_projectile_hp(projectile, hp):
    return responses.set_projectile_hp(projectile, hp)
