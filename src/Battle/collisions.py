import math
from dataclasses import dataclass

import src.const as const
from src.Battle.effects import BattleEffect
from src.collision_capabilities import CollisionRole, ShipImpactContext
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip
from src.toroidal import (
    nearest_position as _nearest_position,
    view_center_and_size as _view_center_and_size,
    wrapped_delta as _wrapped_delta,
    wrapped_distance,
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
    effects = []
    all_asteroids = [obj for obj in game_objects if isinstance(obj, Asteroid)]
    _handle_area_damage(game_objects, effects)

    ships = [
        obj for obj in game_objects
        if isinstance(obj, SpaceShip) and obj.currently_alive and obj.current_hp > 0
    ]
    projectiles = [obj for obj in game_objects if _is_live_projectile(obj)]
    fighters = [obj for obj in game_objects if _is_live_fighter(obj)]
    lasers = [obj for obj in game_objects if _is_live_laser(obj)]
    asteroids = [obj for obj in game_objects if isinstance(obj, Asteroid) and obj.currently_alive]
    planets = [obj for obj in game_objects if isinstance(obj, Planet)]

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
    _spawn_replacement_asteroids(game_objects, all_asteroids, ships, planets)

    game_objects.extend(effects)
    _remove_dead_collision_objects(game_objects)


def _handle_area_damage(game_objects, effects):
    area_abilities = [
        obj for obj in game_objects
        if (
            obj.area_damage_capabilities.emits and
            obj.currently_alive and
            obj.area_damage_pending
        )
    ]

    for ability in area_abilities:
        ability.area_damage_pending = False
        for target in game_objects:
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
    capabilities = target.area_damage_capabilities
    return (
        target is not source
        and capabilities.targetable
        and capabilities.vulnerable
        and target.currently_alive
        and target.collision_capabilities.role in _AREA_DAMAGE_IMPACT_POLICIES
    )


@_area_damage_impact_policy(CollisionRole.SHIP)
def _area_damage_impacts_ship(target, effects, delta, distance, damage):
    if target.current_hp <= 0:
        return
    target.current_hp = max(0, target.current_hp - damage)


@_area_damage_impact_policy(CollisionRole.NONE)
@_area_damage_impact_policy(CollisionRole.PROJECTILE)
@_area_damage_impact_policy(CollisionRole.FIGHTER)
def _area_damage_impacts_ability(target, effects, delta, distance, damage):
    if target.current_hp <= 0:
        return
    remaining_hp = target.current_hp - damage
    if remaining_hp <= 0:
        direction = (
            [delta[0] / distance, delta[1] / distance]
            if distance > 0 else [0, -1]
        )
        _destroy_projectile(target, effects, direction, damage)
    else:
        _set_projectile_hp(target, remaining_hp)


@_area_damage_impact_policy(CollisionRole.ASTEROID)
def _area_damage_impacts_asteroid(target, effects, delta, distance, damage):
    _destroy_asteroid(target, effects)


def _handle_ship_ship_collisions(ships, effects=None):
    if effects is None:
        effects = []
    _dispatch_unique_collision_pairs(ships, effects, lambda ship: True)


@_pair_collision_handler(CollisionRole.SHIP, CollisionRole.SHIP)
def _ship_impacts_ship(ship, other, effects, environment):
    normal, distance, overlap = _collision_info(ship, other)
    if not _objects_overlap(ship, other, overlap):
        return False

    closing_speed = max(0.0, -_dot([
        ship.velocity[0] - other.velocity[0],
        ship.velocity[1] - other.velocity[1],
    ], normal))
    impact = ShipImpactContext(
        normal=(normal[0], normal[1]),
        distance=distance,
        overlap=overlap,
        closing_speed=closing_speed,
    )
    other_impact = ShipImpactContext(
        normal=(-normal[0], -normal[1]),
        distance=distance,
        overlap=overlap,
        closing_speed=closing_speed,
    )

    _elastic_bounce(ship, other, normal, distance, overlap)

    ship_result = ship.on_ship_impact(other, impact)
    other_result = other.on_ship_impact(ship, other_impact)
    _apply_ship_impact_damage(other, ship_result.damage_to_other)
    _apply_ship_impact_damage(ship, other_result.damage_to_other)
    return True


def _apply_ship_impact_damage(ship, damage):
    damage = max(0.0, damage)
    if damage <= 0 or ship.current_hp <= 0:
        return
    ship.current_hp = max(0, ship.current_hp - damage)
    BattleEffect.play_boom(damage)


def _handle_ship_asteroid_collisions(ships, asteroids):
    _dispatch_collision_pairs(
        ships,
        asteroids,
        [],
        stop_after_handled=False,
    )


@_pair_collision_handler(CollisionRole.SHIP, CollisionRole.ASTEROID)
def _ship_impacts_asteroid(ship, asteroid, effects, environment):
    if not asteroid.currently_alive:
        return False

    normal, distance, overlap = _collision_info(ship, asteroid)
    if not _objects_overlap(ship, asteroid, overlap):
        return False

    _elastic_bounce(ship, asteroid, normal, distance, overlap)
    return True


def _handle_ship_planet_collisions(ships, planets):
    _dispatch_collision_pairs(
        ships,
        planets,
        [],
        stop_after_handled=False,
    )


@_pair_collision_handler(CollisionRole.SHIP, CollisionRole.PLANET)
def _ship_impacts_planet(ship, planet, effects, environment):
    normal, distance, overlap = _collision_info(ship, planet)
    contact_id = id(planet)
    objects_overlap = _objects_overlap(ship, planet, overlap)
    if (
        contact_id in ship.planet_contacts
        and not objects_overlap
        and _planet_contact_has_ended(ship, planet, distance)
    ):
        ship.planet_contacts.remove(contact_id)

    if not objects_overlap:
        return False

    new_contact = contact_id not in ship.planet_contacts
    ship.planet_contacts.add(contact_id)
    if new_contact:
        collided_while_approaching = _bounce_off_static_body(
            ship,
            planet,
            normal,
            overlap,
        )
    else:
        collided_while_approaching = False
        _stop_at_static_body(ship, planet, normal, overlap)
    if new_contact and collided_while_approaching and ship.current_hp > 0:
        damage = max(1, math.ceil(ship.current_hp * 0.15))
        ship.current_hp = max(0, ship.current_hp - damage)
        BattleEffect.play_boom(damage)
    return True


def _planet_contact_has_ended(ship, planet, distance):
    contact_distance = _radius(ship) + _radius(planet)
    return distance > contact_distance + PLANET_CONTACT_EXIT_MARGIN


def _handle_asteroid_planet_collisions(asteroids, planets, ships, effects):
    _dispatch_collision_pairs(
        asteroids,
        planets,
        effects,
        environment=CollisionEnvironment(ships=tuple(ships)),
    )


@_pair_collision_handler(CollisionRole.ASTEROID, CollisionRole.PLANET)
def _asteroid_impacts_planet(asteroid, planet, effects, environment):
    if not asteroid.currently_alive:
        return False

    normal, _, overlap = _collision_info(asteroid, planet)
    if not _objects_overlap(asteroid, planet, overlap):
        return False

    if _object_on_screen(asteroid, environment.ships):
        BattleEffect.play_boom(1)
    _destroy_asteroid(asteroid, effects)
    return True


def _handle_projectile_projectile_collisions(projectiles, effects):
    _dispatch_unique_collision_pairs(projectiles, effects, _is_live_projectile)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.PROJECTILE)
def _projectile_impacts_projectile(projectile, other, effects, environment):
    if not _is_live_projectile(other):
        return False

    normal, _, overlap = _collision_info(projectile, other)

    if not _projectiles_can_hit_each_other(projectile, other):
        return False

    contact, impact_normal = _projectile_impact(projectile, other, overlap)
    if contact is None:
        return False

    if projectile.projectile_name == other.projectile_name:
        BattleEffect.play_boom(max(projectile.current_damage, other.current_damage))
        _destroy_projectile(projectile, effects, impact_normal, projectile.current_damage, contact)
        _destroy_projectile(other, effects, [-impact_normal[0], -impact_normal[1]], other.current_damage, contact)
        return True

    projectile_damage = projectile.current_damage
    other_damage = other.current_damage
    projectile_hp = projectile.current_hp - other_damage
    other_hp = other.current_hp - projectile_damage

    BattleEffect.play_boom(max(projectile_damage, other_damage))

    if projectile_hp <= 0 and other_hp <= 0:
        _destroy_projectile(projectile, effects, impact_normal, projectile_damage, contact)
        _destroy_projectile(other, effects, [-impact_normal[0], -impact_normal[1]], other_damage, contact)
    elif projectile_hp > 0 and projectile_hp > other_hp:
        _set_projectile_hp(projectile, projectile_hp)
        _destroy_projectile(other, effects, [-impact_normal[0], -impact_normal[1]], other_damage, contact)
    elif other_hp > 0 and other_hp > projectile_hp:
        _destroy_projectile(projectile, effects, impact_normal, projectile_damage, contact)
        _set_projectile_hp(other, other_hp)
    else:
        _destroy_projectile(projectile, effects, impact_normal, projectile_damage, contact)
        _destroy_projectile(other, effects, [-impact_normal[0], -impact_normal[1]], other_damage, contact)
    return True


def _projectiles_can_hit_each_other(projectile, other):
    if projectile.player != other.player:
        return True

    return (
        projectile.projectile_name == other.projectile_name and
        projectile.hit_self and
        other.hit_self
    )


def _handle_projectile_ship_collisions(projectiles, ships, effects):
    _dispatch_collision_pairs(projectiles, ships, effects)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.SHIP)
def _projectile_impacts_ship(projectile, ship, effects, environment):
    if (
        not _is_live_projectile(projectile)
        or ship.current_hp <= 0
        or not _projectile_can_hit_ship(projectile, ship)
    ):
        return False

    _, _, overlap = _collision_info(projectile, ship)
    contact, impact_normal = _projectile_impact(projectile, ship, overlap)
    if contact is None:
        return False

    damage = projectile.current_damage
    ship.current_hp = max(0, ship.current_hp - damage)
    projectile.on_ship_impact(ship)
    BattleEffect.play_boom(damage)
    _destroy_projectile(projectile, effects, impact_normal, damage, contact)
    return True


def _handle_projectile_asteroid_collisions(projectiles, asteroids, effects):
    _dispatch_collision_pairs(projectiles, asteroids, effects)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.ASTEROID)
def _projectile_impacts_asteroid(projectile, asteroid, effects, environment):
    if not _is_live_projectile(projectile) or not asteroid.currently_alive:
        return False

    _, _, overlap = _collision_info(projectile, asteroid)
    contact, impact_normal = _projectile_impact(projectile, asteroid, overlap)
    if contact is None:
        return False

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    _destroy_projectile(projectile, effects, impact_normal, damage, contact)
    _destroy_asteroid(asteroid, effects)
    return True


def _handle_projectile_planet_collisions(projectiles, planets, effects):
    _dispatch_collision_pairs(projectiles, planets, effects)


@_pair_collision_handler(CollisionRole.PROJECTILE, CollisionRole.PLANET)
def _projectile_impacts_planet(projectile, planet, effects, environment):
    if not _is_live_projectile(projectile):
        return False

    _, _, overlap = _collision_info(projectile, planet)
    contact, impact_normal = _projectile_impact(projectile, planet, overlap)
    if contact is None:
        return False

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    _destroy_projectile(projectile, effects, impact_normal, damage, contact)
    return True


def _handle_fighter_fighter_collisions(fighters, effects):
    _dispatch_unique_collision_pairs(fighters, effects, _is_live_fighter)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.FIGHTER)
def _fighter_impacts_fighter(fighter, other, effects, environment):
    if not _is_live_fighter(other):
        return False

    fighter_hits = (
        fighter.fighter_collision_capabilities.collides_with_fighters
    )
    other_hits = other.fighter_collision_capabilities.collides_with_fighters
    if not fighter_hits and not other_hits:
        return False

    _, _, overlap = _collision_info(fighter, other)
    contact, normal = _projectile_impact(fighter, other, overlap)
    if contact is None:
        return False

    if fighter_hits:
        other.current_hp = max(0, other.current_hp - fighter.current_damage)
    if other_hits:
        fighter.current_hp = max(0, fighter.current_hp - other.current_damage)
    BattleEffect.play_boom(max(fighter.current_damage, other.current_damage))
    if fighter.current_hp <= 0:
        _destroy_projectile(
            fighter,
            effects,
            normal,
            fighter.current_damage,
            contact,
        )
    if other.current_hp <= 0:
        _destroy_projectile(
            other,
            effects,
            [-normal[0], -normal[1]],
            other.current_damage,
            contact,
        )
    return True


def _handle_fighter_projectile_collisions(fighters, projectiles, effects):
    _dispatch_collision_pairs(fighters, projectiles, effects)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.PROJECTILE)
def _fighter_impacts_projectile(fighter, projectile, effects, environment):
    if (
        not _is_live_fighter(fighter)
        or not fighter.fighter_collision_capabilities.collides_with_projectiles
        or not _is_live_projectile(projectile)
    ):
        return False

    _, _, overlap = _collision_info(fighter, projectile)
    contact, normal = _projectile_impact(fighter, projectile, overlap)
    if contact is None:
        return False

    if fighter.fighter_collision_capabilities.damages_projectiles:
        projectile_hp = projectile.current_hp - fighter.current_damage
        if projectile_hp <= 0:
            _destroy_projectile(
                projectile,
                effects,
                [-normal[0], -normal[1]],
                fighter.current_damage,
                contact,
            )
        else:
            _set_projectile_hp(projectile, projectile_hp)
    BattleEffect.play_boom(fighter.current_damage)
    _destroy_projectile(
        fighter,
        effects,
        normal,
        fighter.current_damage,
        contact,
    )
    return True


def _handle_fighter_ship_collisions(fighters, ships, effects):
    _dispatch_collision_pairs(fighters, ships, effects)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.SHIP)
def _fighter_impacts_ship(fighter, ship, effects, environment):
    if not _is_live_fighter(fighter) or ship.current_hp <= 0:
        return False

    if ship is fighter.parent:
        if not fighter.can_recover_with_parent():
            return False
    elif ship.player == fighter.player:
        if not fighter.fighter_collision_capabilities.collides_with_friendly_ships:
            return False
    elif not fighter.fighter_collision_capabilities.collides_with_enemy_ships:
        return False

    _, _, overlap = _collision_info(fighter, ship)
    contact, normal = _projectile_impact(fighter, ship, overlap)
    if contact is None:
        return False

    if ship is fighter.parent:
        fighter.recover_with_parent()
    else:
        damage = fighter.current_damage
        ship.current_hp = max(0, ship.current_hp - damage)
        BattleEffect.play_boom(damage)
        _destroy_projectile(fighter, effects, normal, damage, contact)
    return True


def _handle_fighter_asteroid_collisions(fighters, asteroids, effects):
    _dispatch_collision_pairs(fighters, asteroids, effects)


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.ASTEROID)
def _fighter_impacts_asteroid(fighter, asteroid, effects, environment):
    if (
        not _is_live_fighter(fighter)
        or not fighter.fighter_collision_capabilities.collides_with_asteroids
        or not asteroid.currently_alive
    ):
        return False

    _, _, overlap = _collision_info(fighter, asteroid)
    contact, normal = _projectile_impact(fighter, asteroid, overlap)
    if contact is None:
        return False

    BattleEffect.play_boom(fighter.current_damage)
    _destroy_projectile(
        fighter,
        effects,
        normal,
        fighter.current_damage,
        contact,
    )
    if fighter.fighter_collision_capabilities.damages_asteroids:
        _destroy_asteroid(asteroid, effects)
    return True


def _handle_fighter_planet_collisions(fighters, planets):
    _dispatch_collision_pairs(fighters, planets, [])


@_pair_collision_handler(CollisionRole.FIGHTER, CollisionRole.PLANET)
def _fighter_impacts_planet(fighter, planet, effects, environment):
    if (
        not _is_live_fighter(fighter)
        or not fighter.fighter_collision_capabilities.collides_with_planets
    ):
        return False

    normal, _, overlap = _collision_info(fighter, planet)
    contact, _ = _projectile_impact(fighter, planet, overlap)
    if contact is None:
        return False

    _separate_from_static_body(
        fighter,
        planet,
        normal,
        overlap,
        extra_clearance=1.0,
    )
    fighter.begin_planet_avoidance(planet, normal)
    return True


def _is_live_projectile(obj):
    return (
        isinstance(obj, Ability) and
        obj.type == "projectile" and
        obj.can_collide and
        obj.currently_alive and
        obj.current_hp > 0
    )


def _is_live_fighter(obj):
    return (
        isinstance(obj, Ability) and
        obj.type == "fighter" and
        obj.can_collide and
        obj.currently_alive and
        obj.current_hp > 0
    )


def _is_live_laser(obj):
    return (
        isinstance(obj, Ability) and
        obj.type == "laser" and
        obj.can_collide and
        obj.currently_alive and
        obj.current_hp > 0
    )


def _projectile_can_hit_ship(projectile, ship):
    if ship.player != projectile.player:
        return True

    if ship != projectile.parent or not projectile.hit_parent:
        return False

    if projectile.has_left_parent:
        return True

    _, _, overlap = _collision_info(projectile, ship)
    if _objects_overlap(projectile, ship, overlap):
        return False

    projectile.has_left_parent = True
    return False


def _handle_laser_collisions(lasers, ships, projectiles, fighters, asteroids, planets, effects):
    for laser in lasers:
        if not _is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        if hasattr(laser, "calculate_end_position"):
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
        damage = laser.current_damage
        laser.end_position = [contact[0] % const.ARENA_SIZE, contact[1] % const.ARENA_SIZE]
        laser.intercepted = True

        effects.append(BattleEffect.from_blast(contact, normal, damage, align_edge=True))
        BattleEffect.play_boom(damage)

        _apply_laser_impact(target, effects, normal, damage, contact)


def _apply_laser_impact(target, effects, normal, damage, contact):
    policy = _LASER_IMPACT_POLICIES.get(target.collision_capabilities.role)
    if policy is not None:
        policy(target, effects, normal, damage, contact)


@_laser_impact_policy(CollisionRole.SHIP)
def _laser_impacts_ship(target, effects, normal, damage, contact):
    target.current_hp = max(0, target.current_hp - damage)


@_laser_impact_policy(CollisionRole.PROJECTILE)
@_laser_impact_policy(CollisionRole.FIGHTER)
def _laser_impacts_ability(target, effects, normal, damage, contact):
    target.current_hp = max(0, target.current_hp - damage)
    if target.current_hp <= 0:
        _destroy_projectile(target, effects, normal, damage, contact)


@_laser_impact_policy(CollisionRole.ASTEROID)
def _laser_impacts_asteroid(target, effects, normal, damage, contact):
    _destroy_asteroid(target, effects)


@_laser_impact_policy(CollisionRole.PLANET)
def _laser_impacts_planet(target, effects, normal, damage, contact):
    pass


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
    if target.current_hp <= 0:
        return False
    if explicit:
        return True
    return target.player != laser.player or (
        target is laser.parent and laser.hit_parent
    )


@_laser_target_policy(CollisionRole.PROJECTILE)
def _projectile_is_laser_target(laser, target, explicit):
    if not (
        target.can_collide
        and target.currently_alive
        and target.current_hp > 0
    ):
        return False
    return explicit or target.player != laser.player or laser.hit_self


@_laser_target_policy(CollisionRole.FIGHTER)
def _fighter_is_laser_target(laser, target, explicit):
    if not (
        target.can_collide
        and target.currently_alive
        and target.current_hp > 0
    ):
        return False
    if explicit:
        return True
    return (
        target is not laser.parent
        and target.laser_target_capabilities.vulnerable
    )


@_laser_target_policy(CollisionRole.ASTEROID)
def _asteroid_is_laser_target(laser, target, explicit):
    return target.currently_alive


@_laser_target_policy(CollisionRole.PLANET)
def _planet_is_laser_target(laser, target, explicit):
    return True


@_laser_target_policy(CollisionRole.NONE)
def _generic_is_laser_target(laser, target, explicit):
    return explicit and target.currently_alive


def _laser_hit_info(laser, target):
    start = getattr(laser, "start_position", laser.parent.position)
    end = getattr(laser, "end_position", laser.position)
    segment = _wrapped_segment(start, end)
    distance = _distance_from_segment_to_point(segment[0], segment[1], target.position)
    if distance > _radius(target):
        return None

    target_mask = _get_collision_mask(target)
    if target_mask is not None:
        contact = _sample_laser_mask_hit(segment[0], segment[1], target, target_mask)
    else:
        contact = _segment_circle_intercept(segment[0], segment[1], target.position, _radius(target))

    if contact is None:
        return None

    normal = _normal_from_target(target, contact, _segment_direction(segment[0], segment[1]))
    return {
        "target": target,
        "contact": contact,
        "normal": normal,
        "distance": math.hypot(contact[0] - segment[0][0], contact[1] - segment[0][1]),
    }



def _wrapped_segment(start, end):
    delta = _wrapped_delta(start, end)
    return start, [start[0] + delta[0], start[1] + delta[1]]


def _sample_laser_mask_hit(start, end, target, target_mask):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    steps = max(1, int(math.hypot(dx, dy) / 8))
    target_center = _nearest_position(target.position, start)
    target_size = _collision_size(target)
    left = target_center[0] - target_size[0] / 2
    top = target_center[1] - target_size[1] / 2

    for step in range(steps + 1):
        ratio = step / steps
        x = start[0] + dx * ratio
        y = start[1] + dy * ratio
        mask_x = int(x - left)
        mask_y = int(y - top)
        if 0 <= mask_x < target_mask.get_size()[0] and 0 <= mask_y < target_mask.get_size()[1]:
            if target_mask.get_at((mask_x, mask_y)):
                return [x, y]

    return None


def ship_rotation_blocked(ship):
    candidates = []
    if ship.opponent and ship.opponent.current_hp > 0:
        candidates.append(ship.opponent)
    candidates.extend([obj for obj in ship.asteroids if obj.currently_alive])
    if ship.planet:
        candidates.append(ship.planet)

    for candidate in candidates:
        _, _, overlap = _collision_info(ship, candidate)
        if _objects_overlap(ship, candidate, overlap):
            return True

    return False


def _remove_dead_collision_objects(game_objects):
    game_objects[:] = [
        obj for obj in game_objects
        if not isinstance(obj, (Ability, Asteroid)) or obj.currently_alive
    ]


def _spawn_replacement_asteroids(game_objects, asteroids, ships, planets):
    if not planets:
        return

    dead_count = sum(1 for asteroid in asteroids if not asteroid.currently_alive)
    if dead_count <= 0:
        return

    planet = planets[0]
    avoid_bodies = [
        obj for obj in game_objects
        if _is_asteroid_spawn_avoid_body(obj)
    ]

    for _ in range(dead_count):
        asteroid = Asteroid()
        asteroid.set_planet(planet)
        asteroid.position = asteroid.get_respawn_position(planet, ships, avoid_bodies)
        avoid_bodies.append(asteroid)
        game_objects.append(asteroid)


def _is_asteroid_spawn_avoid_body(obj):
    if isinstance(obj, Planet):
        return False
    if isinstance(obj, Asteroid):
        return obj.currently_alive
    if isinstance(obj, SpaceShip):
        return obj.current_hp > 0
    if isinstance(obj, Ability):
        return obj.can_collide and obj.currently_alive and obj.current_hp > 0
    return getattr(obj, "can_collide", False) and getattr(obj, "currently_alive", True)


def _destroy_projectile(projectile, effects, direction, damage, contact_position=None):
    if not projectile.currently_alive:
        return

    effect_position = contact_position if contact_position is not None else projectile.position
    animation = getattr(projectile, "death_animation", None)
    if animation:
        effects.append(BattleEffect.from_animation(
            effect_position,
            animation,
            direction_vector=direction,
            align_edge=contact_position is not None
        ))
    else:
        effects.append(BattleEffect.from_blast(
            effect_position,
            direction,
            damage,
            align_edge=contact_position is not None
        ))

    projectile.current_hp = 0
    projectile.currently_alive = False


def _destroy_asteroid(asteroid, effects):
    if not asteroid.currently_alive:
        return

    if asteroid.death_animation:
        effects.append(BattleEffect.from_animation(asteroid.position, asteroid.death_animation))
    asteroid.currently_alive = False


def _object_on_screen(obj, ships):
    if len(ships) != 2:
        return True

    view_center, view_size = _view_center_and_size([ship.position for ship in ships])
    delta = _wrapped_delta(view_center, obj.position)
    margin = max(_collision_size(obj)) / 2
    return abs(delta[0]) <= view_size / 2 + margin and abs(delta[1]) <= view_size / 2 + margin


def _set_projectile_hp(projectile, hp):
    if hasattr(projectile, "set_hp"):
        projectile.set_hp(max(0, hp))
    else:
        projectile.current_hp = max(0, hp)


def _collision_info(obj, other):
    delta = _wrapped_delta(other.position, obj.position)
    distance = math.hypot(delta[0], delta[1])
    radius_sum = _radius(obj) + _radius(other)
    overlap = radius_sum - distance

    if distance == 0:
        return [1.0, 0.0], distance, overlap

    return [delta[0] / distance, delta[1] / distance], distance, overlap


def _objects_overlap(obj, other, overlap):
    if overlap <= 0 and not _mask_broadphase_overlap(obj, other):
        return False

    obj_mask = _get_collision_mask(obj)
    other_mask = _get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return True

    obj_size = _collision_size(obj)
    other_size = _collision_size(other)
    delta = _wrapped_delta(other.position, obj.position)
    offset = (
        int(round(-delta[0] + obj_size[0] / 2 - other_size[0] / 2)),
        int(round(-delta[1] + obj_size[1] / 2 - other_size[1] / 2)),
    )
    return obj_mask.overlap(other_mask, offset) is not None


def _projectile_impact(projectile, other, overlap):
    swept_impact = _swept_impact(projectile, other)
    if swept_impact[0] is not None:
        return swept_impact

    if _objects_overlap(projectile, other, overlap):
        return _estimated_impact(projectile, other)

    normal, _, _ = _collision_info(projectile, other)
    return None, normal


def _swept_impact(obj, other):
    obj_previous = _sweep_previous_position(obj)
    other_previous = _sweep_previous_position(other)
    obj_delta = _wrapped_delta(obj_previous, obj.position)
    other_delta = _wrapped_delta(other_previous, other.position)
    relative_delta = [
        obj_delta[0] - other_delta[0],
        obj_delta[1] - other_delta[1],
    ]
    relative_distance = math.hypot(relative_delta[0], relative_delta[1])
    if relative_distance <= 0:
        normal, _, _ = _collision_info(obj, other)
        return None, normal

    steps = max(1, int(math.ceil(relative_distance / _sweep_step_size(obj, other))))
    for step in range(1, steps + 1):
        ratio = step / steps
        obj_position = [
            (obj_previous[0] + obj_delta[0] * ratio) % const.ARENA_SIZE,
            (obj_previous[1] + obj_delta[1] * ratio) % const.ARENA_SIZE,
        ]
        other_position = [
            (other_previous[0] + other_delta[0] * ratio) % const.ARENA_SIZE,
            (other_previous[1] + other_delta[1] * ratio) % const.ARENA_SIZE,
        ]
        if _objects_overlap_at_positions(obj, other, obj_position, other_position):
            return _estimated_impact_at_positions(obj, other, obj_position, other_position)

    normal, _, _ = _collision_info(obj, other)
    return None, normal


def _sweep_previous_position(obj):
    if not getattr(obj, "can_move", True):
        return obj.position
    return getattr(obj, "previous_position", obj.position)


def _sweep_step_size(obj, other):
    obj_size = _collision_size(obj)
    other_size = _collision_size(other)
    min_dimension = min(obj_size[0], obj_size[1], other_size[0], other_size[1])
    return max(2, min(12, min_dimension / 3))


def _estimated_impact(obj, other):
    normal, _, _ = _collision_info(obj, other)
    return _contact_point(other, normal), normal


def _estimated_impact_at_positions(obj, other, obj_position, other_position):
    delta = _wrapped_delta(other_position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    if distance == 0:
        normal = [1.0, 0.0]
    else:
        normal = [delta[0] / distance, delta[1] / distance]
    return [
        (other_position[0] + normal[0] * _radius(other)) % const.ARENA_SIZE,
        (other_position[1] + normal[1] * _radius(other)) % const.ARENA_SIZE,
    ], normal


def _contact_point(target, normal):
    return [
        (target.position[0] + normal[0] * _radius(target)) % const.ARENA_SIZE,
        (target.position[1] + normal[1] * _radius(target)) % const.ARENA_SIZE,
    ]


def _objects_overlap_at_positions(obj, other, obj_position, other_position):
    delta = _wrapped_delta(other_position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    overlap = _radius(obj) + _radius(other) - distance
    if overlap <= 0 and not _mask_broadphase_overlap_at_positions(obj, other, obj_position, other_position):
        return False

    obj_mask = _get_collision_mask(obj)
    other_mask = _get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return True

    obj_size = _collision_size(obj)
    other_size = _collision_size(other)
    offset = (
        int(round(-delta[0] + obj_size[0] / 2 - other_size[0] / 2)),
        int(round(-delta[1] + obj_size[1] / 2 - other_size[1] / 2)),
    )
    return obj_mask.overlap(other_mask, offset) is not None


def _get_collision_mask(obj):
    if hasattr(obj, "get_collision_mask"):
        return obj.get_collision_mask()
    return None


def _collision_size(obj):
    mask = _get_collision_mask(obj)
    if mask is not None:
        return mask.get_size()
    if isinstance(obj, Planet):
        return [obj.diameter, obj.diameter]
    return obj.size


def _mask_broadphase_overlap(obj, other):
    return _mask_broadphase_overlap_at_positions(obj, other, obj.position, other.position)


def _mask_broadphase_overlap_at_positions(obj, other, obj_position, other_position):
    obj_mask = _get_collision_mask(obj)
    other_mask = _get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return False

    delta = _wrapped_delta(other_position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    return _mask_radius(obj) + _mask_radius(other) - distance > 0


def _distance_between(obj, other):
    return wrapped_distance(obj.position, other.position)


def _distance_from_segment_to_point(start, end, point):
    point = _nearest_position(point, start)
    sx, sy = start
    ex, ey = end
    px, py = point
    dx = ex - sx
    dy = ey - sy

    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)

    ratio = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    ratio = max(0, min(1, ratio))
    nearest_x = sx + dx * ratio
    nearest_y = sy + dy * ratio
    return math.hypot(px - nearest_x, py - nearest_y)


def _segment_circle_intercept(start, end, center, radius):
    center = _nearest_position(center, start)
    sx, sy = start
    ex, ey = end
    cx, cy = center
    dx = ex - sx
    dy = ey - sy
    fx = sx - cx
    fy = sy - cy

    a = dx * dx + dy * dy
    if a == 0:
        return None

    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_discriminant = math.sqrt(discriminant)
    candidates = [
        (-b - sqrt_discriminant) / (2 * a),
        (-b + sqrt_discriminant) / (2 * a),
    ]
    candidates = [value for value in candidates if 0 <= value <= 1]
    if not candidates:
        return None

    ratio = min(candidates)
    return [sx + dx * ratio, sy + dy * ratio]


def _normal_from_target(target, contact, fallback):
    target_center = _nearest_position(target.position, contact)
    dx = contact[0] - target_center[0]
    dy = contact[1] - target_center[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return [-fallback[0], -fallback[1]]
    return [dx / length, dy / length]


def _segment_direction(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return [0, -1]
    return [dx / length, dy / length]


def _radius(obj):
    if isinstance(obj, Planet):
        return obj.diameter / 2
    return max(_collision_size(obj)) / 2


def _mask_radius(obj):
    size = _collision_size(obj)
    return math.hypot(size[0], size[1]) / 2


def _mass(obj):
    return max(0.1, getattr(obj, "mass", _radius(obj) / 50))


def _elastic_bounce(obj, other, normal, distance, overlap):
    if not hasattr(obj, "velocity") or not hasattr(other, "velocity"):
        return

    obj_mass = _mass(obj)
    other_mass = _mass(other)
    relative_velocity = [
        obj.velocity[0] - other.velocity[0],
        obj.velocity[1] - other.velocity[1],
    ]
    velocity_along_normal = _dot(relative_velocity, normal)

    if velocity_along_normal < 0:
        impulse = -(1 + 1.0) * velocity_along_normal
        impulse /= (1 / obj_mass) + (1 / other_mass)

        obj.velocity[0] += impulse * normal[0] / obj_mass
        obj.velocity[1] += impulse * normal[1] / obj_mass
        other.velocity[0] -= impulse * normal[0] / other_mass
        other.velocity[1] -= impulse * normal[1] / other_mass

    _separate_dynamic_bodies(obj, other, normal, overlap, obj_mass, other_mass)


def _bounce_off_static_body(obj, static_body, normal, overlap, extra_clearance=0.0):
    if not hasattr(obj, "velocity"):
        return False

    velocity_along_normal = _dot(obj.velocity, normal)
    if velocity_along_normal < 0:
        impulse = [
            -2 * velocity_along_normal * normal[0],
            -2 * velocity_along_normal * normal[1],
        ]
        obj.velocity[0] += impulse[0]
        obj.velocity[1] += impulse[1]
        if not getattr(obj, "inertia", True) and hasattr(obj, "collision_velocity"):
            obj.collision_velocity = obj.velocity.copy()
        collided_while_approaching = True
    else:
        collided_while_approaching = False

    _separate_from_static_body(obj, static_body, normal, overlap, extra_clearance)
    return collided_while_approaching


def _stop_at_static_body(obj, static_body, normal, overlap):
    if hasattr(obj, "velocity"):
        velocity_along_normal = _dot(obj.velocity, normal)
        if velocity_along_normal < 0:
            obj.velocity[0] -= velocity_along_normal * normal[0]
            obj.velocity[1] -= velocity_along_normal * normal[1]

    _separate_from_static_body(obj, static_body, normal, overlap)


def _separate_from_static_body(obj, static_body, normal, overlap, extra_clearance=0.0):
    if overlap <= 0 and not _mask_broadphase_overlap(obj, static_body):
        return

    if _get_collision_mask(obj) is None or _get_collision_mask(static_body) is None:
        if overlap <= 0:
            return
        separation = overlap + extra_clearance
        obj.position[0] = (obj.position[0] + normal[0] * separation) % const.ARENA_SIZE
        obj.position[1] = (obj.position[1] + normal[1] * separation) % const.ARENA_SIZE
        return

    max_separation = int(math.ceil(_mask_radius(obj) + _mask_radius(static_body))) + 1
    moved = 0
    step = 2
    while moved <= max_separation:
        _, _, current_overlap = _collision_info(obj, static_body)
        if not _objects_overlap(obj, static_body, current_overlap):
            if extra_clearance > 0:
                obj.position[0] = (obj.position[0] + normal[0] * extra_clearance) % const.ARENA_SIZE
                obj.position[1] = (obj.position[1] + normal[1] * extra_clearance) % const.ARENA_SIZE
            return

        obj.position[0] = (obj.position[0] + normal[0] * step) % const.ARENA_SIZE
        obj.position[1] = (obj.position[1] + normal[1] * step) % const.ARENA_SIZE
        moved += step

    if extra_clearance > 0:
        obj.position[0] = (obj.position[0] + normal[0] * extra_clearance) % const.ARENA_SIZE
        obj.position[1] = (obj.position[1] + normal[1] * extra_clearance) % const.ARENA_SIZE


def _separate_dynamic_bodies(obj, other, normal, overlap, obj_mass, other_mass):
    if overlap <= 0:
        return

    total_mass = obj_mass + other_mass
    obj_push = overlap * (other_mass / total_mass)
    other_push = overlap * (obj_mass / total_mass)

    obj.position[0] = (obj.position[0] + normal[0] * obj_push) % const.ARENA_SIZE
    obj.position[1] = (obj.position[1] + normal[1] * obj_push) % const.ARENA_SIZE
    other.position[0] = (other.position[0] - normal[0] * other_push) % const.ARENA_SIZE
    other.position[1] = (other.position[1] - normal[1] * other_push) % const.ARENA_SIZE


def _dot(vector, other):
    return vector[0] * other[0] + vector[1] * other[1]
