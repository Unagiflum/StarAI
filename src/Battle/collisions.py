import math

import src.const as const
from src.Battle.effects import BattleEffect
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


def handle_collisions(game_objects):
    ships = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
    projectiles = [obj for obj in game_objects if _is_live_projectile(obj)]
    asteroids = [obj for obj in game_objects if isinstance(obj, Asteroid) and obj.currently_alive]
    planets = [obj for obj in game_objects if isinstance(obj, Planet)]
    effects = []

    _handle_ship_ship_collisions(ships)
    _handle_ship_asteroid_collisions(ships, asteroids)
    _handle_ship_planet_collisions(ships, planets)
    _handle_projectile_projectile_collisions(projectiles, effects)
    _handle_projectile_ship_collisions(projectiles, ships, effects)
    _handle_projectile_asteroid_collisions(projectiles, asteroids, effects)
    _handle_projectile_planet_collisions(projectiles, planets, effects)

    game_objects.extend(effects)
    _remove_dead_collision_objects(game_objects)


def _handle_ship_ship_collisions(ships):
    for i, ship in enumerate(ships):
        for other in ships[i + 1:]:
            normal, distance, overlap = _collision_info(ship, other)
            if overlap > 0:
                _elastic_bounce(ship, other, normal, distance, overlap)


def _handle_ship_asteroid_collisions(ships, asteroids):
    for ship in ships:
        for asteroid in asteroids:
            if not asteroid.currently_alive:
                continue

            normal, distance, overlap = _collision_info(ship, asteroid)
            if overlap > 0:
                _elastic_bounce(ship, asteroid, normal, distance, overlap)


def _handle_ship_planet_collisions(ships, planets):
    for ship in ships:
        for planet in planets:
            normal, distance, overlap = _collision_info(ship, planet)
            if overlap <= 0:
                continue

            collided_while_approaching = _bounce_off_static_body(ship, normal, overlap)
            if collided_while_approaching and ship.current_hp > 0:
                damage = max(1, math.ceil(ship.current_hp * 0.2))
                ship.current_hp = max(0, ship.current_hp - damage)
                BattleEffect.play_boom(damage)


def _handle_projectile_projectile_collisions(projectiles, effects):
    for i, projectile in enumerate(projectiles):
        if not _is_live_projectile(projectile):
            continue

        for other in projectiles[i + 1:]:
            if not _is_live_projectile(other):
                continue

            normal, _, overlap = _collision_info(projectile, other)
            if overlap <= 0:
                continue

            if projectile.player == other.player:
                if (projectile.projectile_name == other.projectile_name and
                        projectile.hit_self and other.hit_self):
                    damage = max(projectile.current_damage, other.current_damage)
                    BattleEffect.play_boom(damage)
                    _destroy_projectile(projectile, effects, normal, projectile.current_damage)
                    _destroy_projectile(other, effects, [-normal[0], -normal[1]], other.current_damage)
                continue

            projectile_damage = projectile.current_damage
            other_damage = other.current_damage

            _set_projectile_hp(projectile, projectile.current_hp - other_damage)
            _set_projectile_hp(other, other.current_hp - projectile_damage)
            BattleEffect.play_boom(max(projectile_damage, other_damage))

            if projectile.current_hp <= 0:
                _destroy_projectile(projectile, effects, normal, projectile_damage)
            if other.current_hp <= 0:
                _destroy_projectile(other, effects, [-normal[0], -normal[1]], other_damage)


def _handle_projectile_ship_collisions(projectiles, ships, effects):
    for projectile in projectiles:
        if not _is_live_projectile(projectile):
            continue

        for ship in ships:
            if ship.current_hp <= 0:
                continue

            normal, _, overlap = _collision_info(projectile, ship)
            if overlap <= 0:
                continue

            can_hit_ship = ship.player != projectile.player
            can_hit_parent = ship == projectile.parent and projectile.hit_parent
            if not can_hit_ship and not can_hit_parent:
                continue

            damage = projectile.current_damage
            ship.current_hp = max(0, ship.current_hp - damage)
            BattleEffect.play_boom(damage)
            _destroy_projectile(projectile, effects, normal, damage)
            break


def _handle_projectile_asteroid_collisions(projectiles, asteroids, effects):
    for projectile in projectiles:
        if not _is_live_projectile(projectile):
            continue

        for asteroid in asteroids:
            if not asteroid.currently_alive:
                continue

            normal, _, overlap = _collision_info(projectile, asteroid)
            if overlap <= 0:
                continue

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            _destroy_projectile(projectile, effects, normal, damage)
            _destroy_asteroid(asteroid, effects)
            break


def _handle_projectile_planet_collisions(projectiles, planets, effects):
    for projectile in projectiles:
        if not _is_live_projectile(projectile):
            continue

        for planet in planets:
            normal, _, overlap = _collision_info(projectile, planet)
            if overlap <= 0:
                continue

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            _destroy_projectile(projectile, effects, normal, damage)
            break


def _is_live_projectile(obj):
    return (
        isinstance(obj, Ability) and
        obj.type == "projectile" and
        obj.can_collide and
        obj.currently_alive and
        obj.current_hp > 0
    )


def _remove_dead_collision_objects(game_objects):
    game_objects[:] = [
        obj for obj in game_objects
        if not isinstance(obj, (Ability, Asteroid)) or obj.currently_alive
    ]


def _destroy_projectile(projectile, effects, direction, damage):
    if not projectile.currently_alive:
        return

    animation = getattr(projectile, "death_animation", None)
    if animation:
        effects.append(BattleEffect.from_animation(projectile.position, animation))
    else:
        effects.append(BattleEffect.from_blast(projectile.position, direction, damage))

    projectile.current_hp = 0
    projectile.currently_alive = False


def _destroy_asteroid(asteroid, effects):
    if not asteroid.currently_alive:
        return

    if asteroid.death_animation:
        effects.append(BattleEffect.from_animation(asteroid.position, asteroid.death_animation))
    asteroid.currently_alive = False


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


def _wrapped_delta(from_position, to_position):
    dx = to_position[0] - from_position[0]
    dy = to_position[1] - from_position[1]

    if abs(dx) > const.ARENA_SIZE / 2:
        dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
    if abs(dy) > const.ARENA_SIZE / 2:
        dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

    return [dx, dy]


def _radius(obj):
    if isinstance(obj, Planet):
        return obj.diameter / 2
    return max(obj.size) / 2


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


def _bounce_off_static_body(obj, normal, overlap):
    if not hasattr(obj, "velocity"):
        return False

    velocity_along_normal = _dot(obj.velocity, normal)
    if velocity_along_normal < 0:
        obj.velocity[0] -= 2 * velocity_along_normal * normal[0]
        obj.velocity[1] -= 2 * velocity_along_normal * normal[1]
        collided_while_approaching = True
    else:
        collided_while_approaching = False

    obj.position[0] = (obj.position[0] + normal[0] * overlap) % const.ARENA_SIZE
    obj.position[1] = (obj.position[1] + normal[1] * overlap) % const.ARENA_SIZE
    return collided_while_approaching


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
