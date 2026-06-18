import math

import src.const as const
from src.Battle.effects import BattleEffect
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


def handle_collisions(game_objects):
    ships = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
    projectiles = [obj for obj in game_objects if _is_live_projectile(obj)]
    lasers = [obj for obj in game_objects if _is_live_laser(obj)]
    asteroids = [obj for obj in game_objects if isinstance(obj, Asteroid) and obj.currently_alive]
    planets = [obj for obj in game_objects if isinstance(obj, Planet)]
    effects = []

    _handle_laser_collisions(lasers, ships, projectiles, asteroids, planets, effects)
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
            if _objects_overlap(ship, other, overlap):
                _elastic_bounce(ship, other, normal, distance, overlap)


def _handle_ship_asteroid_collisions(ships, asteroids):
    for ship in ships:
        for asteroid in asteroids:
            if not asteroid.currently_alive:
                continue

            normal, distance, overlap = _collision_info(ship, asteroid)
            if _objects_overlap(ship, asteroid, overlap):
                _elastic_bounce(ship, asteroid, normal, distance, overlap)


def _handle_ship_planet_collisions(ships, planets):
    for ship in ships:
        for planet in planets:
            normal, distance, overlap = _collision_info(ship, planet)
            if not _objects_overlap(ship, planet, overlap):
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
            if not _projectile_overlap(projectile, other, overlap):
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
            if not _projectile_overlap(projectile, ship, overlap):
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
            if not _projectile_overlap(projectile, asteroid, overlap):
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
            if not _projectile_overlap(projectile, planet, overlap):
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


def _is_live_laser(obj):
    return (
        isinstance(obj, Ability) and
        obj.type == "laser" and
        obj.can_collide and
        obj.currently_alive and
        obj.current_hp > 0
    )


def _handle_laser_collisions(lasers, ships, projectiles, asteroids, planets, effects):
    for laser in lasers:
        if not _is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        if hasattr(laser, "calculate_end_position"):
            laser.calculate_end_position()

        targets = _laser_targets(laser, ships, projectiles, asteroids, planets)
        for target in targets:
            if isinstance(target, Planet):
                if _laser_hits_target(laser, target):
                    BattleEffect.play_boom(laser.current_damage)
                    break
                continue

            if not _laser_hits_target(laser, target):
                continue

            damage = laser.current_damage
            if hasattr(target, "current_hp"):
                target.current_hp = max(0, target.current_hp - damage)
                if isinstance(target, Ability) and target.current_hp <= 0:
                    normal, _, _ = _collision_info(laser, target)
                    _destroy_projectile(target, effects, normal, damage)
                BattleEffect.play_boom(damage)
            elif isinstance(target, Asteroid):
                _destroy_asteroid(target, effects)
                BattleEffect.play_boom(damage)

            break


def _laser_targets(laser, ships, projectiles, asteroids, planets):
    explicit_target = getattr(laser, "target", None)
    if explicit_target is not None:
        return [explicit_target]

    targets = []
    for ship in ships:
        if ship.player != laser.player or (ship == laser.parent and laser.hit_parent):
            targets.append(ship)

    for projectile in projectiles:
        if projectile.player != laser.player or laser.hit_self:
            targets.append(projectile)

    targets.extend(asteroids)
    targets.extend(planets)
    return sorted(targets, key=lambda target: _distance_between(laser.parent, target))


def _laser_hits_target(laser, target):
    start = laser.parent.position
    end = getattr(laser, "end_position", laser.position)
    segment = _wrapped_segment(start, end)
    distance = _distance_from_segment_to_point(segment[0], segment[1], target.position)
    if distance > _radius(target):
        return False

    target_mask = _get_collision_mask(target)
    if target_mask is None:
        return True

    return _sample_laser_mask_hits(segment[0], segment[1], target, target_mask)


def _wrapped_segment(start, end):
    delta = _wrapped_delta(start, end)
    return start, [
        (start[0] + delta[0]) % const.ARENA_SIZE,
        (start[1] + delta[1]) % const.ARENA_SIZE,
    ]


def _sample_laser_mask_hits(start, end, target, target_mask):
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
                return True

    return False


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


def _objects_overlap(obj, other, overlap):
    if overlap <= 0:
        return False

    if isinstance(obj, Planet) or isinstance(other, Planet):
        return True

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


def _projectile_overlap(projectile, other, overlap):
    if _objects_overlap(projectile, other, overlap):
        return True
    return _swept_projectile_overlap(projectile, other)


def _swept_projectile_overlap(projectile, other):
    previous = getattr(projectile, "previous_position", projectile.position)
    current = projectile.position
    delta = _wrapped_delta(previous, current)
    distance = math.hypot(delta[0], delta[1])
    if distance <= 0:
        return False

    step_size = max(4, min(_radius(projectile), _radius(other)) / 2)
    steps = max(1, int(math.ceil(distance / step_size)))
    for step in range(1, steps):
        ratio = step / steps
        position = [
            (previous[0] + delta[0] * ratio) % const.ARENA_SIZE,
            (previous[1] + delta[1] * ratio) % const.ARENA_SIZE,
        ]
        if _objects_overlap_at_position(projectile, other, position):
            projectile.position = position
            return True

    return False


def _objects_overlap_at_position(obj, other, obj_position):
    delta = _wrapped_delta(other.position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    overlap = _radius(obj) + _radius(other) - distance
    if overlap <= 0:
        return False

    if isinstance(other, Planet):
        return True

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


def _nearest_position(position, reference):
    delta = _wrapped_delta(reference, position)
    return [
        reference[0] + delta[0],
        reference[1] + delta[1],
    ]


def _distance_between(obj, other):
    delta = _wrapped_delta(obj.position, other.position)
    return math.hypot(delta[0], delta[1])


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
    return max(_collision_size(obj)) / 2


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
