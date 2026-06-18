import math

import src.const as const
from src.Battle.effects import BattleEffect
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


def handle_collisions(game_objects):
    ships = [
        obj for obj in game_objects
        if isinstance(obj, SpaceShip) and obj.currently_alive and obj.current_hp > 0
    ]
    projectiles = [obj for obj in game_objects if _is_live_projectile(obj)]
    lasers = [obj for obj in game_objects if _is_live_laser(obj)]
    asteroids = [obj for obj in game_objects if isinstance(obj, Asteroid) and obj.currently_alive]
    planets = [obj for obj in game_objects if isinstance(obj, Planet)]
    effects = []

    _handle_laser_collisions(lasers, ships, projectiles, asteroids, planets, effects)
    _handle_ship_ship_collisions(ships)
    _handle_ship_asteroid_collisions(ships, asteroids)
    _handle_ship_planet_collisions(ships, planets)
    _handle_asteroid_planet_collisions(asteroids, planets, ships, effects)
    _handle_projectile_projectile_collisions(projectiles, effects)
    _handle_projectile_ship_collisions(projectiles, ships, effects)
    _handle_projectile_asteroid_collisions(projectiles, asteroids, effects)
    _handle_projectile_planet_collisions(projectiles, planets, effects)
    _spawn_replacement_asteroids(game_objects, asteroids, ships, planets)

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

            bounce_clearance = max(_collision_size(ship)) * 1.5
            collided_while_approaching = _bounce_off_static_body(
                ship,
                planet,
                normal,
                overlap,
                extra_clearance=bounce_clearance,
            )
            if collided_while_approaching and ship.current_hp > 0:
                damage = max(1, math.ceil(ship.current_hp * 0.15))
                ship.current_hp = max(0, ship.current_hp - damage)
                BattleEffect.play_boom(damage)


def _handle_asteroid_planet_collisions(asteroids, planets, ships, effects):
    for asteroid in asteroids:
        if not asteroid.currently_alive:
            continue

        for planet in planets:
            normal, _, overlap = _collision_info(asteroid, planet)
            if _objects_overlap(asteroid, planet, overlap):
                if _object_on_screen(asteroid, ships):
                    BattleEffect.play_boom(1)
                _destroy_asteroid(asteroid, effects)
                break


def _handle_projectile_projectile_collisions(projectiles, effects):
    for i, projectile in enumerate(projectiles):
        if not _is_live_projectile(projectile):
            continue

        for other in projectiles[i + 1:]:
            if not _is_live_projectile(other):
                continue

            normal, _, overlap = _collision_info(projectile, other)

            if projectile.player == other.player:
                if not (projectile.projectile_name == other.projectile_name and
                        projectile.hit_self and other.hit_self):
                    continue

                contact, impact_normal = _projectile_impact(projectile, other, overlap)
                if contact is None:
                    continue

                damage = max(projectile.current_damage, other.current_damage)
                BattleEffect.play_boom(damage)
                _destroy_projectile(projectile, effects, impact_normal, projectile.current_damage, contact)
                _destroy_projectile(other, effects, [-impact_normal[0], -impact_normal[1]], other.current_damage, contact)
                continue

            contact, impact_normal = _projectile_impact(projectile, other, overlap)
            if contact is None:
                continue

            projectile_damage = projectile.current_damage
            other_damage = other.current_damage
            projectile_hp = projectile.current_hp - other_damage
            other_hp = other.current_hp - projectile_damage

            BattleEffect.play_boom(max(projectile_damage, other_damage))

            if projectile_hp <= 0:
                _destroy_projectile(projectile, effects, impact_normal, projectile_damage, contact)
            else:
                _set_projectile_hp(projectile, projectile_hp)

            if other_hp <= 0:
                _destroy_projectile(other, effects, [-impact_normal[0], -impact_normal[1]], other_damage, contact)
            else:
                _set_projectile_hp(other, other_hp)


def _handle_projectile_ship_collisions(projectiles, ships, effects):
    for projectile in projectiles:
        if not _is_live_projectile(projectile):
            continue

        for ship in ships:
            if ship.current_hp <= 0:
                continue

            if not _projectile_can_hit_ship(projectile, ship):
                continue

            _, _, overlap = _collision_info(projectile, ship)
            contact, impact_normal = _projectile_impact(projectile, ship, overlap)
            if contact is None:
                continue

            damage = projectile.current_damage
            ship.current_hp = max(0, ship.current_hp - damage)
            _apply_druuge_projectile_momentum(projectile, ship)
            BattleEffect.play_boom(damage)
            _destroy_projectile(projectile, effects, impact_normal, damage, contact)
            break


def _handle_projectile_asteroid_collisions(projectiles, asteroids, effects):
    for projectile in projectiles:
        if not _is_live_projectile(projectile):
            continue

        for asteroid in asteroids:
            if not asteroid.currently_alive:
                continue

            _, _, overlap = _collision_info(projectile, asteroid)
            contact, impact_normal = _projectile_impact(projectile, asteroid, overlap)
            if contact is None:
                continue

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            _destroy_projectile(projectile, effects, impact_normal, damage, contact)
            _destroy_asteroid(asteroid, effects)
            break


def _handle_projectile_planet_collisions(projectiles, planets, effects):
    for projectile in projectiles:
        if not _is_live_projectile(projectile):
            continue

        for planet in planets:
            _, _, overlap = _collision_info(projectile, planet)
            contact, impact_normal = _projectile_impact(projectile, planet, overlap)
            if contact is None:
                continue

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            _destroy_projectile(projectile, effects, impact_normal, damage, contact)
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


def _apply_druuge_projectile_momentum(projectile, ship):
    if projectile.projectile_name != "DruugeA1" or not hasattr(ship, "add_impulse"):
        return

    speed = math.hypot(projectile.velocity[0], projectile.velocity[1])
    if speed <= 0:
        return

    projectile_direction = [
        projectile.velocity[0] / speed,
        projectile.velocity[1] / speed,
    ]
    parent_mass = _mass(projectile.parent)
    ship_mass = _mass(ship)
    momentum = parent_mass * projectile.RECOIL_INCREMENT
    ship.add_impulse(
        projectile_direction[0] * momentum / ship_mass,
        projectile_direction[1] * momentum / ship_mass,
    )


def _projectile_can_hit_ship(projectile, ship):
    if ship.player != projectile.player:
        return True

    if ship != projectile.parent or not projectile.hit_parent:
        return False

    if getattr(projectile, "has_left_parent", False):
        return True

    _, _, overlap = _collision_info(projectile, ship)
    if _objects_overlap(projectile, ship, overlap):
        return False

    projectile.has_left_parent = True
    return False


def _handle_laser_collisions(lasers, ships, projectiles, asteroids, planets, effects):
    for laser in lasers:
        if not _is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        if hasattr(laser, "calculate_end_position"):
            laser.calculate_end_position()

        targets = _laser_targets(laser, ships, projectiles, asteroids, planets)
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

        if isinstance(target, Planet):
            continue

        if hasattr(target, "current_hp"):
            target.current_hp = max(0, target.current_hp - damage)
            if isinstance(target, Ability) and target.current_hp <= 0:
                _destroy_projectile(target, effects, normal, damage, contact)
        elif isinstance(target, Asteroid):
            _destroy_asteroid(target, effects)


def _laser_targets(laser, ships, projectiles, asteroids, planets):
    explicit_target = getattr(laser, "target", None)
    if explicit_target is not None:
        return [explicit_target] if _is_live_laser_target(explicit_target) else []

    targets = []
    for ship in ships:
        if not _is_live_laser_target(ship):
            continue
        if ship.player != laser.player or (ship == laser.parent and laser.hit_parent):
            targets.append(ship)

    for projectile in projectiles:
        if not _is_live_laser_target(projectile):
            continue
        if projectile.player != laser.player or laser.hit_self:
            targets.append(projectile)

    targets.extend(asteroid for asteroid in asteroids if _is_live_laser_target(asteroid))
    targets.extend(planets)
    return sorted(targets, key=lambda target: _distance_between(laser.parent, target))


def _is_live_laser_target(target):
    if isinstance(target, Planet):
        return True
    if isinstance(target, SpaceShip):
        return target.current_hp > 0
    if isinstance(target, Ability):
        return _is_live_projectile(target)
    if isinstance(target, Asteroid):
        return target.currently_alive
    return getattr(target, "currently_alive", True)


def _laser_hit_info(laser, target):
    start = laser.parent.position
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


def _view_center_and_size(positions):
    p1_pos, p2_pos = positions
    delta = _wrapped_delta(p1_pos, p2_pos)
    view_center = [
        (p1_pos[0] + delta[0] / 2) % const.ARENA_SIZE,
        (p1_pos[1] + delta[1] / 2) % const.ARENA_SIZE,
    ]
    distance = math.hypot(delta[0], delta[1])
    min_view_size = const.SCREEN_HEIGHT / const.MAX_ZOOM
    view_size = min(max(distance / 0.8, min_view_size), const.ARENA_SIZE / 2)
    scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
    return view_center, const.SCREEN_HEIGHT / scale_factor


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
        obj.velocity[0] -= 2 * velocity_along_normal * normal[0]
        obj.velocity[1] -= 2 * velocity_along_normal * normal[1]
        collided_while_approaching = True
    else:
        collided_while_approaching = False

    _separate_from_static_body(obj, static_body, normal, overlap, extra_clearance)
    return collided_while_approaching


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
