"""Read-only collision geometry for the wrapped battle arena."""

import math

import src.const as const
from src.Objects.object import Object
from src.Objects.Space.space_obj import Planet
from src.toroidal import nearest_position, wrapped_delta, wrapped_distance


def laser_hit_info(laser, target):
    start = getattr(laser, "start_position", laser.parent.position)
    end = getattr(laser, "end_position", laser.position)
    segment = wrapped_segment(start, end)
    distance = distance_from_segment_to_point(segment[0], segment[1], target.position)
    
    laser_width = getattr(laser, "LASER_WIDTH", 1)
    
    if distance > radius(target) + (laser_width / 2.0):
        return None

    target_mask = get_collision_mask(target)
    if target_mask is not None:
        contact = sample_laser_mask_hit(segment[0], segment[1], target, target_mask, laser_width)
    else:
        contact = segment_circle_intercept(
            segment[0], segment[1], target.position, radius(target)
        )

    if contact is None:
        return None

    normal = normal_from_target(
        target, contact, segment_direction(segment[0], segment[1])
    )
    return {
        "target": target,
        "contact": contact,
        "normal": normal,
        "distance": math.hypot(contact[0] - segment[0][0], contact[1] - segment[0][1]),
    }


def wrapped_segment(start, end):
    delta = wrapped_delta(start, end)
    return start, [start[0] + delta[0], start[1] + delta[1]]


def sample_laser_mask_hit(start, end, target, target_mask, laser_width=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None
        
    steps = max(1, int(length))
    nx = dy / length
    ny = -dx / length
    
    target_center = nearest_position(target.position, start)
    # Pixel coordinates are relative to the full mask canvas. Gameplay bounds
    # may intentionally exclude transparent sprite padding.
    target_size = target_mask.get_size()
    left = target_center[0] - target_size[0] / 2
    top = target_center[1] - target_size[1] / 2

    half_width = laser_width / 2.0
    width_steps = max(1, int(laser_width))
    offsets = []
    
    if width_steps == 1:
        offsets = [0.0]
    else:
        for i in range(width_steps):
            offsets.append((i / (width_steps - 1)) * laser_width - half_width)
            
    # Sort offsets by absolute distance to center so central hit is favored
    offsets.sort(key=abs)

    for step in range(steps + 1):
        ratio = step / steps
        base_x = start[0] + dx * ratio
        base_y = start[1] + dy * ratio
        
        for offset in offsets:
            x = base_x + nx * offset
            y = base_y + ny * offset
            
            mask_x = int(x - left)
            mask_y = int(y - top)
            
            if (
                0 <= mask_x < target_mask.get_size()[0]
                and 0 <= mask_y < target_mask.get_size()[1]
                and target_mask.get_at((mask_x, mask_y))
            ):
                return [x, y]

    return None


def collision_info(obj, other):
    delta = wrapped_delta(other.position, obj.position)
    distance = math.hypot(delta[0], delta[1])
    radius_sum = radius(obj) + radius(other)
    overlap = radius_sum - distance

    if distance == 0:
        return [1.0, 0.0], distance, overlap

    return [delta[0] / distance, delta[1] / distance], distance, overlap


def objects_overlap(obj, other, overlap):
    if overlap <= 0 and not mask_broadphase_overlap(obj, other):
        return False

    obj_mask = get_collision_mask(obj)
    other_mask = get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return True

    obj_size = obj_mask.get_size()
    other_size = other_mask.get_size()
    delta = wrapped_delta(other.position, obj.position)
    offset = (
        int(round(-delta[0] + obj_size[0] / 2 - other_size[0] / 2)),
        int(round(-delta[1] + obj_size[1] / 2 - other_size[1] / 2)),
    )
    return obj_mask.overlap(other_mask, offset) is not None


def solid_sweep_overlap(obj, other, step_size=12):
    """
    Checks for collision along the swept path.
    If a collision is found, updates obj and other positions to the point of first contact,
    and returns True. Otherwise returns False.
    """
    obj_previous = sweep_previous_position(obj)
    other_previous = sweep_previous_position(other)
    
    if obj_previous == obj.position and other_previous == other.position:
        _, _, overlap = collision_info(obj, other)
        return objects_overlap(obj, other, overlap)

    obj_delta = wrapped_delta(obj_previous, obj.position)
    other_delta = wrapped_delta(other_previous, other.position)
    
    relative_delta = [
        obj_delta[0] - other_delta[0],
        obj_delta[1] - other_delta[1],
    ]
    relative_distance = math.hypot(relative_delta[0], relative_delta[1])
    
    if relative_distance <= step_size:
        _, _, overlap = collision_info(obj, other)
        return objects_overlap(obj, other, overlap)

    steps = max(1, int(math.ceil(relative_distance / step_size)))
    
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
        
        if objects_overlap_at_positions(obj, other, obj_position, other_position):
            if getattr(obj, "can_move", True):
                obj.position = obj_position
            if getattr(other, "can_move", True):
                other.position = other_position
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
        _, _, overlap = collision_info(ship, candidate)
        if objects_overlap(ship, candidate, overlap):
            return True

    return False


def ship_shape_change_blocked(ship, candidate_masks, candidate_size):
    """Return whether a prospective ship shape overlaps a collidable object."""
    candidates = []
    if ship.opponent and ship.opponent.current_hp > 0:
        candidates.append(ship.opponent)
    candidates.extend(obj for obj in ship.asteroids if obj.currently_alive)
    if ship.planet:
        candidates.append(ship.planet)
    candidates.extend(
        obj
        for obj in (*ship.friendly_objects, *ship.enemy_objects)
        if (
            obj is not ship
            and getattr(obj, "can_collide", False)
            and getattr(obj, "currently_alive", True)
            and getattr(obj, "current_hp", 1) > 0
        )
    )

    original_masks = ship.masks
    original_size = ship.size
    ship.masks = candidate_masks
    ship.size = list(candidate_size)
    try:
        for candidate in dict.fromkeys(candidates):
            _, _, overlap = collision_info(ship, candidate)
            if objects_overlap(ship, candidate, overlap):
                return True
        return False
    finally:
        ship.masks = original_masks
        ship.size = original_size


def projectile_impact(projectile, other, overlap):
    swept = swept_impact(projectile, other)
    if swept[0] is not None:
        return swept

    if objects_overlap(projectile, other, overlap):
        return estimated_impact(projectile, other)

    normal, _, _ = collision_info(projectile, other)
    return None, normal


def swept_impact(obj, other):
    obj_previous = sweep_previous_position(obj)
    other_previous = sweep_previous_position(other)
    obj_delta = wrapped_delta(obj_previous, obj.position)
    other_delta = wrapped_delta(other_previous, other.position)
    relative_delta = [
        obj_delta[0] - other_delta[0],
        obj_delta[1] - other_delta[1],
    ]
    relative_distance = math.hypot(relative_delta[0], relative_delta[1])
    if relative_distance <= 0:
        normal, _, _ = collision_info(obj, other)
        return None, normal

    steps = max(1, int(math.ceil(relative_distance / sweep_step_size(obj, other))))
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
        if objects_overlap_at_positions(obj, other, obj_position, other_position):
            return estimated_impact_at_positions(
                obj, other, obj_position, other_position
            )

    normal, _, _ = collision_info(obj, other)
    return None, normal


def sweep_previous_position(obj):
    # Defaults keep partially initialized collision test doubles compatible.
    if not getattr(obj, "can_move", True):
        return obj.position
    return getattr(obj, "previous_position", obj.position)


def sweep_step_size(obj, other):
    obj_size = collision_size(obj)
    other_size = collision_size(other)
    min_dimension = min(obj_size[0], obj_size[1], other_size[0], other_size[1])
    return max(2, min(12, min_dimension / 3))


def estimated_impact(obj, other):
    normal, _, _ = collision_info(obj, other)
    contact = mask_overlap_contact(obj, other, obj.position, other.position)
    return contact or contact_point(other, normal), normal


def estimated_impact_at_positions(obj, other, obj_position, other_position):
    delta = wrapped_delta(other_position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    if distance == 0:
        normal = [1.0, 0.0]
    else:
        normal = [delta[0] / distance, delta[1] / distance]
    contact = mask_overlap_contact(obj, other, obj_position, other_position)
    return contact or contact_point_at_position(other, normal, other_position), normal


def mask_overlap_contact(obj, other, obj_position, other_position):
    """Return the world-space centroid of the objects' opaque overlap."""
    obj_mask = get_collision_mask(obj)
    other_mask = get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return None

    obj_size = obj_mask.get_size()
    other_size = other_mask.get_size()
    delta = wrapped_delta(other_position, obj_position)
    offset = (
        int(round(-delta[0] + obj_size[0] / 2 - other_size[0] / 2)),
        int(round(-delta[1] + obj_size[1] / 2 - other_size[1] / 2)),
    )
    overlap = obj_mask.overlap_mask(other_mask, offset)
    if overlap.count() == 0:
        return None

    centroid_x, centroid_y = overlap.centroid()
    return [
        (obj_position[0] + centroid_x - obj_size[0] / 2) % const.ARENA_SIZE,
        (obj_position[1] + centroid_y - obj_size[1] / 2) % const.ARENA_SIZE,
    ]


def contact_point(target, normal):
    return contact_point_at_position(target, normal, target.position)


def contact_point_at_position(target, normal, target_position):
    mask = get_collision_mask(target)
    if mask is not None:
        offset = opaque_mask_contact_offset(mask, normal)
        if offset is not None:
            return [
                (target_position[0] + offset[0]) % const.ARENA_SIZE,
                (target_position[1] + offset[1]) % const.ARENA_SIZE,
            ]

    return [
        (target_position[0] + normal[0] * radius(target)) % const.ARENA_SIZE,
        (target_position[1] + normal[1] * radius(target)) % const.ARENA_SIZE,
    ]


def opaque_mask_contact_offset(mask, normal):
    """Find the outermost opaque pixel nearest the ray along ``normal``."""
    length = math.hypot(normal[0], normal[1])
    bounds = mask.get_bounding_rects()
    if length == 0 or not bounds:
        return None

    nx = normal[0] / length
    ny = normal[1] / length
    center_x = mask.get_size()[0] / 2
    center_y = mask.get_size()[1] / 2
    best = None

    for rect in bounds:
        for y in range(rect.top, rect.bottom):
            for x in range(rect.left, rect.right):
                if not mask.get_at((x, y)):
                    continue
                offset_x = x - center_x
                offset_y = y - center_y
                forward = offset_x * nx + offset_y * ny
                if forward < 0:
                    continue
                perpendicular = abs(offset_x * ny - offset_y * nx)
                score = (perpendicular, -forward)
                if best is None or score < best[0]:
                    best = (score, [offset_x, offset_y])

    return best[1] if best is not None else None


def objects_overlap_at_positions(obj, other, obj_position, other_position):
    delta = wrapped_delta(other_position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    overlap = radius(obj) + radius(other) - distance
    if overlap <= 0 and not mask_broadphase_overlap_at_positions(
        obj, other, obj_position, other_position
    ):
        return False

    obj_mask = get_collision_mask(obj)
    other_mask = get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return True

    obj_size = obj_mask.get_size()
    other_size = other_mask.get_size()
    offset = (
        int(round(-delta[0] + obj_size[0] / 2 - other_size[0] / 2)),
        int(round(-delta[1] + obj_size[1] / 2 - other_size[1] / 2)),
    )
    return obj_mask.overlap(other_mask, offset) is not None


def get_collision_mask(obj):
    if isinstance(obj, Object):
        return obj.get_collision_mask()

    # Compatibility boundary for collision test doubles.
    get_mask = getattr(obj, "get_collision_mask", None)
    return get_mask() if get_mask is not None else None


def collision_size(obj):
    if isinstance(obj, Planet):
        return [obj.diameter, obj.diameter]
    size = getattr(obj, "size", None)
    if size is not None:
        return size
    mask = get_collision_mask(obj)
    return mask.get_size() if mask is not None else [0, 0]


def mask_broadphase_overlap(obj, other):
    return mask_broadphase_overlap_at_positions(
        obj, other, obj.position, other.position
    )


def mask_broadphase_overlap_at_positions(obj, other, obj_position, other_position):
    obj_mask = get_collision_mask(obj)
    other_mask = get_collision_mask(other)
    if obj_mask is None or other_mask is None:
        return False

    delta = wrapped_delta(other_position, obj_position)
    distance = math.hypot(delta[0], delta[1])
    return mask_radius(obj) + mask_radius(other) - distance > 0


def distance_between(obj, other):
    return wrapped_distance(obj.position, other.position)


def distance_from_segment_to_point(start, end, point):
    point = nearest_position(point, start)
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


def segment_circle_intercept(start, end, center, radius):
    center = nearest_position(center, start)
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


def normal_from_target(target, contact, fallback):
    target_center = nearest_position(target.position, contact)
    dx = contact[0] - target_center[0]
    dy = contact[1] - target_center[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return [-fallback[0], -fallback[1]]
    return [dx / length, dy / length]


def segment_direction(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return [0, -1]
    return [dx / length, dy / length]


def radius(obj):
    if isinstance(obj, Planet):
        return obj.diameter / 2
    return max(collision_size(obj)) / 2


def mask_radius(obj):
    size = collision_size(obj)
    return math.hypot(size[0], size[1]) / 2
