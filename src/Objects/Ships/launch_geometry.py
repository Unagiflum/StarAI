"""Shared geometry for parent-mounted ability emitters."""

import math

import src.const as const


def _first_surface(sprites):
    surface = sprites[0]
    while isinstance(surface, (tuple, list)):
        surface = surface[0]
    return surface


def gun_world_position(parent, gun_location, *, rotation=None, position=None):
    """Transform a coordinate on the parent's 00 sprite into world space."""
    sprite = _first_surface(parent.sprites)
    width, height = sprite.get_size()
    scale = getattr(parent, "sprite_scale", 1.0)
    local_x = gun_location[0] * scale - width // 2
    local_y = gun_location[1] * scale - height // 2
    angle = math.radians(parent.rotation if rotation is None else rotation)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    origin = parent.position if position is None else position
    return [
        (origin[0] + cosine * local_x - sine * local_y) % const.ARENA_SIZE,
        (origin[1] + sine * local_x + cosine * local_y) % const.ARENA_SIZE,
    ]


def absolute_direction(parent, relative_direction):
    return (parent.rotation + relative_direction) % 360


def direction_vector(direction):
    angle = math.radians(direction)
    return math.sin(angle), -math.cos(angle)


def anchored_sprite_position(
    parent,
    gun_location,
    relative_direction,
    anchor_offset,
    *,
    rotation=None,
    position=None,
):
    """Place a sprite center so its heading-00 anchor lands on a gun point."""
    gun_rotation = parent.rotation if rotation is None else rotation
    muzzle = gun_world_position(
        parent,
        gun_location,
        rotation=gun_rotation,
        position=position,
    )
    direction = (gun_rotation + relative_direction) % 360
    angle = math.radians(direction)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    anchor_x = cosine * anchor_offset[0] - sine * anchor_offset[1]
    anchor_y = sine * anchor_offset[0] + cosine * anchor_offset[1]
    return [
        (muzzle[0] - anchor_x) % const.ARENA_SIZE,
        (muzzle[1] - anchor_y) % const.ARENA_SIZE,
    ]


def mask_projection_bounds(mask, direction):
    """Project opaque pixel centers onto a world-space firing direction."""
    width, height = mask.get_size()
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    forward_x, forward_y = direction_vector(direction)
    bounds = mask.get_bounding_rects()
    minimum = None
    maximum = None
    for rect in bounds:
        for y in range(rect.top, rect.bottom):
            for x in range(rect.left, rect.right):
                if not mask.get_at((x, y)):
                    continue
                projection = (
                    (x - center_x) * forward_x + (y - center_y) * forward_y
                )
                minimum = projection if minimum is None else min(minimum, projection)
                maximum = projection if maximum is None else max(maximum, projection)
    if minimum is None:
        return 0.0, 0.0
    return minimum, maximum


def launch_mask(ability, direction):
    if ability.omnidirectional:
        return ability.get_collision_mask()
    index = round(direction / const.TOTAL_SPRITE_STEP) % len(ability.masks)
    return ability.masks[index]


def place_projectile_at_gun(
    projectile,
    gun_location,
    relative_direction,
    *,
    gap_multiplier=None,
    inherit_parent_velocity=True,
    gun_rotation=None,
    launch_direction=None,
):
    """Place and propel a projectile using its opaque rear-most pixel span."""
    direction = (
        absolute_direction(projectile.parent, relative_direction)
        if launch_direction is None
        else launch_direction % 360
    )
    muzzle = gun_world_position(
        projectile.parent, gun_location, rotation=gun_rotation
    )
    mask = launch_mask(projectile, direction)
    rear_projection, _ = mask_projection_bounds(mask, direction)
    if gap_multiplier is None:
        gap_multiplier = 2 if projectile.omnidirectional else 1
    distance = const.PROJ_GAP * gap_multiplier - rear_projection
    forward_x, forward_y = direction_vector(direction)
    projectile.position = [
        (muzzle[0] + forward_x * distance) % const.ARENA_SIZE,
        (muzzle[1] + forward_y * distance) % const.ARENA_SIZE,
    ]
    projectile.previous_position = projectile.position.copy()
    projectile.rotation = direction
    projectile.heading = (
        0
        if projectile.omnidirectional
        else round(direction / const.TURN_ANGLE) % const.SHIP_DIRECTIONS
    )
    parent_x, parent_y = (
        projectile.parent.velocity if inherit_parent_velocity else (0, 0)
    )
    projectile.velocity = [
        forward_x * projectile.speed + parent_x * projectile.parent_vel,
        forward_y * projectile.speed + parent_y * projectile.parent_vel,
    ]
    return direction
