"""Frame interpolation utilities for VIDEO_FPS_MULTIPLIER > 1."""

import src.const as const
from src.toroidal import wrapped_delta


def interpolated_position(obj, t):
    """Lerp between previous_position and position, wrapping toroidally."""
    if const.VIDEO_FPS_MULTIPLIER <= 1:
        return obj.position
    prev = getattr(obj, "previous_position", obj.position)
    dx, dy = wrapped_delta(prev, obj.position)
    return [
        (prev[0] + dx * t) % const.ARENA_SIZE,
        (prev[1] + dy * t) % const.ARENA_SIZE,
    ]


def interpolated_sprite_index(obj, t):
    """Return the sprite index for an interpolated rotation.

    Heading is a gameplay index (0..SHIP_DIRECTIONS-1).
    Sprite index is (0..TOTAL_SPRITE_DIRECTIONS-1).
    Between physics frames, we interpolate across video sub-steps.
    """
    if const.VIDEO_FPS_MULTIPLIER <= 1:
        return const.heading_to_sprite_index(getattr(obj, "heading", 0))

    prev_heading = getattr(obj, "previous_heading", obj.heading)
    curr_heading = getattr(obj, "heading", 0)

    # Find shortest rotation direction.
    diff = curr_heading - prev_heading
    if diff > const.SHIP_DIRECTIONS // 2:
        diff -= const.SHIP_DIRECTIONS
    elif diff < -(const.SHIP_DIRECTIONS // 2):
        diff += const.SHIP_DIRECTIONS

    # Interpolated heading in total-sprite-index space.
    prev_sprite = prev_heading * const.VIDEO_FPS_MULTIPLIER
    sprite_diff = diff * const.VIDEO_FPS_MULTIPLIER
    interp_sprite = prev_sprite + sprite_diff * t
    return round(interp_sprite) % const.TOTAL_SPRITE_DIRECTIONS
