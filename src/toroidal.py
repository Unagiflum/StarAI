"""Geometry helpers for the wrapped battle arena."""

import math

import src.const as const


def wrapped_delta(from_position, to_position, arena_size=const.ARENA_SIZE):
    """Return the shortest signed displacement from one position to another."""
    dx = to_position[0] - from_position[0]
    dy = to_position[1] - from_position[1]
    half_size = arena_size / 2

    if abs(dx) > half_size:
        dx += -arena_size if dx > 0 else arena_size
    if abs(dy) > half_size:
        dy += -arena_size if dy > 0 else arena_size

    return [dx, dy]


def wrapped_distance(from_position, to_position, arena_size=const.ARENA_SIZE):
    delta = wrapped_delta(from_position, to_position, arena_size)
    return math.hypot(delta[0], delta[1])


def nearest_position(position, reference, arena_size=const.ARENA_SIZE):
    """Return the wrapped image of position nearest to reference."""
    delta = wrapped_delta(reference, position, arena_size)
    return [reference[0] + delta[0], reference[1] + delta[1]]


def wrapped_midpoint(first, second, arena_size=const.ARENA_SIZE):
    delta = wrapped_delta(first, second, arena_size)
    return [
        (first[0] + delta[0] / 2) % arena_size,
        (first[1] + delta[1] / 2) % arena_size,
    ]


def view_center_and_size(
    positions,
    screen_height=const.SCREEN_HEIGHT,
    max_zoom=const.MAX_ZOOM,
    arena_size=const.ARENA_SIZE,
):
    """Return the wrapped midpoint and world-space size for a two-target view."""
    first, second = positions
    center = wrapped_midpoint(first, second, arena_size)
    distance = wrapped_distance(first, second, arena_size)
    minimum_size = screen_height / max_zoom
    requested_size = min(max(distance / 0.8, minimum_size), arena_size / 2)
    scale = min(max_zoom, screen_height / requested_size)
    return center, screen_height / scale
