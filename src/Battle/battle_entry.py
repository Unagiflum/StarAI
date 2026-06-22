"""State, geometry, and rendering for ships entering a battle round."""

from dataclasses import dataclass, field
import math

import pygame

import src.const as const


YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


def _interpolate_color(start, end, progress):
    return tuple(
        round(start_channel + (end_channel - start_channel) * progress)
        for start_channel, end_channel in zip(start, end)
    )


def silhouette_color(age):
    """Return the yellow-to-red-to-black color for a trail silhouette."""
    if age < 0 or age > const.ENTRY_TRAIL_FADE_FRAMES:
        return None

    midpoint = const.ENTRY_TRAIL_FADE_FRAMES / 2
    if age <= midpoint:
        return _interpolate_color(YELLOW, RED, age / midpoint)
    return _interpolate_color(
        RED,
        BLACK,
        (age - midpoint) / midpoint,
    )


def silhouette_positions(ship, arrival_position=None):
    """Return the standard trail ordered from farthest to nearest."""
    arrival_position = arrival_position or ship.position
    direction_x, direction_y = _direction(ship.rotation)
    count = const.ENTRY_TRAIL_SILHOUETTES

    return tuple(
        (
            (arrival_position[0] - direction_x * distance) % const.ARENA_SIZE,
            (arrival_position[1] - direction_y * distance) % const.ARENA_SIZE,
        )
        for distance in (
            const.ENTRY_TRAIL_SPACING * index
            for index in range(count - 1, -1, -1)
        )
    )


def pkunk_rebirth_silhouette_lines(ship, arrival_position=None):
    """Return the Pkunk rebirth's four close diagonal trails."""
    arrival_position = arrival_position or ship.position
    count = const.ENTRY_TRAIL_SILHOUETTES
    spacing = max(ship.size) + const.PKUNK_REBIRTH_TRAIL_GAP
    distances = tuple(
        spacing * index for index in range(count - 1, -1, -1)
    )

    return tuple(
        tuple(
            (
                (arrival_position[0] - direction_x * distance)
                % const.ARENA_SIZE,
                (arrival_position[1] - direction_y * distance)
                % const.ARENA_SIZE,
            )
            for distance in distances
        )
        for direction_x, direction_y in (
            _direction(ship.rotation + angle)
            for angle in const.PKUNK_REBIRTH_TRAIL_ANGLES
        )
    )


def _direction(rotation):
    angle = math.radians(rotation)
    direction_x = math.sin(angle)
    direction_y = -math.cos(angle)
    if abs(direction_x) < 1e-12:
        direction_x = 0.0
    if abs(direction_y) < 1e-12:
        direction_y = 0.0
    return direction_x, direction_y


@dataclass(frozen=True)
class FixedCameraTarget:
    position: tuple[float, float]


@dataclass
class EntryState:
    started_frame: int
    entering_ships: tuple
    arrival_targets: dict
    camera_targets: tuple
    trackable_states: dict
    diagonal_trail_ships: frozenset = field(default_factory=frozenset)


def start_entry(
    entering_ships,
    player1,
    player2,
    frame_id,
    *,
    diagonal_trail_ships=(),
):
    entering_ships = tuple(entering_ships)
    if not entering_ships:
        return None

    entering_set = set(entering_ships)
    arrival_targets = {
        ship: tuple(ship.position)
        for ship in entering_ships
    }
    trackable_states = {
        ship: getattr(ship, "trackable", True)
        for ship in entering_ships
    }
    for ship in entering_ships:
        ship.trackable = False
    camera_targets = tuple(
        FixedCameraTarget(arrival_targets[ship])
        if ship in entering_set else ship
        for ship in (player1, player2)
    )
    return EntryState(
        started_frame=frame_id,
        entering_ships=entering_ships,
        arrival_targets=arrival_targets,
        camera_targets=camera_targets,
        trackable_states=trackable_states,
        diagonal_trail_ships=frozenset(diagonal_trail_ships),
    )


def finish_entry(entry):
    for ship, trackable in entry.trackable_states.items():
        ship.trackable = trackable


def entry_duration_frames():
    return (
        (const.ENTRY_TRAIL_SILHOUETTES - 1)
        * const.ENTRY_TRAIL_STAGGER_FRAMES
        + const.ENTRY_TRAIL_FADE_FRAMES
        + 1
    )


def entry_complete(entry, frame_id):
    return frame_id - entry.started_frame >= entry_duration_frames()


def visible_silhouettes(entry, ship, frame_id):
    visible = []
    elapsed = frame_id - entry.started_frame
    if ship in entry.diagonal_trail_ships:
        lines = pkunk_rebirth_silhouette_lines(
            ship, entry.arrival_targets[ship]
        )
    else:
        lines = (silhouette_positions(ship, entry.arrival_targets[ship]),)
    for index in range(const.ENTRY_TRAIL_SILHOUETTES):
        age = elapsed - index * const.ENTRY_TRAIL_STAGGER_FRAMES
        color = (
            YELLOW
            if index == const.ENTRY_TRAIL_SILHOUETTES - 1 and age >= 0
            else silhouette_color(age)
        )
        if color is not None:
            visible.extend((line[index], color) for line in lines)
    return visible


def draw_entry_silhouettes(
    screen,
    entry,
    frame_id,
    scale_factor,
    translation,
):
    for ship in entry.entering_ships:
        mask = ship.masks[ship.heading]
        for position, color in visible_silhouettes(
            entry, ship, frame_id
        ):
            silhouette = mask.to_surface(
                setcolor=(*color, 255),
                unsetcolor=(0, 0, 0, 0),
            )
            scaled = pygame.transform.smoothscale_by(
                silhouette, scale_factor
            )
            _draw_wrapped(
                screen,
                scaled,
                position,
                scale_factor,
                translation,
            )


def _draw_wrapped(screen, image, position, scale_factor, translation):
    rect = image.get_rect()
    screen_x = int((position[0] + translation[0]) * scale_factor)
    screen_y = int((position[1] + translation[1]) * scale_factor)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
            pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor
            if (
                -rect.width <= pos_x <= const.SCREEN_HEIGHT + rect.width
                and -rect.height <= pos_y <= const.SCREEN_HEIGHT + rect.height
            ):
                screen.blit(
                    image,
                    (
                        const.SCREEN_LEFT + pos_x - rect.width // 2,
                        pos_y - rect.height // 2,
                    ),
                )
