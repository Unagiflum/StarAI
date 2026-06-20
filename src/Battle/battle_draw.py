import pygame
from src.UI import ui
from src.Battle.effects import BattleEffect
from src.Objects.object import ThrustMarker
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ability import Ability
from src.Objects.Space.space_obj import Star, Planet, Asteroid
from src.Battle.status_bar import draw_player_status
import src.const as const
from src.toroidal import view_center_and_size, wrapped_midpoint


def calculate_view_parameters(game_objects, camera_targets=None):
    targets = camera_targets
    if targets is None:
        targets = [
            obj for obj in game_objects
            if isinstance(obj, SpaceShip) and obj.currently_alive and obj.current_hp > 0
        ]

    if len(targets) == 1:
        view_size = (const.SCREEN_HEIGHT / const.MAX_ZOOM) * 1.5
        scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
        translation = [
            const.SCREEN_HEIGHT / (2 * scale_factor) - targets[0].position[0],
            const.SCREEN_HEIGHT / (2 * scale_factor) - targets[0].position[1],
        ]
        return scale_factor, translation
    if len(targets) < 2:
        return 1.0, [0, 0]

    center, view_size = view_center_and_size(
        [targets[0].position, targets[1].position]
    )

    scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
    translation = [
        const.SCREEN_HEIGHT / (2 * scale_factor) - center[0],
        const.SCREEN_HEIGHT / (2 * scale_factor) - center[1]
    ]

    return scale_factor, translation


def draw_battle(screen, game_objects, border_rect, border_color, camera_targets=None):
    scale_factor, translation = calculate_view_parameters(game_objects, camera_targets)

    players = [
        obj for obj in game_objects
        if isinstance(obj, SpaceShip) and obj.currently_alive and obj.current_hp > 0
    ]
    if len(players) == 2:
        p1_pos, p2_pos = players[0].position, players[1].position
        midpoint = wrapped_midpoint(p1_pos, p2_pos)
    else:
        midpoint = [const.ARENA_SIZE / 2, const.ARENA_SIZE / 2]

    screen.fill(ui.BLACK)
    screen.set_clip(border_rect)

    # Update and draw star layers
    stars = [obj for obj in game_objects if isinstance(obj, Star)]
    for depth in range(const.STAR_DEPTHS):
        parallax_factor = 0.5 + 0.5 * (depth / (const.STAR_DEPTHS - 1))
        Star.update_depth_surface(depth, stars, scale_factor, translation, midpoint, parallax_factor)
        screen.blit(Star.depth_surfaces[depth], (0, 0))

    # Draw other objects normally
    for obj in game_objects:
        if isinstance(obj, Planet):
            obj.draw(screen, scale_factor, translation)

    for obj in game_objects:
        if isinstance(obj, ThrustMarker):
            obj.draw(screen, scale_factor, translation)

    for obj in game_objects:
        if isinstance(obj, Asteroid):
            obj.draw(screen, scale_factor, translation)

    for obj in game_objects:
        if isinstance(obj, Ability):
            obj.draw(screen, scale_factor, translation)

    for obj in game_objects:
        if isinstance(obj, SpaceShip):
            obj.draw(screen, scale_factor, translation)

    for obj in game_objects:
        if isinstance(obj, BattleEffect):
            obj.draw(screen, scale_factor, translation)

    pygame.draw.rect(screen, border_color, border_rect, 2)
    screen.set_clip(None)

    # Draw status bars for each surviving player. During the post-fight winner
    # view only one ship remains alive, but its crew and battery are still
    # useful and should remain visible.
    if players:
        BAR_WIDTH = 30
        BAR_SPACING = 5

        # Calculate total width of both bars + spacing
        TOTAL_BAR_WIDTH = (BAR_WIDTH * 2) + BAR_SPACING

        # Calculate panel widths (space between arena edge and screen edge)
        LEFT_PANEL_WIDTH = const.SCREEN_LEFT
        RIGHT_PANEL_WIDTH = const.SCREEN_WIDTH - (const.SCREEN_LEFT + const.SCREEN_HEIGHT)

        # Center bars in panels
        P1_X = const.SCREEN_LEFT - TOTAL_BAR_WIDTH - ((LEFT_PANEL_WIDTH - TOTAL_BAR_WIDTH) // 2)
        P2_X = (const.SCREEN_LEFT + const.SCREEN_HEIGHT) + ((RIGHT_PANEL_WIDTH - TOTAL_BAR_WIDTH) // 2)

        BASE_Y = const.SCREEN_HEIGHT // 2

        for ship in players:
            status_x = P1_X if ship.player == 1 else P2_X
            draw_player_status(screen, ship, status_x, BASE_Y, BAR_WIDTH, BAR_SPACING)

    pygame.display.flip()
