import pygame
from src.UI import ui
from src.Battle.effects import BattleEffect
from src.Objects.object import ThrustMarker
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ability import Ability
from src.Objects.Space.space_obj import Star, Planet, Asteroid
from src.Battle.status_bar import draw_player_status
import src.const as const


def calculate_view_parameters(game_objects):
    players = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
    if len(players) != 2:
        return 1.0, [0, 0]

    p1_pos = players[0].position
    p2_pos = players[1].position

    dx = p2_pos[0] - p1_pos[0]
    dy = p2_pos[1] - p1_pos[1]

    if abs(dx) > const.ARENA_SIZE / 2:
        dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
    if abs(dy) > const.ARENA_SIZE / 2:
        dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

    mid_x = (p1_pos[0] + dx / 2) % const.ARENA_SIZE
    mid_y = (p1_pos[1] + dy / 2) % const.ARENA_SIZE

    distance = (dx ** 2 + dy ** 2) ** 0.5
    view_size = min(distance / 0.8, const.ARENA_SIZE / 2)

    scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
    translation = [
        const.SCREEN_HEIGHT / (2 * scale_factor) - mid_x,
        const.SCREEN_HEIGHT / (2 * scale_factor) - mid_y
    ]

    return scale_factor, translation


def draw_battle(screen, game_objects, border_rect, border_color):
    scale_factor, translation = calculate_view_parameters(game_objects)

    players = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
    if len(players) == 2:
        p1_pos, p2_pos = players[0].position, players[1].position
        dx = p2_pos[0] - p1_pos[0]
        dy = p2_pos[1] - p1_pos[1]

        if abs(dx) > const.ARENA_SIZE / 2:
            dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
        if abs(dy) > const.ARENA_SIZE / 2:
            dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE

        midpoint = [(p1_pos[0] + dx / 2) % const.ARENA_SIZE,
                    (p1_pos[1] + dy / 2) % const.ARENA_SIZE]
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

    # Draw status bars for both players
    if len(players) == 2:
        BAR_WIDTH = 30
        BAR_SPACING = 5
        DASH_HEIGHT = 8

        # Calculate total width of both bars + spacing
        TOTAL_BAR_WIDTH = (BAR_WIDTH * 2) + BAR_SPACING

        # Calculate bar height using same formula as StatusBar
        max_height = max(players[0].max_hp, players[0].max_energy,
                         players[1].max_hp, players[1].max_energy)
        bar_height = ((max_height + 1) // 2) * (DASH_HEIGHT + 2) + 2  # Using new dash_height of 6

        # Calculate total width of both bars + spacing
        TOTAL_BAR_WIDTH = (BAR_WIDTH * 2) + BAR_SPACING

        # Calculate panel widths (space between arena edge and screen edge)
        LEFT_PANEL_WIDTH = const.SCREEN_LEFT
        RIGHT_PANEL_WIDTH = const.SCREEN_WIDTH - (const.SCREEN_LEFT + const.SCREEN_HEIGHT)

        # Center bars in panels
        P1_X = const.SCREEN_LEFT - TOTAL_BAR_WIDTH - ((LEFT_PANEL_WIDTH - TOTAL_BAR_WIDTH) // 2)
        P2_X = (const.SCREEN_LEFT + const.SCREEN_HEIGHT) + ((RIGHT_PANEL_WIDTH - TOTAL_BAR_WIDTH) // 2)

        BASE_Y = const.SCREEN_HEIGHT // 2

        draw_player_status(screen, players[0], P1_X, BASE_Y, BAR_WIDTH, BAR_SPACING)
        draw_player_status(screen, players[1], P2_X, BASE_Y, BAR_WIDTH, BAR_SPACING)

    pygame.display.flip()
