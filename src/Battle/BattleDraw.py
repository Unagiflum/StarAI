import pygame
from src.UI import UI
from src.Objects.Object import ThrustMarker
from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Space.SpaceObject import Planet, Star
from src.Const import ARENA_SIZE
from src.UI.UI import SCREEN_HEIGHT


def calculate_view_parameters(game_objects):
    players = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
    if len(players) != 2:
        return 1.0, [0, 0]

    p1_pos = players[0].position
    p2_pos = players[1].position

    # Calculate shortest distance considering wrap-around
    dx = p2_pos[0] - p1_pos[0]
    dy = p2_pos[1] - p1_pos[1]

    # Adjust for wrap-around
    if abs(dx) > ARENA_SIZE / 2:
        dx = dx - ARENA_SIZE if dx > 0 else dx + ARENA_SIZE
    if abs(dy) > ARENA_SIZE / 2:
        dy = dy - ARENA_SIZE if dy > 0 else dy + ARENA_SIZE

    # Midpoint between ships (considering wrap-around)
    mid_x = (p1_pos[0] + dx / 2) % ARENA_SIZE
    mid_y = (p1_pos[1] + dy / 2) % ARENA_SIZE

    # Calculate required view size (with 10% margin)
    distance = (dx ** 2 + dy ** 2) ** 0.5
    view_size = min(distance / 0.8, ARENA_SIZE/2)  # 10% margin on each side

    # Calculate scale factor
    scale_factor = min(1.0, UI.SCREEN_HEIGHT / view_size)

    # Calculate translation to center the view
    translation = [
        UI.SCREEN_HEIGHT / (2 * scale_factor) - mid_x,
        UI.SCREEN_HEIGHT / (2 * scale_factor) - mid_y
    ]

    return scale_factor, translation

def draw_battle(screen, game_objects, border_rect, border_color):
    scale_factor, translation = calculate_view_parameters(game_objects)

    screen.fill(UI.BLACK)
    screen.set_clip(border_rect)

    # Draw stars first (background)
    for obj in game_objects:
        if isinstance(obj, Star):
            obj.draw(screen, scale_factor, translation)

    # Draw planet
    for obj in game_objects:
        if isinstance(obj, Planet):
            obj.draw(screen, scale_factor, translation)

    # Draw thrust markers
    for obj in game_objects:
        if isinstance(obj, ThrustMarker):
            obj.draw(screen, scale_factor, translation)

    # Draw spaceships
    for obj in game_objects:
        if isinstance(obj, SpaceShip):
            obj.draw(screen, scale_factor, translation)

    pygame.draw.rect(screen, border_color, border_rect, 2)
    screen.set_clip(None)
    pygame.display.flip()