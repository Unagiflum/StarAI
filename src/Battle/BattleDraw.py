import pygame
from src.UI import UI
from src.Objects.Object import ThrustMarker
from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Space.SpaceObject import Planet, Star
from src.Const import ARENA_SIZE
from src.UI.UI import SCREEN_HEIGHT


def calculate_view_parameters(game_objects):
    # Find players
    players = [obj for obj in game_objects if isinstance(obj, SpaceShip)]
    if len(players) != 2:
        return 1.0, [0, 0]  # Default values if not enough players

    p1_pos = players[0].position
    p2_pos = players[1].position

    arena_width = ARENA_SIZE
    arena_height = ARENA_SIZE
    view_width = SCREEN_HEIGHT
    view_height = SCREEN_HEIGHT

    return UI.SCREEN_HEIGHT/ARENA_SIZE, [0.0,0.0]

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