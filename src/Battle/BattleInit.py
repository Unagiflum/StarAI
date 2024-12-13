import pygame
import json
import random
import math

from src.UI import UI
import src.Const as Const
from src.Objects.Space.SpaceObject import Planet, Star
from src.Objects.Ships.SpaceShip import SpaceShip


def load_settings():
    try:
        with open(Const.GAME_JSON_PATH, 'r') as f:
            loaded_settings = json.load(f)
            return {key: value for key, value in loaded_settings.items()}
    except Exception as e:
        print(f"Error loading settings: {e}. Using default settings.")
        return Const.DEFAULT_KEYS


def get_random_position():
    while True:
        x = random.randint(0, Const.ARENA_SIZE)
        y = random.randint(0, Const.ARENA_SIZE)
        center = Const.ARENA_SIZE // 2
        dx = abs(x - center)
        dy = abs(y - center)
        if dx > Const.CENTER_BUFFER or dy > Const.CENTER_BUFFER:
            return x, y


def validate_ship_positions(pos1, pos2):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, Const.ARENA_SIZE - dx)
    dy = min(dy, Const.ARENA_SIZE - dy)
    return math.sqrt(dx * dx + dy * dy) >= Const.MIN_SHIP_SEPARATION


def get_valid_ship_positions():
    while True:
        pos1 = get_random_position()
        pos2 = get_random_position()
        if validate_ship_positions(pos1, pos2):
            return pos1, pos2


def initialize_battle(screen, ship1: SpaceShip, ship2: SpaceShip):
    settings = load_settings()
    scale_factor = Const.SCREEN_HEIGHT / Const.ARENA_SIZE
    game_objects = []

    # Create stars
    game_objects.extend(Star.create_random_stars(Const.STAR_COUNT))

    # Initialize ships
    pos1, pos2 = get_valid_ship_positions()

    player1 = ship1
    player1.initialize_in_battle(pos1, random.randint(0, 15))
    player2 = ship2
    player2.initialize_in_battle(pos2, random.randint(0, 15))

    game_objects.append(player1)
    game_objects.append(player2)

    # Create planet
    planet = Planet.create_center()
    game_objects.append(planet)

    player1.set_planet(planet)
    player2.set_planet(planet)

    # Create border
    border_rect = pygame.Rect(0, 0, Const.SCREEN_HEIGHT, Const.SCREEN_HEIGHT)
    border_color = (50, 50, 50)

    return {
        'settings': settings,
        'scale_factor': scale_factor,
        'game_objects': game_objects,
        'border_rect': border_rect,
        'border_color': border_color,
        'player1': player1,
        'player2': player2
    }