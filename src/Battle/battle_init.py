import pygame
import json
import random
import math

import src.const as const
from src.Objects.Space.space_obj import Planet, Star, Asteroid
from src.Objects.Ships.space_ship import SpaceShip


def load_settings():
    try:
        with open(const.GAME_JSON_PATH, 'r') as f:
            loaded_settings = json.load(f)
            return {key: value for key, value in loaded_settings.items()}
    except Exception as e:
        print(f"Error loading settings: {e}. Using default settings.")
        return const.DEFAULT_KEYS


def get_random_position():
    while True:
        x = random.randint(0, const.ARENA_SIZE)
        y = random.randint(0, const.ARENA_SIZE)
        center = const.ARENA_SIZE // 2
        dx = abs(x - center)
        dy = abs(y - center)
        if dx > const.CENTER_BUFFER or dy > const.CENTER_BUFFER:
            return x, y


def validate_ship_positions(pos1, pos2):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, const.ARENA_SIZE - dx)
    dy = min(dy, const.ARENA_SIZE - dy)
    return math.sqrt(dx * dx + dy * dy) >= const.MIN_SHIP_SEPARATION


def get_valid_ship_positions():
    while True:
        pos1 = get_random_position()
        pos2 = get_random_position()
        if validate_ship_positions(pos1, pos2):
            return pos1, pos2


def initialize_battle(screen, ship1: SpaceShip, ship2: SpaceShip):
    settings = load_settings()
    game_objects = []

    # Create stars
    game_objects.extend(Star.create_random_stars(const.STAR_COUNT))

    # Initialize ships
    pos1, pos2 = get_valid_ship_positions()

    player1 = ship1
    player1.initialize_in_battle(pos1, random.randint(0, 15))
    player2 = ship2
    player2.initialize_in_battle(pos2, random.randint(0, 15))
    player1.opponent = player2
    player2.opponent = player1

    game_objects.append(player1)
    game_objects.append(player2)

    # Create planet
    planet = Planet.create_center()
    game_objects.append(planet)

    asteroid_positions = []
    for _ in range(const.ASTEROID_COUNT):
        asteroid = Asteroid()
        asteroid.set_planet(planet)
        pos = asteroid.get_valid_asteroid_position(planet.position, [pos1, pos2], asteroid_positions)
        asteroid.position = pos
        asteroid_positions.append(pos)
        game_objects.append(asteroid)

    player1.set_planet(planet)
    player2.set_planet(planet)

    # Create border
    border_rect = pygame.Rect(const.SCREEN_LEFT, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)
    border_color = (50, 50, 50)

    return {
        'settings': settings,
        'game_objects': game_objects,
        'border_rect': border_rect,
        'border_color': border_color,
        'player1': player1,
        'player2': player2
    }