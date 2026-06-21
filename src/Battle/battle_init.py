import pygame
import random

import src.const as const
from src.configuration import GameSettingsRepository
from src.toroidal import wrapped_distance
from src.Objects.Space.space_obj import Planet, Star, Asteroid
from src.Objects.Ships.space_ship import SpaceShip
from src.Battle.world import World


def load_settings():
    repository = GameSettingsRepository(const.GAME_JSON_PATH, const.DEFAULT_KEYS)
    return repository.load().key_codes()


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
    return wrapped_distance(pos1, pos2) >= const.MIN_SHIP_SEPARATION


def get_valid_ship_positions():
    while True:
        pos1 = get_random_position()
        pos2 = get_random_position()
        if validate_ship_positions(pos1, pos2):
            return pos1, pos2


def initialize_battle(screen, ship1: SpaceShip, ship2: SpaceShip):
    settings = load_settings()
    world = World()

    # Create stars
    world.add_all(Star.create_random_stars(const.STAR_COUNT))

    # Initialize ships
    pos1, pos2 = get_valid_ship_positions()

    player1 = ship1
    player1.initialize_in_battle(pos1, random.randint(0, 15))
    player2 = ship2
    player2.initialize_in_battle(pos2, random.randint(0, 15))
    player1.opponent = player2
    player2.opponent = player1

    world.add(player1)
    world.add(player2)

    # Create planet
    planet = Planet.create_center()
    world.add(planet)

    asteroids = []
    for _ in range(const.ASTEROID_COUNT):
        asteroid = Asteroid()
        asteroid.set_planet(planet)
        pos = asteroid.get_valid_asteroid_position(planet, [player1, player2], [player1, player2, *asteroids])
        asteroid.position = pos
        asteroids.append(asteroid)
        world.add(asteroid)

    player1.set_planet(planet)
    player2.set_planet(planet)

    # Create border
    border_rect = pygame.Rect(const.SCREEN_LEFT, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)
    border_color = (50, 50, 50)

    return {
        'settings': settings,
        'world': world,
        'game_objects': world.objects,
        'border_rect': border_rect,
        'border_color': border_color,
        'player1': player1,
        'player2': player2
    }
