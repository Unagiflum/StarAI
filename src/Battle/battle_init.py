import pygame
import random

import src.const as const
from src.configuration import GameSettingsRepository
from src.toroidal import wrapped_distance
from src.Objects.Space.space_obj import Planet, Star, Asteroid
from src.Objects.Ships.space_ship import SpaceShip
from src.Battle.world import World


def load_settings():
    repository = GameSettingsRepository(
        const.GAME_JSON_PATH, const.DEFAULT_KEYS, const.DEFAULT_GAMEPLAY
    )
    return repository.load().key_codes()


def get_random_position(rng=None):
    rng = rng or random
    while True:
        x = rng.randint(0, const.ARENA_SIZE)
        y = rng.randint(0, const.ARENA_SIZE)
        center = const.ARENA_SIZE // 2
        dx = abs(x - center)
        dy = abs(y - center)
        if dx > const.CENTER_BUFFER or dy > const.CENTER_BUFFER:
            return x, y


def validate_ship_positions(pos1, pos2):
    return wrapped_distance(pos1, pos2) >= const.MIN_SHIP_SEPARATION


def get_valid_ship_positions(rng=None):
    rng = rng or random
    while True:
        pos1 = get_random_position(rng)
        pos2 = get_random_position(rng)
        if validate_ship_positions(pos1, pos2):
            return pos1, pos2


def apply_vux_starting_conditions(player1, player2, preserved_ships=None, rng=None):
    import math
    from src.Objects.Ships.catalog import ABILITIES_DATA
    from src.toroidal import wrapped_delta

    rng = rng or random
    preserved_ships = preserved_ships or set()

    for p, opponent in [(player1, player2), (player2, player1)]:
        if p.name == "Vux" and p.battles_fought == 1 and p not in preserved_ships:
            if opponent in preserved_ships:
                anchor = opponent
                mover = p
            else:
                anchor = p
                mover = opponent

            laser_range = ABILITIES_DATA.get("VuxA1", {}).get("LASER_RANGE", 644)
            min_dist = min(300, laser_range - 50) if laser_range > 300 else 0
            dist = rng.randint(min_dist, laser_range)
            angle = rng.uniform(0, 2 * math.pi)

            new_pos = [
                (anchor.position[0] + math.sin(angle) * dist) % const.ARENA_SIZE,
                (anchor.position[1] - math.cos(angle) * dist) % const.ARENA_SIZE,
            ]
            mover.position = new_pos
            if hasattr(mover, "previous_position"):
                mover.previous_position = new_pos.copy()

            dx, dy = wrapped_delta(p.position, opponent.position)
            target_angle = math.degrees(math.atan2(dx, -dy))
            if target_angle < 0:
                target_angle += 360
            direction_step = 360 / const.SHIP_DIRECTIONS
            p.heading = round(target_angle / direction_step) % const.SHIP_DIRECTIONS
            p.rotation = p.heading * const.TURN_ANGLE


def initialize_battle(
    screen,
    ship1: SpaceShip,
    ship2: SpaceShip,
    *,
    rng=None,
    resources=None,
    include_stars=True,
):
    explicit_runtime = rng is not None or resources is not None
    rng = rng or random
    resources = resources or getattr(ship1, "resources", None)
    settings = load_settings()
    world = World()

    # Create stars
    if include_stars:
        if explicit_runtime:
            world.add_all(Star.create_random_stars(const.STAR_COUNT, resources, rng))
        else:
            world.add_all(Star.create_random_stars(const.STAR_COUNT))

    # Initialize ships
    pos1, pos2 = (
        get_valid_ship_positions(rng)
        if explicit_runtime
        else get_valid_ship_positions()
    )

    player1 = ship1
    player1.initialize_in_battle(pos1, rng.randint(0, 15))
    player2 = ship2
    player2.initialize_in_battle(pos2, rng.randint(0, 15))
    player1.opponent = player2
    player2.opponent = player1

    apply_vux_starting_conditions(player1, player2, rng=rng)

    world.add(player1)
    world.add(player2)

    # Create planet
    planet = (
        Planet.create_center(resources, rng)
        if explicit_runtime
        else Planet.create_center()
    )
    world.add(planet)

    asteroids = []
    for _ in range(const.ASTEROID_COUNT):
        asteroid = Asteroid(resources, rng) if explicit_runtime else Asteroid()
        asteroid.set_planet(planet)
        pos = asteroid.get_valid_asteroid_position(
            planet, [player1, player2], [player1, player2, *asteroids]
        )
        asteroid.position = pos
        asteroids.append(asteroid)
        world.add(asteroid)

    player1.set_planet(planet)
    player2.set_planet(planet)

    # Create border
    border_rect = pygame.Rect(
        const.SCREEN_LEFT, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT
    )
    border_color = (50, 50, 50)

    return {
        "settings": settings,
        "world": world,
        "game_objects": world.objects,
        "border_rect": border_rect,
        "border_color": border_color,
        "player1": player1,
        "player2": player2,
    }
