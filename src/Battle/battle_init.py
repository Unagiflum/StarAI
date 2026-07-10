import pygame
import random

import src.const as const
from src.configuration import GameSettingsRepository
from src.toroidal import wrapped_distance
from src.Objects.Space.space_obj import Planet, Asteroid
from src.Objects.Ships.space_ship import SpaceShip
from src.Battle.world import World


def load_settings():
    repository = GameSettingsRepository(
        const.GAME_JSON_PATH, const.DEFAULT_KEYS, const.DEFAULT_GAMEPLAY
    )
    return repository.load().key_codes()


def get_random_position(rng=None):
    rng = rng or random
    return (
        rng.randint(0, const.ARENA_SIZE),
        rng.randint(0, const.ARENA_SIZE),
    )


def validate_ship_positions(pos1, pos2):
    return wrapped_distance(pos1, pos2) >= const.SHIP_SPAWN_SEPARATION


def _placement_radius(obj):
    size = getattr(obj, "size", None)
    return max(size) / 2 if size else 0


def validate_ship_position(position, arena_objects=(), ship=None):
    """Return whether a ship spawn is clear of the supplied arena objects."""
    objects = tuple(arena_objects)
    planets = [obj for obj in objects if isinstance(obj, Planet)]
    gravity_centers = (
        [planet.position for planet in planets]
        if planets
        else [const.PLANET_POSITION]
    )
    if any(
        wrapped_distance(position, center) < const.GRAVITY_RANGE
        for center in gravity_centers
    ):
        return False

    ship_player = getattr(ship, "player", None)
    ship_radius = _placement_radius(ship)
    for obj in objects:
        if getattr(obj, "position", None) is None or isinstance(obj, Planet):
            continue
        distance = wrapped_distance(position, obj.position)
        if isinstance(obj, SpaceShip):
            if distance < const.SHIP_SPAWN_SEPARATION:
                return False
            continue
        if isinstance(obj, Asteroid):
            if distance < ship_radius + _placement_radius(obj):
                return False
            continue
        if getattr(obj, "type", None) in ("projectile", "special_object"):
            if ship is not None and getattr(obj, "player", ship_player) == ship_player:
                continue
            if distance < const.OBJECT_SPAWN_SEPARATION:
                return False
    return True


def validate_vux_start_position(position, vux, arena_objects=()):
    """Validate the Vux close-start exception without using normal clearance."""
    planets = [obj for obj in arena_objects if isinstance(obj, Planet)]
    gravity_centers = (
        [planet.position for planet in planets]
        if planets
        else [const.PLANET_POSITION]
    )
    if any(
        wrapped_distance(position, center) < const.GRAVITY_RANGE
        for center in gravity_centers
    ):
        return False

    vux_player = getattr(vux, "player", None)
    for obj in arena_objects:
        if getattr(obj, "type", None) not in ("projectile", "special_object"):
            continue
        if not getattr(obj, "currently_alive", True):
            continue
        if getattr(obj, "player", vux_player) == vux_player:
            continue
        if (
            wrapped_distance(position, obj.position)
            < const.VUX_OBJECT_SPAWN_CLEARANCE
        ):
            return False
    return True


def fallback_ship_positions():
    """Yield deterministic candidates throughout the arena."""
    step = max(1, const.SHIP_SPAWN_SEPARATION // 4)
    for x in range(0, const.ARENA_SIZE, step):
        for y in range(0, const.ARENA_SIZE, step):
            yield x, y


def get_valid_ship_positions(rng=None, arena_objects=(), ships=()):
    rng = rng or random
    ships = tuple(ships)
    ship1 = ships[0] if len(ships) > 0 else None
    ship2 = ships[1] if len(ships) > 1 else None
    for _ in range(1000):
        pos1 = get_random_position(rng)
        pos2 = get_random_position(rng)
        if (
            validate_ship_positions(pos1, pos2)
            and validate_ship_position(pos1, arena_objects, ship1)
            and validate_ship_position(pos2, arena_objects, ship2)
        ):
            return pos1, pos2

    clear_positions = [
        position
        for position in fallback_ship_positions()
        if validate_ship_position(position, arena_objects, ship1)
    ]
    for index, pos1 in enumerate(clear_positions):
        for pos2 in clear_positions[index + 1 :]:
            if validate_ship_positions(pos1, pos2) and validate_ship_position(
                pos2, arena_objects, ship2
            ):
                return pos1, pos2

    raise RuntimeError("Unable to find clear ship positions")


def apply_vux_starting_conditions(
    player1,
    player2,
    preserved_ships=None,
    rng=None,
    arena_objects=(),
):
    import math
    from src.Objects.Ships.catalog import ABILITIES_DATA
    from src.toroidal import wrapped_delta

    rng = rng or random
    preserved_ships = tuple(preserved_ships or ())
    close_start_vux = []

    for p, opponent in [(player1, player2), (player2, player1)]:
        if p.name == "Vux" and p.battles_fought == 1 and p not in preserved_ships:
            close_start_vux.append((p, opponent))
            # The Vux is the ship receiving the close-start exception. Keep the
            # opponent fixed so only the Vux bypasses normal spawn clearance.
            anchor = opponent
            mover = p

            laser_range = ABILITIES_DATA.get("VuxA1", {}).get("range", 644)
            min_dist = laser_range * 0.75
            preferred_max_dist = laser_range * 1.2
            new_pos = None
            phase = rng.uniform(0, 2 * math.pi)
            arena_max_dist = math.ceil(const.ARENA_SIZE / math.sqrt(2))
            first_dist = math.ceil(min_dist / const.VUX_SPAWN_SEARCH_STEP)
            first_dist *= const.VUX_SPAWN_SEARCH_STEP
            preferred_last_dist = (
                math.floor(preferred_max_dist / const.VUX_SPAWN_SEARCH_STEP)
                * const.VUX_SPAWN_SEARCH_STEP
            )
            distance_ranges = [
                (first_dist, preferred_last_dist),
                (
                    preferred_last_dist + const.VUX_SPAWN_SEARCH_STEP,
                    arena_max_dist,
                ),
            ]
            for range_start, range_end in distance_ranges:
                if range_start > range_end:
                    continue
                for dist in range(
                    range_start,
                    range_end + const.VUX_SPAWN_SEARCH_STEP,
                    const.VUX_SPAWN_SEARCH_STEP,
                ):
                    sample_count = max(
                        32,
                        math.ceil(
                            2
                            * math.pi
                            * max(dist, 1)
                            / const.VUX_SPAWN_ANGULAR_SPACING
                        ),
                    )
                    for index in range(sample_count):
                        angle = phase + index * 2 * math.pi / sample_count
                        candidate = [
                            (anchor.position[0] + math.sin(angle) * dist)
                            % const.ARENA_SIZE,
                            (anchor.position[1] - math.cos(angle) * dist)
                            % const.ARENA_SIZE,
                        ]
                        if validate_vux_start_position(candidate, p, arena_objects):
                            new_pos = candidate
                            break
                    if new_pos is not None:
                        break
                if new_pos is not None:
                    break

            if new_pos is not None:
                mover.position = new_pos
                if hasattr(mover, "previous_position"):
                    mover.previous_position = new_pos.copy()

    # Aim after every close-start ship has been moved. When both ships are Vux,
    # aiming inside the placement loop makes the first Vux face the opponent's
    # old position before the second Vux receives its own close-start move.
    for p, opponent in close_start_vux:
        dx, dy = wrapped_delta(p.position, opponent.position)
        target_angle = math.degrees(math.atan2(dx, -dy))
        if target_angle < 0:
            target_angle += 360
        asset_direction_step = 360 / const.ASSET_SPRITE_DIRECTIONS
        asset_heading = (
            round(target_angle / asset_direction_step)
            % const.ASSET_SPRITE_DIRECTIONS
        )
        p.heading = (asset_heading * const.DIRECTIONS_MULTIPLIER) % (
            const.SHIP_DIRECTIONS
        )
        p.previous_heading = p.heading
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

    # Stars are display-owned. ``include_stars`` is retained only as a
    # compatibility parameter for older callers.

    # Initialize ships
    pos1, pos2 = (
        get_valid_ship_positions(rng, ships=(ship1, ship2))
        if explicit_runtime
        else get_valid_ship_positions(ships=(ship1, ship2))
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
