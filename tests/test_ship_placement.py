import os
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

import src.const as const
from src.Battle.battle import random_position_away_from, ship_spawn_obstacles
from src.Battle.battle_init import (
    apply_vux_starting_conditions,
    validate_ship_position,
)
from src.Battle.world import World
from src.configuration import GameSettingsCodec
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.space_ship import SpaceShip
from src.toroidal import wrapped_delta


def arena_object(
    object_type,
    position,
    *,
    ability_type=None,
    alive=True,
    player=None,
    size=(100, 100),
):
    obj = object_type.__new__(object_type)
    obj.position = list(position)
    obj.currently_alive = alive
    obj.current_hp = 1 if alive else 0
    obj.size = list(size)
    if player is not None:
        obj.player = player
    if ability_type is not None:
        obj.type = ability_type
    return obj


def heading_toward(source, target):
    dx, dy = wrapped_delta(source.position, target.position)
    target_angle = math.degrees(math.atan2(dx, -dy)) % 360
    return round(target_angle / const.TURN_ANGLE) % const.SHIP_DIRECTIONS


def asset_quantized_heading_toward(source, target):
    dx, dy = wrapped_delta(source.position, target.position)
    target_angle = math.degrees(math.atan2(dx, -dy)) % 360
    asset_step = 360 / const.ASSET_SPRITE_DIRECTIONS
    asset_heading = round(target_angle / asset_step) % const.ASSET_SPRITE_DIRECTIONS
    return (asset_heading * const.DIRECTIONS_MULTIPLIER) % const.SHIP_DIRECTIONS


def apply_ship_directions(ship_directions):
    codec = GameSettingsCodec(const.DEFAULT_KEYS, const.DEFAULT_GAMEPLAY)
    settings = codec.decode(
        {
            "asteroid_count": const.ASTEROID_COUNT,
            "ship_directions": ship_directions,
            "repeat_key_delay": const.INPUT_REPEAT_DELAY_FRAMES,
        }
    )
    const.apply_game_settings(settings)


def vux_laser_range():
    return ABILITY_DEFINITIONS["VuxA1"].range


def stepped_vux_min_distance():
    return (
        math.ceil(vux_laser_range() * 0.75 / const.VUX_SPAWN_SEARCH_STEP)
        * const.VUX_SPAWN_SEARCH_STEP
    )


def stepped_vux_preferred_max_distance():
    return (
        math.floor(vux_laser_range() * 1.2 / const.VUX_SPAWN_SEARCH_STEP)
        * const.VUX_SPAWN_SEARCH_STEP
    )


class ShipPlacementTests(unittest.TestCase):
    def test_position_validation_uses_toroidal_distance_from_arena_objects(self):
        obstacle = arena_object(Asteroid, [50, 100])
        ship = SimpleNamespace(player=1, size=[100, 100])

        self.assertFalse(validate_ship_position([7975, 100], [obstacle], ship))
        self.assertTrue(validate_ship_position([7000, 4000], [obstacle], ship))

    def test_normal_position_stays_outside_circular_gravity_well(self):
        self.assertTrue(validate_ship_position([5020, 4000]))
        self.assertFalse(validate_ship_position([5019, 4000]))

    def test_normal_position_uses_clearance_by_object_type(self):
        ship = SimpleNamespace(player=1, size=[100, 100])
        other_ship = arena_object(SpaceShip, [1000, 1000], player=2)
        enemy_projectile = arena_object(
            Ability,
            [1000, 1000],
            ability_type="projectile",
            player=2,
        )
        friendly_projectile = arena_object(
            Ability,
            [1000, 1000],
            ability_type="projectile",
            player=1,
        )
        asteroid = arena_object(Asteroid, [1000, 1000], size=(100, 100))

        ship_boundary = 1000 + const.SHIP_SPAWN_SEPARATION
        self.assertFalse(
            validate_ship_position([ship_boundary - 1, 1000], [other_ship], ship)
        )
        self.assertTrue(
            validate_ship_position([ship_boundary, 1000], [other_ship], ship)
        )
        object_boundary = 1000 + const.OBJECT_SPAWN_SEPARATION
        self.assertFalse(
            validate_ship_position(
                [object_boundary - 1, 1000], [enemy_projectile], ship
            )
        )
        self.assertTrue(
            validate_ship_position([object_boundary, 1000], [enemy_projectile], ship)
        )
        self.assertTrue(
            validate_ship_position([1000, 1000], [friendly_projectile], ship)
        )
        self.assertFalse(validate_ship_position([1099, 1000], [asteroid], ship))
        self.assertTrue(validate_ship_position([1100, 1000], [asteroid], ship))

    def test_random_position_retries_when_candidate_is_near_arena_object(self):
        obstacle = arena_object(Planet, [3100, 0])
        rng = mock.Mock()
        rng.randint.side_effect = [3100, 0, 6000, 7500]

        position = random_position_away_from([0, 0], rng, [obstacle])

        self.assertEqual(position, (6000, 7500))

    def test_spawn_obstacles_include_all_requested_live_object_kinds(self):
        ship = arena_object(SpaceShip, [0, 0])
        planet = arena_object(Planet, [1, 0])
        asteroid = arena_object(Asteroid, [2, 0])
        projectile = arena_object(Ability, [3, 0], ability_type="projectile")
        special = arena_object(Ability, [4, 0], ability_type="special_object")
        laser = arena_object(Ability, [5, 0], ability_type="laser")
        dead_projectile = arena_object(
            Ability,
            [6, 0],
            ability_type="projectile",
            alive=False,
        )

        obstacles = ship_spawn_obstacles(
            World(
                [
                    ship,
                    planet,
                    asteroid,
                    projectile,
                    special,
                    laser,
                    dead_projectile,
                ]
            )
        )

        self.assertEqual(obstacles, [ship, planet, asteroid, projectile, special])

    def test_vux_close_start_uses_500_pixel_enemy_object_clearance(self):
        vux = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=1,
            position=[1000, 1000],
            previous_position=[1000, 1000],
        )
        opponent = SimpleNamespace(
            name="Earthling",
            battles_fought=1,
            player=2,
            position=[4000, 2000],
            previous_position=[4000, 2000],
        )
        projectile = SimpleNamespace(
            type="projectile",
            player=2,
            position=[4000, 1600],
            currently_alive=True,
        )
        rng = mock.Mock()
        rng.uniform.return_value = 0

        apply_vux_starting_conditions(
            vux,
            opponent,
            preserved_ships=(object(),),
            rng=rng,
            arena_objects=[projectile],
        )

        self.assertAlmostEqual(
            math.dist(vux.position, opponent.position),
            stepped_vux_min_distance(),
        )
        self.assertGreaterEqual(math.dist(vux.position, projectile.position), 500)
        self.assertEqual(opponent.position, [4000, 2000])

    def test_vux_close_start_remains_outside_gravity_well(self):
        vux = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=1,
            position=[1000, 1000],
            previous_position=[1000, 1000],
        )
        opponent = SimpleNamespace(
            name="Earthling",
            battles_fought=1,
            player=2,
            position=[4000, 3500],
            previous_position=[4000, 3500],
        )
        rng = mock.Mock()
        rng.uniform.return_value = 0

        apply_vux_starting_conditions(
            vux,
            opponent,
            preserved_ships=(object(),),
            rng=rng,
        )

        self.assertGreaterEqual(
            math.dist(vux.position, const.PLANET_POSITION),
            const.GRAVITY_RANGE,
        )

    def test_vux_searches_past_surrounding_enemy_projectiles(self):
        vux = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=1,
            position=[7000, 7000],
            previous_position=[7000, 7000],
        )
        opponent = SimpleNamespace(
            name="KohrAh",
            battles_fought=1,
            player=2,
            position=[1000, 1000],
            previous_position=[1000, 1000],
        )
        disks = [
            SimpleNamespace(
                type="projectile",
                player=2,
                position=[
                    opponent.position[0] + math.sin(angle) * 300,
                    opponent.position[1] - math.cos(angle) * 300,
                ],
                currently_alive=True,
            )
            for angle in [index * math.pi / 4 for index in range(8)]
        ]
        rng = mock.Mock()
        rng.uniform.return_value = 0

        apply_vux_starting_conditions(
            vux,
            opponent,
            preserved_ships=(object(),),
            rng=rng,
            arena_objects=disks,
        )

        distance_to_kohr_ah = math.dist(vux.position, opponent.position)
        self.assertLess(distance_to_kohr_ah, 1200)
        for disk in disks:
            self.assertGreaterEqual(math.dist(vux.position, disk.position), 500)

    def test_vux_close_start_prefers_configured_laser_range_band(self):
        vux = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=1,
            position=[7000, 7000],
            previous_position=[7000, 7000],
        )
        opponent = SimpleNamespace(
            name="Earthling",
            battles_fought=1,
            player=2,
            position=[1000, 1000],
            previous_position=[1000, 1000],
        )
        rng = mock.Mock()
        rng.uniform.return_value = 0

        apply_vux_starting_conditions(vux, opponent, rng=rng)

        distance = math.dist(vux.position, opponent.position)
        self.assertGreaterEqual(distance, vux_laser_range() * 0.75)
        self.assertLessEqual(distance, vux_laser_range() * 1.2)
        self.assertAlmostEqual(distance, stepped_vux_min_distance())

    def test_vux_close_start_can_exceed_preferred_band_to_avoid_projectiles(self):
        vux = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=1,
            position=[7000, 7000],
            previous_position=[7000, 7000],
        )
        opponent = SimpleNamespace(
            name="KohrAh",
            battles_fought=1,
            player=2,
            position=[1000, 1000],
            previous_position=[1000, 1000],
        )
        blocking_distances = range(
            stepped_vux_min_distance(),
            stepped_vux_preferred_max_distance() + const.VUX_SPAWN_SEARCH_STEP,
            const.VUX_SPAWN_SEARCH_STEP,
        )
        projectiles = [
            SimpleNamespace(
                type="projectile",
                player=2,
                position=[
                    opponent.position[0] + math.sin(angle) * distance,
                    opponent.position[1] - math.cos(angle) * distance,
                ],
                currently_alive=True,
            )
            for distance in blocking_distances
            for angle in [index * math.pi / 8 for index in range(16)]
        ]
        rng = mock.Mock()
        rng.uniform.return_value = 0

        apply_vux_starting_conditions(
            vux,
            opponent,
            rng=rng,
            arena_objects=projectiles,
        )

        self.assertGreater(
            math.dist(vux.position, opponent.position),
            vux_laser_range() * 1.2,
        )
        for projectile in projectiles:
            self.assertGreaterEqual(math.dist(vux.position, projectile.position), 500)

    def test_simultaneous_vux_close_starts_face_final_opponent_positions(self):
        player1 = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=1,
            position=[1000, 1000],
            previous_position=[1000, 1000],
        )
        player2 = SimpleNamespace(
            name="Vux",
            battles_fought=1,
            player=2,
            position=[4000, 1000],
            previous_position=[4000, 1000],
        )
        rng = mock.Mock()
        rng.uniform.return_value = 0

        apply_vux_starting_conditions(player1, player2, rng=rng)

        self.assertEqual(player1.heading, heading_toward(player1, player2))
        self.assertEqual(player2.heading, heading_toward(player2, player1))
        self.assertEqual(player1.previous_heading, player1.heading)
        self.assertEqual(player2.previous_heading, player2.heading)
        self.assertLessEqual(math.dist(player1.position, player2.position), 600)

    def test_vux_close_start_facing_is_limited_to_asset_directions(self):
        original_directions = const.SHIP_DIRECTIONS
        original_asteroid_count = const.ASTEROID_COUNT
        original_repeat_delay = const.INPUT_REPEAT_DELAY_FRAMES
        try:
            apply_ship_directions(64)
            vux = SimpleNamespace(
                name="Vux",
                battles_fought=1,
                player=1,
                position=[1000, 1000],
                previous_position=[1000, 1000],
            )
            opponent = SimpleNamespace(
                name="Earthling",
                battles_fought=1,
                player=2,
                position=[4000, 2000],
                previous_position=[4000, 2000],
            )
            rng = mock.Mock()
            rng.uniform.return_value = math.radians(10)

            apply_vux_starting_conditions(vux, opponent, rng=rng)

            self.assertEqual(vux.heading, asset_quantized_heading_toward(vux, opponent))
            self.assertNotEqual(vux.heading, heading_toward(vux, opponent))
            self.assertEqual(vux.heading % const.DIRECTIONS_MULTIPLIER, 0)
        finally:
            apply_ship_directions(original_directions)
            const.ASTEROID_COUNT = original_asteroid_count
            const.INPUT_REPEAT_DELAY_FRAMES = original_repeat_delay


if __name__ == "__main__":
    unittest.main()
