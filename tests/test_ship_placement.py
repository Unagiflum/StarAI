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
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


def arena_object(object_type, position, *, ability_type=None, alive=True):
    obj = object_type.__new__(object_type)
    obj.position = list(position)
    obj.currently_alive = alive
    obj.current_hp = 1 if alive else 0
    if ability_type is not None:
        obj.type = ability_type
    return obj


class ShipPlacementTests(unittest.TestCase):
    def test_position_validation_uses_toroidal_distance_from_arena_objects(self):
        obstacle = arena_object(Asteroid, [50, 100])

        self.assertFalse(validate_ship_position([7950, 100], [obstacle]))
        self.assertTrue(validate_ship_position([4000, 4000], [obstacle]))

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
            300,
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
        self.assertLess(distance_to_kohr_ah, 1000)
        for disk in disks:
            self.assertGreaterEqual(math.dist(vux.position, disk.position), 500)


if __name__ == "__main__":
    unittest.main()
