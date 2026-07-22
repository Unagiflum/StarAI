import os
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle.collision_geometry import collision_info, objects_overlap
from src.Objects.Ships.Earthling.A1.EarthlingA1 import EarthlingA1
from src.Objects.Ships.Mycon.A1.MyconA1 import MyconA1
from src.Objects.Ships.Pkunk.A1.PkunkA1 import PkunkA1
from src.Objects.Ships.Spathi.A2.SpathiA2 import SpathiA2
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships import launch_geometry
from src.Objects.Ships.launch_geometry import (
    direction_vector,
    gun_world_position,
    launch_mask,
    mask_projection_bounds,
)
from src.Objects.Ships.registry import create_ship
from src.resources import AssetManager
from src.toroidal import wrapped_delta


class LaunchGeometryTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.resources = AssetManager()

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled
        launch_geometry._cached_mask_projection_bounds.cache_clear()

    def test_mask_projection_bounds_reuses_mask_and_normalized_direction(self):
        mask = pygame.mask.Mask((8, 8), fill=True)
        launch_geometry._cached_mask_projection_bounds.cache_clear()

        with mock.patch.object(
            launch_geometry,
            "_compute_mask_projection_bounds",
            wraps=launch_geometry._compute_mask_projection_bounds,
        ) as compute:
            first = mask_projection_bounds(mask, 0)
            second = mask_projection_bounds(mask, 360)

        self.assertEqual(first, second)
        compute.assert_called_once_with(mask, 0.0)

    def test_mask_projection_bounds_caches_directions_independently(self):
        mask = pygame.mask.Mask((8, 8), fill=True)
        launch_geometry._cached_mask_projection_bounds.cache_clear()

        with mock.patch.object(
            launch_geometry,
            "_compute_mask_projection_bounds",
            wraps=launch_geometry._compute_mask_projection_bounds,
        ) as compute:
            mask_projection_bounds(mask, 0)
            mask_projection_bounds(mask, 90)
            mask_projection_bounds(mask, 90)

        self.assertEqual(compute.call_count, 2)

    def assert_rear_gap(self, projectile, location, direction, expected_gap):
        muzzle = gun_world_position(projectile.parent, location)
        forward = direction_vector(direction)
        displacement = wrapped_delta(muzzle, projectile.position)
        distance = displacement[0] * forward[0] + displacement[1] * forward[1]
        perpendicular = (
            displacement[0] * -forward[1] + displacement[1] * forward[0]
        )
        rear, _ = mask_projection_bounds(
            launch_mask(projectile, direction), direction
        )
        self.assertAlmostEqual(distance + rear, expected_gap)
        self.assertAlmostEqual(perpendicular, 0.0)

    def test_directional_projectile_uses_one_projectile_gap(self):
        ship = create_ship("Earthling", 1, resources=self.resources)
        ship.position = [1000.0, 1000.0]
        projectile = EarthlingA1(ship)
        definition = ABILITY_DEFINITIONS["EarthlingA1"]

        self.assert_rear_gap(
            projectile,
            definition.gun_locations[0],
            ship.rotation + definition.gun_directions[0],
            const.PROJ_GAP,
        )

    def test_omnidirectional_side_shots_project_opaque_span_along_travel(self):
        ship = create_ship("Pkunk", 1, resources=self.resources)
        ship.position = [1000.0, 1000.0]
        definition = ABILITY_DEFINITIONS["PkunkA1"]
        projectiles = PkunkA1.create_projectiles(ship)

        self.assertEqual(len(projectiles), 3)
        for projectile, location, relative_direction in zip(
            projectiles,
            definition.gun_locations,
            definition.gun_directions,
        ):
            self.assert_rear_gap(
                projectile,
                location,
                (ship.rotation + relative_direction) % 360,
                2 * const.PROJ_GAP,
            )

    def test_projectile_gap_is_preserved_across_arena_wrap(self):
        ship = create_ship("Earthling", 1, resources=self.resources)
        ship.position = [1000.0, 2.0]
        projectile = EarthlingA1(ship)
        definition = ABILITY_DEFINITIONS["EarthlingA1"]

        self.assert_rear_gap(
            projectile,
            definition.gun_locations[0],
            ship.rotation + definition.gun_directions[0],
            const.PROJ_GAP,
        )
        self.assertGreater(projectile.position[1], const.ARENA_SIZE / 2)

    def test_parent_damaging_projectiles_spawn_clear_of_parent(self):
        cases = (
            ("Earthling", EarthlingA1),
            ("Mycon", MyconA1),
            ("Spathi", SpathiA2),
        )
        for ship_name, projectile_type in cases:
            with self.subTest(ship=ship_name):
                ship = create_ship(ship_name, 1, resources=self.resources)
                ship.position = [1000.0, 1000.0]
                projectile = projectile_type(ship)

                _, _, overlap = collision_info(projectile, ship)
                self.assertFalse(objects_overlap(projectile, ship, overlap))


if __name__ == "__main__":
    unittest.main()
