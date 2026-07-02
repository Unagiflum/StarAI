import math
import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ship


class ZoqFotPikA1Tests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.parent = create_ship("ZoqFotPik", 1)
        self.parent.initialize_in_battle([500, 500], 0)
        self.parent.rng = mock.Mock()
        self.parent.rng.randrange.return_value = 5
        self.parent.rng.choice.return_value = 1

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_random_variant_is_selected_once_without_evolving(self):
        projectile = self.parent.perform_action1()

        self.assertEqual(ABILITY_DEFINITIONS["ZoqFotPikA1"].frames, 16)
        self.assertEqual(len(projectile.masks), 16)
        self.assertEqual(projectile.variant_index, 5)
        self.assertIs(projectile.get_sprite(), projectile.sprites[0][5])
        self.assertIs(projectile.get_collision_mask(), projectile.masks[5])

        projectile.update()
        self.assertEqual(projectile.variant_index, 5)
        self.assertIs(projectile.get_sprite(), projectile.sprites[0][5])
        self.parent.rng.randrange.assert_called_once_with(16)

    def test_speed_stages_and_first_change_veer(self):
        projectile = self.parent.perform_action1()
        speeds = []

        for _ in range(7):
            projectile.update()
            speeds.append(round(math.hypot(*projectile.velocity)))

        self.assertEqual(speeds, [96, 96, 88, 88, 80, 80, 72])
        self.assertAlmostEqual(projectile.rotation, 5.625)
        self.parent.rng.choice.assert_called_once_with((-1, 0, 1))

    def test_projectile_does_not_inherit_parent_velocity(self):
        self.parent.velocity = [30.0, 20.0]

        projectile = self.parent.perform_action1()

        self.assertEqual(ABILITY_DEFINITIONS["ZoqFotPikA1"].parent_vel, 0.0)
        self.assertAlmostEqual(projectile.velocity[0], 0.0)
        self.assertAlmostEqual(projectile.velocity[1], -96.0)
