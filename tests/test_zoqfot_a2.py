import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from collision_test_support import CollisionTestCase
from src.Battle import collisions
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.ZoqFot.A2.ZoqFotA2 import ZoqFotA2


class ZoqFotA2Tests(CollisionTestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.parent = create_ship("ZoqFot", 1)
        self.parent.initialize_in_battle([500, 500], 0)

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_action2_spawns_area_and_commits_configured_cost_and_cooldown(self):
        initial_energy = self.parent.current_energy

        area = self.parent.perform_action2()

        self.assertIsInstance(area, ZoqFotA2)
        self.assertEqual(
            self.parent.current_energy,
            initial_energy - self.parent.a2_cost,
        )
        self.assertEqual(
            self.parent.action2_timer,
            const.cooldown_frames(self.parent.a2_wait),
        )
        self.assertTrue(area.area_damage_pending)
        self.assertFalse(area.can_collide)
        self.assertFalse(area.area_damage_capabilities.vulnerable)
        self.assertTrue(area.area_damage_capabilities.persistent)
        self.assertTrue(area.area_damage_capabilities.plays_impact_sound)
        definition = ABILITY_DEFINITIONS["ZoqFotA2"]
        self.assertEqual(area.size, [12, 50])
        self.assertEqual(area.base_offset, definition.offset)
        self.assertEqual(
            area.position,
            area._mounted_position(self.parent.position, self.parent.rotation),
        )

