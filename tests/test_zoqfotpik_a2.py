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
from src.Objects.Ships.ZoqFotPik.A2.ZoqFotPikA2 import ZoqFotPikA2


class ZoqFotPikA2Tests(CollisionTestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.parent = create_ship("ZoqFotPik", 1)
        self.parent.initialize_in_battle([500, 500], 0)

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_action2_spawns_area_and_commits_configured_cost_and_cooldown(self):
        initial_energy = self.parent.current_energy

        area = self.parent.perform_action2()

        self.assertIsInstance(area, ZoqFotPikA2)
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
        definition = ABILITY_DEFINITIONS["ZoqFotPikA2"]
        self.assertEqual(area.size, [12, 50])
        self.assertEqual(area.base_offset, definition.offset)
        self.assertEqual(
            area.position,
            area._mounted_position(self.parent.position, self.parent.rotation),
        )

    def test_shape_hits_inside_target_once_and_misses_nearby_outside_target(self):
        area = self.parent.perform_action2()
        inside = self.make_ship()
        inside.player = 2
        inside.position = area.position.copy()
        outside = self.make_ship()
        outside.player = 2
        outside.position = [area.position[0] + 20, area.position[1]]
        full_mask = pygame.mask.Mask((20, 20), fill=True)
        inside.get_collision_mask = lambda: full_mask
        outside.get_collision_mask = lambda: full_mask
        inside_starting_hp = inside.current_hp
        outside_starting_hp = outside.current_hp

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_area_damage([area, inside, outside], [])

        self.assertLess(inside.current_hp, inside_starting_hp)
        self.assertEqual(outside.current_hp, outside_starting_hp)
        hp_after_first_hit = inside.current_hp

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_area_damage([area, inside, outside], [])

        self.assertEqual(inside.current_hp, hp_after_first_hit)

