import os
import unittest
from dataclasses import replace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from collision_test_support import CollisionTestCase
from src.Battle import collisions
from src.collision_capabilities import ProjectileContactPolicy
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
        self.assertEqual(area.current_hp, 1)
        self.assertTrue(area.area_damage_capabilities.vulnerable)
        self.assertTrue(area.laser_target_capabilities.targetable)
        self.assertTrue(area.laser_target_capabilities.vulnerable)
        self.assertTrue(area.area_damage_capabilities.persistent)
        self.assertTrue(area.area_damage_capabilities.plays_impact_sound)
        definition = ABILITY_DEFINITIONS["ZoqFotPikA2"]
        self.assertEqual(area.size, [10, 28])
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
        second_inside = self.make_ship()
        second_inside.player = 2
        second_inside.position = area.position.copy()
        outside = self.make_ship()
        outside.player = 2
        outside.position = [area.position[0] + 20, area.position[1]]
        full_mask = pygame.mask.Mask((20, 20), fill=True)
        inside.get_collision_mask = lambda: full_mask
        second_inside.get_collision_mask = lambda: full_mask
        outside.get_collision_mask = lambda: full_mask
        inside_starting_hp = inside.current_hp
        second_inside_starting_hp = second_inside.current_hp
        outside_starting_hp = outside.current_hp

        effects = []
        with (
            mock.patch.object(collisions.BattleEffect, "from_blast") as from_blast,
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_area_damage(
                [area, inside, second_inside, outside],
                effects,
            )

        self.assertLess(inside.current_hp, inside_starting_hp)
        self.assertEqual(second_inside.current_hp, second_inside_starting_hp)
        self.assertEqual(outside.current_hp, outside_starting_hp)
        self.assertFalse(area.currently_alive)
        from_blast.assert_called_once()
        play_boom.assert_called_once_with(12)
        hp_after_first_hit = inside.current_hp

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_area_damage(
                [area, inside, second_inside, outside],
                [],
            )

        self.assertEqual(inside.current_hp, hp_after_first_hit)

    def test_fragile_pass_through_does_not_consume_the_single_blocking_hit(self):
        area = self.parent.perform_action2()
        fragile = self.make_special_object()
        fragile.player = 2
        fragile.position = area.position.copy()
        fragile.special_object_collision_capabilities = replace(
            fragile.special_object_collision_capabilities,
            projectile_contact_policy=ProjectileContactPolicy.FRAGILE,
        )
        fragile.physical_collision_capabilities = replace(
            fragile.physical_collision_capabilities,
            is_fragile=True,
        )
        blocking_ship = self.make_ship()
        blocking_ship.player = 2
        blocking_ship.position = area.position.copy()
        blocking_starting_hp = blocking_ship.current_hp

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_area_damage(
                [area, fragile, blocking_ship],
                [],
            )

        self.assertFalse(fragile.currently_alive)
        self.assertLess(blocking_ship.current_hp, blocking_starting_hp)
        self.assertFalse(area.currently_alive)

    def test_advances_for_five_frames_then_retracts_for_five(self):
        area = self.parent.perform_action2()
        lengths = [area.current_length]

        for _ in range(9):
            area.update()
            lengths.append(area.current_length)

        self.assertEqual(lengths, [28, 56, 84, 112, 140, 112, 84, 56, 28, 1])

