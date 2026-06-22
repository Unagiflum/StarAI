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
            int(self.parent.a2_wait * const.ACTION_WAIT_SCALE),
        )
        self.assertTrue(area.area_damage_pending)
        self.assertFalse(area.can_collide)
        self.assertFalse(area.area_damage_capabilities.vulnerable)
        definition = ABILITY_DEFINITIONS["ZoqFotA2"]
        self.assertEqual(
            area.size,
            [
                round(12 * definition.sprite_scale_x),
                round(40 * definition.sprite_scale_y),
            ],
        )
        self.assertEqual(area.sprite_scale_x, definition.sprite_scale_x)
        self.assertEqual(area.sprite_scale_y, definition.sprite_scale_y)
        self.assertEqual(area.base_offset, definition.offset)

        parent_forward = area._projection_bounds(
            self.parent.get_collision_mask(), area.heading
        )[1]
        effect_rear = area._projection_bounds(
            area._source_masks[area.heading], area.heading
        )[0]
        effect_center_distance = self.parent.position[1] - area.position[1]
        self.assertAlmostEqual(
            effect_center_distance + effect_rear,
            (parent_forward + const.PROJ_GAP) * area.base_offset,
        )

    def test_area_hits_each_valid_overlapping_target_once(self):
        area = ZoqFotA2(self.parent)
        enemy_ship = self.make_ship()
        enemy_ship.current_hp = 30
        enemy_ship.position = area.position.copy()
        enemy_projectile = self.make_projectile(self.parent)
        enemy_projectile.player = 2
        enemy_projectile.position = area.position.copy()
        enemy_fighter = self.make_fighter()
        enemy_fighter.player = 2
        enemy_fighter.position = area.position.copy()
        asteroid = self.make_asteroid(area.position)
        friendly_ship = self.make_ship()
        friendly_ship.player = 1
        friendly_ship.current_hp = 30
        friendly_ship.position = area.position.copy()
        objects = [
            area,
            enemy_ship,
            enemy_projectile,
            enemy_fighter,
            asteroid,
            friendly_ship,
        ]

        with mock.patch.object(collisions.BattleEffect, "play_boom") as boom:
            collisions._handle_area_damage(objects, [])
            collisions._handle_area_damage(objects, [])

        self.assertEqual(enemy_ship.current_hp, 18)
        self.assertFalse(enemy_projectile.currently_alive)
        self.assertFalse(enemy_fighter.currently_alive)
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(friendly_ship.current_hp, 30)
        self.assertTrue(area.currently_alive)
        self.assertTrue(area.area_damage_pending)
        self.assertEqual(boom.call_args_list, [mock.call(12)] * 4)

    def test_target_can_first_contact_during_retraction(self):
        area = ZoqFotA2(self.parent)
        target = self.make_ship()
        target.current_hp = 30
        target.position = [800, 800]

        collisions._handle_area_damage([area, target], [])
        for _ in range(5):
            self.assertTrue(area.update())
        target.position = area.position.copy()
        collisions._handle_area_damage([area, target], [])
        collisions._handle_area_damage([area, target], [])

        self.assertEqual(target.current_hp, 18)

    def test_area_follows_parent_heading_while_visible_mask_retracts(self):
        area = ZoqFotA2(self.parent)
        full_pixel_count = area.get_collision_mask().count()

        self.assertTrue(area.update())
        self.parent.position = [700, 600]
        self.parent.heading = 4
        self.parent.rotation = 4 * const.TURN_ANGLE
        self.assertTrue(area.update())

        self.assertEqual(area.heading, 4)
        self.assertGreater(area.position[0], self.parent.position[0])
        self.assertAlmostEqual(area.position[1], self.parent.position[1])
        self.assertLess(area.get_collision_mask().count(), full_pixel_count)

        for _ in range(area._duration - 2):
            self.assertTrue(area.update())
        self.assertFalse(area.update())

    def test_directional_scaling_keeps_y_forward_and_x_lateral(self):
        area = ZoqFotA2(self.parent)
        area.sprite_scale_x = 2.0
        area.sprite_scale_y = 1.0

        up_source = area.sprites[0]
        right_source = area.sprites[4]
        facing_up = area._scale_directional_sprite(up_source, 0)
        facing_right = area._scale_directional_sprite(right_source, 4)

        self.assertEqual(
            facing_up.get_size(),
            (up_source.get_width() * 2, up_source.get_height()),
        )
        self.assertEqual(
            facing_right.get_size(),
            (right_source.get_width(), right_source.get_height() * 2),
        )


if __name__ == "__main__":
    unittest.main()
