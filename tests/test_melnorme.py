import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle import collision_responses, collisions
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.resources import AssetManager


class MelnormeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        cls.resources = AssetManager()

    @classmethod
    def tearDownClass(cls):
        Ability.sound_enabled = cls.sound_enabled

    def make_ship(self, player=1):
        ship = create_ship("Melnorme", player, resources=self.resources)
        ship.initialize_in_battle([500, 500], 0)
        return ship

    def test_catalog_and_a2_launch_geometry(self):
        ship = SHIP_DEFINITIONS["Melnorme"]
        a1 = ABILITY_DEFINITIONS["MelnormeA1"]
        a2 = ABILITY_DEFINITIONS["MelnormeA2"]

        self.assertEqual((ship.cost, ship.max_hp, ship.max_energy), (18, 20, 42))
        self.assertEqual(
            (a1.gun_levels, a1.gun_level_mult, a1.gun_level_timer),
            (4, 2, 72),
        )
        self.assertEqual(a2.gun_locations, a1.gun_locations)
        self.assertEqual(a2.gun_directions, a1.gun_directions)

    def test_a1_charges_while_mounted_then_releases(self):
        ship = self.make_ship()
        projectile = ship.perform_action1()

        self.assertIs(projectile, ship.held_a1)
        self.assertTrue(projectile.held)
        self.assertEqual(ship.current_energy, 37)
        start_offset = [
            projectile.position[0] - ship.position[0],
            projectile.position[1] - ship.position[1],
        ]

        for _ in range(72 * 3):
            self.assertTrue(projectile.update())

        self.assertEqual(projectile.level, 3)
        self.assertEqual(projectile.current_damage, 16)
        self.assertEqual(projectile.current_hp, 16)
        self.assertEqual(
            [
                projectile.position[0] - ship.position[0],
                projectile.position[1] - ship.position[1],
            ],
            start_offset,
        )

        ship.perform_action1_release()
        self.assertFalse(projectile.held)
        self.assertIsNone(ship.held_a1)
        self.assertEqual(projectile.velocity, [0.0, -180.0])

    def test_held_a1_repairs_nonlethal_damage_and_rebuilds_after_destruction(self):
        ship = self.make_ship()
        projectile = ship.perform_action1()
        projectile.set_hp(1)
        self.assertEqual(projectile.current_hp, 1)

        projectile.update()
        self.assertEqual(projectile.current_hp, 2)

        ship.current_energy = 5
        projectile.set_hp(0)
        effects = []
        collision_responses.destroy_projectile(
            projectile, effects, [0, 1], projectile.current_damage
        )
        self.assertTrue(projectile.is_alive())
        self.assertTrue(projectile.held)
        self.assertEqual(projectile.level, 0)
        self.assertEqual(ship.current_energy, 0)

        projectile.set_hp(0)
        collision_responses.destroy_projectile(
            projectile, effects, [0, 1], projectile.current_damage
        )
        self.assertFalse(projectile.is_alive())
        self.assertIsNone(ship.held_a1)

    def test_held_a1_exchanges_projectile_damage_then_repairs(self):
        projectile = self.make_ship().perform_action1()
        enemy_projectile = SimpleNamespace(
            current_damage=1,
            current_hp=3,
        )

        self.assertTrue(projectile.handle_projectile_contact(enemy_projectile))
        self.assertEqual(projectile.current_hp, 1)
        self.assertEqual(enemy_projectile.current_hp, 1)

        projectile.update()
        self.assertEqual(projectile.current_hp, 2)

    def test_a1_hold_pauses_battery_timer(self):
        ship = self.make_ship()
        ship.current_energy = 0
        ship.energy_timer = 0
        ship.action1_active = True

        for _ in range(20):
            ship.update_timers()
        self.assertEqual((ship.current_energy, ship.energy_timer), (0, 0))

        ship.action1_active = False
        for _ in range(5):
            ship.update_timers()
        self.assertEqual(ship.current_energy, 1)

    def test_a2_animation_holds_final_frame(self):
        ship = self.make_ship()
        pulse = ship.perform_action2()

        self.assertEqual(pulse.current_frame, 0)
        for _ in range(14):
            self.assertTrue(pulse.update())
        self.assertEqual(pulse.current_frame, 7)
        for _ in range(5):
            self.assertTrue(pulse.update())
            self.assertEqual(pulse.current_frame, 7)
        self.assertFalse(pulse.update())

    def test_a2_only_contacts_enemy_projectiles_without_stopping_them(self):
        pulse = self.make_ship().perform_action2()
        enemy_projectile = SimpleNamespace(
            type="projectile", player=2, current_damage=12, current_hp=3
        )
        friendly_projectile = SimpleNamespace(type="projectile", player=1)
        enemy_special = SimpleNamespace(type="special_object", player=2)

        self.assertTrue(pulse.should_collide_with_projectile_like(enemy_projectile))
        self.assertFalse(
            pulse.should_collide_with_projectile_like(friendly_projectile)
        )
        self.assertFalse(pulse.should_collide_with_projectile_like(enemy_special))

        pulse.handle_projectile_contact(enemy_projectile)
        self.assertEqual(pulse.current_hp, 188)
        self.assertEqual(enemy_projectile.current_hp, 3)

    def test_a2_confuses_for_480_frames_forces_right_and_blocks_a2(self):
        pulse = self.make_ship().perform_action2()
        target = create_ship("Earthling", 2, resources=self.resources)
        target.initialize_in_battle([800, 800], 0)

        pulse.handle_ship_contact(target)
        self.assertTrue(target.is_confused)
        self.assertEqual(target.confused_timer, 480)
        self.assertEqual(len(target.confused_frames), 8)

        target.set_control_state("action2", True, frame_id=1)
        energy = target.current_energy
        spawned = target.process_controls(frame_id=1)
        self.assertEqual(target.heading, 1)
        self.assertEqual(spawned, [])
        self.assertEqual(target.current_energy, energy)

        for _ in range(479):
            target.update()
        self.assertTrue(target.is_confused)
        target.update()
        self.assertFalse(target.is_confused)

    def test_a2_is_unaffected_by_lasers(self):
        pulse = self.make_ship().perform_action2()
        laser = SimpleNamespace(
            parent=None,
            hit_parent=False,
            current_damage=10,
            end_position=[0, 0],
            intercepted=False,
            attached_target=None,
        )

        self.assertFalse(
            collisions.LASER_TARGET_REGISTRY.is_eligible(laser, pulse)
        )


if __name__ == "__main__":
    unittest.main()
