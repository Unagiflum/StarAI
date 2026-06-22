import math
import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ability, create_ship
from src.resources import AssetManager


class OrzAbilityTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.ship = create_ship("Orz", 1)
        self.ship.initialize_in_battle([500, 500], 4)

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_a1_uses_configured_projectile_characteristics(self):
        projectile = create_ability("OrzA1", self.ship)

        self.assertEqual(projectile.current_hp, 2)
        self.assertEqual(projectile.current_damage, 3)
        self.assertEqual(projectile.speed, 120 * const.PROJ_SPEED_SCALE)
        self.assertEqual(
            projectile.expiration_timer,
            12 * const.PROJ_LIFE_SCALE,
        )
        self.assertEqual(len(projectile.death_animation), 6)
        self.assertTrue(ABILITY_DEFINITIONS["OrzA1"].has_sound)

    def test_a2_turns_turret_relative_to_hull_and_wraps(self):
        self.assertEqual(self.ship.turret.relative_heading, 0)
        self.assertEqual(self.ship.turret_heading, self.ship.heading)

        self.ship.set_control_state("turn_right", True)
        for _ in range(const.SHIP_DIRECTIONS + 1):
            self.ship.action2_timer = 0
            self.ship.perform_action2()

        self.assertEqual(self.ship.turret.relative_heading, 1)
        self.assertEqual(
            self.ship.turret_heading,
            (self.ship.heading + 1) % const.SHIP_DIRECTIONS,
        )

        self.ship.heading = 7
        self.ship.rotation = self.ship.heading * const.TURN_ANGLE
        self.assertEqual(self.ship.turret_heading, 8)

    def test_a2_requires_one_direction_and_turns_each_way(self):
        self.assertIsNone(self.ship.perform_action2())
        self.assertEqual(self.ship.turret.relative_heading, 0)
        self.assertEqual(self.ship.action2_timer, 0)

        self.ship.set_control_state("turn_left", True)
        self.ship.perform_action2()
        self.assertEqual(
            self.ship.turret.relative_heading,
            const.SHIP_DIRECTIONS - 1,
        )

        self.ship.action2_timer = 0
        self.ship.set_control_state("turn_right", True)
        self.assertIsNone(self.ship.perform_action2())
        self.assertEqual(
            self.ship.turret.relative_heading,
            const.SHIP_DIRECTIONS - 1,
        )

    def test_a2_direction_controls_turret_without_turning_hull(self):
        initial_heading = self.ship.heading
        self.ship.set_control_state("action2", True, 10)
        self.ship.set_control_state("turn_right", True, 10)

        self.ship.process_controls(10)

        self.assertEqual(self.ship.heading, initial_heading)
        self.assertEqual(self.ship.turret.relative_heading, 1)

    def test_a1_fires_along_current_turret_heading_with_parent_velocity(self):
        self.ship.turret.relative_heading = 3
        self.ship.velocity = [5, -7]

        projectile = self.ship.perform_action1()
        expected_heading = (self.ship.heading + 3) % const.SHIP_DIRECTIONS
        angle = math.radians(expected_heading * const.TURN_ANGLE)

        self.assertEqual(projectile.heading, expected_heading)
        self.assertEqual(projectile.rotation, expected_heading * const.TURN_ANGLE)
        self.assertAlmostEqual(
            projectile.velocity[0],
            math.sin(angle) * projectile.speed + 5,
        )
        self.assertAlmostEqual(
            projectile.velocity[1],
            -math.cos(angle) * projectile.speed - 7,
        )

    def test_turret_sprite_is_centered_over_ship_and_not_a_collision_mask(self):
        base = self.ship.sprites[self.ship.heading]
        composite = self.ship.set_sprite()

        self.assertEqual(composite.get_size(), base.get_size())
        self.assertIs(self.ship.get_collision_mask(), self.ship.masks[self.ship.heading])
        self.assertFalse(ABILITY_DEFINITIONS["OrzA2"].has_sound)

    def test_battle_initialization_retains_winners_turret_orientation(self):
        self.ship.turret.relative_heading = 9
        self.ship.initialize_in_battle([100, 200], 12)

        self.assertEqual(self.ship.turret.relative_heading, 9)
        self.assertEqual(
            self.ship.turret_heading,
            (12 + 9) % const.SHIP_DIRECTIONS,
        )

    def test_holding_a2_suppresses_a1_when_a1_is_then_pressed(self):
        self.ship.set_control_state("action2", True, 10)
        self.assertIsNone(self.ship.perform_action1())
        self.assertEqual(self.ship.process_controls(10), [])

        self.ship.set_control_state("action1", True, 11)
        self.assertEqual(self.ship.process_controls(11), [])

        self.assertEqual(self.ship.action1_timer, 0)
        self.assertGreater(self.ship.action3_timer, 0)

    def test_menu_icon_includes_forward_turret(self):
        resources = AssetManager()
        icon = resources.menu_ship_sprite("Orz")
        ship = resources._image(
            const.source_path("Objects/Ships/Orz/Orz00.png")
        )

        self.assertEqual(icon.get_size(), ship.get_size())
        self.assertNotEqual(
            pygame.image.tobytes(icon, "RGBA"),
            pygame.image.tobytes(ship, "RGBA"),
        )


if __name__ == "__main__":
    unittest.main()
