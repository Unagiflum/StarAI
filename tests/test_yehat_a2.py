import os
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.Yehat.A2.YehatA2 import YehatA2


class YehatA2Tests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.ship = create_ship("Yehat", 1)

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_action_activates_timed_shield_and_uses_alternate_sprite(self):
        original_mask = self.ship.get_collision_mask()
        original_sprite = self.ship.set_sprite()

        shield = self.ship.perform_action2()

        self.assertIsInstance(shield, YehatA2)
        self.assertTrue(self.ship.damage_shield_is_active())
        self.assertIs(self.ship.set_sprite(), shield.sprites[self.ship.heading])
        self.assertIsNot(self.ship.set_sprite(), original_sprite)
        self.assertIs(self.ship.get_collision_mask(), original_mask)

        duration = int(ABILITY_DEFINITIONS["YehatA2"].life_time)
        for _ in range(duration):
            self.assertTrue(shield.update())
            self.assertTrue(self.ship.damage_shield_is_active())
        self.assertFalse(shield.update())
        self.assertFalse(self.ship.damage_shield_is_active())
        self.assertIs(self.ship.set_sprite(), original_sprite)

    def test_shield_blocks_combat_damage_but_supports_explicit_bypass(self):
        shield = self.ship.perform_action2()
        starting_hp = self.ship.current_hp

        self.assertEqual(self.ship.take_damage(4), 0)

        self.assertEqual(self.ship.current_hp, starting_hp)
        self.assertEqual(
            self.ship.take_damage(3, shieldable=False),
            3,
        )
        self.assertEqual(self.ship.current_hp, starting_hp - 3)
        shield.deactivate()
        self.assertEqual(self.ship.take_damage(2), 2)

    def test_reactivation_replaces_old_shield_without_old_expiry_clearing_new(self):
        first = self.ship.perform_action2()
        self.ship.action2_timer = 0
        second = self.ship.perform_action2()

        self.assertFalse(first.currently_alive)
        self.assertIs(self.ship._active_damage_shield, second)
        first.deactivate()
        self.assertIs(self.ship._active_damage_shield, second)
        self.assertTrue(self.ship.damage_shield_is_active())

    def test_activation_sound_is_owned_by_the_shield_action(self):
        self.ship.audio_service = mock.Mock()
        sound = mock.Mock()
        self.ship.audio_service.load_effect.return_value = sound

        shield = self.ship.perform_action2()

        self.assertIs(shield.launch_sound, sound)
        sound.play.assert_called_once_with()

    def test_held_shield_refreshes_on_uqm_two_frame_counter(self):
        self.ship.set_control_state("action2", True, 0)

        first = self.ship.process_controls(0)[0]
        self.assertEqual(self.ship.current_energy, 7)
        self.assertEqual(self.ship.action2_timer, 2)

        self.assertEqual(self.ship.process_controls(1), [])
        second = self.ship.process_controls(2)[0]

        self.assertFalse(first.currently_alive)
        self.assertIs(self.ship._active_damage_shield, second)
        self.assertEqual(self.ship.current_energy, 4)

    def test_releasing_special_does_not_end_fixed_duration_shield(self):
        self.ship.set_control_state("action2", True, 0)
        shield = self.ship.process_controls(0)[0]

        self.ship.set_control_state("action2", False, 1)
        self.ship.process_controls(1)

        self.assertTrue(shield.currently_alive)
        self.assertTrue(self.ship.damage_shield_is_active())


if __name__ == "__main__":
    unittest.main()
