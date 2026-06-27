import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

from src.Battle import collision_responses
from src.collision_capabilities import SpecialObjectCollisionCapabilities
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
        for _ in range(duration - 1):
            self.assertTrue(shield.update())
            self.assertTrue(self.ship.damage_shield_is_active())
        self.assertFalse(shield.update())
        self.assertFalse(self.ship.damage_shield_is_active())
        self.assertIs(self.ship.set_sprite(), original_sprite)

    def test_shield_blocks_combat_damage_but_supports_explicit_bypass(self):
        shield = self.ship.perform_action2()
        starting_hp = self.ship.current_hp

        collision_responses.area_damage_impacts_ship(
            self.ship, [], [0, 0], 0, 4
        )
        collision_responses.laser_impacts_ship(
            self.ship, [], [0, -1], 4, self.ship.position
        )
        collision_responses.apply_ship_impact_damage(self.ship, 4)

        self.assertEqual(self.ship.current_hp, starting_hp)
        self.assertEqual(
            self.ship.take_damage(3, shieldable=False),
            3,
        )
        self.assertEqual(self.ship.current_hp, starting_hp - 3)
        shield.deactivate()
        self.assertEqual(self.ship.take_damage(2), 2)

    def test_projectile_fighter_and_planet_contacts_resolve_without_damage(self):
        self.ship.perform_action2()
        self.ship.position = [100, 100]
        self.ship.previous_position = [100, 100]
        starting_hp = self.ship.current_hp
        contact = [100, 100]
        normal = [0, -1]

        projectile = SimpleNamespace(
            player=2,
            parent=object(),
            current_damage=4,
            on_ship_impact=mock.Mock(),
        )
        with (
            mock.patch.object(collision_responses, "is_live_projectile", return_value=True),
            mock.patch.object(collision_responses, "collision_info", return_value=(normal, 0, 1)),
            mock.patch.object(collision_responses, "projectile_impact", return_value=(contact, normal)),
            mock.patch.object(collision_responses, "destroy_projectile") as destroy,
            mock.patch.object(collision_responses.BattleEffect, "play_boom"),
        ):
            self.assertTrue(
                collision_responses.projectile_impacts_ship(
                    projectile, self.ship, [], None
                )
            )
        self.assertEqual(self.ship.current_hp, starting_hp)
        projectile.on_ship_impact.assert_called_once_with(self.ship)
        destroy.assert_called_once()

        special_object = SimpleNamespace(
            player=2,
            parent=object(),
            current_damage=4,
            special_object_collision_capabilities=SpecialObjectCollisionCapabilities(),
        )
        with (
            mock.patch.object(collision_responses, "is_live_fighter", return_value=True),
            mock.patch.object(collision_responses, "collision_info", return_value=(normal, 0, 1)),
            mock.patch.object(collision_responses, "projectile_impact", return_value=(contact, normal)),
            mock.patch.object(collision_responses, "destroy_projectile") as destroy,
            mock.patch.object(collision_responses.BattleEffect, "play_boom"),
        ):
            self.assertTrue(
                collision_responses.fighter_impacts_ship(
                    special_object, self.ship, [], None
                )
            )
        self.assertEqual(self.ship.current_hp, starting_hp)
        destroy.assert_called_once()

        planet = SimpleNamespace()
        with (
            mock.patch.object(collision_responses, "collision_info", return_value=(normal, 0, 1)),
            mock.patch.object(collision_responses, "objects_overlap", return_value=True),
            mock.patch.object(collision_responses, "bounce_off_static_body", return_value=True),
            mock.patch.object(collision_responses.BattleEffect, "play_boom"),
        ):
            self.assertTrue(
                collision_responses.ship_impacts_planet(
                    self.ship, planet, [], None
                )
            )
        self.assertEqual(self.ship.current_hp, starting_hp)

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


if __name__ == "__main__":
    unittest.main()
