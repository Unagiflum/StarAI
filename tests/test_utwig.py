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
from src.Objects.Ships.catalog import (
    ABILITIES_DATA,
    ABILITY_DEFINITIONS,
    parse_ability_definition,
)
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.Utwig.Utwig import Utwig


class UtwigShieldTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    @staticmethod
    def make_ship(*, recharge_on_planet):
        ship = Utwig.__new__(Utwig)
        ship.current_hp = 10
        ship.current_energy = 0
        ship.max_energy = 20
        shield = mock.Mock()
        shield.currently_alive = True
        shield.blocks_damage = True
        shield.recharge_on_planet = recharge_on_planet
        ship._active_damage_shield = shield
        return ship, shield

    def test_catalog_disables_utwig_planet_recharge(self):
        self.assertFalse(ABILITY_DEFINITIONS["UtwigA2"].recharge_on_planet)

    def test_missing_setting_preserves_previous_recharge_behavior(self):
        raw_definition = dict(ABILITIES_DATA["UtwigA2"])
        raw_definition.pop("recharge_on_planet")

        definition = parse_ability_definition("UtwigA2", raw_definition)

        self.assertTrue(definition.recharge_on_planet)

    def test_disabled_planet_recharge_still_blocks_damage(self):
        ship, shield = self.make_ship(recharge_on_planet=False)

        applied = ship.take_planet_impact_damage(3)

        self.assertEqual(applied, 0)
        self.assertEqual(ship.current_hp, 10)
        shield.absorb_damage.assert_not_called()

    def test_disabled_planet_recharge_does_not_affect_weapon_absorption(self):
        ship, shield = self.make_ship(recharge_on_planet=False)

        applied = ship.take_damage(3)

        self.assertEqual(applied, 0)
        self.assertEqual(ship.current_hp, 10)
        shield.absorb_damage.assert_called_once_with(3)

    def test_enabled_planet_recharge_absorbs_damage_normally(self):
        ship, shield = self.make_ship(recharge_on_planet=True)

        applied = ship.take_planet_impact_damage(3)

        self.assertEqual(applied, 0)
        self.assertEqual(ship.current_hp, 10)
        shield.absorb_damage.assert_called_once_with(3)

    def test_shield_is_held_and_drains_one_energy_every_six_frames(self):
        ship = create_ship("Utwig", 1)
        ship.initialize_in_battle([500.0, 500.0], 0)
        ship.set_control_state("action2", True, 0)

        spawned = ship.process_controls(0)
        shield = spawned[0]
        self.assertTrue(ship.damage_shield_is_active())
        self.assertEqual(ship.current_energy, ship.start_energy - 1)

        for frame in range(1, 6):
            self.assertEqual(ship.process_controls(frame), [])
        self.assertEqual(ship.current_energy, ship.start_energy - 1)

        ship.process_controls(6)
        self.assertEqual(ship.current_energy, ship.start_energy - 2)
        self.assertIs(ship._active_damage_shield, shield)

        ship.set_control_state("action2", False, 7)
        ship.process_controls(7)
        self.assertFalse(ship.damage_shield_is_active())

    def test_shield_blocks_firing_including_activation_frame(self):
        ship = create_ship("Utwig", 1)
        ship.initialize_in_battle([500.0, 500.0], 0)
        ship.set_control_state("action1", True, 0)
        ship.set_control_state("action2", True, 0)

        spawned = ship.process_controls(0)

        self.assertEqual(len(spawned), 1)
        self.assertEqual(spawned[0].name, "UtwigA2")
        self.assertIsNone(ship.perform_action1())

    def test_absorbed_damage_converts_on_following_preprocess(self):
        ship = create_ship("Utwig", 1)
        ship.initialize_in_battle([500.0, 500.0], 0)
        ship.set_control_state("action2", True, 0)
        shield = ship.process_controls(0)[0]
        shield.gain_sound = mock.Mock()
        energy_after_activation = ship.current_energy

        self.assertEqual(ship.take_damage(3), 0)
        self.assertEqual(ship.current_energy, energy_after_activation)

        ship.process_controls(1)

        self.assertEqual(ship.current_energy, energy_after_activation + 3)
        shield.gain_sound.play.assert_called_once_with()

    def test_failed_midpoint_drain_keeps_uqm_half_cycle_grace(self):
        ship = create_ship("Utwig", 1)
        ship.initialize_in_battle([500.0, 500.0], 0)
        ship.current_energy = 1
        ship.set_control_state("action2", True, 0)
        ship.process_controls(0)

        for frame in range(1, 12):
            ship.process_controls(frame)
        self.assertTrue(ship.damage_shield_is_active())
        self.assertEqual(ship.current_energy, 0)

        ship.process_controls(12)
        self.assertFalse(ship.damage_shield_is_active())


if __name__ == "__main__":
    unittest.main()
