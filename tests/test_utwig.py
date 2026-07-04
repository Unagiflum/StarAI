import unittest
from unittest import mock

from src.Objects.Ships.catalog import (
    ABILITIES_DATA,
    ABILITY_DEFINITIONS,
    parse_ability_definition,
)
from src.Objects.Ships.Utwig.Utwig import Utwig


class UtwigShieldTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
