import json
import unittest
from dataclasses import FrozenInstanceError

import src.const as const
from src.Objects.Ships.catalog import (
    ABILITIES_DATA,
    ABILITY_DEFINITIONS,
    SHIPS_DATA,
    SHIP_DEFINITIONS,
    AbilityDefinition,
    CatalogValidationError,
    ShipDefinition,
    build_catalogs,
    parse_ability_definition,
    parse_ship_definition,
)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class CatalogDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_ships = load_json(const.SHIPS_JSON_PATH)
        cls.raw_abilities = load_json(const.ABILITIES_JSON_PATH)

    def test_catalogs_contain_typed_immutable_definitions(self):
        ship = SHIP_DEFINITIONS["Earthling"]
        ability = ABILITY_DEFINITIONS["EarthlingA1"]

        self.assertIsInstance(ship, ShipDefinition)
        self.assertIsInstance(ability, AbilityDefinition)
        self.assertIsInstance(ability.start_hp, tuple)
        self.assertIsInstance(ability.damage, tuple)
        with self.assertRaises(FrozenInstanceError):
            ship.cost = 0
        with self.assertRaises(TypeError):
            SHIP_DEFINITIONS["Earthling"] = ship

    def test_definitions_preserve_the_existing_json_shape_and_values(self):
        self.assertEqual(
            {name: definition.to_json_dict() for name, definition in SHIP_DEFINITIONS.items()},
            self.raw_ships,
        )
        self.assertEqual(
            {
                name: definition.to_json_dict()
                for name, definition in ABILITY_DEFINITIONS.items()
            },
            self.raw_abilities,
        )

        copied_hp = ABILITY_DEFINITIONS["EarthlingA1"]["start_hp"]
        copied_hp.append(99)
        self.assertEqual(ABILITY_DEFINITIONS["EarthlingA1"].start_hp, (1,))

    def test_defaults_match_existing_constructor_and_resource_defaults(self):
        ship_data = dict(self.raw_ships["Earthling"])
        ship_data.pop("sprite_scale")
        ship = parse_ship_definition("TestShip", ship_data)
        self.assertEqual(ship.sprite_scale, 1.0)
        self.assertIsNone(ship.menu_overlay_path)
        self.assertEqual(ship.fade_duration, 8)
        self.assertEqual(ship.saw_count, 8)
        self.assertEqual(ship.gas_count, 16)

        ability_data = dict(self.raw_abilities["EarthlingA1"])
        for key in (
            "turn_wait", "end_anim", "sprite_scale", "frames", "frame_delay",
            "has_sprites", "has_sound",
        ):
            ability_data.pop(key, None)
        ability = parse_ability_definition("TestAbility", ability_data)
        self.assertEqual(ability.turn_wait, 0)
        self.assertEqual(ability.end_anim, 0)
        self.assertEqual(ability.sprite_scale, 1.0)
        self.assertEqual(ability.sprite_scale_x, 1.0)
        self.assertEqual(ability.sprite_scale_y, 1.0)
        self.assertEqual(ability.frames, 1)
        self.assertEqual(ability.frame_delay, 0)
        self.assertTrue(ability.has_sprites)
        self.assertTrue(ability.has_sound)
        self.assertTrue(ability.laser_vulnerable)
        self.assertTrue(ability.blocks_lasers)
        self.assertFalse(ability.collide_friendly_ships)

    def test_special_object_collision_contracts_are_configured(self):
        for name in ("VuxA2", "SyreenCrew", "KzerZaA2"):
            with self.subTest(name=name):
                self.assertFalse(ABILITY_DEFINITIONS[name].blocks_lasers)

        self.assertFalse(ABILITY_DEFINITIONS["VuxA2"].damage_asteroids)
        self.assertFalse(ABILITY_DEFINITIONS["VuxA2"].damage_projectiles)
        self.assertTrue(ABILITY_DEFINITIONS["OrzA3"].collide_asteroids)
        self.assertFalse(ABILITY_DEFINITIONS["OrzA3"].damage_asteroids)

    def test_missing_unknown_and_wrongly_typed_fields_are_rejected(self):
        cases = []
        missing = dict(self.raw_ships["Earthling"])
        missing.pop("cost")
        cases.append((missing, "missing field.*cost"))
        unknown = dict(self.raw_ships["Earthling"])
        unknown["typo_cost"] = 18
        cases.append((unknown, "unknown field.*typo_cost"))
        wrong_type = dict(self.raw_ships["Earthling"])
        wrong_type["max_hp"] = "18"
        cases.append((wrong_type, "max_hp.*must be int"))

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(CatalogValidationError, message):
                    parse_ship_definition("BrokenShip", data)

    def test_invalid_ability_arrays_and_unknown_ship_references_are_rejected(self):
        bad_array = dict(self.raw_abilities["EarthlingA1"])
        bad_array["damage"] = []
        with self.assertRaisesRegex(CatalogValidationError, "damage.*non-empty array"):
            parse_ability_definition("BrokenAbility", bad_array)

        ship_entries = {"Earthling": self.raw_ships["Earthling"]}
        unknown_ship = dict(self.raw_abilities["EarthlingA1"])
        unknown_ship["ship_name"] = "MissingShip"
        with self.assertRaisesRegex(CatalogValidationError, "references unknown ship"):
            build_catalogs(ship_entries, {"BrokenAbility": unknown_ship})

    def test_silhouette_colors_require_rgba_channels(self):
        ability_data = dict(self.raw_abilities["ChmmrA2"])
        ability_data["SILHOUETTE_COLORS"] = [[0, 0, 200]] * 5

        with self.assertRaisesRegex(CatalogValidationError, "must contain 4 values"):
            parse_ability_definition("BrokenAbility", ability_data)

    def test_invalid_gun_direction_contracts_are_rejected(self):
        mismatched = dict(self.raw_abilities["EarthlingA1"])
        mismatched["gun_directions"] = [0.0, 10.0]
        with self.assertRaisesRegex(
            CatalogValidationError, "gun_directions must match gun_locations"
        ):
            parse_ability_definition("BrokenAbility", mismatched)

        out_of_range = dict(self.raw_abilities["EarthlingA1"])
        out_of_range["gun_directions"] = [360.0]
        with self.assertRaisesRegex(CatalogValidationError, r"must be in \[0, 360\)"):
            parse_ability_definition("BrokenAbility", out_of_range)

        without_locations = dict(self.raw_abilities["EarthlingA1"])
        without_locations.pop("gun_locations")
        with self.assertRaisesRegex(
            CatalogValidationError, "gun_directions without gun_locations"
        ):
            parse_ability_definition("BrokenAbility", without_locations)

    def test_compatibility_aliases_are_the_typed_catalogs(self):
        self.assertIs(SHIPS_DATA, SHIP_DEFINITIONS)
        self.assertIs(ABILITIES_DATA, ABILITY_DEFINITIONS)
        self.assertEqual(
            SHIPS_DATA["Earthling"]["cost"], self.raw_ships["Earthling"]["cost"]
        )
        self.assertEqual(ABILITIES_DATA["EarthlingA1"]["ship_name"], "Earthling")


if __name__ == "__main__":
    unittest.main()
