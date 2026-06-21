import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import (
    ABILITIES_DATA,
    ABILITY_DEFINITIONS,
    SHIPS_DATA,
    SHIP_DEFINITIONS,
    AbilityDefinition,
    ShipDefinition,
)
from src.Objects.Ships.registry import (
    ability_names_for_ship,
    create_ability,
    create_ship,
    get_ability_class,
    get_ship_class,
    preload_ship_ability_resources,
)
from src.Menus.pick_fleet import load_fleets, load_ships
from src.resources import AssetManager


class ConfigurationPathTests(unittest.TestCase):
    def test_resource_and_configuration_paths_are_absolute(self):
        paths = (
            const.GAME_JSON_PATH,
            const.TRAINING_JSON_PATH,
            const.SHIPS_JSON_PATH,
            const.ABILITIES_JSON_PATH,
            const.PLANETS_JSON_PATH,
            const.STARS_JSON_PATH,
            const.BATTLE_MUSIC_PATH,
            const.MAIN_BG_PATH,
            const.MENU_BG_PATH,
            const.MENU_WAV_PATH,
        )

        self.assertTrue(all(path.is_absolute() for path in paths))
        self.assertTrue(all(path.exists() for path in paths))

    def test_source_path_preserves_absolute_paths(self):
        absolute = const.SOURCE_ROOT / "Objects"
        self.assertEqual(const.source_path(absolute), absolute)

    def test_saved_fleets_load_through_absolute_configuration_path(self):
        class Fleet:
            def __init__(self):
                self.ships = []

            def add_ship(self, sprite, name, cost):
                self.ships.append((sprite, name, cost, None))

        ships_data = load_ships()
        sprites = {name: object() for name in ships_data}
        left_fleet = Fleet()
        right_fleet = Fleet()

        left_ai, right_ai = load_fleets(
            left_fleet, right_fleet, sprites, ships_data
        )

        self.assertIsInstance(left_ai, bool)
        self.assertIsInstance(right_ai, bool)
        self.assertTrue(left_fleet.ships)
        self.assertTrue(right_fleet.ships)


class ShipRegistryTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_catalogs_and_registry_cover_known_types(self):
        self.assertEqual(get_ship_class("Earthling").__name__, "Earthling")
        self.assertEqual(get_ability_class("EarthlingA1").__name__, "EarthlingA1")
        self.assertIn("Earthling", SHIPS_DATA)
        self.assertIn("EarthlingA1", ABILITIES_DATA)
        self.assertIsInstance(SHIP_DEFINITIONS["Earthling"], ShipDefinition)
        self.assertIsInstance(
            ABILITY_DEFINITIONS["EarthlingA1"], AbilityDefinition
        )
        self.assertEqual(
            set(ability_names_for_ship("Earthling")),
            {"EarthlingA1", "EarthlingA2"},
        )

    def test_factory_works_outside_source_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(temporary_directory)
                ship = create_ship("Earthling", 1)
                ability = create_ability("EarthlingA1", ship)
            finally:
                os.chdir(previous_directory)

        self.assertEqual(ship.name, "Earthling")
        self.assertEqual(ability.projectile_name, "EarthlingA1")
        self.assertTrue(ship.sprite_location.is_absolute())
        self.assertTrue(ability.sprite_location.is_absolute())

    def test_preloading_resources_does_not_construct_gameplay_objects(self):
        resources = AssetManager()
        with (
            mock.patch("src.Objects.Ships.registry.get_ability_class") as get_class,
            mock.patch.object(resources, "ability", wraps=resources.ability) as ability,
        ):
            preload_ship_ability_resources("Earthling", resources)

        get_class.assert_not_called()
        expected_names = ability_names_for_ship("Earthling")
        self.assertEqual(
            ability.call_args_list,
            [mock.call(name) for name in expected_names],
        )

    def test_unknown_catalog_names_fail_clearly(self):
        with self.assertRaisesRegex(KeyError, "Unknown ship"):
            get_ship_class("MissingShip")
        with self.assertRaisesRegex(KeyError, "Unknown ability"):
            get_ability_class("MissingAbility")


if __name__ == "__main__":
    unittest.main()
