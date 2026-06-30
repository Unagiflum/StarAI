import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

import src.const as const
from src.configuration import (
    Fleets,
    FleetsRepository,
    GameSettingsRepository,
    PlayerFleet,
    TrainingSettings,
    TrainingSettingsRepository,
)
from src.persistence import PersistenceValidationError, atomic_write_json


class GameSettingsPersistenceTests(unittest.TestCase):
    def repository(self, directory):
        return GameSettingsRepository(Path(directory) / "game.json", const.DEFAULT_KEYS)

    def test_missing_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.repository(directory).load()
        self.assertEqual(settings.key_codes(), const.DEFAULT_KEYS)

    def test_malformed_json_uses_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            path.write_text("{not json", encoding="utf-8")
            settings = self.repository(directory).load()
        self.assertEqual(settings.key_codes(), const.DEFAULT_KEYS)

    def test_partial_invalid_and_unknown_values_fall_back_per_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            path.write_text(json.dumps({
                "Player 1: Left": pygame.K_z,
                "Player 1: Right": "not-a-code",
                "future setting": 123,
            }), encoding="utf-8")
            settings = self.repository(directory).load().key_codes()

        self.assertEqual(settings["Player 1: Left"], pygame.K_z)
        self.assertEqual(settings["Player 1: Right"], const.DEFAULT_KEYS["Player 1: Right"])
        self.assertNotIn("future setting", settings)
        self.assertEqual(set(settings), set(const.DEFAULT_KEYS))

    def test_key_names_round_trip_through_existing_json_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            names = repository.default().key_names()
            names["Player 1: Left"] = "z"
            repository.save(repository.codec.from_key_names(names))
            saved = json.loads(repository.path.read_text(encoding="utf-8"))
            loaded = repository.load()

        self.assertEqual(saved["Player 1: Left"], pygame.K_z)
        self.assertEqual(list(saved), list(const.DEFAULT_KEYS))
        self.assertEqual(loaded.key_names()["Player 1: Left"], "z")


class TrainingSettingsPersistenceTests(unittest.TestCase):
    def repository(self, directory):
        return TrainingSettingsRepository(
            Path(directory) / "training.json", const.DEFAULT_TRAINING
        )

    def test_partial_settings_merge_defaults_and_invalid_ranges_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text(json.dumps({
                "learning_rate": 0.005,
                "epsilon": 2.0,
                "batch_size": "64",
            }), encoding="utf-8")
            settings = self.repository(directory).load().to_dict()

        self.assertEqual(settings["learning_rate"], 0.005)
        self.assertEqual(settings["epsilon"], const.DEFAULT_TRAINING["epsilon"])
        self.assertEqual(settings["batch_size"], const.DEFAULT_TRAINING["batch_size"])
        self.assertEqual(
            settings["number_of_hidden_layers"],
            const.DEFAULT_TRAINING["number_of_hidden_layers"],
        )

    def test_round_trip_preserves_training_json_shape(self):
        settings = TrainingSettings(0.002, 0.95, 0.5, 4, 256, 128)
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            repository.save(settings)
            saved = json.loads(repository.path.read_text(encoding="utf-8"))
            loaded = repository.load()

        self.assertEqual(saved, settings.to_dict())
        self.assertEqual(loaded, settings)


class FleetPersistenceTests(unittest.TestCase):
    catalog = {"Earthling": object(), "Arilou": object(), "Spathi": object()}

    def repository(self, directory):
        return FleetsRepository(Path(directory) / "fleets.json", self.catalog)

    def test_missing_and_malformed_files_produce_empty_fleets(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            self.assertEqual(repository.load(), Fleets())
            repository.path.write_text("{broken", encoding="utf-8")
            self.assertEqual(repository.load(), Fleets())
            repository.path.write_text("[]", encoding="utf-8")
            self.assertEqual(repository.load(), Fleets())

    def test_unknown_ships_are_skipped_without_reordering_known_ships(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            repository.path.write_text(json.dumps({
                "Player1": {
                    "ships": ["Spathi", "Unknown", "Earthling", "Spathi"],
                    "ai": True,
                },
                "Player2": {"ships": ["Arilou"], "ai": "yes"},
                "FuturePlayer": {"ships": ["Earthling"]},
            }), encoding="utf-8")
            fleets = repository.load()

        self.assertEqual(fleets.player1.ships, ("Spathi", "Earthling", "Spathi"))
        self.assertTrue(fleets.player1.ai)
        self.assertEqual(fleets.player2.ships, ("Arilou",))
        self.assertFalse(fleets.player2.ai)

    def test_round_trip_preserves_shape_order_and_ai_flags(self):
        fleets = Fleets(
            PlayerFleet(("Spathi", None, "Earthling", "Spathi"), True),
            PlayerFleet(("Arilou",), False),
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            repository.save(fleets)
            saved = json.loads(repository.path.read_text(encoding="utf-8"))
            loaded = repository.load()

        self.assertEqual(saved, fleets.to_json_dict())
        self.assertEqual(loaded, fleets)

    def test_unknown_ship_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self.repository(directory)
            with self.assertRaises(PersistenceValidationError):
                repository.save(Fleets(PlayerFleet(("Unknown",)), PlayerFleet()))


class AtomicJsonWriteTests(unittest.TestCase):
    def test_failed_replace_does_not_corrupt_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            original = '{"preserved": true}'
            path.write_text(original, encoding="utf-8")

            with mock.patch("src.persistence.os.replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    atomic_write_json(path, {"replacement": True})

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_unexpected_programming_errors_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = GameSettingsRepository(
                Path(directory) / "game.json", const.DEFAULT_KEYS
            )
            repository.path.write_text("{}", encoding="utf-8")
            with mock.patch.object(repository.codec, "decode", side_effect=RuntimeError("bug")):
                with self.assertRaisesRegex(RuntimeError, "bug"):
                    repository.load()


if __name__ == "__main__":
    unittest.main()
