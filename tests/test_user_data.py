import tempfile
import unittest
from pathlib import Path

import src.const as const


class UserDataInitializationTests(unittest.TestCase):
    def test_missing_configuration_is_seeded_from_bundled_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = const.initialize_user_data(directory)

            self.assertEqual(
                paths["game_settings"].read_bytes(),
                const.DEFAULT_GAME_JSON_PATH.read_bytes(),
            )
            self.assertEqual(
                paths["display_settings"].read_bytes(),
                const.DEFAULT_DISPLAY_JSON_PATH.read_bytes(),
            )
            self.assertEqual(
                paths["fleets"].read_bytes(),
                const.DEFAULT_FLEETS_JSON_PATH.read_bytes(),
            )

    def test_existing_user_configuration_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            game_settings = Path(directory) / "game_settings.json"
            game_settings.write_text('{"custom": true}', encoding="utf-8")

            const.initialize_user_data(directory)

            self.assertEqual(
                game_settings.read_text(encoding="utf-8"), '{"custom": true}'
            )


if __name__ == "__main__":
    unittest.main()
