import tempfile
import unittest
from pathlib import Path

import src.const as const
from src.Battle.battle_draw import _should_show_crosshairs
from src.configuration import DisplaySettingsCodec, DisplaySettingsRepository
from src.resources import AssetManager


class DisplaySettingsTests(unittest.TestCase):
    def setUp(self):
        self.codec = DisplaySettingsCodec(const.DEFAULT_DISPLAY)

    def test_supported_display_settings_round_trip(self):
        settings = self.codec.decode(
            {
                "video_frame_rate": 48,
                "ship_crosshairs": "mirror_match_only",
                "show_planet_gravity_marker": False,
            }
        )
        self.assertEqual(self.codec.encode(settings), settings.to_dict())

    def test_unsupported_frame_rate_falls_back_to_default(self):
        settings = self.codec.decode({"video_frame_rate": 60})
        self.assertEqual(settings.video_frame_rate, 120)

    def test_repository_uses_defaults_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "display_settings.json"
            path.write_text("not json", encoding="utf-8")
            settings = DisplaySettingsRepository(path, const.DEFAULT_DISPLAY).load()
        self.assertEqual(settings.to_dict(), const.DEFAULT_DISPLAY)

    def test_runtime_constants_follow_frame_rate(self):
        original = self.codec.decode(const.DEFAULT_DISPLAY)
        changed = self.codec.decode(
            {
                "video_frame_rate": 48,
                "ship_crosshairs": "never",
                "show_planet_gravity_marker": False,
            }
        )
        try:
            const.apply_display_settings(changed)
            self.assertEqual(const.VIDEO_FPS_MULTIPLIER, 2)
            self.assertEqual(const.TOTAL_SPRITE_DIRECTIONS, const.SHIP_DIRECTIONS * 2)
            self.assertEqual(const.SHIP_CROSSHAIRS, "never")
            self.assertFalse(const.SHOW_PLANET_GRAVITY_MARKER)
        finally:
            const.apply_display_settings(original)

    def test_frame_rate_cache_invalidation_preserves_audio(self):
        manager = AssetManager()
        manager._ships["ship"] = object()
        manager._abilities["ability"] = object()
        manager._sounds["sound"] = object()
        manager.invalidate_interpolated_graphics()
        self.assertFalse(manager._ships)
        self.assertFalse(manager._abilities)
        self.assertIn("sound", manager._sounds)

    def test_crosshair_modes(self):
        self.assertFalse(_should_show_crosshairs("never", True))
        self.assertFalse(_should_show_crosshairs("mirror_match_only", False))
        self.assertTrue(_should_show_crosshairs("mirror_match_only", True))
        self.assertTrue(_should_show_crosshairs("always", False))


if __name__ == "__main__":
    unittest.main()
