import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.UI.key_display import draw_pressed_keys, standard_key_abbreviation
from src.UI import ui


class InputKeyDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))

    def test_standard_abbreviations_cover_common_keycap_names(self):
        self.assertEqual(standard_key_abbreviation(pygame.K_ESCAPE), "Esc")
        self.assertEqual(standard_key_abbreviation(pygame.K_PAGEUP), "PgUp")
        self.assertEqual(standard_key_abbreviation(pygame.K_LCTRL), "L Ctrl")
        self.assertEqual(standard_key_abbreviation(pygame.K_KP_PLUS), "Num +")
        self.assertEqual(standard_key_abbreviation(pygame.K_a), "A")
        self.assertEqual(standard_key_abbreviation(pygame.K_F5), "F5")

    def test_held_keys_draw_as_white_keycaps_in_a_gray_panel(self):
        surface = pygame.Surface((500, 180))
        panel = pygame.Rect(20, 20, 460, 140)
        font = pygame.font.SysFont(None, 24)

        keycaps = draw_pressed_keys(
            surface,
            (pygame.K_LCTRL, pygame.K_a),
            panel,
            font,
            label="Key Tester",
        )

        self.assertEqual(len(keycaps), 2)
        self.assertEqual(surface.get_at(panel.midtop)[:3], ui.BLACK)
        self.assertEqual(surface.get_at((panel.left + 10, panel.centery))[:3], ui.GREY)
        for keycap in keycaps:
            self.assertTrue(panel.contains(keycap))
            self.assertEqual(surface.get_at(keycap.midleft)[:3], ui.WHITE)


if __name__ == "__main__":
    unittest.main()
