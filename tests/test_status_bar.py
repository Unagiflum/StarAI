import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle.status_bar import draw_limpet_count, draw_special_indicator


class LimpetStatusTests(unittest.TestCase):
    def setUp(self):
        self.icon = pygame.Surface((12, 12), pygame.SRCALPHA)
        ability_assets = SimpleNamespace(sprites=(self.icon,))
        self.resources = mock.Mock()
        self.resources.ability.return_value = ability_assets

    def test_draws_sprite_and_count_centered_in_bottom_region(self):
        screen = mock.Mock()
        ship = SimpleNamespace(limpets_attached=13, resources=self.resources)
        text = pygame.Surface((20, 14), pygame.SRCALPHA)

        with mock.patch(
            "src.Battle.status_bar._get_limpet_counter_font"
        ) as get_font:
            get_font.return_value.render.return_value = text
            draw_limpet_count(screen, ship, 10, 100, 65, 20)

        self.assertEqual(screen.blit.call_count, 2)
        icon_call, text_call = screen.blit.call_args_list
        get_font.return_value.render.assert_called_once_with(
            "x13", True, (255, 255, 255)
        )
        self.assertIs(icon_call.args[0], self.icon)
        self.assertIs(text_call.args[0], text)
        self.assertEqual(icon_call.args[1][1], 104)
        self.assertEqual(text_call.args[1][1], 103)

        left = icon_call.args[1][0]
        right = text_call.args[1][0] + text_call.args[0].get_width()
        self.assertLessEqual(abs((left + right) - (2 * 10 + 65)), 1)

    def test_omits_counter_when_active_form_has_no_limpets(self):
        screen = mock.Mock()
        ship = SimpleNamespace(limpets_attached=0, resources=self.resources)

        draw_limpet_count(screen, ship, 10, 100, 65, 20)

        screen.blit.assert_not_called()
        self.resources.ability.assert_not_called()


class SpecialIndicatorTests(unittest.TestCase):
    def test_draws_antialiased_light_with_one_pixel_panel_inset(self):
        panel = pygame.Surface((20, 20))
        panel.fill((12, 34, 56))
        ship = SimpleNamespace(
            hud_indicator_color=(255, 255, 0),
            hud_indicator_size=8,
            hud_indicator_gap=1,
        )

        with mock.patch(
            "src.Battle.status_bar.pygame.gfxdraw.aacircle",
            wraps=pygame.gfxdraw.aacircle,
        ) as aacircle:
            draw_special_indicator(panel, ship)

        self.assertEqual(aacircle.call_count, 2)
        self.assertEqual(panel.get_at((0, 5))[:3], (12, 34, 56))
        self.assertEqual(panel.get_at((1, 5))[:3], (0, 0, 0))
        self.assertEqual(panel.get_at((5, 5))[:3], (255, 255, 0))

    def test_size_and_gap_control_circle_placement(self):
        panel = pygame.Surface((24, 24))
        panel.fill((12, 34, 56))
        ship = SimpleNamespace(
            hud_indicator_color=(255, 0, 0),
            hud_indicator_size=12,
            hud_indicator_gap=3,
        )

        draw_special_indicator(panel, ship)

        self.assertEqual(panel.get_at((2, 9))[:3], (12, 34, 56))
        self.assertEqual(panel.get_at((3, 9))[:3], (0, 0, 0))
        self.assertEqual(panel.get_at((9, 9))[:3], (255, 0, 0))

    def test_omits_light_for_ships_without_an_indicator(self):
        screen = mock.Mock()

        draw_special_indicator(screen, SimpleNamespace())

        screen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
