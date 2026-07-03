import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.UI.ui_slider import Slider


class DiscreteSliderTests(unittest.TestCase):
    def test_discrete_values_skip_unavailable_numbers(self):
        slider = Slider(0, 0, 300, 16, 64, 16, "Directions", values=(16, 32, 64))

        slider.adjust_value(True)
        self.assertEqual(slider.value, 32)
        slider.adjust_value(True)
        self.assertEqual(slider.value, 64)
        slider.adjust_value(False)
        self.assertEqual(slider.value, 32)

    def test_discrete_positions_snap_to_available_values(self):
        slider = Slider(0, 0, 300, 16, 64, 16, "Directions", values=(16, 32, 64))

        self.assertEqual(slider.position_to_value(slider.line_rect.centerx), 32)
        self.assertEqual(slider.position_to_value(slider.line_rect.right), 64)


if __name__ == "__main__":
    unittest.main()
