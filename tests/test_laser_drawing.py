import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.Objects.Ships.ability import Ability


class LaserDrawingTests(unittest.TestCase):
    def test_fill_overlaps_aa_edges_without_moving_outline(self):
        screen = pygame.Surface((40, 20))

        with (
            mock.patch.object(pygame.draw, "aacircle") as aacircle,
            mock.patch.object(pygame.draw, "polygon") as polygon,
            mock.patch.object(pygame.draw, "aaline") as aaline,
        ):
            Ability.draw_aa_laser(screen, (255, 0, 0), (10, 10), (30, 10), 4)

        aacircle.assert_called_once_with(screen, (255, 0, 0), (30, 10), 2.0)
        polygon.assert_called_once_with(
            screen,
            (255, 0, 0),
            ((10.0, 12.25), (10.0, 7.75), (30.0, 7.75), (30.0, 12.25)),
        )
        self.assertEqual(
            [call.args[2:] for call in aaline.call_args_list],
            [((10.0, 12.0), (30.0, 12.0)), ((10.0, 8.0), (30.0, 8.0))],
        )


if __name__ == "__main__":
    unittest.main()
