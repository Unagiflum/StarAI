import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

import src.const as const
from src.Battle.battle_draw import StarFieldRenderer
from src.Objects.Space.space_obj import Star


class StarGenerationTests(unittest.TestCase):
    def test_creating_second_collection_does_not_affect_first(self):
        depths = iter([0, 1, 2, 1])

        def initialize(star, resources=None):
            star.depth = next(depths)

        with mock.patch.object(
            Star, "__init__", autospec=True, side_effect=initialize
        ):
            first = Star.create_random_stars(2)
            first_snapshot = [(star, list(star.position)) for star in first]
            second = Star.create_random_stars(2)

        self.assertEqual(
            [(star, star.position) for star in first], first_snapshot
        )
        self.assertTrue(all(star not in second for star in first))
        self.assertFalse(hasattr(Star, "stars_by_depth"))
        self.assertFalse(hasattr(Star, "depth_surfaces"))


class StarFieldRendererTests(unittest.TestCase):
    def test_renderer_instances_do_not_share_surfaces(self):
        first = StarFieldRenderer()
        second = StarFieldRenderer()

        for first_surface, second_surface in zip(
            first.depth_surfaces, second.depth_surfaces
        ):
            self.assertIsNot(first_surface, second_surface)

        first.depth_surfaces[0].fill((255, 0, 0, 255))
        self.assertNotEqual(
            first.depth_surfaces[0].get_at((0, 0)),
            second.depth_surfaces[0].get_at((0, 0)),
        )

    def test_only_supplied_stars_are_rendered_in_each_depth(self):
        renderer = StarFieldRenderer()
        screen = mock.Mock()
        stars = [
            SimpleNamespace(depth=2),
            SimpleNamespace(depth=0),
            SimpleNamespace(depth=2),
        ]

        with mock.patch.object(renderer, "update_depth_surface") as update:
            renderer.draw(screen, stars, 1.25, [10, 20], [30, 40])

        rendered_by_depth = {
            call.args[0]: call.args[1] for call in update.call_args_list
        }
        self.assertEqual(rendered_by_depth[0], [stars[1]])
        self.assertEqual(rendered_by_depth[1], [])
        self.assertEqual(rendered_by_depth[2], [stars[0], stars[2]])
        self.assertEqual(update.call_count, const.STAR_DEPTHS)


if __name__ == "__main__":
    unittest.main()
