import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

import src.const as const
from src.Battle.battle_draw import StarFieldRenderer, _render_world_to_surface
from src.Objects.Space.space_obj import Star
from src.Objects.Space.space_obj import _circle_outline_intersects_rect


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
    def test_renderer_does_not_allocate_full_screen_depth_surfaces(self):
        renderer = StarFieldRenderer()

        self.assertFalse(hasattr(renderer, "depth_surfaces"))

    def test_only_supplied_stars_are_rendered_in_each_depth(self):
        renderer = StarFieldRenderer()
        screen = mock.Mock()
        stars = [
            SimpleNamespace(depth=2),
            SimpleNamespace(depth=0),
            SimpleNamespace(depth=2),
        ]

        with mock.patch.object(renderer, "draw_depth_stars") as draw_depth:
            renderer.draw(screen, stars, 1.25, [10, 20], [30, 40])

        rendered_by_depth = [call.args[1] for call in draw_depth.call_args_list]
        self.assertEqual(rendered_by_depth[0], [stars[1]])
        self.assertEqual(rendered_by_depth[1], [])
        self.assertEqual(rendered_by_depth[2], [stars[0], stars[2]])
        self.assertEqual(draw_depth.call_count, const.STAR_DEPTHS)

    def test_hud_world_render_omits_gravity_range_overlay(self):
        planet = mock.Mock()
        world = SimpleNamespace(
            stars=[],
            planets=[planet],
            thrust_markers=[],
            asteroids=[],
            abilities=[],
            ships=[],
            effects=[],
        )

        _render_world_to_surface(
            pygame.Surface((320, 240)),
            world,
            1.0,
            [0, 0],
            [0, 0],
            None,
            0,
            StarFieldRenderer(),
            skip_stars=True,
            show_gravity_range=False,
        )

        planet.draw_gravity_range.assert_not_called()
        planet.draw.assert_called_once()

    def test_gravity_ring_culling_rejects_a_ring_surrounding_the_view(self):
        viewport = pygame.Rect(0, 0, 100, 100)

        self.assertFalse(
            _circle_outline_intersects_rect((50, 50), 200, 10, viewport)
        )
        self.assertTrue(
            _circle_outline_intersects_rect((150, 50), 50, 2, viewport)
        )


if __name__ == "__main__":
    unittest.main()
