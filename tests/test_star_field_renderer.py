import os
import math
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

import src.const as const
from src.Battle.battle_draw import (
    RenderSnapshot,
    StarFieldRenderer,
    _render_world_to_surface,
)
from src.Battle.effects import BattleEffect
from src.Objects.object import ThrustMarker
from src.Objects.Space.space_obj import Asteroid, Planet, Star
from src.Objects.Space.space_obj import (
    _annular_sector_points,
    _circle_outline_intersects_rect,
    _draw_antialiased_dashed_circle,
)
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


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
    def test_render_snapshot_classifies_the_world_once_into_stable_tuples(self):
        star = Star.__new__(Star)
        planet = Planet.__new__(Planet)
        marker = ThrustMarker.__new__(ThrustMarker)
        asteroid = Asteroid.__new__(Asteroid)
        ability = Ability.__new__(Ability)
        ship = SpaceShip.__new__(SpaceShip)
        ship.currently_alive = True
        ship.current_hp = 1
        effect = BattleEffect.__new__(BattleEffect)
        objects = [star, planet, marker, asteroid, ability, ship, effect]

        snapshot = RenderSnapshot.capture(objects)
        objects.clear()

        self.assertEqual(snapshot.stars, (star,))
        self.assertEqual(snapshot.planets, (planet,))
        self.assertEqual(snapshot.thrust_markers, (marker,))
        self.assertEqual(snapshot.asteroids, (asteroid,))
        self.assertEqual(snapshot.abilities, (ability,))
        self.assertEqual(snapshot.ships, (ship,))
        self.assertEqual(snapshot.effects, (effect,))
        self.assertEqual(snapshot.live_ships, (ship,))

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

    def test_world_objects_render_in_explicit_combat_layers(self):
        order = []

        class Drawable:
            def __init__(self, name, ability_type=None, render_priority=0):
                self.name = name
                self.type = ability_type
                self.render_priority = render_priority
                self.physical_collision_capabilities = None
                self.cloaked = False

            def draw(self, *args, **kwargs):
                order.append(self.name)

        class RecordingStars:
            def draw(self, *args, **kwargs):
                order.append("stars")

        planet = Drawable("planet")
        marker = Drawable("thrust marker")
        asteroid = Drawable("asteroid")
        ship = Drawable("ship")
        effect = Drawable("battle effect")
        abilities = [
            Drawable("Zoq-Fot area", "area", render_priority=1),
            Drawable("laser", "laser"),
            Drawable("other ability", "other"),
            Drawable("Shofixti area", "area"),
            Drawable("projectile", "projectile"),
            Drawable("special object", "special_object"),
        ]
        world = SimpleNamespace(
            stars=[object()],
            planets=[planet],
            thrust_markers=[marker],
            asteroids=[asteroid],
            abilities=abilities,
            ships=[ship],
            effects=[effect],
        )

        _render_world_to_surface(
            mock.sentinel.surface,
            world,
            1.0,
            [0, 0],
            [0, 0],
            None,
            0,
            RecordingStars(),
            show_gravity_range=False,
        )

        self.assertEqual(
            order,
            [
                "stars",
                "planet",
                "thrust marker",
                "asteroid",
                "other ability",
                "ship",
                "special object",
                "projectile",
                "laser",
                "Shofixti area",
                "Zoq-Fot area",
                "battle effect",
            ],
        )

    def test_gravity_ring_culling_rejects_a_ring_surrounding_the_view(self):
        viewport = pygame.Rect(0, 0, 100, 100)

        self.assertFalse(
            _circle_outline_intersects_rect((50, 50), 200, 10, viewport)
        )
        self.assertTrue(
            _circle_outline_intersects_rect((150, 50), 50, 2, viewport)
        )

    def test_gravity_dash_polygon_has_radial_closing_edges(self):
        polygon = _annular_sector_points(
            (100, 100),
            80,
            100,
            0,
            math.pi / 2,
        )
        midpoint = len(polygon) // 2

        self.assertEqual((polygon[0], polygon[-1]), ((200, 100), (180, 100)))
        self.assertEqual(
            (polygon[midpoint - 1], polygon[midpoint]),
            ((100, 200), (100, 180)),
        )

    def test_gravity_dashes_do_not_draw_streaks_inside_the_annulus(self):
        surface = pygame.Surface((400, 400), pygame.SRCALPHA)

        _draw_antialiased_dashed_circle(
            surface,
            (255, 0, 0, 150),
            (200, 200),
            150,
            10,
        )

        self.assertEqual(surface.get_at((200, 200)).a, 0)
        self.assertEqual(surface.get_at((200, 100)).a, 0)
        self.assertGreater(surface.get_at((350, 200)).a, 0)


if __name__ == "__main__":
    unittest.main()
