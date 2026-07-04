import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

import src.resources as resources_module
from src.UI.loading_screen import (
    BLACK,
    GREEN,
    WHITE,
    draw_loading_screen,
    preload_assets,
)
from src.resources import AssetManager


class LoadingScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((200, 100))

    def test_loading_bar_has_white_border_green_progress_and_black_remainder(self):
        surface = pygame.Surface((200, 100))

        outer, inner = draw_loading_screen(surface, 1, 2)

        self.assertEqual(surface.get_at((0, 0))[:3], BLACK)
        self.assertEqual(surface.get_at((outer.left, outer.centery))[:3], WHITE)
        self.assertEqual(surface.get_at((inner.left, inner.centery))[:3], GREEN)
        self.assertEqual(surface.get_at((inner.right - 1, inner.centery))[:3], BLACK)
        self.assertEqual(GREEN, (0, 128, 0))

    def test_loading_bar_has_rounded_corners(self):
        surface = pygame.Surface((400, 200))

        outer, _ = draw_loading_screen(surface, 1, 2)

        self.assertEqual(surface.get_at(outer.topleft)[:3], BLACK)
        self.assertEqual(surface.get_at((outer.left, outer.centery))[:3], WHITE)

    def test_preload_redraws_for_each_reported_work_update(self):
        class FakeResources:
            def preload_all(self, progress_callback):
                progress_callback(0, 2)
                progress_callback(1, 2)
                progress_callback(2, 2)
                return ["error"]

        screen = pygame.display.get_surface()
        with (
            mock.patch("pygame.event.pump") as pump,
            mock.patch("pygame.display.flip") as flip,
        ):
            errors = preload_assets(screen, FakeResources())

        self.assertEqual(errors, ["error"])
        self.assertEqual(pump.call_count, 3)
        self.assertEqual(flip.call_count, 3)

    def test_asset_manager_reports_weighted_progress_through_all_phases(self):
        manager = AssetManager()
        definitions = {
            "First": SimpleNamespace(forms={}),
            "Second": SimpleNamespace(forms={}),
        }
        loaded_asset = SimpleNamespace(ditty_path=Path(__file__))
        progress = []

        with (
            mock.patch.object(resources_module, "SHIP_DEFINITIONS", definitions),
            mock.patch.object(resources_module, "ABILITY_DEFINITIONS", {}),
            mock.patch.object(manager, "ship", return_value=loaded_asset),
            mock.patch.object(manager, "menu_ship_sprite"),
            mock.patch.object(manager, "asteroid"),
            mock.patch.object(manager, "image"),
            mock.patch.object(manager, "animation"),
            mock.patch.object(manager, "sound"),
            mock.patch.object(manager, "background"),
        ):
            manager.preload_all(
                progress_callback=lambda completed, total: progress.append(
                    (completed, total)
                )
            )

        completed = [value for value, _ in progress]
        totals = {total for _, total in progress}
        self.assertEqual(completed[0], 0)
        self.assertEqual(len(totals), 1)
        self.assertEqual(completed, sorted(completed))
        self.assertLess(completed[2], progress[2][1])
        self.assertEqual(completed[-1], progress[-1][1])


if __name__ == "__main__":
    unittest.main()
