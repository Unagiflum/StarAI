import os
import unittest
from types import SimpleNamespace


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Menus.pick_fleet import PICKER_CAPACITY, ShipPickerModal
from src.UI.ship_sprites import scale_ship_sprites
from src.UI.ui_box import Fleet


class FleetSlotTests(unittest.TestCase):
    def setUp(self):
        self.fleet = Fleet(0, 0, 640, 360, "Fleet", (0, 0))
        self.sprite = pygame.Surface((20, 10), pygame.SRCALPHA)

    def test_slot_hit_testing_includes_empty_slots(self):
        empty_slot = 17
        self.assertEqual(
            self.fleet.slot_index_at_pos(self.fleet.slot_rect(empty_slot).center),
            empty_slot,
        )

    def test_occupied_slot_is_replaced_and_empty_slot_appends(self):
        self.assertTrue(self.fleet.add_ship(self.sprite, "First", 1))
        self.assertTrue(self.fleet.set_ship_at_slot(0, self.sprite, "Replacement", 3))
        self.assertEqual(self.fleet.model.ship_names, ("Replacement",))
        self.assertEqual(self.fleet.model.total_cost, 3)

        self.assertTrue(self.fleet.set_ship_at_slot(10, self.sprite, "Second", 2))
        self.assertEqual(self.fleet.model.ship_names, ("Replacement", "Second"))


class ShipPickerModalTests(unittest.TestCase):
    def test_fixed_grid_has_25_cells_and_maps_catalog_entries(self):
        catalog = {
            f"Ship {index}": SimpleNamespace(cost=index)
            for index in range(PICKER_CAPACITY - 3)
        }
        sprites = {
            name: pygame.Surface((20, 10), pygame.SRCALPHA) for name in catalog
        }
        picker = ShipPickerModal(2, 7, catalog, sprites)

        self.assertEqual(len(picker.cell_rects), PICKER_CAPACITY)
        self.assertEqual(picker.ship_at_pos(picker.cell_rects[4].center)[0], "Ship 4")
        self.assertIsNone(picker.ship_at_pos(picker.cell_rects[-1].center))

    def test_catalog_larger_than_grid_is_rejected(self):
        catalog = {
            f"Ship {index}": SimpleNamespace(cost=index)
            for index in range(PICKER_CAPACITY + 1)
        }
        sprites = {
            name: pygame.Surface((20, 10), pygame.SRCALPHA) for name in catalog
        }
        with self.assertRaisesRegex(ValueError, "supports 25 ships"):
            ShipPickerModal(1, 0, catalog, sprites)


class UniformScalingTests(unittest.TestCase):
    def test_all_sprites_use_one_scale_factor(self):
        originals = {
            "Small": pygame.Surface((10, 20), pygame.SRCALPHA),
            "Large": pygame.Surface((20, 40), pygame.SRCALPHA),
        }

        scaled = scale_ship_sprites(originals, 80, {})

        self.assertEqual(scaled["Small"].get_size(), (20, 40))
        self.assertEqual(scaled["Large"].get_size(), (40, 80))


if __name__ == "__main__":
    unittest.main()
