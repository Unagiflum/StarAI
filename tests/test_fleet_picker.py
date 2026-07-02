import os
import unittest
from types import SimpleNamespace


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import src.const as Const

pygame.init()
pygame.display.set_mode((1, 1))

from src.Menus.pick_fleet import (
    FLEET_CONTROL_WEIGHTS,
    PICKER_BORDER_WIDTH,
    PICKER_BOX_COLOR,
    PICKER_CAPACITY,
    PICKER_CELL_COLOR,
    PICKER_CELL_GAP,
    ShipPickerModal,
    fleet_control_rects,
)
from src.Menus.pick_ship import (
    fleet_slot_indices_for_ships,
    fleet_slots_for_ships,
    selection_prompt,
)
from src.menu_state import ShipSelectionState
from src.UI.ship_sprites import fit_ship_sprites, scale_ship_sprites
from src.UI.ship_sprites import populate_fleet_panel
from src.UI import ui
from src.UI.ui_box import (
    FLEET_BORDER_WIDTH,
    FLEET_BOX_COLOR,
    FLEET_CONTENT_INSET,
    FLEET_EDGE_LINE_WIDTH,
    FLEET_SLOT_COLOR,
    FLEET_SLOT_SPACING,
    FLEET_TITLE_HEIGHT,
    Fleet,
    SHIP_SELECTION_HOVER_FADE_MS,
    ShipSelectionFleet,
    ship_selection_hover_alpha,
)


class SelectionPromptTests(unittest.TestCase):
    def test_reincarnated_first_chooser_leaves_other_player_selecting_alone(self):
        shofixti_replacement = SimpleNamespace(
            name="Shofixti", currently_alive=True, current_hp=10
        )
        reborn_pkunk = SimpleNamespace(
            name="Pkunk", currently_alive=True, current_hp=1
        )
        state = ShipSelectionState(
            {1: [shofixti_replacement], 2: [reborn_pkunk]},
            {1: ["Shofixti"], 2: ["Pkunk"]},
            preselected={1: None, 2: reborn_pkunk},
            choose_second_player=1,
        )

        self.assertEqual(
            selection_prompt(state),
            ("Player 2 Survives - Player 1: Select Ship", "SELECT SHIP"),
        )


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

    def test_occupied_slot_hover_uses_full_square(self):
        self.fleet.set_ship_at_slot(0, self.sprite, "Shofixti", 5)
        hovered = self.fleet.occupied_slot_at_pos(self.fleet.slot_rect(0).topleft)

        self.assertEqual(hovered[0], 0)
        self.assertEqual(hovered[1][1:3], ("Shofixti", 5))
        self.assertIsNone(
            self.fleet.occupied_slot_at_pos(self.fleet.slot_rect(1).center)
        )

    def test_shared_ship_tooltip_format(self):
        self.assertEqual(
            ui.format_ship_tooltip("Shofixti", "Scout", 5),
            "Shofixti Scout: 5",
        )
        self.assertEqual(
            ui.format_ship_tooltip(
                "Shofixti", "Scout", 5, include_cost=False
            ),
            "Shofixti Scout",
        )

    def test_shared_ship_tooltip_style_and_cursor_offset(self):
        screen = pygame.Surface((400, 300))
        screen_color = (20, 40, 60)
        screen.fill(screen_color)
        font = pygame.font.SysFont(None, Const.SHIP_TOOLTIP_FONT_SIZE)
        mouse_pos = (200, 100)
        anchor_rect = pygame.Rect(150, 75, 100, 100)

        tooltip_rect = ui.draw_ship_tooltip(
            screen, font, "Shofixti Scout: 5", mouse_pos, anchor_rect
        )

        self.assertEqual(tooltip_rect.centerx, mouse_pos[0])
        self.assertEqual(
            tooltip_rect.top,
            mouse_pos[1] + Const.SHIP_TOOLTIP_VERTICAL_OFFSET,
        )
        self.assertEqual(
            screen.get_at(tooltip_rect.topleft)[:3],
            screen_color,
        )
        self.assertEqual(
            screen.get_at((tooltip_rect.centerx, tooltip_rect.top))[:3],
            Const.SHIP_TOOLTIP_BORDER_COLOR,
        )
        background_pixel = screen.get_at(
            (
                tooltip_rect.left + Const.SHIP_TOOLTIP_BORDER_RADIUS + 1,
                tooltip_rect.top + 2,
            )
        )[:3]
        self.assertNotEqual(background_pixel, screen_color)
        self.assertNotEqual(background_pixel, Const.SHIP_TOOLTIP_BACKGROUND_COLOR)

    def test_visible_text_is_vertically_centered(self):
        font = pygame.font.SysFont(None, 40)
        text = font.render("Cancel", True, ui.WHITE)
        container = pygame.Rect(20, 30, 160, 50)

        text_rect = ui.centered_text_rect(text, container)
        visible_rect = text.get_bounding_rect().move(text_rect.topleft)

        self.assertEqual(visible_rect.centery, container.centery)

    def test_catalog_cost_font_is_three_times_original_size(self):
        self.assertEqual(Const.SHIP_CATALOG_COST_FONT_SIZE, 30)

    def test_slots_use_requested_background_and_gap(self):
        self.assertEqual(self.fleet.spacing, FLEET_SLOT_SPACING)
        self.assertEqual(FLEET_SLOT_SPACING, 3)

        screen = pygame.Surface((640, 360))
        font = pygame.font.SysFont(None, 20)
        self.fleet.draw(screen, font)

        self.assertEqual(screen.get_at(self.fleet.slot_rect(0).center)[:3], FLEET_SLOT_COLOR)
        self.assertEqual(FLEET_SLOT_COLOR, Const.SHIP_PANEL_BACKGROUND_COLOR)
        self.assertEqual(FLEET_BOX_COLOR, Const.SHIP_BOX_BACKGROUND_COLOR)

    def test_fleet_grid_is_seven_by_seven(self):
        self.assertEqual(self.fleet.icons_per_row, 7)
        self.assertEqual(self.fleet.max_fleet_size, 49)

    def test_grid_meets_edge_separator_without_outer_padding(self):
        icon_size = 50
        grid_size = 7 * icon_size + 6 * FLEET_SLOT_SPACING
        fleet = Fleet(
            0,
            0,
            grid_size + 2 * FLEET_CONTENT_INSET,
            FLEET_TITLE_HEIGHT + grid_size + FLEET_CONTENT_INSET,
            "Fleet",
            (0, 0),
        )
        first_slot = fleet.slot_rect(0)
        last_slot = fleet.slot_rect(fleet.max_fleet_size - 1)

        self.assertEqual(first_slot.left, FLEET_CONTENT_INSET)
        self.assertEqual(first_slot.top, FLEET_TITLE_HEIGHT)
        self.assertEqual(fleet.rect.right - fleet.slot_rect(6).right, FLEET_CONTENT_INSET)
        self.assertEqual(fleet.rect.bottom - last_slot.bottom, FLEET_CONTENT_INSET)
        self.assertEqual(first_slot.size, (icon_size, icon_size))
        self.assertEqual(first_slot.left - FLEET_BORDER_WIDTH, 3)
        self.assertEqual(
            fleet.slot_rect(1).left - first_slot.right,
            FLEET_SLOT_SPACING,
        )
        self.assertEqual(FLEET_EDGE_LINE_WIDTH, FLEET_SLOT_SPACING)

        screen = pygame.Surface(fleet.rect.size)
        fleet.draw(screen, pygame.font.SysFont(None, 20))
        for x in range(FLEET_BORDER_WIDTH, first_slot.left):
            self.assertEqual(
                screen.get_at((x, first_slot.centery))[:3],
                FLEET_BOX_COLOR,
            )
        for x in range(first_slot.right, fleet.slot_rect(1).left):
            self.assertEqual(
                screen.get_at((x, first_slot.centery))[:3],
                FLEET_BOX_COLOR,
            )

    def test_occupied_slot_is_replaced_and_empty_slot_appends(self):
        self.assertTrue(self.fleet.add_ship(self.sprite, "First", 1))
        self.assertTrue(self.fleet.set_ship_at_slot(0, self.sprite, "Replacement", 3))
        self.assertEqual(self.fleet.model.ship_names, ("Replacement",))
        self.assertEqual(self.fleet.model.total_cost, 3)

        self.assertTrue(self.fleet.set_ship_at_slot(10, self.sprite, "Second", 2))
        self.assertEqual(self.fleet.model.ship_names, ("Replacement", "Second"))

    def test_removing_ship_preserves_empty_slot(self):
        for name in ("First", "Second", "Third"):
            self.fleet.add_ship(self.sprite, name, 1)
        old_third_center = self.fleet.ships[2][3].center

        self.assertTrue(self.fleet.remove_ship_at_index(1))

        self.assertEqual(self.fleet.model.ship_names, ("First", "Third"))
        self.assertIsNone(self.fleet.ships[1])
        self.assertEqual(self.fleet.ships[2][3].center, old_third_center)

    def test_bulk_add_updates_the_ordered_sparse_view(self):
        self.fleet.set_ship_at_slot(0, self.sprite, "First", 1)
        self.fleet.set_ship_at_slot(47, self.sprite, "Last", 1)

        added = self.fleet.add_ships_after_last(
            self.sprite, "Bulk", 2, 4
        )

        self.assertEqual(added, 4)
        self.assertEqual(
            tuple(index for index, _ in self.fleet.occupied_slots()),
            (0, 1, 2, 3, 47, 48),
        )
        self.assertEqual(self.fleet.model.ship_slots[48], "Bulk")
        self.assertEqual(self.fleet.model.ship_slots[1:4], ("Bulk",) * 3)

    def test_population_preserves_persisted_empty_slots(self):
        sprites = {
            "First": self.sprite,
            "Second": self.sprite,
        }
        catalog = {
            "First": SimpleNamespace(cost=1),
            "Second": SimpleNamespace(cost=2),
        }

        populate_fleet_panel(
            self.fleet, ("First", None, "Second"), sprites, catalog
        )

        self.assertIsNotNone(self.fleet.ships[0])
        self.assertIsNone(self.fleet.ships[1])
        self.assertIsNotNone(self.fleet.ships[2])
        self.assertEqual(
            tuple(index for index, _ in self.fleet.occupied_slots()), (0, 2)
        )


class ShipPickerModalTests(unittest.TestCase):
    def test_fixed_grid_has_25_cells_and_maps_catalog_entries(self):
        catalog = {
            f"Ship {index}": SimpleNamespace(cost=index, ship_type="Scout")
            for index in range(PICKER_CAPACITY - 3)
        }
        sprites = {
            name: pygame.Surface((20, 10), pygame.SRCALPHA) for name in catalog
        }
        picker = ShipPickerModal(2, 7, catalog, sprites)

        self.assertEqual(len(picker.cell_rects), PICKER_CAPACITY)
        self.assertEqual(picker.ship_at_pos(picker.cell_rects[4].center)[0], "Ship 4")
        self.assertIsNone(picker.ship_at_pos(picker.cell_rects[-1].center))

    def test_cells_have_three_pixel_gaps_and_requested_background(self):
        catalog = {"Shofixti": SimpleNamespace(cost=5, ship_type="Scout")}
        sprites = {"Shofixti": pygame.Surface((20, 10), pygame.SRCALPHA)}
        picker = ShipPickerModal(1, 0, catalog, sprites)
        screen = pygame.Surface((1920, 1080))
        title_font = pygame.font.SysFont(None, 40)
        tooltip_font = pygame.font.SysFont(None, 24)

        old_mouse_pos = pygame.mouse.get_pos
        pygame.mouse.get_pos = lambda: (-1, -1)
        try:
            picker.draw(screen, title_font, tooltip_font)
        finally:
            pygame.mouse.get_pos = old_mouse_pos

        self.assertEqual(
            picker.cell_rects[1].left - picker.cell_rects[0].right,
            PICKER_CELL_GAP,
        )
        self.assertTrue(all(rect.width == rect.height for rect in picker.cell_rects))
        self.assertEqual(
            picker.cell_rects[0].left - picker.content_rect.left,
            PICKER_CELL_GAP,
        )
        self.assertEqual(
            picker.content_rect.right - picker.cell_rects[4].right,
            PICKER_CELL_GAP,
        )
        self.assertEqual(
            picker.content_rect.bottom - picker.cell_rects[-1].bottom,
            PICKER_CELL_GAP,
        )
        self.assertEqual(picker.content_rect.left - picker.rect.left, 5)
        self.assertEqual(PICKER_BORDER_WIDTH, 5)
        self.assertEqual(screen.get_at(picker.rect.topleft)[:3], picker.color)
        self.assertLess(picker.cancel_rect.bottom, picker.cell_rects[0].top)
        self.assertEqual(
            screen.get_at(picker.cell_rects[0].center)[:3], PICKER_CELL_COLOR
        )
        self.assertEqual(PICKER_CELL_COLOR, Const.SHIP_PANEL_BACKGROUND_COLOR)
        self.assertEqual(PICKER_BOX_COLOR, Const.SHIP_BOX_BACKGROUND_COLOR)
        self.assertEqual(
            screen.get_at(picker.title_rect.topleft)[:3], PICKER_BOX_COLOR
        )

    def test_tooltip_uses_constant_cursor_offset_and_can_overlap_ship(self):
        catalog = {"Shofixti": SimpleNamespace(cost=5, ship_type="Scout")}
        sprites = {"Shofixti": pygame.Surface((20, 10), pygame.SRCALPHA)}
        picker = ShipPickerModal(1, 0, catalog, sprites)
        cell_rect = picker.cell_rects[0]
        text = pygame.font.SysFont(None, 24).render(
            "Shofixti Scout: 5", True, (255, 255, 255)
        )

        tooltip_rect = picker._tooltip_rect(
            text, cell_rect.center, cell_rect, pygame.Rect(0, 0, 1920, 1080)
        )

        self.assertEqual(tooltip_rect.centerx, cell_rect.centerx)
        self.assertEqual(
            tooltip_rect.top,
            cell_rect.centery + Const.SHIP_TOOLTIP_VERTICAL_OFFSET,
        )
        self.assertLess(tooltip_rect.top, cell_rect.bottom)

    def test_catalog_draws_cost_in_cell_top_left(self):
        catalog = {"Shofixti": SimpleNamespace(cost=5, ship_type="Scout")}
        sprites = {"Shofixti": pygame.Surface((20, 10), pygame.SRCALPHA)}
        picker = ShipPickerModal(1, 0, catalog, sprites)
        screen = pygame.Surface((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT))

        old_mouse_pos = pygame.mouse.get_pos
        pygame.mouse.get_pos = lambda: (-1, -1)
        try:
            picker.draw(
                screen,
                pygame.font.SysFont(None, 40),
                pygame.font.SysFont(None, Const.SHIP_TOOLTIP_FONT_SIZE),
            )
        finally:
            pygame.mouse.get_pos = old_mouse_pos

        cell_rect = picker.cell_rects[0]
        cost_area = pygame.Rect(cell_rect.left, cell_rect.top, 15, 15)
        self.assertTrue(
            any(
                screen.get_at((x, y))[:3] == Const.SHIP_CATALOG_COST_COLOR
                for x in range(cost_area.left, cost_area.right)
                for y in range(cost_area.top, cost_area.bottom)
            )
        )

    def test_catalog_larger_than_grid_is_rejected(self):
        catalog = {
            f"Ship {index}": SimpleNamespace(cost=index, ship_type="Scout")
            for index in range(PICKER_CAPACITY + 1)
        }
        sprites = {
            name: pygame.Surface((20, 10), pygame.SRCALPHA) for name in catalog
        }
        with self.assertRaisesRegex(ValueError, "supports 25 ships"):
            ShipPickerModal(1, 0, catalog, sprites)


class FleetControlLayoutTests(unittest.TestCase):
    def test_six_controls_fit_exactly_above_the_fleet(self):
        left = 50
        width = ui.SELECTION_WIDTH
        gap = ui.button_spaceH

        rects = fleet_control_rects(left, 100, width, 36, gap)

        self.assertEqual(tuple(rects), tuple(FLEET_CONTROL_WEIGHTS))
        self.assertEqual(rects["ai"].left, left)
        self.assertEqual(rects["clear"].right, left + width)
        for previous, current in zip(rects.values(), tuple(rects.values())[1:]):
            self.assertEqual(current.left - previous.right, gap)


class UniformScalingTests(unittest.TestCase):
    def test_all_sprites_use_one_scale_factor(self):
        originals = {
            "Small": pygame.Surface((10, 20), pygame.SRCALPHA),
            "Large": pygame.Surface((20, 40), pygame.SRCALPHA),
        }

        scaled = scale_ship_sprites(originals, 80, {})

        self.assertEqual(scaled["Small"].get_size(), (20, 40))
        self.assertEqual(scaled["Large"].get_size(), (40, 80))

    def test_panel_fitting_never_upscales_source_art(self):
        originals = {
            "Small": pygame.Surface((10, 20), pygame.SRCALPHA),
            "Large": pygame.Surface((100, 200), pygame.SRCALPHA),
        }

        fitted = fit_ship_sprites(originals, 80)

        self.assertEqual(fitted["Small"].get_size(), (10, 20))
        self.assertEqual(fitted["Large"].get_size(), (40, 80))


class ShipSelectionFleetLayoutTests(unittest.TestCase):
    def test_ship_selection_fleet_is_black_without_slot_panels(self):
        fleet = ShipSelectionFleet(0, 0, 640, 360, "Fleet", (0, 0))
        first_slot = fleet.slot_rect(0)
        second_slot = fleet.slot_rect(1)
        screen = pygame.Surface(fleet.rect.size)

        fleet.draw(screen, pygame.font.SysFont(None, 20))

        self.assertEqual(screen.get_at(first_slot.center)[:3], ui.BLACK)
        self.assertEqual(screen.get_at(second_slot.center)[:3], ui.BLACK)
        self.assertEqual(screen.get_at(fleet.slot_rect(2).center)[:3], ui.BLACK)

    def test_hover_outline_fades_to_zero_and_back(self):
        half_period = SHIP_SELECTION_HOVER_FADE_MS // 2

        self.assertEqual(ship_selection_hover_alpha(0), 0)
        self.assertEqual(ship_selection_hover_alpha(half_period), 255)
        self.assertEqual(ship_selection_hover_alpha(SHIP_SELECTION_HOVER_FADE_MS), 0)

    def test_runtime_order_maps_back_to_sparse_display_slots(self):
        ships = [
            SimpleNamespace(name="First", fleet_slot_index=7),
            SimpleNamespace(name="Second", fleet_slot_index=2),
        ]

        self.assertEqual(fleet_slot_indices_for_ships(ships), (7, 2))
        slots = fleet_slots_for_ships(ships)
        self.assertEqual(slots[2], "Second")
        self.assertEqual(slots[7], "First")
        self.assertIsNone(slots[0])


if __name__ == "__main__":
    unittest.main()
