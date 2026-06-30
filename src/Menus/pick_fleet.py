import pygame
import sys

from src.UI import ui, ui_button, ui_box
from src.UI.ship_sprites import (
    load_menu_ship_sprites,
    populate_fleet_panel,
    scale_ship_sprites,
)
import src.const as const
from src.configuration import Fleets, FleetsRepository, PlayerFleet
from src.Menus import pick_ship
from src.persistence import PersistenceValidationError
from src.Objects.Ships.catalog import SHIP_DEFINITIONS

# Display settings
SELECTION_ICON_SIZE = const.SELECTION_ICON_SIZE
FLEET_ICON_SIZE = const.FLEET_ICON_SIZE
cost_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.03)
TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.08)
PICKER_COLS = 5
PICKER_ROWS = 5
PICKER_CAPACITY = PICKER_COLS * PICKER_ROWS
PICKER_ICON_SIZE = int(const.SCREEN_HEIGHT * 0.075)
MODAL_SHADE_ALPHA = 165


class ShipPickerModal:
    """Centered, fixed 5x5 catalog used to fill one fleet slot."""

    def __init__(self, player, slot_index, ships_data, sprites):
        self.player = player
        self.slot_index = slot_index
        self.color = const.P1_COLOR if player == 1 else const.P2_COLOR
        self.ships = [
            (name, definition.cost, sprites[name])
            for name, definition in ships_data.items()
        ]
        if len(self.ships) > PICKER_CAPACITY:
            raise ValueError(
                f"The ship picker supports {PICKER_CAPACITY} ships; "
                f"the catalog contains {len(self.ships)}"
            )

        modal_size = min(
            int(const.SCREEN_WIDTH * 0.70), int(const.SCREEN_HEIGHT * 0.82)
        )
        self.rect = pygame.Rect(0, 0, modal_size, modal_size)
        self.rect.center = (const.SCREEN_WIDTH // 2, const.SCREEN_HEIGHT // 2)

        padding = int(const.SCREEN_HEIGHT * 0.02)
        title_height = int(const.SCREEN_HEIGHT * 0.055)
        footer_height = int(const.SCREEN_HEIGHT * 0.05)
        footer_gap = int(const.SCREEN_HEIGHT * 0.0125)
        button_width = int(self.rect.width * 0.32)
        button_gap = int(const.SCREEN_WIDTH * 0.012)
        button_top = self.rect.bottom - padding - footer_height
        self.cancel_rect = pygame.Rect(
            self.rect.centerx - button_width // 2,
            button_top,
            button_width,
            footer_height,
        )

        grid_rect = pygame.Rect(
            self.rect.left + padding,
            self.rect.top + padding + title_height,
            self.rect.width - 2 * padding,
            self.cancel_rect.top
            - footer_gap
            - (self.rect.top + padding + title_height),
        )
        cell_width = grid_rect.width // PICKER_COLS
        cell_height = grid_rect.height // PICKER_ROWS
        self.cell_rects = []
        for index in range(PICKER_CAPACITY):
            row, col = divmod(index, PICKER_COLS)
            self.cell_rects.append(
                pygame.Rect(
                    grid_rect.left + col * cell_width,
                    grid_rect.top + row * cell_height,
                    cell_width,
                    cell_height,
                ).inflate(-6, -6)
            )

    def ship_at_pos(self, pos):
        for index, ship in enumerate(self.ships):
            if self.cell_rects[index].collidepoint(pos):
                return ship
        return None

    def draw(self, screen, title_font, name_font, cost_font):
        pygame.draw.rect(screen, ui.BLACK, self.rect)
        pygame.draw.rect(screen, self.color, self.rect, 4)

        title = title_font.render(
            f"Player {self.player}: Select a Ship", True, ui.WHITE
        )
        screen.blit(
            title,
            title.get_rect(
                centerx=self.rect.centerx,
                top=self.rect.top + int(const.SCREEN_HEIGHT * 0.018),
            ),
        )

        mouse_pos = pygame.mouse.get_pos()
        quiet_color = tuple(max(20, int(channel * 0.42)) for channel in self.color)
        for index, cell_rect in enumerate(self.cell_rects):
            if index < len(self.ships) and cell_rect.collidepoint(mouse_pos):
                hover_surface = pygame.Surface(cell_rect.size, pygame.SRCALPHA)
                hover_surface.fill((*self.color, 65))
                screen.blit(hover_surface, cell_rect)
            pygame.draw.rect(screen, quiet_color, cell_rect, 1)

            if index >= len(self.ships):
                continue
            name, cost, sprite = self.ships[index]
            sprite_area_center = (
                cell_rect.centerx,
                cell_rect.top + PICKER_ICON_SIZE // 2 + 5,
            )
            screen.blit(sprite, sprite.get_rect(center=sprite_area_center))

            name_surface = name_font.render(name, True, ui.WHITE)
            cost_surface = cost_font.render(f"Cost: {cost}", True, ui.WHITE)
            screen.blit(
                name_surface,
                name_surface.get_rect(
                    centerx=cell_rect.centerx, bottom=cell_rect.bottom - 20
                ),
            )
            screen.blit(
                cost_surface,
                cost_surface.get_rect(
                    centerx=cell_rect.centerx, bottom=cell_rect.bottom - 3
                ),
            )

        cancel_color = (
            ui.CAN_RED_HI
            if self.cancel_rect.collidepoint(mouse_pos)
            else ui.CAN_RED
        )
        pygame.draw.rect(screen, cancel_color, self.cancel_rect, border_radius=5)
        cancel_text = cost_font.render("Cancel", True, ui.WHITE)
        screen.blit(cancel_text, cancel_text.get_rect(center=self.cancel_rect.center))


def load_ships():
    """Return the authoritative typed catalog (compatibility entry point)."""
    return SHIP_DEFINITIONS


def load_ship_sprites(ships_data, resources=None):
    def fallback(_ship_name):
        surface = pygame.Surface(SELECTION_ICON_SIZE, pygame.SRCALPHA)
        surface.fill(ui.GREY)
        return surface

    return load_menu_ship_sprites(ships_data, resources=resources, fallback=fallback)


def create_sprite_sets(ships_data):
    """Create picker and fleet sprites using one scale factor per view."""
    original_sprites = load_ship_sprites(ships_data)
    selection_sprites = scale_ship_sprites(
        original_sprites, PICKER_ICON_SIZE, ships_data
    )

    # Calculate fleet icon size first
    fleet = ui_box.Fleet(0, 0, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT, "", (0, 0))
    fleet_size = fleet.icon_size[0]
    fleet_sprites = scale_ship_sprites(original_sprites, fleet_size, ships_data)

    return selection_sprites, fleet_sprites


scale_sprites = scale_ship_sprites


def save_fleets(
    left_fleet: ui_box.Fleet, right_fleet: ui_box.Fleet, left_ai: bool, right_ai: bool
):
    """Save the current fleets and AI settings to fleets.json."""
    fleets = Fleets(
        PlayerFleet(left_fleet.model.ship_names, left_ai),
        PlayerFleet(right_fleet.model.ship_names, right_ai),
    )
    try:
        FleetsRepository(const.FLEETS_JSON_PATH, SHIP_DEFINITIONS).save(fleets)
        print("Fleets and AI settings saved to fleets.json")
    except (OSError, PersistenceValidationError) as e:
        print(f"Error saving fleets.json: {e}")


def load_fleets(left_fleet, right_fleet, fleet_sprites, ships_data):
    """Load fleets and AI settings from fleets.json if it exists."""
    if not const.FLEETS_JSON_PATH.exists():
        print("fleets.json does not exist. Starting with empty fleets.")
        return False, False

    fleets = FleetsRepository(const.FLEETS_JSON_PATH, ships_data).load()
    for fleet_panel, player_fleet in (
        (left_fleet, fleets.player1),
        (right_fleet, fleets.player2),
    ):
        populate_fleet_panel(fleet_panel, player_fleet.ships, fleet_sprites, ships_data)

    print("Fleets loaded from fleets.json")
    return fleets.player1.ai, fleets.player2.ai


def run(screen: pygame.Surface, menu_sound_manager=None, audio_service=None):
    """Run the Pick Fleet module."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, cost_FONT_SIZE)
    title_font = pygame.font.SysFont(None, TITLE_FONT_SIZE)
    picker_title_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.042))
    picker_name_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.021))
    picker_cost_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.019))
    background = ui.load_background(
        const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT
    )

    # Load ships data and sprites
    ships_data = load_ships()

    # Scale sprites for selection and fleet views, maintaining proportions
    selection_sprites, fleet_sprites = create_sprite_sets(ships_data)

    # Create UI components

    left_column_start = int(0.033 * const.SCREEN_WIDTH)
    top_button_start = int(0.1 * const.SCREEN_HEIGHT)
    AI_toggle_width = int(0.075 * const.SCREEN_WIDTH)
    top_button_height = int(0.0375 * const.SCREEN_HEIGHT)
    each_button_width = int(
        0.5 * (ui.SELECTION_WIDTH - AI_toggle_width - 2 * ui.button_spaceH)
    )

    right_column_start = int(const.SCREEN_WIDTH // 2 + (0.016 * const.SCREEN_WIDTH))
    columns = {1: left_column_start, 2: right_column_start}
    fleet_top = top_button_start + top_button_height + 2 * ui.button_spaceV
    fleets = ui_box.create_player_fleet_panels(
        columns,
        fleet_top,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        FLEET_ICON_SIZE,
    )
    ai_toggles = {
        player: ui_button.ToggleButton(
            columns[player],
            top_button_start,
            AI_toggle_width,
            top_button_height,
            "AI",
            initial_state=False,
            bg_color=(*const.P1_COLOR, 75) if player == 1 else (*const.P2_COLOR, 75),
            hover_color=(
                (*const.P1_COLOR, 255) if player == 1 else (*const.P2_COLOR, 255)
            ),
        )
        for player in (1, 2)
    }

    def create_one_of_each(player):
        fleets[player].clear()
        populate_fleet_panel(fleets[player], ships_data, fleet_sprites, ships_data)

    one_of_each_buttons = {
        player: ui_button.Button(
            columns[player] + AI_toggle_width + ui.button_spaceH,
            top_button_start,
            each_button_width,
            top_button_height,
            "One of Each Ship",
            lambda player=player: create_one_of_each(player),
            bg_color=(*const.P1_COLOR, 75) if player == 1 else (*const.P2_COLOR, 75),
            hover_color=(
                (*const.P1_COLOR, 255) if player == 1 else (*const.P2_COLOR, 255)
            ),
        )
        for player in (1, 2)
    }
    clear_buttons = {
        player: ui_button.Button(
            columns[player]
            + AI_toggle_width
            + 2 * ui.button_spaceH
            + each_button_width,
            top_button_start,
            each_button_width,
            top_button_height,
            f"Clear Fleet {player}",
            fleets[player].clear,
            bg_color=(*const.P1_COLOR, 75) if player == 1 else (*const.P2_COLOR, 75),
            hover_color=(
                (*const.P1_COLOR, 255) if player == 1 else (*const.P2_COLOR, 255)
            ),
        )
        for player in (1, 2)
    }

    loaded_ai = load_fleets(fleets[1], fleets[2], fleet_sprites, ships_data)
    for player, ai in zip((1, 2), loaded_ai):
        ai_toggles[player].is_on = ai

    # Create buttons
    def confirm_callback():
        if all(not fleet.model.is_empty for fleet in fleets.values()):
            save_fleets(
                fleets[1],
                fleets[2],
                ai_toggles[1].value,
                ai_toggles[2].value,
            )
            print("Fleets confirmed.")

            pick_ship.run(
                screen=screen,
                menu_sound_manager=menu_sound_manager,
                audio_service=audio_service,
            )

    running = True
    ship_picker = None

    def cancel_callback():
        nonlocal running
        print("Cancel button clicked. Returning to main menu.")
        running = False

    confirm_button = ui_button.Button(
        ui.ok_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Confirm",
        confirm_callback,
        bg_color=ui.DISABLED_BUTTON,
        hover_color=ui.DISABLED_BUTTON,
        text_color=ui.WHITE,
    )

    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Cancel",
        cancel_callback,
        bg_color=ui.CAN_RED,
        hover_color=ui.CAN_RED_HI,
        text_color=ui.WHITE,
    )

    while running:
        clock.tick(const.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ship_picker is not None:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    ship_picker = None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if (
                        not ship_picker.rect.collidepoint(event.pos)
                        or ship_picker.cancel_rect.collidepoint(event.pos)
                    ):
                        ship_picker = None
                    else:
                        selected = ship_picker.ship_at_pos(event.pos)
                        if selected is not None:
                            name, cost, _ = selected
                            fleet = fleets[ship_picker.player]
                            if fleet.set_ship_at_slot(
                                ship_picker.slot_index,
                                fleet_sprites[name],
                                name,
                                cost,
                            ):
                                print(f"Added {name} in {fleet.title}")
                                if menu_sound_manager:
                                    menu_sound_manager.play_sound("menu")
                            ship_picker = None
                continue

            for controls in (ai_toggles, one_of_each_buttons, clear_buttons):
                for control in controls.values():
                    control.handle_event(event, menu_sound_manager)

            confirm_button.handle_event(event, menu_sound_manager)
            cancel_button.handle_event(event, menu_sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for player, fleet in fleets.items():
                    slot_index = fleet.slot_index_at_pos(event.pos)
                    if slot_index is not None:
                        if fleet.ships[slot_index] is not None:
                            fleet.remove_ship_at_index(slot_index)
                            print(f"Removed a ship from {fleet.title}")
                            if menu_sound_manager:
                                menu_sound_manager.play_sound("menu")
                        else:
                            ship_picker = ShipPickerModal(
                                player,
                                slot_index,
                                ships_data,
                                selection_sprites,
                            )
                            if menu_sound_manager:
                                menu_sound_manager.play_sound("menu")
                        break

        # Update confirm button state
        if all(not fleet.model.is_empty for fleet in fleets.values()):
            confirm_button.bg_color = ui.OK_GREEN
            confirm_button.hover_color = ui.OK_GREEN_HI
        else:
            confirm_button.bg_color = ui.DISABLED_BUTTON
            confirm_button.hover_color = ui.DISABLED_BUTTON

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)

        ui.draw_title(
            screen,
            "Players: Pick your Fleets",
            TITLE_FONT_SIZE,
            int(0.05 * const.SCREEN_HEIGHT),
        )

        for controls in (ai_toggles, one_of_each_buttons, clear_buttons):
            for control in controls.values():
                control.draw(screen, font)
        for fleet in fleets.values():
            fleet.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        if ship_picker is not None:
            shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, MODAL_SHADE_ALPHA))
            screen.blit(shade, (0, 0))
            for player, fleet in fleets.items():
                color = const.P1_COLOR if player == 1 else const.P2_COLOR
                dark_border = tuple(int(channel * 0.35) for channel in color)
                pygame.draw.rect(screen, dark_border, fleet.rect, 3)
            ship_picker.draw(
                screen, picker_title_font, picker_name_font, picker_cost_font
            )

        pygame.display.flip()
