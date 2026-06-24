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
cost_FONT_SIZE = int(const.SCREEN_HEIGHT*0.03)
TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT*0.08)
PLAYER_FONT_SIZE = int(const.SCREEN_HEIGHT*0.03)


def load_ships():
    """Return the authoritative typed catalog (compatibility entry point)."""
    return SHIP_DEFINITIONS


def load_ship_sprites(ships_data, resources=None):
    def fallback(_ship_name):
        surface = pygame.Surface(SELECTION_ICON_SIZE, pygame.SRCALPHA)
        surface.fill(ui.GREY)
        return surface

    return load_menu_ship_sprites(
        ships_data, resources=resources, fallback=fallback
    )

def create_sprite_sets(ships_data):
    """Create selection and fleet sprites from a single load."""
    original_sprites = load_ship_sprites(ships_data)
    selection_sprites = scale_ship_sprites(
        original_sprites, SELECTION_ICON_SIZE[0], ships_data
    )

    # Calculate fleet icon size first
    fleet = ui_box.Fleet(0, 0, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT, "", (0, 0))
    fleet_size = fleet.icon_size[0]
    fleet_sprites = scale_ship_sprites(original_sprites, fleet_size, ships_data)

    return selection_sprites, fleet_sprites

scale_sprites = scale_ship_sprites

def save_fleets(left_fleet: ui_box.Fleet, right_fleet: ui_box.Fleet, left_ai: bool, right_ai: bool):
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
        populate_fleet_panel(
            fleet_panel, player_fleet.ships, fleet_sprites, ships_data
        )

    print("Fleets loaded from fleets.json")
    return fleets.player1.ai, fleets.player2.ai


def run(screen: pygame.Surface, menu_sound_manager=None, audio_service=None):
    """Run the Pick Fleet module."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, cost_FONT_SIZE)
    title_font = pygame.font.SysFont(None, TITLE_FONT_SIZE)
    player_font = pygame.font.SysFont(None, PLAYER_FONT_SIZE)
    background = ui.load_background(const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT)

    # Load ships data and sprites
    ships_data = load_ships()

    # Scale sprites for selection and fleet views, maintaining proportions
    selection_sprites, fleet_sprites = create_sprite_sets(ships_data)

    # Create UI components

    left_column_start =  int(0.033*const.SCREEN_WIDTH)
    top_button_start = int(0.1*const.SCREEN_HEIGHT)
    AI_toggle_width = int(0.075*const.SCREEN_WIDTH)
    top_button_height = int(0.0375*const.SCREEN_HEIGHT)
    each_button_width = int(0.5*(ui.SELECTION_WIDTH-AI_toggle_width-2*ui.button_spaceH))

    right_column_start = int(const.SCREEN_WIDTH//2+(0.016*const.SCREEN_WIDTH))
    columns = {1: left_column_start, 2: right_column_start}
    fleet_top = (
        top_button_start + top_button_height + ui.SELECTION_HEIGHT
        + 2 * ui.button_spaceV
    )
    fleets = ui_box.create_player_fleet_panels(
        columns, fleet_top, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT,
        FLEET_ICON_SIZE,
    )
    ship_lists = {
        player: ui_box.ShipList(
            columns[player],
            top_button_start + top_button_height + ui.button_spaceV,
            ui.SELECTION_WIDTH,
            ui.SELECTION_HEIGHT,
            f"Player {player}: Pick your fleet",
            SELECTION_ICON_SIZE,
            color=const.P1_COLOR if player == 1 else const.P2_COLOR
        )
        for player in (1, 2)
    }
    ai_toggles = {
        player: ui_button.ToggleButton(
            columns[player], top_button_start, AI_toggle_width,
            top_button_height, "AI", initial_state=False,
        )
        for player in (1, 2)
    }

    def create_one_of_each(player):
        fleets[player].clear()
        populate_fleet_panel(
            fleets[player], ships_data, fleet_sprites, ships_data
        )

    one_of_each_buttons = {
        player: ui_button.Button(
            columns[player] + AI_toggle_width + ui.button_spaceH,
            top_button_start, each_button_width, top_button_height,
            "One of Each Ship",
            lambda player=player: create_one_of_each(player),
            bg_color=ui.MENU_BUTTON_COLOR,
            hover_color=ui.MENU_BUTTON_COLOR_HI,
        )
        for player in (1, 2)
    }
    clear_buttons = {
        player: ui_button.Button(
            columns[player] + AI_toggle_width + 2 * ui.button_spaceH
            + each_button_width,
            top_button_start, each_button_width, top_button_height,
            f"Clear Fleet {player}",
            fleets[player].clear,
            bg_color=ui.MENU_BUTTON_COLOR,
            hover_color=ui.MENU_BUTTON_COLOR_HI,
        )
        for player in (1, 2)
    }

    for ship_name, definition in ships_data.items():
        for ship_list in ship_lists.values():
            ship_list.add_ship(
                selection_sprites[ship_name], ship_name, definition.cost
            )

    loaded_ai = load_fleets(
        fleets[1], fleets[2], fleet_sprites, ships_data
    )
    for player, ai in zip((1, 2), loaded_ai):
        ai_toggles[player].is_on = ai

    # Create buttons
    def confirm_callback():
        if all(not fleet.model.is_empty for fleet in fleets.values()):
            save_fleets(
                fleets[1], fleets[2],
                ai_toggles[1].value, ai_toggles[2].value,
            )
            print("Fleets confirmed.")

            pick_ship.run(
                screen=screen,
                menu_sound_manager=menu_sound_manager,
                audio_service=audio_service,
            )

    running = True

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
        text_color=ui.WHITE
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
        text_color=ui.WHITE
    )

    while running:
        clock.tick(const.FPS)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
                for ship_list in ship_lists.values():
                    if ship_list.rect.collidepoint(event.pos):
                        ship_list.handle_event(event)

            for controls in (ai_toggles, one_of_each_buttons, clear_buttons):
                for control in controls.values():
                    control.handle_event(event, menu_sound_manager)

            confirm_button.handle_event(event, menu_sound_manager)
            cancel_button.handle_event(event, menu_sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                added = False
                for player in (1, 2):
                    ships_list = ship_lists[player]
                    fleet = fleets[player]
                    if ships_list.rect.collidepoint(mouse_pos):
                        for sprite, name, cost, rect in ships_list.ships:
                            if rect and rect.collidepoint(mouse_pos):
                                success = fleet.add_ship(fleet_sprites[name], name, cost)
                                if success:
                                    print(f"Added {name} to {fleet.title}")
                                    if menu_sound_manager:
                                        menu_sound_manager.play_sound('menu')
                                else:
                                    print(f"{fleet.title} is full. Cannot add {name}.")
                                added = True
                                break
                        if added:
                            break

                if not added:
                    for player, fleet in fleets.items():
                        if (fleet.rect.collidepoint(mouse_pos)
                                and fleet.remove_ship_at_pos(mouse_pos)):
                            print(f"Removed a ship from Player {player} Fleet")
                            if menu_sound_manager:
                                menu_sound_manager.play_sound('menu')
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

        ui.draw_title(screen, "Players: Pick your Fleets", TITLE_FONT_SIZE, int(0.05*const.SCREEN_HEIGHT))

        for controls in (ai_toggles, one_of_each_buttons, clear_buttons):
            for control in controls.values():
                control.draw(screen, font)
        for ship_list in ship_lists.values():
            ship_list.draw(screen, font, player_font)
        for fleet in fleets.values():
            fleet.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()
