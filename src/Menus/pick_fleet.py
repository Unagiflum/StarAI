import pygame
import sys

from src.UI import ui, ui_button, ui_box
import src.const as const
from src.configuration import Fleets, FleetsRepository, PlayerFleet
from src.Menus import pick_ship
from src.persistence import PersistenceValidationError
from typing import Dict, Tuple
from src.Objects.Ships.catalog import SHIPS_DATA
from src.resources import default_assets


# Display settings
SELECTION_ICON_SIZE = const.SELECTION_ICON_SIZE
FLEET_ICON_SIZE = const.FLEET_ICON_SIZE
cost_FONT_SIZE = int(const.SCREEN_HEIGHT*0.03)
TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT*0.08)
PLAYER_FONT_SIZE = int(const.SCREEN_HEIGHT*0.03)


def load_ships() -> Dict:
    simplified_data = {}
    for ship_name, stats in SHIPS_DATA.items():
        simplified_data[ship_name] = {
            stats['ship_type']: stats['cost'],
            'sprite_scale': stats['sprite_scale'],
            'sprite_path': stats['sprite_path']
        }
    return simplified_data


def load_ship_sprites(ships_data: Dict, resources=None) -> Dict[str, pygame.Surface]:
    """Load all ship sprites once."""
    sprites = {}
    resources = resources or default_assets()
    for ship_name in ships_data:
        try:
            sprites[ship_name] = resources.menu_ship_sprite(ship_name)
        except (OSError, pygame.error) as e:
            print(f"Error loading sprite for {ship_name}: {e}")
            surface = pygame.Surface(SELECTION_ICON_SIZE, pygame.SRCALPHA)
            surface.fill(ui.GREY)
            sprites[ship_name] = surface
    return sprites

def create_sprite_sets(ships_data: Dict) -> Tuple[Dict[str, pygame.Surface], Dict[str, pygame.Surface]]:
    """Create selection and fleet sprites from a single load."""
    original_sprites = load_ship_sprites(ships_data)
    selection_sprites = scale_sprites(original_sprites, SELECTION_ICON_SIZE[0], ships_data)

    # Calculate fleet icon size first
    fleet = ui_box.Fleet(0, 0, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT, "", (0, 0))
    fleet_size = fleet.icon_size[0]
    fleet_sprites = scale_sprites(original_sprites, fleet_size, ships_data)

    return selection_sprites, fleet_sprites

def scale_sprites(original_sprites: Dict[str, pygame.Surface], target_size: int, ships_data: Dict) -> Dict[str, pygame.Surface]:
    # Find the maximum dimension across all sprites after applying sprite_scale
    max_dim = 1
    for name, sprite in original_sprites.items():
        width, height = sprite.get_size()
        sprite_scale = ships_data[name]['sprite_scale']
        scaled_width = width * sprite_scale
        scaled_height = height * sprite_scale
        max_dim = max(max_dim, scaled_width, scaled_height)

    # Calculate uniform scaling factor
    scale_factor = target_size / max_dim

    # Scale all sprites using both factors
    scaled_sprites = {}
    for name, sprite in original_sprites.items():
        width, height = sprite.get_size()
        sprite_scale = ships_data[name]['sprite_scale']
        new_width = int(width * sprite_scale * scale_factor)
        new_height = int(height * sprite_scale * scale_factor)
        scaled_sprites[name] = pygame.transform.scale(sprite, (new_width, new_height))

    return scaled_sprites

def save_fleets(left_fleet: ui_box.Fleet, right_fleet: ui_box.Fleet, left_ai: bool, right_ai: bool):
    """Save the current fleets and AI settings to fleets.json."""
    fleets = Fleets(
        PlayerFleet(tuple(name for _, name, _, _ in left_fleet.ships), left_ai),
        PlayerFleet(tuple(name for _, name, _, _ in right_fleet.ships), right_ai),
    )
    try:
        FleetsRepository(const.FLEETS_JSON_PATH, SHIPS_DATA).save(fleets)
        print("Fleets and AI settings saved to fleets.json")
    except (OSError, PersistenceValidationError) as e:
        print(f"Error saving fleets.json: {e}")


def load_fleets(left_fleet: ui_box.Fleet, right_fleet: ui_box.Fleet, fleet_sprites: Dict[str, pygame.Surface], ships_data: Dict):
    """Load fleets and AI settings from fleets.json if it exists."""
    if not const.FLEETS_JSON_PATH.exists():
        print("fleets.json does not exist. Starting with empty fleets.")
        return False, False

    fleets = FleetsRepository(const.FLEETS_JSON_PATH, ships_data).load()
    for ship_name in fleets.player1.ships:
        ship_info = ships_data[ship_name]
        ship_type = next(iter(ship_info))
        left_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_info[ship_type])
    for ship_name in fleets.player2.ships:
        ship_info = ships_data[ship_name]
        ship_type = next(iter(ship_info))
        right_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_info[ship_type])

    print("Fleets loaded from fleets.json")
    return fleets.player1.ai, fleets.player2.ai


def run(screen: pygame.Surface):
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

    # Create AI toggle buttons
    left_ai_toggle = ui_button.ToggleButton(
        left_column_start,
        top_button_start,
        AI_toggle_width,
        top_button_height,
        "AI",
        initial_state=False
    )
    right_ai_toggle = ui_button.ToggleButton(
        right_column_start,
        top_button_start,
        AI_toggle_width,
        top_button_height,
        "AI",
        initial_state=False
    )

    # Create "One of Each" buttons
    def create_one_of_each(fleet: ui_box.Fleet, ships_data: dict, fleet_sprites: dict):
        fleet.ships.clear()  # Remove all ships
        for ship_name, ship_info in ships_data.items():
            ship_type = list(ship_info.keys())[0]
            ship_cost = ship_info[ship_type]
            fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

    def clear_fleet(fleet: ui_box.Fleet, ships_data: dict, fleet_sprites: dict):
        fleet.ships.clear()  # Remove all ships

    left_one_of_each = ui_button.Button(
        left_column_start+AI_toggle_width+ui.button_spaceH,
        top_button_start, each_button_width, top_button_height,
        "One of Each Ship",
        lambda: create_one_of_each(left_fleet, ships_data, fleet_sprites),
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI
    )

    right_one_of_each = ui_button.Button(
        right_column_start+AI_toggle_width+ui.button_spaceH,
        top_button_start, each_button_width, top_button_height,
        "One of Each Ship",
        lambda: create_one_of_each(right_fleet, ships_data, fleet_sprites),
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI
    )

    left_clear_button = ui_button.Button(
        left_column_start+AI_toggle_width+2*ui.button_spaceH+each_button_width,
        top_button_start, each_button_width, top_button_height,
        "Clear Fleet 1",
        lambda: clear_fleet(left_fleet, ships_data, fleet_sprites),
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI
    )

    right_clear_button = ui_button.Button(
        right_column_start+AI_toggle_width+2*ui.button_spaceH+each_button_width,
        top_button_start, each_button_width, top_button_height,
        "Clear Fleet 2",
        lambda: clear_fleet(right_fleet, ships_data, fleet_sprites),
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI
    )

    left_ships = ui_box.ShipList(
        left_column_start,
        top_button_start+top_button_height+ui.button_spaceV,
        ui.SELECTION_WIDTH,
        ui.SELECTION_HEIGHT,
        "Player 1: Pick your fleet",
        SELECTION_ICON_SIZE
    )

    right_ships = ui_box.ShipList(
        right_column_start,
        top_button_start + top_button_height + ui.button_spaceV,
        ui.SELECTION_WIDTH,
        ui.SELECTION_HEIGHT,
        "Player 2: Pick your fleet",
        SELECTION_ICON_SIZE
    )

    left_fleet = ui_box.Fleet(
        left_column_start,
        top_button_start + top_button_height + ui.SELECTION_HEIGHT + 2*ui.button_spaceV,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        "Player 1 Fleet",
        FLEET_ICON_SIZE
    )
    right_fleet = ui_box.Fleet(
        right_column_start,
        top_button_start + top_button_height + ui.SELECTION_HEIGHT + 2 * ui.button_spaceV,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        "Player 2 Fleet",
        FLEET_ICON_SIZE
    )

    # Add ships to selection lists
    for ship_name, ship_info in ships_data.items():
        ship_type = list(ship_info.keys())[0]
        ship_cost = ship_info[ship_type]
        left_ships.add_ship(selection_sprites[ship_name], ship_name, ship_cost)
        right_ships.add_ship(selection_sprites[ship_name], ship_name, ship_cost)

    # Load saved fleets and AI settings
    left_ai, right_ai = load_fleets(left_fleet, right_fleet, fleet_sprites, ships_data)
    left_ai_toggle.is_on = left_ai
    right_ai_toggle.is_on = right_ai

    # Create buttons
    def confirm_callback():
        if len(left_fleet.ships) > 0 and len(right_fleet.ships) > 0:
            save_fleets(left_fleet, right_fleet, left_ai_toggle.value, right_ai_toggle.value)
            print("Fleets confirmed.")

            pick_ship.run(screen)

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
                if left_ships.rect.collidepoint(event.pos):
                    left_ships.handle_event(event)
                if right_ships.rect.collidepoint(event.pos):
                    right_ships.handle_event(event)

            # Handle toggle button events
            left_ai_toggle.handle_event(event, ui.sound_manager)
            right_ai_toggle.handle_event(event, ui.sound_manager)

            left_one_of_each.handle_event(event, ui.sound_manager)
            right_one_of_each.handle_event(event, ui.sound_manager)

            left_clear_button.handle_event(event, ui.sound_manager)
            right_clear_button.handle_event(event, ui.sound_manager)

            confirm_button.handle_event(event, ui.sound_manager)
            cancel_button.handle_event(event, ui.sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                added = False
                for ships_list, fleet in [(left_ships, left_fleet), (right_ships, right_fleet)]:
                    if ships_list.rect.collidepoint(mouse_pos):
                        for sprite, name, cost, rect in ships_list.ships:
                            if rect and rect.collidepoint(mouse_pos):
                                success = fleet.add_ship(fleet_sprites[name], name, cost)
                                if success:
                                    print(f"Added {name} to {fleet.title}")
                                    ui.sound_manager.play_sound('menu')
                                else:
                                    print(f"{fleet.title} is full. Cannot add {name}.")
                                added = True
                                break
                        if added:
                            break

                if not added:
                    if left_fleet.rect.collidepoint(mouse_pos):
                        if left_fleet.remove_ship_at_pos(mouse_pos):
                            print("Removed a ship from Player 1 Fleet")
                            ui.sound_manager.play_sound('menu')
                    elif right_fleet.rect.collidepoint(mouse_pos):
                        if right_fleet.remove_ship_at_pos(mouse_pos):
                            print("Removed a ship from Player 2 Fleet")
                            ui.sound_manager.play_sound('menu')

        # Update confirm button state
        if len(left_fleet.ships) > 0 and len(right_fleet.ships) > 0:
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

        # Draw AI toggles
        left_ai_toggle.draw(screen, font)
        right_ai_toggle.draw(screen, font)

        left_one_of_each.draw(screen, font)
        right_one_of_each.draw(screen, font)

        left_clear_button.draw(screen, font)
        right_clear_button.draw(screen, font)

        left_ships.draw(screen, font, player_font)
        right_ships.draw(screen, font, player_font)
        left_fleet.draw(screen, font)
        right_fleet.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()
