import pygame
import sys
import json
import os
from src.UI import UI, UIButton, UIBox
import src.GameConstants as GameConstants
from src.Menus import PickShip
from typing import Dict, Tuple

# Display settings
SELECTION_ICON_SIZE = (int(UI.SCREEN_WIDTH*0.075), int(UI.SCREEN_WIDTH*0.075))
COST_FONT_SIZE = int(UI.SCREEN_HEIGHT*0.03)
TITLE_FONT_SIZE = int(UI.SCREEN_HEIGHT*0.08)
PLAYER_FONT_SIZE = int(UI.SCREEN_HEIGHT*0.03)


def load_ships() -> Dict:
    try:
        with open(GameConstants.SHIPS_JSON_PATH, 'r') as f:
            ships_data = json.load(f)

        simplified_data = {}
        for ship_name, stats in ships_data.items():
            simplified_data[ship_name] = {
                stats['ShipType']: stats['Cost'],
                'SpriteScale': stats['SpriteScale'],
                'SpriteLocation': stats['SpriteLocation']
            }

        return simplified_data
    except Exception as e:
        print(f"Error loading Ships.json: {e}")
        return {}

def load_ship_sprite(ship_name: str, ships_data: Dict) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """Load the ship sprite without scaling and return its surface and size."""
    try:
        sprite_path = os.path.join(ships_data[ship_name]['SpriteLocation'], f'{ship_name}00.png')
        sprite = pygame.image.load(sprite_path).convert_alpha()
        return sprite, sprite.get_size()
    except Exception as e:
        print(f"Error loading sprite for {ship_name}: {e}")
        surface = pygame.Surface(SELECTION_ICON_SIZE, pygame.SRCALPHA)
        surface.fill(UI.GREY)
        return surface, SELECTION_ICON_SIZE

def scale_sprites(original_sprites: Dict[str, pygame.Surface], target_size: int, ships_data: Dict) -> Dict[str, pygame.Surface]:
    # Find the maximum dimension across all sprites after applying SpriteScale
    max_dim = 1
    for name, sprite in original_sprites.items():
        width, height = sprite.get_size()
        sprite_scale = ships_data[name]['SpriteScale']
        scaled_width = width * sprite_scale
        scaled_height = height * sprite_scale
        max_dim = max(max_dim, scaled_width, scaled_height)

    # Calculate uniform scaling factor
    scale_factor = target_size / max_dim

    # Scale all sprites using both factors
    scaled_sprites = {}
    for name, sprite in original_sprites.items():
        width, height = sprite.get_size()
        sprite_scale = ships_data[name]['SpriteScale']
        new_width = int(width * sprite_scale * scale_factor)
        new_height = int(height * sprite_scale * scale_factor)
        scaled_sprites[name] = pygame.transform.scale(sprite, (new_width, new_height))

    return scaled_sprites


def save_fleets(left_fleet: UIBox.Fleet, right_fleet: UIBox.Fleet, left_ai: bool, right_ai: bool):
    """Save the current fleets and AI settings to Fleets.json."""
    fleets_data = {
        "Player1": {
            "ships": [name for _, name, _, _ in left_fleet.ships],
            "ai": left_ai
        },
        "Player2": {
            "ships": [name for _, name, _, _ in right_fleet.ships],
            "ai": right_ai
        }
    }
    try:
        os.makedirs('Config', exist_ok=True)
        with open(GameConstants.FLEETS_JSON_PATH, 'w') as f:
            json.dump(fleets_data, f, indent=4)
        print("Fleets and AI settings saved to Fleets.json")
    except Exception as e:
        print(f"Error saving Fleets.json: {e}")


def load_fleets(left_fleet: UIBox.Fleet, right_fleet: UIBox.Fleet, fleet_sprites: Dict[str, pygame.Surface], ships_data: Dict):
    """Load fleets and AI settings from Fleets.json if it exists."""
    if not os.path.exists(GameConstants.FLEETS_JSON_PATH):
        print("Fleets.json does not exist. Starting with empty fleets.")
        return False, False

    try:
        with open(GameConstants.FLEETS_JSON_PATH, 'r') as f:
            fleets_data = json.load(f)

        # Load Player 1 fleet
        player1_data = fleets_data.get("Player1", {})
        for ship_name in player1_data.get("ships", []):
            ship_info = ships_data.get(ship_name, {})
            if ship_info:
                ship_type = list(ship_info.keys())[0]
                ship_cost = ship_info[ship_type]
                left_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

        # Load Player 2 fleet
        player2_data = fleets_data.get("Player2", {})
        for ship_name in player2_data.get("ships", []):
            ship_info = ships_data.get(ship_name, {})
            if ship_info:
                ship_type = list(ship_info.keys())[0]
                ship_cost = ship_info[ship_type]
                right_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

        print("Fleets loaded from Fleets.json")
        return player1_data.get("ai", False), player2_data.get("ai", False)
    except Exception as e:
        print(f"Error loading Fleets.json: {e}")
        return False, False


def run(screen: pygame.Surface):
    """Run the Pick Fleet module."""
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, COST_FONT_SIZE)
    title_font = pygame.font.SysFont(None, TITLE_FONT_SIZE)
    player_font = pygame.font.SysFont(None, PLAYER_FONT_SIZE)
    background = UI.load_background(GameConstants.MENU_BG_PATH, UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT)

    # Load ships data and sprites
    ships_data = load_ships()
    original_sprites = {ship_name: load_ship_sprite(ship_name, ships_data)[0] for ship_name in ships_data}

    # Scale sprites for selection and fleet views, maintaining proportions
    selection_sprites = scale_sprites(original_sprites, SELECTION_ICON_SIZE[0], ships_data)
    fleet_sprites = scale_sprites(original_sprites, UI.FLEET_ICON_SIZE[0], ships_data)

    # Create UI components

    left_column_start =  int(0.033*UI.SCREEN_WIDTH)
    top_button_start = int(0.1*UI.SCREEN_HEIGHT)
    AI_toggle_width = int(0.075*UI.SCREEN_WIDTH)
    top_button_height = int(0.0375*UI.SCREEN_HEIGHT)
    each_button_width = int(0.5*(UI.SELECTION_WIDTH-AI_toggle_width-2*UI.button_spaceH))

    right_column_start = int(UI.SCREEN_WIDTH//2+(0.016*UI.SCREEN_WIDTH))

    # Create AI toggle buttons
    left_ai_toggle = UIButton.ToggleButton(
        left_column_start,
        top_button_start,
        AI_toggle_width,
        top_button_height,
        "AI",
        initial_state=False
    )
    right_ai_toggle = UIButton.ToggleButton(
        right_column_start,
        top_button_start,
        AI_toggle_width,
        top_button_height,
        "AI",
        initial_state=False
    )

    # Create "One of Each" buttons
    def create_one_of_each(fleet: UIBox.Fleet, ships_data: dict, fleet_sprites: dict):
        fleet.ships.clear()  # Remove all ships
        for ship_name, ship_info in ships_data.items():
            ship_type = list(ship_info.keys())[0]
            ship_cost = ship_info[ship_type]
            fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

    def clear_fleet(fleet: UIBox.Fleet, ships_data: dict, fleet_sprites: dict):
        fleet.ships.clear()  # Remove all ships

    left_one_of_each = UIButton.Button(
        left_column_start+AI_toggle_width+UI.button_spaceH,
        top_button_start, each_button_width, top_button_height,
        "One of Each Ship",
        lambda: create_one_of_each(left_fleet, ships_data, fleet_sprites),
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    right_one_of_each = UIButton.Button(
        right_column_start+AI_toggle_width+UI.button_spaceH,
        top_button_start, each_button_width, top_button_height,
        "One of Each Ship",
        lambda: create_one_of_each(right_fleet, ships_data, fleet_sprites),
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    left_clear_button = UIButton.Button(
        left_column_start+AI_toggle_width+2*UI.button_spaceH+each_button_width,
        top_button_start, each_button_width, top_button_height,
        "Clear Fleet 1",
        lambda: clear_fleet(left_fleet, ships_data, fleet_sprites),
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    right_clear_button = UIButton.Button(
        right_column_start+AI_toggle_width+2*UI.button_spaceH+each_button_width,
        top_button_start, each_button_width, top_button_height,
        "Clear Fleet 2",
        lambda: clear_fleet(right_fleet, ships_data, fleet_sprites),
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    left_ships = UIBox.ShipList(
        left_column_start,
        top_button_start+top_button_height+UI.button_spaceV,
        UI.SELECTION_WIDTH,
        UI.SELECTION_HEIGHT,
        "Player 1: Pick your fleet",
        SELECTION_ICON_SIZE
    )

    right_ships = UIBox.ShipList(
        right_column_start,
        top_button_start + top_button_height + UI.button_spaceV,
        UI.SELECTION_WIDTH,
        UI.SELECTION_HEIGHT,
        "Player 2: Pick your fleet",
        SELECTION_ICON_SIZE
    )

    left_fleet = UIBox.Fleet(
        left_column_start,
        top_button_start + top_button_height + UI.SELECTION_HEIGHT + 2*UI.button_spaceV,
        UI.SELECTION_WIDTH,
        UI.FLEET_HEIGHT,
        "Player 1 Fleet",
        UI.FLEET_ICON_SIZE
    )
    right_fleet = UIBox.Fleet(
        right_column_start,
        top_button_start + top_button_height + UI.SELECTION_HEIGHT + 2 * UI.button_spaceV,
        UI.SELECTION_WIDTH,
        UI.FLEET_HEIGHT,
        "Player 2 Fleet",
        UI.FLEET_ICON_SIZE
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

            PickShip.run(screen)

    running = True

    def cancel_callback():
        nonlocal running
        print("Cancel button clicked. Returning to main menu.")
        running = False

    confirm_button = UIButton.Button(
        UI.ok_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Confirm",
        confirm_callback,
        bg_color=UI.DISABLED_BUTTON,
        hover_color=UI.DISABLED_BUTTON,
        text_color=UI.WHITE
    )

    cancel_button = UIButton.Button(
        UI.can_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Cancel",
        cancel_callback,
        bg_color=UI.CAN_RED,
        hover_color=UI.CAN_RED_HI,
        text_color=UI.WHITE
    )

    while running:
        clock.tick(UI.FPS)
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
            left_ai_toggle.handle_event(event, UI.sound_manager)
            right_ai_toggle.handle_event(event, UI.sound_manager)

            left_one_of_each.handle_event(event, UI.sound_manager)
            right_one_of_each.handle_event(event, UI.sound_manager)

            left_clear_button.handle_event(event, UI.sound_manager)
            right_clear_button.handle_event(event, UI.sound_manager)

            confirm_button.handle_event(event, UI.sound_manager)
            cancel_button.handle_event(event, UI.sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                added = False
                for ships_list, fleet in [(left_ships, left_fleet), (right_ships, right_fleet)]:
                    if ships_list.rect.collidepoint(mouse_pos):
                        for sprite, name, cost, rect in ships_list.ships:
                            if rect and rect.collidepoint(mouse_pos):
                                success = fleet.add_ship(fleet_sprites[name], name, cost)
                                if success:
                                    print(f"Added {name} to {fleet.title}")
                                    UI.sound_manager.play_sound('menu')
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
                            UI.sound_manager.play_sound('menu')
                    elif right_fleet.rect.collidepoint(mouse_pos):
                        if right_fleet.remove_ship_at_pos(mouse_pos):
                            print("Removed a ship from Player 2 Fleet")
                            UI.sound_manager.play_sound('menu')

        # Update confirm button state
        if len(left_fleet.ships) > 0 and len(right_fleet.ships) > 0:
            confirm_button.bg_color = UI.OK_GREEN
            confirm_button.hover_color = UI.OK_GREEN_HI
        else:
            confirm_button.bg_color = UI.DISABLED_BUTTON
            confirm_button.hover_color = UI.DISABLED_BUTTON

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(UI.BG_COLOR)

        UI.draw_title(screen, "Pick Fleets", TITLE_FONT_SIZE, int(0.05*UI.SCREEN_HEIGHT))

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