import pygame
import sys
import json
import os
from UI import UIBox, UI
from typing import Dict, Tuple

# Display settings
SELECTION_ICON_SIZE = (int(UI.SCREEN_WIDTH*0.075), int(UI.SCREEN_WIDTH*0.075))
FLEET_ICON_SIZE = (int(UI.SCREEN_WIDTH*0.060), int(UI.SCREEN_WIDTH*0.060))
COST_FONT_SIZE = int(UI.SCREEN_HEIGHT*0.03)
TITLE_FONT_SIZE = int(UI.SCREEN_HEIGHT*0.08)
PLAYER_FONT_SIZE = int(UI.SCREEN_HEIGHT*0.03)

def load_ships() -> Dict:
    try:
        with open('Ships/Ships.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading Ships.json: {e}")
        return {}

def load_ship_sprite(ship_name: str) -> Tuple[pygame.Surface, Tuple[int, int]]:
    """Load ship sprite without scaling and return its surface and size."""
    try:
        sprite_path = os.path.join('Ships', ship_name, f'{ship_name}00.png')
        sprite = pygame.image.load(sprite_path).convert_alpha()
        return sprite, sprite.get_size()
    except Exception as e:
        print(f"Error loading sprite for {ship_name}: {e}")
        surface = pygame.Surface(SELECTION_ICON_SIZE, pygame.SRCALPHA)
        surface.fill(UI.GREY)
        return surface, SELECTION_ICON_SIZE

def scale_sprites(original_sprites: Dict[str, pygame.Surface], target_size: int) -> Dict[str, pygame.Surface]:
    """Scale sprites proportionally based on the largest dimension among all sprites."""
    # Find the maximum dimension across all sprites
    max_dim = 1  # Avoid division by zero
    for sprite in original_sprites.values():
        width, height = sprite.get_size()
        max_dim = max(max_dim, width, height)

    # Calculate scaling factor based on target size
    scale_factor = target_size / max_dim

    # Scale all sprites using the same factor
    scaled_sprites = {}
    for name, sprite in original_sprites.items():
        width, height = sprite.get_size()
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
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
        with open('Config/Fleets.json', 'w') as f:
            json.dump(fleets_data, f, indent=4)
        print("Fleets and AI settings saved to Fleets.json")
    except Exception as e:
        print(f"Error saving Fleets.json: {e}")


def load_fleets(left_fleet: UIBox.Fleet, right_fleet: UIBox.Fleet, fleet_sprites: Dict[str, pygame.Surface], ships_data: Dict):
    """Load fleets and AI settings from Fleets.json if it exists."""
    if not os.path.exists('Config/Fleets.json'):
        print("Fleets.json does not exist. Starting with empty fleets.")
        return False, False

    try:
        with open('Config/Fleets.json', 'r') as f:
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

    # Load ships data and sprites
    ships_data = load_ships()
    original_sprites = {ship_name: load_ship_sprite(ship_name)[0] for ship_name in ships_data}

    # Scale sprites for selection and fleet views, maintaining proportions
    selection_sprites = scale_sprites(original_sprites, SELECTION_ICON_SIZE[0])
    fleet_sprites = scale_sprites(original_sprites, FLEET_ICON_SIZE[0])

    # Create UI components

    selection_width = int(0.45*UI.SCREEN_WIDTH)
    selection_height = int(.35*UI.SCREEN_HEIGHT)
    fleet_height = int(.4*UI.SCREEN_HEIGHT)

    left_column_start =  int(0.033*UI.SCREEN_WIDTH)
    top_button_start = int(0.1*UI.SCREEN_HEIGHT)
    AI_toggle_width = int(0.075*UI.SCREEN_WIDTH)
    top_button_height = int(0.0375*UI.SCREEN_HEIGHT)
    each_button_width = selection_width-AI_toggle_width-UI.button_spaceH

    right_column_start = int(UI.SCREEN_WIDTH//2+(0.016*UI.SCREEN_WIDTH))

    # Create AI toggle buttons
    left_ai_toggle = UI.ToggleButton(
        left_column_start,
        top_button_start,
        AI_toggle_width,
        top_button_height,
        "AI",
        initial_state=False
    )
    right_ai_toggle = UI.ToggleButton(
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

    left_one_of_each = UI.Button(
        left_column_start+AI_toggle_width+UI.button_spaceH,
        top_button_start, each_button_width, top_button_height,
        "Pick One of Each Ship",
        lambda: create_one_of_each(left_fleet, ships_data, fleet_sprites),
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    right_one_of_each = UI.Button(
        right_column_start+AI_toggle_width+UI.button_spaceH,
        top_button_start, each_button_width, top_button_height,
        "Pick One of Each Ship",
        lambda: create_one_of_each(right_fleet, ships_data, fleet_sprites),
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    left_ships = UIBox.ShipList(
        left_column_start,
        top_button_start+top_button_height+UI.button_spaceV,
        selection_width,
        selection_height,
        "Player 1: Pick your fleet",
        SELECTION_ICON_SIZE
    )

    right_ships = UIBox.ShipList(
        right_column_start,
        top_button_start + top_button_height + UI.button_spaceV,
        selection_width,
        selection_height,
        "Player 2: Pick your fleet",
        SELECTION_ICON_SIZE
    )

    left_fleet = UIBox.Fleet(
        left_column_start,
        top_button_start + top_button_height + selection_height + 2*UI.button_spaceV,
        selection_width,
        fleet_height,
        "Player 1 Fleet",
        FLEET_ICON_SIZE
    )
    right_fleet = UIBox.Fleet(
        right_column_start,
        top_button_start + top_button_height + selection_height + 2 * UI.button_spaceV,
        selection_width,
        fleet_height,
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
    ok_button_width = int(0.150*UI.SCREEN_WIDTH)
    ok_button_height = int(0.05*UI.SCREEN_HEIGHT)

    def confirm_callback():
        if len(left_fleet.ships) > 0 and len(right_fleet.ships) > 0:
            save_fleets(left_fleet, right_fleet, left_ai_toggle.value, right_ai_toggle.value)
            print("Fleets confirmed.")
            UI.sound_manager.play_sound('menu')

    running = True

    def cancel_callback():
        nonlocal running
        print("Cancel button clicked. Returning to main menu.")
        UI.sound_manager.play_sound('menu')
        running = False

    confirm_button = UI.Button(
        UI.ok_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Confirm",
        confirm_callback,
        bg_color=UI.GREY,
        hover_color=UI.GREY,
        text_color=UI.WHITE
    )

    cancel_button = UI.Button(
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
            confirm_button.bg_color = UI.GREY
            confirm_button.hover_color = UI.GREY

        # Draw everything
        screen.fill(UI.BG_COLOR)
        UI.draw_title(screen, "Pick Fleets", TITLE_FONT_SIZE, int(0.05*UI.SCREEN_HEIGHT))

        # Draw AI toggles
        left_ai_toggle.draw(screen, font)
        right_ai_toggle.draw(screen, font)

        left_one_of_each.draw(screen, font)
        right_one_of_each.draw(screen, font)

        left_ships.draw(screen, font, player_font)
        right_ships.draw(screen, font, player_font)
        left_fleet.draw(screen, font)
        right_fleet.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()