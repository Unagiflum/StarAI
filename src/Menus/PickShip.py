import pygame
import json
import os
import sys
import random
from src.UI import UI, UIButton, UIBox
from src.Ships.SpaceShip import SpaceShip
from src.Battle import Battle

TITLE_FONT_SIZE = int(UI.SCREEN_HEIGHT * 0.08)
HIGHLIGHT_COLOR = (50, 50, 0)
FLEET_ICON_WIDTH = int(UI.SCREEN_WIDTH * 0.06)
FLEET_ICON_SIZE = (FLEET_ICON_WIDTH, FLEET_ICON_WIDTH)
X_COLOR = (255, 0, 0)  # Red X for dead ships
X_THICKNESS = 2


def draw_x(surface, rect):
    """Draw a red X in a square box sized to the largest ship dimension."""
    size = FLEET_ICON_WIDTH
    x_rect = pygame.Rect(
        rect.centerx - size//2,
        rect.centery - size//2,
        size,
        size
    )
    pygame.draw.line(surface, X_COLOR, x_rect.topleft, x_rect.bottomright, X_THICKNESS)
    pygame.draw.line(surface, X_COLOR, x_rect.bottomleft, x_rect.topright, X_THICKNESS)

def load_fleet_data():
    try:
        with open('Config/Fleets.json', 'r') as f:
            fleet_data = json.load(f)

        # Create SpaceShip objects
        player1_ships = [SpaceShip(ship_name, 1) for ship_name in fleet_data["Player1"]["ships"]]
        player2_ships = [SpaceShip(ship_name, 2) for ship_name in fleet_data["Player2"]["ships"]]

        return fleet_data, player1_ships, player2_ships
    except Exception as e:
        print(f"Error loading Fleets.json: {e}")
        return None, [], []


def load_ship_sprite(ship_name):
    try:
        sprite_path = os.path.join('Ships', ship_name, f'{ship_name}00.png')
        sprite = pygame.image.load(sprite_path).convert_alpha()
        return sprite, sprite.get_size()
    except Exception as e:
        print(f"Error loading sprite for {ship_name}: {e}")
        return None, None


def scale_sprites(original_sprites, target_size):
    max_dim = 1
    for sprite in original_sprites.values():
        width, height = sprite.get_size()
        max_dim = max(max_dim, width, height)

    scale_factor = target_size / max_dim
    scaled_sprites = {}
    for name, sprite in original_sprites.items():
        width, height = sprite.get_size()
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_sprites[name] = pygame.transform.scale(sprite, (new_width, new_height))

    return scaled_sprites


def load_ships_data():
    try:
        with open('Ships/Ships.json', 'r') as f:
            ships_data = json.load(f)

        original_sprites = {}
        simplified_data = {}
        for ship_name, stats in ships_data.items():
            simplified_data[ship_name] = {stats['ShipType']: stats['Cost']}
            sprite, _ = load_ship_sprite(ship_name)
            if sprite:
                original_sprites[ship_name] = sprite

        return simplified_data, original_sprites
    except Exception as e:
        print(f"Error loading Ships.json: {e}")
        return None, None


def run(screen):
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(UI.SCREEN_HEIGHT * 0.03))
    background = UI.load_background("UI/Menu.png", UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT)

    fleet_data, player1_ships, player2_ships = load_fleet_data()
    ships_data, original_sprites = load_ships_data()
    if not fleet_data or not ships_data or not original_sprites:
        return

    player2_ships[0].currently_alive = False

    LEFT_COLUMN_START = int(0.033 * UI.SCREEN_WIDTH)
    RIGHT_COLUMN_START = int(UI.SCREEN_WIDTH // 2 + (0.016 * UI.SCREEN_WIDTH))

    FLEET_TOP = int(0.15 * UI.SCREEN_HEIGHT)
    FLEET_WIDTH = int(0.45 * UI.SCREEN_WIDTH)
    FLEET_HEIGHT = int(0.4 * UI.SCREEN_HEIGHT)

    SELECTION_BOX_SIZE = int(UI.SCREEN_WIDTH * 0.165)
    SELECTION_TOP = FLEET_TOP + FLEET_HEIGHT + int(0.05 * UI.SCREEN_HEIGHT)
    SELECTION_L_LEFT = LEFT_COLUMN_START + int(0.5 * (FLEET_WIDTH - SELECTION_BOX_SIZE))
    SELECTION_R_LEFT = RIGHT_COLUMN_START + int(0.5 * (FLEET_WIDTH - SELECTION_BOX_SIZE))
    RAND_TOP = SELECTION_TOP + SELECTION_BOX_SIZE + int(0.01 * UI.SCREEN_HEIGHT)

    fleet_sprites = scale_sprites(original_sprites, int(UI.SCREEN_WIDTH * 0.048))
    selection_sprites = scale_sprites(original_sprites, SELECTION_BOX_SIZE)

    left_fleet = UIBox.Fleet(
        LEFT_COLUMN_START,
        FLEET_TOP,
        FLEET_WIDTH,
        FLEET_HEIGHT,
        "Player 1 Fleet",
        FLEET_ICON_SIZE
    )

    right_fleet = UIBox.Fleet(
        RIGHT_COLUMN_START,
        FLEET_TOP,
        FLEET_WIDTH,
        FLEET_HEIGHT,
        "Player 2 Fleet",
        FLEET_ICON_SIZE
    )

    # Load ships into fleets
    for ship, ship_name in zip(player1_ships, fleet_data["Player1"]["ships"]):
        ship_info = ships_data[ship_name]
        ship_type = list(ship_info.keys())[0]
        ship_cost = ship_info[ship_type]
        left_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

    for ship, ship_name in zip(player2_ships, fleet_data["Player2"]["ships"]):
        ship_info = ships_data[ship_name]
        ship_type = list(ship_info.keys())[0]
        ship_cost = ship_info[ship_type]
        right_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

    left_selection = {"rect": pygame.Rect(SELECTION_L_LEFT,
                                          SELECTION_TOP,
                                          SELECTION_BOX_SIZE,
                                          SELECTION_BOX_SIZE),
                      "ship": None,
                      "sprite": None,
                      "ship_obj": None}

    right_selection = {"rect": pygame.Rect(SELECTION_R_LEFT,
                                           SELECTION_TOP,
                                           SELECTION_BOX_SIZE,
                                           SELECTION_BOX_SIZE),
                       "ship": None,
                       "sprite": None,
                       "ship_obj": None}

    def pick_random_left():
        alive_ships = [(i, ship) for i, ship in enumerate(player1_ships) if ship.currently_alive]
        if alive_ships:
            idx, ship_obj = random.choice(alive_ships)
            sprite, name, cost, _ = left_fleet.ships[idx]
            left_selection["ship"] = name
            left_selection["sprite"] = selection_sprites[name]
            left_selection["ship_obj"] = ship_obj

    def pick_random_right():
        alive_ships = [(i, ship) for i, ship in enumerate(player2_ships) if ship.currently_alive]
        if alive_ships:
            idx, ship_obj = random.choice(alive_ships)
            sprite, name, cost, _ = right_fleet.ships[idx]
            right_selection["ship"] = name
            right_selection["sprite"] = selection_sprites[name]
            right_selection["ship_obj"] = ship_obj

    random_left = UIButton.Button(
        SELECTION_L_LEFT,
        RAND_TOP,
        SELECTION_BOX_SIZE,
        int(0.05 * UI.SCREEN_HEIGHT),
        "Pick Random",
        pick_random_left,
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    random_right = UIButton.Button(
        SELECTION_R_LEFT,
        RAND_TOP,
        SELECTION_BOX_SIZE,
        int(0.05 * UI.SCREEN_HEIGHT),
        "Pick Random",
        pick_random_right,
        bg_color=UI.MENU_BUTTON_COLOR,
        hover_color=UI.MENU_BUTTON_COLOR_HI
    )

    def confirm_callback():
        if (left_selection["ship_obj"] and right_selection["ship_obj"] and
                left_selection["ship_obj"].currently_alive and right_selection["ship_obj"].currently_alive):
            print("Ships selected:", left_selection["ship"], "vs", right_selection["ship"])
            # Call Battle.run() with the selected ships
            Battle.run(screen, left_selection["ship_obj"], right_selection["ship_obj"])
            return None, None  # Return None to maintain compatibility with existing code

    confirm_button = UIButton.Button(
        UI.ok_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Confirm",
        confirm_callback,
        bg_color=UI.DISABLED_BUTTON,
        hover_color=UI.DISABLED_BUTTON
    )

    cancel_button = UIButton.Button(
        UI.can_button_left,
        UI.ok_button_top,
        UI.ok_button_width,
        UI.ok_button_height,
        "Cancel",
        lambda: None,
        bg_color=UI.CAN_RED,
        hover_color=UI.CAN_RED_HI
    )

    running = True
    while running:
        clock.tick(UI.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                if left_fleet.rect.collidepoint(mouse_pos):
                    for i, (_, name, _, rect) in enumerate(left_fleet.ships):
                        if rect and rect.collidepoint(mouse_pos) and player1_ships[i].currently_alive:
                            left_selection["ship"] = name
                            left_selection["sprite"] = selection_sprites[name]
                            left_selection["ship_obj"] = player1_ships[i]
                            UI.sound_manager.play_sound('menu')
                            break

                elif right_fleet.rect.collidepoint(mouse_pos):
                    for i, (_, name, _, rect) in enumerate(right_fleet.ships):
                        if rect and rect.collidepoint(mouse_pos) and player2_ships[i].currently_alive:
                            right_selection["ship"] = name
                            right_selection["sprite"] = selection_sprites[name]
                            right_selection["ship_obj"] = player2_ships[i]
                            UI.sound_manager.play_sound('menu')
                            break

            random_left.handle_event(event, UI.sound_manager)
            random_right.handle_event(event, UI.sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button.rect.collidepoint(event.pos):
                    UI.sound_manager.play_sound('menu')
                    running = False
                elif (confirm_button.rect.collidepoint(event.pos) and
                      left_selection["ship_obj"] and right_selection["ship_obj"] and
                      left_selection["ship_obj"].currently_alive and
                      right_selection["ship_obj"].currently_alive):
                    UI.sound_manager.play_sound('menu')
                    return confirm_callback()

        if (left_selection["ship_obj"] and right_selection["ship_obj"] and
                left_selection["ship_obj"].currently_alive and
                right_selection["ship_obj"].currently_alive):
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

        UI.draw_title(screen, "Players: Pick your Ship", TITLE_FONT_SIZE, int(0.05 * UI.SCREEN_HEIGHT))

        # Draw fleets
        left_fleet.draw(screen, font)
        right_fleet.draw(screen, font)

        # Draw highlight boxes under selected ships and X's over dead ships
        if left_selection["ship"]:
            for i, (_, name, _, rect) in enumerate(left_fleet.ships):
                if name == left_selection["ship"]:
                    highlight_rect = pygame.Rect(rect.centerx - FLEET_ICON_SIZE[0] // 2,
                                                 rect.centery - FLEET_ICON_SIZE[1] // 2,
                                                 FLEET_ICON_SIZE[0],
                                                 FLEET_ICON_SIZE[1])
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)
                    break

        if right_selection["ship"]:
            for i, (_, name, _, rect) in enumerate(right_fleet.ships):
                if name == right_selection["ship"]:
                    highlight_rect = pygame.Rect(rect.centerx - FLEET_ICON_SIZE[0] // 2,
                                                 rect.centery - FLEET_ICON_SIZE[1] // 2,
                                                 FLEET_ICON_SIZE[0],
                                                 FLEET_ICON_SIZE[1])
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)
                    break

        # Redraw ships to appear above highlights
        for i, (sprite, _, _, rect) in enumerate(left_fleet.ships):
            screen.blit(sprite, rect)
            if not player1_ships[i].currently_alive:
                draw_x(screen, rect)

        for i, (sprite, _, _, rect) in enumerate(right_fleet.ships):
            screen.blit(sprite, rect)
            if not player2_ships[i].currently_alive:
                draw_x(screen, rect)

        # Draw selection boxes
        for selection in [left_selection, right_selection]:
            pygame.draw.rect(screen, UI.BLACK, selection["rect"])
            pygame.draw.rect(screen, UI.WHITE, selection["rect"], 2)
            if selection["sprite"]:
                sprite_rect = selection["sprite"].get_rect(center=selection["rect"].center)
                screen.blit(selection["sprite"], sprite_rect)

        random_left.draw(screen, font)
        random_right.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()

    return None, None