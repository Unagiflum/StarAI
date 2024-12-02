import pygame
import json
import os
import sys
import random
from UI import UI, UIButton, UIBox
from src.UI.UI import SCREEN_HEIGHT

TITLE_FONT_SIZE = int(UI.SCREEN_HEIGHT * 0.08)
HIGHLIGHT_COLOR = (50, 50, 0)
FLEET_ICON_SIZE = (int(UI.SCREEN_WIDTH * 0.048), int(UI.SCREEN_WIDTH * 0.048))

def load_fleet_data():
    try:
        with open('Config/Fleets.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading Fleets.json: {e}")
        return None

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
        for ship_name in ships_data:
            sprite, _ = load_ship_sprite(ship_name)
            if sprite:
                original_sprites[ship_name] = sprite

        return ships_data, original_sprites
    except Exception as e:
        print(f"Error loading Ships.json: {e}")
        return None, None

def run(screen):
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(UI.SCREEN_HEIGHT * 0.03))
    background = UI.load_background("UI/Menu.png", UI.SCREEN_WIDTH, UI.SCREEN_HEIGHT)

    fleet_data = load_fleet_data()
    ships_data, original_sprites = load_ships_data()
    if not fleet_data or not ships_data or not original_sprites:
        return

    LEFT_COLUMN_START =  int(0.033*UI.SCREEN_WIDTH)
    RIGHT_COLUMN_START = int(UI.SCREEN_WIDTH//2+(0.016*UI.SCREEN_WIDTH))

    FLEET_TOP = int(0.15 * UI.SCREEN_HEIGHT)

    FLEET_WIDTH = int(0.45*UI.SCREEN_WIDTH)
    FLEET_HEIGHT = int(0.4*UI.SCREEN_HEIGHT)

    SELECTION_BOX_SIZE = int(UI.SCREEN_WIDTH * 0.165)
    SELECTION_TOP = FLEET_TOP+FLEET_HEIGHT+int(0.05*UI.SCREEN_HEIGHT)
    SELECTION_L_LEFT = LEFT_COLUMN_START+int(0.5*(FLEET_WIDTH-SELECTION_BOX_SIZE))
    SELECTION_R_LEFT = RIGHT_COLUMN_START+int(0.5*(FLEET_WIDTH-SELECTION_BOX_SIZE))
    RAND_TOP = SELECTION_TOP+SELECTION_BOX_SIZE+int(0.01*UI.SCREEN_HEIGHT)

    fleet_sprites = scale_sprites(original_sprites, int(UI.SCREEN_WIDTH * 0.048))
    selection_sprites = scale_sprites(original_sprites, SELECTION_BOX_SIZE)

    left_fleet = UIBox.Fleet(
        LEFT_COLUMN_START,
        FLEET_TOP,
        FLEET_WIDTH,
        FLEET_HEIGHT,
        "Player 1 Fleet",
        (int(UI.SCREEN_WIDTH * 0.048), int(UI.SCREEN_WIDTH * 0.048))
    )

    right_fleet = UIBox.Fleet(
        RIGHT_COLUMN_START,
        FLEET_TOP,
        FLEET_WIDTH,
        FLEET_HEIGHT,
        "Player 2 Fleet",
        (int(UI.SCREEN_WIDTH * 0.048), int(UI.SCREEN_WIDTH * 0.048))
    )

    # Load ships into fleets
    for ship_name in fleet_data["Player1"]["ships"]:
        ship_info = ships_data[ship_name]
        ship_type = list(ship_info.keys())[0]
        ship_cost = ship_info[ship_type]
        left_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

    for ship_name in fleet_data["Player2"]["ships"]:
        ship_info = ships_data[ship_name]
        ship_type = list(ship_info.keys())[0]
        ship_cost = ship_info[ship_type]
        right_fleet.add_ship(fleet_sprites[ship_name], ship_name, ship_cost)

    left_selection = {"rect": pygame.Rect(SELECTION_L_LEFT,
                                        SELECTION_TOP,
                                        SELECTION_BOX_SIZE,
                                        SELECTION_BOX_SIZE),
                     "ship": None,
                     "sprite": None}

    right_selection = {"rect": pygame.Rect(SELECTION_R_LEFT,
                                         SELECTION_TOP,
                                         SELECTION_BOX_SIZE,
                                         SELECTION_BOX_SIZE),
                      "ship": None,
                      "sprite": None}

    def pick_random_left():
        if left_fleet.ships:
            selected_idx = random.randrange(len(left_fleet.ships))
            sprite, name, cost, _ = left_fleet.ships[selected_idx]
            left_selection["ship"] = name
            left_selection["sprite"] = selection_sprites[name]

    def pick_random_right():
        if right_fleet.ships:
            selected_idx = random.randrange(len(right_fleet.ships))
            sprite, name, cost, _ = right_fleet.ships[selected_idx]
            right_selection["ship"] = name
            right_selection["sprite"] = selection_sprites[name]

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
        if left_selection["ship"] and right_selection["ship"]:
            print("Ships selected:", left_selection["ship"], "vs", right_selection["ship"])
            # TODO: Call Battle module with selected ships

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
                    for _, name, _, rect in left_fleet.ships:
                        if rect and rect.collidepoint(mouse_pos):
                            left_selection["ship"] = name
                            left_selection["sprite"] = selection_sprites[name]
                            UI.sound_manager.play_sound('menu')
                            break

                elif right_fleet.rect.collidepoint(mouse_pos):
                    for _, name, _, rect in right_fleet.ships:
                        if rect and rect.collidepoint(mouse_pos):
                            right_selection["ship"] = name
                            right_selection["sprite"] = selection_sprites[name]
                            UI.sound_manager.play_sound('menu')
                            break

            random_left.handle_event(event, UI.sound_manager)
            random_right.handle_event(event, UI.sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button.rect.collidepoint(event.pos):
                    UI.sound_manager.play_sound('menu')
                    running = False
                elif confirm_button.rect.collidepoint(event.pos) and left_selection["ship"] and right_selection["ship"]:
                    UI.sound_manager.play_sound('menu')
                    confirm_callback()

        if left_selection["ship"] and right_selection["ship"]:
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

        # Draw highlight boxes under selected ships
        if left_selection["ship"]:
            for _, name, _, rect in left_fleet.ships:
                if name == left_selection["ship"]:
                    highlight_rect = pygame.Rect(rect.centerx - FLEET_ICON_SIZE[0]//2,
                                              rect.centery - FLEET_ICON_SIZE[1]//2,
                                              FLEET_ICON_SIZE[0],
                                              FLEET_ICON_SIZE[1])
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)
                    break

        if right_selection["ship"]:
            for _, name, _, rect in right_fleet.ships:
                if name == right_selection["ship"]:
                    highlight_rect = pygame.Rect(rect.centerx - FLEET_ICON_SIZE[0]//2,
                                              rect.centery - FLEET_ICON_SIZE[1]//2,
                                              FLEET_ICON_SIZE[0],
                                              FLEET_ICON_SIZE[1])
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)
                    break

        # Redraw ships to appear above highlights
        for sprite, _, _, rect in left_fleet.ships:
            screen.blit(sprite, rect)
        for sprite, _, _, rect in right_fleet.ships:
            screen.blit(sprite, rect)

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