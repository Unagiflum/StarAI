import pygame
import json
import os
import sys
import random
import math

from src.UI import ui, ui_button, ui_box
import src.const as Const

from src.Battle import battle

with open(Const.SHIPS_JSON_PATH, 'r') as f:
    SHIPS_DATA = json.load(f)

TITLE_FONT_SIZE = int(Const.SCREEN_HEIGHT * 0.08)
HIGHLIGHT_COLOR = (50, 50, 75)
ACTIVE_SELECTION_COLOR = (255, 196, 64)
SELECTED_COLOR = (0, 175, 175)
LOCKED_COLOR = (120, 120, 120)
FLEET_ICON_SIZE = Const.FLEET_ICON_SIZE
X_COLOR = (255, 100, 100, 100)
X_THICKNESS = int(0.2*FLEET_ICON_SIZE[0])

def draw_x(surface, rect):
    """Draw a red X in a square box sized to the largest ship dimension."""
    size = FLEET_ICON_SIZE[0]
    x_rect = pygame.Rect(
        rect.centerx - size // 2,
        rect.centery - size // 2,
        size,
        size
    )

    x_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.line(x_surface, X_COLOR, (0, 0), (size, size), X_THICKNESS)
    pygame.draw.line(x_surface, X_COLOR, (0, size), (size, 0), X_THICKNESS)

    surface.blit(x_surface, x_rect)


def load_fleet_data():
    try:
        with open(Const.FLEETS_JSON_PATH, 'r') as f:
            fleet_data = json.load(f)

        with open(Const.ABILITIES_JSON_PATH, 'r') as f:
            projectiles_data = json.load(f)


        def get_ship_class(ship_name, player_num):
            # Import and instantiate ship
            ship_module = __import__(f"src.Objects.Ships.{ship_name}.{ship_name}", fromlist=[''])
            ship = getattr(ship_module, ship_name)(ship_name, player_num)

            # Import and instantiate associated projectiles to load resources
            for proj_name, proj_data in projectiles_data.items():
                if proj_data["ship_name"] == ship_name:
                    proj_module = __import__(f"src.Objects.Ships.{ship_name}.{proj_data['action']}.{proj_name}",
                                             fromlist=[''])
                    getattr(proj_module, proj_name)(ship)
            return ship


        player1_ships = [get_ship_class(ship_name, 1) for ship_name in fleet_data["Player1"]["ships"]]
        player2_ships = [get_ship_class(ship_name, 2) for ship_name in fleet_data["Player2"]["ships"]]
        return fleet_data, player1_ships, player2_ships

    except Exception as e:
        print(f"Error loading Fleets.json: {e}")
        return None, [], []


def load_ship_sprite(ship_name):
    try:
        sprite_path = os.path.join(SHIPS_DATA[ship_name]['sprite_path'], f'{ship_name}00.png')
        sprite = pygame.image.load(sprite_path).convert_alpha()
        return sprite, sprite.get_size()
    except Exception as e:
        print(f"Error loading sprite for {ship_name}: {e}")
        return None, None


def scale_sprites(original_sprites, target_size, ships_data):
    # First find max dimension after applying sprite_scale
    max_dim = 1
    for name, sprite in original_sprites.items():
        sprite_scale = ships_data[name]['sprite_scale']
        width, height = sprite.get_size()
        scaled_width = width * sprite_scale
        scaled_height = height * sprite_scale
        max_dim = max(max_dim, scaled_width, scaled_height)

    base_scale_factor = target_size / max_dim
    scaled_sprites = {}

    # Now scale each sprite by both its sprite_scale and the base_scale_factor
    for name, sprite in original_sprites.items():
        sprite_scale = ships_data[name]['sprite_scale']
        width, height = sprite.get_size()
        new_width = int(width * sprite_scale * base_scale_factor)
        new_height = int(height * sprite_scale * base_scale_factor)
        scaled_sprites[name] = pygame.transform.scale(sprite, (new_width, new_height))

    return scaled_sprites


def load_ships_data(ships_data):
   try:
       simplified_data = {}
       original_sprites = {}
       for ship_name, stats in ships_data.items():
           simplified_data[ship_name] = {stats['ship_type']: stats['cost']}
           sprite, _ = load_ship_sprite(ship_name)
           if sprite:
               original_sprites[ship_name] = sprite

       return simplified_data, original_sprites
   except Exception as e:
       print(f"Error loading ships data: {e}")
       return None, None


def run(screen, player1_ships=None, player2_ships=None, start_battle=True,
        preselect_player1=None, preselect_player2=None, choose_second_player=None):
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.03))
    state_font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.025))
    background = ui.load_background(Const.MENU_BG_PATH, Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT)

    if player1_ships is None or player2_ships is None:
        fleet_data, player1_ships, player2_ships = load_fleet_data()
    else:
        fleet_data = {
            "Player1": {"ships": [ship.name for ship in player1_ships]},
            "Player2": {"ships": [ship.name for ship in player2_ships]},
        }
    ships_data, original_sprites = load_ships_data(SHIPS_DATA)
    if not fleet_data or not ships_data or not original_sprites:
        return

    # Create temporary fleet to get icon size
    temp_fleet = ui_box.Fleet(0, 0, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT, "", (0, 0))
    fleet_size = temp_fleet.icon_size[0]

    # Update globals that use fleet size
    global X_THICKNESS
    X_THICKNESS = int(0.2 * fleet_size)


    LEFT_COLUMN_START = int(0.033 * Const.SCREEN_WIDTH)
    RIGHT_COLUMN_START = int(Const.SCREEN_WIDTH // 2 + (0.016 * Const.SCREEN_WIDTH))

    FLEET_TOP = int(0.15 * Const.SCREEN_HEIGHT)

    SELECTION_BOX_SIZE = int(Const.SCREEN_WIDTH * 0.165)
    SELECTION_TOP = FLEET_TOP + ui.FLEET_HEIGHT + int(0.025 * Const.SCREEN_HEIGHT)
    SELECTION_L_LEFT = LEFT_COLUMN_START + ui.SELECTION_WIDTH - SELECTION_BOX_SIZE
    SELECTION_R_LEFT = RIGHT_COLUMN_START
    RAND_TOP = SELECTION_TOP + SELECTION_BOX_SIZE + int(0.01 * Const.SCREEN_HEIGHT)

    left_fleet = ui_box.Fleet(
        LEFT_COLUMN_START,
        FLEET_TOP,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        "Player 1 Fleet",
        FLEET_ICON_SIZE
    )

    right_fleet = ui_box.Fleet(
        RIGHT_COLUMN_START,
        FLEET_TOP,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        "Player 2 Fleet",
        FLEET_ICON_SIZE
    )

    fleet_sprites = scale_sprites(original_sprites, fleet_size, SHIPS_DATA)
    selection_sprites = scale_sprites(original_sprites, SELECTION_BOX_SIZE, SHIPS_DATA)

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
                      "ship_obj": None,
                      "index": None}

    right_selection = {"rect": pygame.Rect(SELECTION_R_LEFT,
                                           SELECTION_TOP,
                                           SELECTION_BOX_SIZE,
                                           SELECTION_BOX_SIZE),
                       "ship": None,
                       "sprite": None,
                       "ship_obj": None,
                       "index": None}

    selection_order = {
        "active_player": 3 - choose_second_player
        if choose_second_player in (1, 2) else None,
        "first_player": 3 - choose_second_player
        if choose_second_player in (1, 2) else None,
        "first_locked": False,
    }
    survivor_locked_players = {
        player
        for player, ship in ((1, preselect_player1), (2, preselect_player2))
        if ship is not None and ship.currently_alive and ship.current_hp > 0
    }

    def selection_allowed(player):
        if player in survivor_locked_players:
            return False
        active_player = selection_order["active_player"]
        return active_player is None or active_player == player

    def advance_selection_order(player):
        if selection_order["active_player"] == player and choose_second_player in (1, 2):
            if player == selection_order["first_player"]:
                selection_order["first_locked"] = True
            selection_order["active_player"] = choose_second_player

    def draw_panel_badge(panel, text, color):
        text_surface = state_font.render(text, True, ui.WHITE)
        badge_rect = text_surface.get_rect(
            top=panel.rect.top + 8,
            right=panel.rect.right - 8,
        ).inflate(16, 8)
        pygame.draw.rect(screen, color, badge_rect, border_radius=4)
        screen.blit(text_surface, text_surface.get_rect(center=badge_rect.center))

    def draw_locked_panel(panel):
        overlay = pygame.Surface(panel.rect.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 145))
        screen.blit(overlay, panel.rect)
        pygame.draw.rect(screen, LOCKED_COLOR, panel.rect, 4)
        draw_panel_badge(panel, "LOCKED", LOCKED_COLOR)

    def draw_selection_label(selection, text, color, show_lock=False):
        label_surface = state_font.render(text, True, ui.WHITE)
        label_rect = pygame.Rect(
            selection["rect"].left,
            selection["rect"].bottom - label_surface.get_height() - 12,
            selection["rect"].width,
            label_surface.get_height() + 12,
        )
        pygame.draw.rect(screen, color, label_rect)
        text_rect = label_surface.get_rect(center=label_rect.center)
        if show_lock:
            text_rect.x += 7
            lock_center_x = text_rect.left - 14
            pygame.draw.arc(
                screen,
                ui.WHITE,
                pygame.Rect(lock_center_x - 4, label_rect.centery - 8, 8, 10),
                math.pi,
                2 * math.pi,
                2,
            )
            pygame.draw.rect(
                screen,
                ui.WHITE,
                pygame.Rect(lock_center_x - 5, label_rect.centery - 1, 10, 8),
                border_radius=2,
            )
        screen.blit(label_surface, text_rect)

    def set_selection(selection, selection_sprites, fleet, ships, ship_obj):
        if not ship_obj or not ship_obj.currently_alive:
            return

        for idx, candidate in enumerate(ships):
            if candidate == ship_obj:
                _, name, _, _ = fleet.ships[idx]
                selection["ship"] = name
                selection["sprite"] = selection_sprites[name]
                selection["ship_obj"] = candidate
                selection["index"] = idx
                return

    set_selection(left_selection, selection_sprites, left_fleet, player1_ships, preselect_player1)
    set_selection(right_selection, selection_sprites, right_fleet, player2_ships, preselect_player2)

    def pick_random_left():
        if not selection_allowed(1):
            return
        alive_ships = [(i, ship) for i, ship in enumerate(player1_ships) if ship.currently_alive]
        if alive_ships:
            idx, ship_obj = random.choice(alive_ships)
            sprite, name, cost, _ = left_fleet.ships[idx]
            left_selection["ship"] = name
            left_selection["sprite"] = selection_sprites[name]
            left_selection["ship_obj"] = ship_obj
            left_selection["index"] = idx  # Add this line
            advance_selection_order(1)

    def pick_random_right():
        if not selection_allowed(2):
            return
        alive_ships = [(i, ship) for i, ship in enumerate(player2_ships) if ship.currently_alive]
        if alive_ships:
            idx, ship_obj = random.choice(alive_ships)
            sprite, name, cost, _ = right_fleet.ships[idx]
            right_selection["ship"] = name
            right_selection["sprite"] = selection_sprites[name]
            right_selection["ship_obj"] = ship_obj
            right_selection["index"] = idx  # Add this line
            advance_selection_order(2)

    random_left = ui_button.Button(
        SELECTION_L_LEFT,
        RAND_TOP,
        SELECTION_BOX_SIZE,
        int(0.05 * Const.SCREEN_HEIGHT),
        "Pick Random",
        pick_random_left,
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI
    )

    random_right = ui_button.Button(
        SELECTION_R_LEFT,
        RAND_TOP,
        SELECTION_BOX_SIZE,
        int(0.05 * Const.SCREEN_HEIGHT),
        "Pick Random",
        pick_random_right,
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI
    )

    def confirm_callback():
        if (left_selection["ship_obj"] and right_selection["ship_obj"] and
                left_selection["ship_obj"].currently_alive and right_selection["ship_obj"].currently_alive):
            print("Ships selected:", left_selection["ship"], "vs", right_selection["ship"])
            if start_battle:
                battle.run(
                    screen,
                    left_selection["ship_obj"],
                    right_selection["ship_obj"],
                    player1_ships,
                    player2_ships,
                )
                return None, None

            return left_selection["ship_obj"], right_selection["ship_obj"]

    confirm_button = ui_button.Button(
        ui.ok_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Confirm",
        confirm_callback,
        bg_color=ui.DISABLED_BUTTON,
        hover_color=ui.DISABLED_BUTTON
    )

    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Cancel",
        lambda: None,
        bg_color=ui.CAN_RED,
        hover_color=ui.CAN_RED_HI
    )

    running = True
    while running:
        clock.tick(Const.FPS)

        random_left.enabled = selection_allowed(1)
        random_right.enabled = selection_allowed(2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                if left_fleet.rect.collidepoint(mouse_pos) and selection_allowed(1):
                    for i, (_, name, _, rect) in enumerate(left_fleet.ships):
                        if rect and rect.collidepoint(mouse_pos) and player1_ships[i].currently_alive:
                            left_selection["ship"] = name
                            left_selection["sprite"] = selection_sprites[name]
                            left_selection["ship_obj"] = player1_ships[i]
                            left_selection["index"] = i
                            ui.sound_manager.play_sound('menu')
                            advance_selection_order(1)
                            break

                elif right_fleet.rect.collidepoint(mouse_pos) and selection_allowed(2):
                    for i, (_, name, _, rect) in enumerate(right_fleet.ships):
                        if rect and rect.collidepoint(mouse_pos) and player2_ships[i].currently_alive:
                            right_selection["ship"] = name
                            right_selection["sprite"] = selection_sprites[name]
                            right_selection["ship_obj"] = player2_ships[i]
                            right_selection["index"] = i
                            ui.sound_manager.play_sound('menu')
                            advance_selection_order(2)
                            break

            random_left.handle_event(event, ui.sound_manager)
            random_right.handle_event(event, ui.sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button.rect.collidepoint(event.pos):
                    ui.sound_manager.play_sound('menu')
                    running = False
                elif (confirm_button.rect.collidepoint(event.pos) and
                      left_selection["ship_obj"] and right_selection["ship_obj"] and
                      left_selection["ship_obj"].currently_alive and
                      right_selection["ship_obj"].currently_alive):
                    ui.sound_manager.play_sound('menu')
                    return confirm_callback()

        if (left_selection["ship_obj"] and right_selection["ship_obj"] and
                left_selection["ship_obj"].currently_alive and
                right_selection["ship_obj"].currently_alive):
            confirm_button.bg_color = ui.OK_GREEN
            confirm_button.hover_color = ui.OK_GREEN_HI
        else:
            confirm_button.bg_color = ui.DISABLED_BUTTON
            confirm_button.hover_color = ui.DISABLED_BUTTON

        # A first-player click can lock that side during this event loop.
        random_left.enabled = selection_allowed(1)
        random_right.enabled = selection_allowed(2)

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)

        active_player = selection_order["active_player"]
        both_selected = bool(left_selection["ship_obj"] and right_selection["ship_obj"])
        title_size = TITLE_FONT_SIZE
        if choose_second_player in (1, 2):
            if not selection_order["first_locked"]:
                title = f"Player {active_player}: Select First"
            elif both_selected:
                title = "Both Players Ready"
            else:
                title = (
                    f"Player {selection_order['first_player']} Locked - "
                    f"Player {choose_second_player}: Select Second"
                )
                title_size = int(Const.SCREEN_HEIGHT * 0.055)
        elif survivor_locked_players:
            selecting_players = [player for player in (1, 2) if player not in survivor_locked_players]
            if both_selected:
                title = "Both Players Ready"
            elif selecting_players:
                survivor = next(iter(survivor_locked_players))
                title = f"Player {survivor} Survives - Player {selecting_players[0]}: Select Ship"
                title_size = int(Const.SCREEN_HEIGHT * 0.055)
            else:
                title = "Both Players Ready"
        else:
            title = "Players: Select Your Ships"
        ui.draw_title(screen, title, title_size, int(0.05 * Const.SCREEN_HEIGHT))

        # Draw fleets
        left_fleet.draw(screen, font)
        right_fleet.draw(screen, font)

        # Draw highlight boxes under selected ships and X's over dead ships

        if left_selection["ship"]:
            rect = left_fleet.ships[left_selection["index"]][3]  # Get rect from stored index
            highlight_rect = pygame.Rect(rect.centerx - FLEET_ICON_SIZE[0] // 2,
                                         rect.centery - FLEET_ICON_SIZE[1] // 2,
                                         FLEET_ICON_SIZE[0],
                                         FLEET_ICON_SIZE[1])
            pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)

        if right_selection["ship"]:
            rect = right_fleet.ships[right_selection["index"]][3]  # Get rect from stored index
            highlight_rect = pygame.Rect(rect.centerx - FLEET_ICON_SIZE[0] // 2,
                                         rect.centery - FLEET_ICON_SIZE[1] // 2,
                                         FLEET_ICON_SIZE[0],
                                         FLEET_ICON_SIZE[1])
            pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)

        # Redraw ships to appear above highlights
        for i, (sprite, _, _, rect) in enumerate(left_fleet.ships):
            screen.blit(sprite, rect)
            if not player1_ships[i].currently_alive:
                draw_x(screen, rect)

        for i, (sprite, _, _, rect) in enumerate(right_fleet.ships):
            screen.blit(sprite, rect)
            if not player2_ships[i].currently_alive:
                draw_x(screen, rect)

        panels = {1: left_fleet, 2: right_fleet}
        for player in survivor_locked_players:
            draw_locked_panel(panels[player])
        if choose_second_player in (1, 2):
            if selection_order["first_locked"]:
                draw_locked_panel(panels[selection_order["first_player"]])
            active_panel = panels[active_player]
            pygame.draw.rect(screen, ACTIVE_SELECTION_COLOR, active_panel.rect, 4)
            active_badge = "SELECT FIRST" if not selection_order["first_locked"] else "SELECT SECOND"
            draw_panel_badge(active_panel, active_badge, ACTIVE_SELECTION_COLOR)
        elif survivor_locked_players:
            for player in (1, 2):
                if player in survivor_locked_players:
                    continue
                pygame.draw.rect(screen, ACTIVE_SELECTION_COLOR, panels[player].rect, 4)
                draw_panel_badge(panels[player], "SELECT SHIP", ACTIVE_SELECTION_COLOR)

        # Draw selection boxes
        for selection in [left_selection, right_selection]:
            pygame.draw.rect(screen, ui.BLACK, selection["rect"])
            if selection["sprite"]:
                sprite_rect = selection["sprite"].get_rect(center=selection["rect"].center)
                screen.blit(selection["sprite"], sprite_rect)
            pygame.draw.rect(screen, ui.WHITE, selection["rect"], 2)

        selections = {1: left_selection, 2: right_selection}
        visually_locked_players = set(survivor_locked_players)
        if choose_second_player in (1, 2) and selection_order["first_locked"]:
            visually_locked_players.add(selection_order["first_player"])
        for player, selection in selections.items():
            if not selection["ship_obj"]:
                continue
            if player in visually_locked_players:
                label = "SURVIVOR LOCKED" if player in survivor_locked_players else "SELECTION LOCKED"
                draw_selection_label(selection, label, LOCKED_COLOR, show_lock=True)
            else:
                draw_selection_label(selection, "SELECTED", SELECTED_COLOR)

        random_left.draw(screen, font)
        random_right.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()

    return None, None
