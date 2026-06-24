import pygame
import sys
import random
import math

from src.UI import ui, ui_button, ui_box
from src.UI.ship_sprites import (
    load_menu_ship_sprites,
    populate_fleet_panel,
    scale_ship_sprites,
)
import src.const as Const
from src.configuration import FleetsRepository
from src.menu_state import ShipSelectionState

from src.Battle import battle
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ship, preload_ship_ability_resources


TITLE_FONT_SIZE = int(Const.SCREEN_HEIGHT * 0.08)
HIGHLIGHT_COLOR = (50, 50, 75)
ACTIVE_SELECTION_COLOR = (255, 196, 64)
SELECTED_COLOR = (0, 175, 175)
LOCKED_COLOR = (120, 120, 120)
BANNER_ALPHA = 175
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


def load_fleet_data(audio_service=None):
    fleets = FleetsRepository(Const.FLEETS_JSON_PATH, SHIP_DEFINITIONS).load()
    ship_names = set(fleets.player1.ships + fleets.player2.ships)
    for ship_name in ship_names:
        preload_ship_ability_resources(ship_name)

    player1_ships = [
        create_ship(ship_name, 1, audio_service=audio_service)
        for ship_name in fleets.player1.ships
    ]
    player2_ships = [
        create_ship(ship_name, 2, audio_service=audio_service)
        for ship_name in fleets.player2.ships
    ]
    return fleets.to_json_dict(), player1_ships, player2_ships


def load_ship_sprite(ship_name, resources=None):
    sprite = load_menu_ship_sprites([ship_name], resources=resources).get(ship_name)
    return (sprite, sprite.get_size()) if sprite is not None else (None, None)


scale_sprites = scale_ship_sprites


def load_ships_data(ships_data):
    return ships_data, load_menu_ship_sprites(ships_data)


def run(screen, player1_ships=None, player2_ships=None, start_battle=True,
        preselect_player1=None, preselect_player2=None, choose_second_player=None,
        audio_service=None, menu_sound_manager=None):
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.03))
    state_font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.025))
    background = ui.load_background(Const.MENU_BG_PATH, Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT)

    if player1_ships is None or player2_ships is None:
        fleet_data, player1_ships, player2_ships = load_fleet_data(audio_service)
    else:
        fleet_data = {
            "Player1": {"ships": [ship.name for ship in player1_ships]},
            "Player2": {"ships": [ship.name for ship in player2_ships]},
        }
    ships_data, original_sprites = load_ships_data(SHIP_DEFINITIONS)
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

    columns = {1: LEFT_COLUMN_START, 2: RIGHT_COLUMN_START}
    panels = ui_box.create_player_fleet_panels(
        columns, FLEET_TOP, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT,
        FLEET_ICON_SIZE,
    )
    fleet_sprites = scale_ship_sprites(
        original_sprites, fleet_size, SHIP_DEFINITIONS
    )
    selection_sprites = scale_ship_sprites(
        original_sprites, SELECTION_BOX_SIZE, SHIP_DEFINITIONS
    )
    player_ships = {1: player1_ships, 2: player2_ships}
    fleet_names = {
        player: tuple(
            name
            for _, name in zip(
                player_ships[player], fleet_data[f"Player{player}"]["ships"]
            )
        )
        for player in (1, 2)
    }
    for player in (1, 2):
        populate_fleet_panel(
            panels[player], fleet_names[player], fleet_sprites, ships_data
        )

    selection_rects = {
        1: pygame.Rect(
            SELECTION_L_LEFT, SELECTION_TOP,
            SELECTION_BOX_SIZE, SELECTION_BOX_SIZE,
        ),
        2: pygame.Rect(
            SELECTION_R_LEFT, SELECTION_TOP,
            SELECTION_BOX_SIZE, SELECTION_BOX_SIZE,
        ),
    }
    selection_state = ShipSelectionState(
        player_ships,
        fleet_names,
        preselected={1: preselect_player1, 2: preselect_player2},
        choose_second_player=choose_second_player,
    )

    def draw_panel_badge(panel, text, color):
        text_surface = state_font.render(text, True, ui.WHITE)
        badge_rect = text_surface.get_rect(
            top=panel.rect.top + 8,
            right=panel.rect.right - 8,
        ).inflate(16, 8)
        badge_surface = pygame.Surface(badge_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            badge_surface,
            (*color, BANNER_ALPHA),
            badge_surface.get_rect(),
            border_radius=4,
        )
        screen.blit(badge_surface, badge_rect)
        screen.blit(text_surface, text_surface.get_rect(center=badge_rect.center))

    def draw_locked_panel(panel):
        overlay = pygame.Surface(panel.rect.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 145))
        screen.blit(overlay, panel.rect)
        pygame.draw.rect(screen, LOCKED_COLOR, panel.rect, 4)
        draw_panel_badge(panel, "LOCKED", LOCKED_COLOR)

    def draw_selection_label(selection_rect, text, color, show_lock=False):
        label_surface = state_font.render(text, True, ui.WHITE)
        label_height = label_surface.get_height() + 12
        label_rect = pygame.Rect(
            selection_rect.left + 3,
            selection_rect.bottom - label_height - 3,
            selection_rect.width - 6,
            label_height,
        )
        label_surface_bg = pygame.Surface(label_rect.size, pygame.SRCALPHA)
        label_surface_bg.fill((*color, BANNER_ALPHA))
        screen.blit(label_surface_bg, label_rect)
        text_rect = label_surface.get_rect(center=label_rect.center)
        if show_lock:
            text_rect.x += 7
            lock_center_x = text_rect.left - 14
            pygame.draw.arc(
                screen,
                ui.WHITE,
                pygame.Rect(lock_center_x - 5, label_rect.centery - 7, 10, 12),
                0,
                math.pi,
                2,
            )
            pygame.draw.rect(
                screen,
                ui.WHITE,
                pygame.Rect(lock_center_x - 7, label_rect.centery - 1, 14, 12),
                border_radius=2,
            )
        screen.blit(label_surface, text_rect)

    def pick_random(player):
        if not selection_state.selection_allowed(player):
            return
        alive_indices = selection_state.alive_indices(player)
        if alive_indices:
            selection_state.select_index(player, random.choice(alive_indices))

    random_buttons = {
        player: ui_button.Button(
            selection_rects[player].left - 1,
            RAND_TOP,
            SELECTION_BOX_SIZE + 2,
            int(0.05 * Const.SCREEN_HEIGHT),
            "Pick Random",
            lambda player=player: pick_random(player),
            bg_color=(*Const.P1_COLOR, 75) if player == 1 else (*Const.P2_COLOR, 75),
            hover_color=(*Const.P1_COLOR, 255) if player == 1 else (*Const.P2_COLOR, 255),
        )
        for player in (1, 2)
    }

    def confirm_callback():
        selected = selection_state.selected_ships()
        if selected is not None:
            left_selection = selection_state.selection(1)
            right_selection = selection_state.selection(2)
            print("Ships selected:", left_selection.name, "vs", right_selection.name)
            if start_battle:
                battle.run(
                    screen,
                    selected[0],
                    selected[1],
                    player1_ships,
                    player2_ships,
                    audio_service=audio_service,
                    menu_sound_manager=menu_sound_manager,
                )
                return None, None

            return selected

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

        for player, button in random_buttons.items():
            button.enabled = selection_state.selection_allowed(player)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos

                for player in (1, 2):
                    panel = panels[player]
                    if (panel.rect.collidepoint(mouse_pos)
                            and selection_state.selection_allowed(player)):
                        for index, (_, _, _, rect) in enumerate(panel.ships):
                            if rect and rect.collidepoint(mouse_pos):
                                if selection_state.select_index(player, index):
                                    if menu_sound_manager:
                                        menu_sound_manager.play_sound('menu')
                                break
                        break

            for button in random_buttons.values():
                button.handle_event(event, menu_sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button.rect.collidepoint(event.pos):
                    if menu_sound_manager:
                        menu_sound_manager.play_sound('menu')
                    running = False
                elif (confirm_button.rect.collidepoint(event.pos)
                      and selection_state.confirmation_ready):
                    if menu_sound_manager:
                        menu_sound_manager.play_sound('menu')
                    return confirm_callback()

        if selection_state.confirmation_ready:
            confirm_button.bg_color = ui.OK_GREEN
            confirm_button.hover_color = ui.OK_GREEN_HI
        else:
            confirm_button.bg_color = ui.DISABLED_BUTTON
            confirm_button.hover_color = ui.DISABLED_BUTTON

        # A first-player click can lock that side during this event loop.
        for player, button in random_buttons.items():
            button.enabled = selection_state.selection_allowed(player)

        # Draw everything
        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)

        active_player = selection_state.active_player
        both_selected = selection_state.both_selected
        survivor_locked_players = selection_state.survivor_locked_players
        title_size = TITLE_FONT_SIZE
        if choose_second_player in (1, 2):
            if not selection_state.first_locked:
                title = f"Player {active_player}: Select First"
            elif both_selected:
                title = "Both Players Ready"
            else:
                title = (
                    f"Player {selection_state.first_player} Locked - "
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

        for panel in panels.values():
            panel.draw(screen, font)

        # Draw highlight boxes under selected ships and X's over dead ships

        for player, panel in panels.items():
            selection = selection_state.selection(player)
            if selection is not None:
                rect = panel.ships[selection.index][3]
                highlight_rect = pygame.Rect(
                    rect.centerx - FLEET_ICON_SIZE[0] // 2,
                    rect.centery - FLEET_ICON_SIZE[1] // 2,
                    FLEET_ICON_SIZE[0], FLEET_ICON_SIZE[1],
                )
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, highlight_rect)

        # Redraw ships to appear above highlights
        for player, panel in panels.items():
            for index, (sprite, _, _, rect) in enumerate(panel.ships):
                screen.blit(sprite, rect)
                if not player_ships[player][index].currently_alive:
                    draw_x(screen, rect)

        for player in survivor_locked_players:
            draw_locked_panel(panels[player])
        if choose_second_player in (1, 2):
            if selection_state.first_locked:
                draw_locked_panel(panels[selection_state.first_player])
            else:
                draw_locked_panel(panels[choose_second_player])
            active_panel = panels[active_player]
            active_color = Const.P1_COLOR if active_player == 1 else Const.P2_COLOR
            pygame.draw.rect(screen, active_color, active_panel.rect, 4)
            active_badge = (
                "SELECT FIRST" if not selection_state.first_locked
                else "SELECT SECOND"
            )
            draw_panel_badge(active_panel, active_badge, active_color)
        elif survivor_locked_players:
            for player in (1, 2):
                if player in survivor_locked_players:
                    continue
                player_color = Const.P1_COLOR if player == 1 else Const.P2_COLOR
                pygame.draw.rect(screen, player_color, panels[player].rect, 4)
                draw_panel_badge(panels[player], "SELECT SHIP", player_color)

        # Draw selection boxes
        for player, selection_rect in selection_rects.items():
            pygame.draw.rect(screen, ui.BLACK, selection_rect)
            selection = selection_state.selection(player)
            if selection is not None:
                sprite = selection_sprites[selection.name]
                sprite_rect = sprite.get_rect(center=selection_rect.center)
                screen.blit(sprite, sprite_rect)
            border_color = Const.P1_COLOR if player == 1 else Const.P2_COLOR
            pygame.draw.rect(screen, border_color, selection_rect, 3)

        visually_locked_players = set(survivor_locked_players)
        if choose_second_player in (1, 2) and selection_state.first_locked:
            visually_locked_players.add(selection_state.first_player)
        for player, selection_rect in selection_rects.items():
            if selection_state.selection(player) is None:
                continue
            if player in visually_locked_players:
                label = "SURVIVOR LOCKED" if player in survivor_locked_players else "SELECTION LOCKED"
                draw_selection_label(
                    selection_rect, label, LOCKED_COLOR, show_lock=True
                )
            else:
                player_color = Const.P1_COLOR if player == 1 else Const.P2_COLOR
                draw_selection_label(selection_rect, "SELECTED", player_color)

        for button in random_buttons.values():
            button.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        pygame.display.flip()

    return None, None
