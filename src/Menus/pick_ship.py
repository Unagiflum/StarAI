import pygame
import sys
import random

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
from src.frame_timing import PresentationClock
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ship, preload_ship_ability_resources

TITLE_FONT_SIZE = int(Const.SCREEN_HEIGHT * 0.08)
LOCKED_COLOR = (120, 120, 120)
BANNER_ALPHA = 175
FLEET_ICON_SIZE = Const.FLEET_ICON_SIZE
X_COLOR = (255, 100, 100, 100)
X_THICKNESS = int(0.2 * FLEET_ICON_SIZE[0])


def show_battle_countdown(screen, steps=None, step_time=None):
    """Show a responsive wall-clock countdown before battle resumes."""
    steps = Const.COUNT_DOWN_STEPS if steps is None else steps
    step_time = Const.COUNT_DOWN_TIME if step_time is None else step_time
    if steps < 0 or step_time < 0:
        raise ValueError("Countdown steps and time must be non-negative")

    pygame.mouse.set_pos((0, 0))
    font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.35))
    clock = PresentationClock(Const.FPS, Const.VIDEO_FPS_MULTIPLIER)

    for number in range(steps, 0, -1):
        screen.fill(ui.BLACK)
        number_surface = font.render(str(number), True, ui.WHITE)
        number_rect = number_surface.get_rect(center=screen.get_rect().center)
        screen.blit(number_surface, number_rect)
        pygame.display.flip()
        clock.reset()

        elapsed_seconds = 0.0
        while elapsed_seconds < step_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            elapsed_seconds += clock.tick()


def draw_x(surface, rect):
    """Draw a red X in a square box sized to the largest ship dimension."""
    size = FLEET_ICON_SIZE[0]
    x_rect = pygame.Rect(rect.centerx - size // 2, rect.centery - size // 2, size, size)

    x_surface = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.line(x_surface, X_COLOR, (0, 0), (size, size), X_THICKNESS)
    pygame.draw.line(x_surface, X_COLOR, (0, size), (size, 0), X_THICKNESS)

    surface.blit(x_surface, x_rect)


def load_fleet_data(audio_service=None):
    fleets = FleetsRepository(Const.FLEETS_JSON_PATH, SHIP_DEFINITIONS).load()
    ship_names = {
        name
        for name in fleets.player1.ships + fleets.player2.ships
        if name is not None
    }
    for ship_name in ship_names:
        preload_ship_ability_resources(ship_name)

    def create_player_ships(player, slots):
        ships = []
        for slot_index, ship_name in enumerate(slots):
            if ship_name is None:
                continue
            ship = create_ship(ship_name, player, audio_service=audio_service)
            ship.fleet_slot_index = slot_index
            ships.append(ship)
        return ships

    player1_ships = create_player_ships(1, fleets.player1.ships)
    player2_ships = create_player_ships(2, fleets.player2.ships)
    return fleets.to_json_dict(), player1_ships, player2_ships


def fleet_slot_indices_for_ships(ships):
    """Map compact runtime ship order back to unique fleet positions."""
    capacity = Const.SHIP_COLS * Const.SHIP_ROWS
    occupied = set()
    slot_indices = []
    next_open_slot = 0
    for ship in ships:
        slot_index = getattr(ship, "fleet_slot_index", None)
        if (
            not isinstance(slot_index, int)
            or not 0 <= slot_index < capacity
            or slot_index in occupied
        ):
            while next_open_slot < capacity and next_open_slot in occupied:
                next_open_slot += 1
            if next_open_slot >= capacity:
                break
            slot_index = next_open_slot
        occupied.add(slot_index)
        slot_indices.append(slot_index)
    return tuple(slot_indices)


def fleet_slots_for_ships(ships):
    """Rebuild sparse display slots from compact runtime ship sequences."""
    capacity = Const.SHIP_COLS * Const.SHIP_ROWS
    slots = [None] * capacity
    for ship, slot_index in zip(ships, fleet_slot_indices_for_ships(ships)):
        slots[slot_index] = ship.name
    return tuple(slots)


def load_ship_sprite(ship_name, resources=None):
    sprite = load_menu_ship_sprites([ship_name], resources=resources).get(ship_name)
    return (sprite, sprite.get_size()) if sprite is not None else (None, None)


scale_sprites = scale_ship_sprites


def selection_prompt(selection_state):
    """Return the title and active-panel badge for the current selection step."""
    if selection_state.both_selected:
        return "Both Players Ready", None

    survivor_locked_players = selection_state.survivor_locked_players
    if selection_state.choose_second_player is not None:
        if selection_state.first_player in survivor_locked_players:
            return (
                f"Player {selection_state.first_player} Survives - "
                f"Player {selection_state.active_player}: Select Ship",
                "SELECT SHIP",
            )
        if not selection_state.first_locked:
            return (
                f"Player {selection_state.active_player}: Select First",
                "SELECT FIRST",
            )
        return (
            f"Player {selection_state.first_player} Locked - "
            f"Player {selection_state.choose_second_player}: Select Second",
            "SELECT SECOND",
        )

    if survivor_locked_players:
        selecting_players = [
            player for player in (1, 2) if player not in survivor_locked_players
        ]
        if selecting_players:
            survivor = next(iter(survivor_locked_players))
            return (
                f"Player {survivor} Survives - "
                f"Player {selecting_players[0]}: Select Ship",
                "SELECT SHIP",
            )
        return "Both Players Ready", None

    return "Players: Select Your Ships", None


def load_ships_data(ships_data):
    return ships_data, load_menu_ship_sprites(ships_data)


def run(
    screen,
    player1_ships=None,
    player2_ships=None,
    start_battle=True,
    preselect_player1=None,
    preselect_player2=None,
    choose_second_player=None,
    audio_service=None,
    menu_sound_manager=None,
):
    clock = PresentationClock(Const.FPS, Const.VIDEO_FPS_MULTIPLIER)
    font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.03))
    state_font = pygame.font.SysFont(None, int(Const.SCREEN_HEIGHT * 0.025))
    tooltip_font = pygame.font.SysFont(None, Const.SHIP_TOOLTIP_FONT_SIZE)
    background = ui.load_background(
        Const.MENU_BG_PATH, Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT
    )

    if player1_ships is None or player2_ships is None:
        fleet_data, player1_ships, player2_ships = load_fleet_data(audio_service)
    else:
        fleet_data = {
            "Player1": {"ships": list(fleet_slots_for_ships(player1_ships))},
            "Player2": {"ships": list(fleet_slots_for_ships(player2_ships))},
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

    columns = {1: LEFT_COLUMN_START, 2: RIGHT_COLUMN_START}
    panels = ui_box.create_player_fleet_panels(
        columns,
        FLEET_TOP,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        FLEET_ICON_SIZE,
        fleet_factory=ui_box.ShipSelectionFleet,
    )
    fleet_sprites = scale_ship_sprites(original_sprites, fleet_size, SHIP_DEFINITIONS)
    player_ships = {1: player1_ships, 2: player2_ships}
    fleet_names = {
        player: tuple(ship.name for ship in player_ships[player])
        for player in (1, 2)
    }
    for player in (1, 2):
        populate_fleet_panel(
            panels[player],
            fleet_data[f"Player{player}"]["ships"],
            fleet_sprites,
            ships_data,
        )
    selectable_panel_slots = {
        player: tuple(
            (slot_index, panels[player].ships[slot_index])
            for slot_index in fleet_slot_indices_for_ships(player_ships[player])
        )
        for player in (1, 2)
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

    def draw_random_locked_panel(panel, color):
        overlay = pygame.Surface(panel.rect.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 145))
        screen.blit(overlay, panel.rect)
        pygame.draw.rect(screen, color, panel.rect, 4)
        draw_panel_badge(panel, "RANDOM SELECTED", color)

    def pick_random(player):
        if not selection_state.selection_allowed(player):
            return
        alive_indices = selection_state.alive_indices(player)
        if alive_indices:
            selection_state.select_random_index(
                player, random.choice(alive_indices)
            )

    random_buttons = {
        player: ui_button.Button(
            columns[player],
            int(0.10 * Const.SCREEN_HEIGHT),
            ui.SELECTION_WIDTH,
            int(0.04 * Const.SCREEN_HEIGHT),
            "Pick Random",
            lambda player=player: pick_random(player),
            bg_color=(*Const.P1_COLOR, 75) if player == 1 else (*Const.P2_COLOR, 75),
            hover_color=(
                (*Const.P1_COLOR, 255) if player == 1 else (*Const.P2_COLOR, 255)
            ),
        )
        for player in (1, 2)
    }

    def confirm_callback():
        selected = selection_state.selected_ships()
        if selected is not None:
            left_selection = selection_state.selection(1)
            right_selection = selection_state.selection(2)
            if selection_state.random_locked_players:
                print("Ships selected; random selection remains hidden.")
            else:
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
        "Continue",
        confirm_callback,
        bg_color=ui.DISABLED_BUTTON,
        hover_color=ui.DISABLED_BUTTON,
    )

    cancel_button = ui_button.Button(
        ui.can_button_left,
        ui.ok_button_top,
        ui.ok_button_width,
        ui.ok_button_height,
        "Cancel",
        lambda: None,
        bg_color=ui.CAN_RED,
        hover_color=ui.CAN_RED_HI,
    )

    running = True
    while running:
        clock.tick()

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
                    if panel.rect.collidepoint(
                        mouse_pos
                    ) and selection_state.selection_allowed(player):
                        slot_index = panel.slot_index_at_pos(mouse_pos)
                        selectable_index = next(
                            (
                                index
                                for index, (candidate_slot, _) in enumerate(
                                    selectable_panel_slots[player]
                                )
                                if candidate_slot == slot_index
                            ),
                            None,
                        )
                        if slot_index is None:
                            changed = False
                        elif selectable_index is None:
                            changed = selection_state.deselect(player)
                        else:
                            changed = selection_state.toggle_index(
                                player, selectable_index
                            )
                        if changed and menu_sound_manager:
                            menu_sound_manager.play_sound("menu")
                        break

            for button in random_buttons.values():
                button.handle_event(event, menu_sound_manager)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cancel_button.rect.collidepoint(event.pos):
                    if menu_sound_manager:
                        menu_sound_manager.play_sound("menu")
                    running = False
                elif (
                    confirm_button.rect.collidepoint(event.pos)
                    and selection_state.confirmation_ready
                ):
                    if menu_sound_manager:
                        menu_sound_manager.play_sound("menu")
                    show_battle_countdown(screen)
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
        random_locked_players = selection_state.random_locked_players
        title, active_badge = selection_prompt(selection_state)
        title_size = (
            int(Const.SCREEN_HEIGHT * 0.055)
            if " - " in title
            else TITLE_FONT_SIZE
        )
        ui.draw_title(screen, title, title_size, int(0.05 * Const.SCREEN_HEIGHT))

        mouse_pos = pygame.mouse.get_pos()
        hovered_slots = {}
        for player, panel in panels.items():
            if not selection_state.selection_allowed(player):
                continue
            hovered = panel.occupied_slot_at_pos(mouse_pos)
            if hovered is not None:
                hovered_slots[player] = hovered[0]

        selected_slots = {}
        for player in (1, 2):
            selection = selection_state.selection(player)
            if selection is not None and player not in random_locked_players:
                selected_slots[player] = selectable_panel_slots[player][
                    selection.index
                ][0]

        for panel in panels.values():
            panel.draw(screen, font)

        # Redraw ships so dead-ship overlays stay above the fleet grid.
        for player, panel in panels.items():
            for index, (_, (sprite, _, _, rect)) in enumerate(
                selectable_panel_slots[player]
            ):
                screen.blit(sprite, rect)
                if not player_ships[player][index].currently_alive:
                    draw_x(screen, rect)

        for player in survivor_locked_players:
            draw_locked_panel(panels[player])
        for player in random_locked_players:
            player_color = Const.P1_COLOR if player == 1 else Const.P2_COLOR
            draw_random_locked_panel(panels[player], player_color)
        if choose_second_player in (1, 2):
            if (
                selection_state.first_locked
                and selection_state.first_player not in random_locked_players
            ):
                draw_locked_panel(panels[selection_state.first_player])
            elif not selection_state.first_locked:
                draw_locked_panel(panels[choose_second_player])
            if not both_selected:
                active_panel = panels[active_player]
                active_color = Const.P1_COLOR if active_player == 1 else Const.P2_COLOR
                pygame.draw.rect(screen, active_color, active_panel.rect, 4)
                draw_panel_badge(active_panel, active_badge, active_color)
        elif survivor_locked_players and not both_selected:
            for player in (1, 2):
                if player in survivor_locked_players:
                    continue
                player_color = Const.P1_COLOR if player == 1 else Const.P2_COLOR
                pygame.draw.rect(screen, player_color, panels[player].rect, 4)
                draw_panel_badge(panels[player], active_badge, player_color)

        # A selected ship is indicated directly in its fleet square.
        for player, slot_index in selected_slots.items():
            pygame.draw.rect(screen, ui.WHITE, panels[player].slot_rect(slot_index), 3)

        hover_alpha = ui_box.ship_selection_hover_alpha(pygame.time.get_ticks())
        for player, slot_index in hovered_slots.items():
            if selected_slots.get(player) == slot_index:
                continue
            ui_box.draw_alpha_rect_outline(
                screen,
                panels[player].slot_rect(slot_index),
                ui.WHITE,
                hover_alpha,
                3,
            )

        for button in random_buttons.values():
            button.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        for player, panel in panels.items():
            if not selection_state.selection_allowed(player):
                continue
            hovered = panel.occupied_slot_at_pos(mouse_pos)
            if hovered is None:
                continue
            slot_index, (_, name, cost, _) = hovered
            definition = ships_data[name]
            label = ui.format_ship_tooltip(
                name,
                getattr(definition, "ship_type", ""),
                cost,
            )
            ui.draw_ship_tooltip(
                screen,
                tooltip_font,
                label,
                mouse_pos,
                panel.slot_rect(slot_index),
            )
            break

        pygame.display.flip()

    return None, None
