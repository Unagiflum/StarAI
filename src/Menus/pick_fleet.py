import pygame
import sys

from src.UI import ui, ui_button, ui_box
from src.UI.ship_sprites import (
    fit_ship_sprites,
    load_menu_ship_sprites,
    populate_fleet_panel,
    scale_ship_sprites,
)
import src.const as const
from src.frame_timing import PresentationClock
from src.configuration import (
    FLEET_AI_MODE_BASELINE,
    FLEET_AI_MODE_TRAINED,
    Fleets,
    FleetsRepository,
    PlayerFleet,
)
from src.Menus import pick_ship
from src.persistence import PersistenceValidationError
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.training.model_registry import (
    TrainingModelRepository,
    trained_model_counts_for_ships,
)

# Display settings
SELECTION_ICON_SIZE = const.SELECTION_ICON_SIZE
FLEET_ICON_SIZE = const.FLEET_ICON_SIZE
cost_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.03)
TITLE_FONT_SIZE = int(const.SCREEN_HEIGHT * 0.08)
PICKER_COLS = 5
PICKER_ROWS = 5
PICKER_CAPACITY = PICKER_COLS * PICKER_ROWS
MODAL_SHADE_ALPHA = 165
PICKER_CELL_COLOR = const.SHIP_PANEL_BACKGROUND_COLOR
PICKER_BOX_COLOR = const.SHIP_BOX_BACKGROUND_COLOR
PICKER_CELL_GAP = 3
PICKER_BORDER_WIDTH = 5
PICKER_TOOLTIP_FONT_SIZE = const.SHIP_TOOLTIP_FONT_SIZE
PICKER_AI_DOT_DIAMETER = 4
PICKER_AI_DOT_GAP = 2
PICKER_AI_DOT_COLOR = (0, 255, 0)
FLEET_CONTROL_WEIGHTS = {
    "ai": 18,
    "one_of_each": 25,
    "2x": 12,
    "4x": 12,
    "fill": 15,
    "clear": 18,
}


TRAINED_AI_LABEL = "Trained AI"
BASELINE_AI_LABEL = "Base AI"
HUMAN_LABEL = "Human"
# Slider stops from left to right. A click on the slider picks the nearest
# stop; a click elsewhere on the button moves one stop right, wrapping, so it
# keeps the old cycle of human -> trained AI -> baseline AI -> human.
AI_SELECTOR_STOPS = (
    (TRAINED_AI_LABEL, True),
    (BASELINE_AI_LABEL, FLEET_AI_MODE_BASELINE),
    (HUMAN_LABEL, False),
)
HUMAN_STOP = len(AI_SELECTOR_STOPS) - 1


class AIModeToggleButton(ui_button.Button):
    """Three-stop selector for a player: trained AI, baseline AI or human."""

    def __init__(
        self,
        x,
        y,
        width,
        height,
        text=None,
        callback=None,
        initial_state=False,
        bg_color=ui.MENU_BUTTON_COLOR,
        hover_color=ui.MENU_BUTTON_COLOR_HI,
        text_color=ui.WHITE,
    ):
        super().__init__(
            x, y, width, height, "", callback, bg_color, hover_color, text_color
        )
        self.stop = HUMAN_STOP
        self.set_setting(initial_state)

    def set_setting(self, setting) -> None:
        if setting == FLEET_AI_MODE_BASELINE:
            self.stop = 1
        elif setting:
            self.stop = 0
        else:
            self.stop = HUMAN_STOP

    def handle_event(self, event, sound_manager=None):
        if not self.enabled:
            return
        if event.type != pygame.MOUSEBUTTONDOWN or not self.rect.collidepoint(
            event.pos
        ):
            return
        if sound_manager:
            sound_manager.play_sound("menu")
        if self._slider_band().collidepoint(event.pos):
            self.stop = self._nearest_stop(event.pos[0])
        else:
            self.stop = (self.stop + 1) % len(AI_SELECTOR_STOPS)
        if self.callback:
            self.callback(self.value)

    @property
    def text(self) -> str:
        return AI_SELECTOR_STOPS[self.stop][0]

    @text.setter
    def text(self, _value) -> None:
        # The label always names the selected stop.
        pass

    @property
    def is_on(self) -> bool:
        return self.stop != HUMAN_STOP

    @property
    def baseline(self) -> bool:
        return AI_SELECTOR_STOPS[self.stop][1] == FLEET_AI_MODE_BASELINE

    @property
    def value(self):
        """``False`` for a human, ``True`` for trained AI, or ``"baseline"``."""

        return AI_SELECTOR_STOPS[self.stop][1]

    @property
    def ai_mode(self) -> str:
        return FLEET_AI_MODE_BASELINE if self.baseline else FLEET_AI_MODE_TRAINED

    @property
    def _knob_radius(self) -> int:
        return max(4, self.rect.height // 8)

    def _stop_positions(self) -> list[int]:
        padding = max(self._knob_radius * 2, self.rect.width // 8)
        left = self.rect.left + padding
        right = self.rect.right - padding
        last = len(AI_SELECTOR_STOPS) - 1
        return [round(left + (right - left) * index / last) for index in range(last + 1)]

    def _slider_center_y(self) -> int:
        return self.rect.bottom - 2 * self._knob_radius

    def _label_center_y(self) -> int:
        """Centre of the space above the slider's knob."""

        return (self.rect.top + self._slider_center_y() - self._knob_radius) // 2

    def _slider_band(self) -> pygame.Rect:
        """The lower part of the button, where a click picks a stop."""

        top = self._slider_center_y() - 2 * self._knob_radius
        return pygame.Rect(
            self.rect.left, top, self.rect.width, self.rect.bottom - top
        )

    def _nearest_stop(self, x) -> int:
        positions = self._stop_positions()
        return min(range(len(positions)), key=lambda index: abs(positions[index] - x))

    def draw(self, surface, font, mouse_pos=None):
        if not self.enabled:
            color = (*ui.DARK_GREY, 255)
        else:
            if mouse_pos is None:
                mouse_pos = pygame.mouse.get_pos()
            color = (
                self.hover_color if self.rect.collidepoint(mouse_pos) else self.bg_color
            )
        button_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            button_surface, color, button_surface.get_rect(), border_radius=5
        )
        surface.blit(button_surface, self.rect)

        slider_y = self._slider_center_y()
        text_color = self.text_color if self.enabled else ui.LIGHT_GREY
        text_surf = font.render(self.text, True, text_color)
        surface.blit(
            text_surf,
            text_surf.get_rect(center=(self.rect.centerx, self._label_center_y())),
        )

        positions = self._stop_positions()
        radius = self._knob_radius
        line_half = max(1, radius // 3)
        pygame.draw.rect(
            surface,
            ui.SLIDER_LINE,
            pygame.Rect(
                positions[0],
                slider_y - line_half,
                positions[-1] - positions[0],
                2 * line_half,
            ),
        )
        for x in positions:
            pygame.draw.circle(
                surface, ui.SLIDER_LINE, (x, slider_y), line_half + 2
            )
        pygame.draw.circle(
            surface,
            ui.HANDLE_COLOR if self.enabled else ui.GREY,
            (positions[self.stop], slider_y),
            radius,
        )


def fleet_control_rects(column_start, top, width, height, gap):
    """Lay out the fleet controls in one row exactly as wide as the fleet."""
    available_width = width - gap * (len(FLEET_CONTROL_WEIGHTS) - 1)
    items = tuple(FLEET_CONTROL_WEIGHTS.items())
    rects = {}
    left = column_start
    remaining_width = available_width
    for index, (name, weight) in enumerate(items):
        if index == len(items) - 1:
            control_width = remaining_width
        else:
            control_width = available_width * weight // 100
        rects[name] = pygame.Rect(left, top, control_width, height)
        left += control_width + gap
        remaining_width -= control_width
    return rects


class ShipPickerModal:
    """Centered, fixed 5x5 catalog used by ship-selection screens."""

    def __init__(
        self,
        player,
        slot_index,
        ships_data,
        sprites,
        quantity=1,
        title_label=None,
        model_counts=None,
    ):
        self.player = player
        self.slot_index = slot_index
        self.quantity = quantity
        self.title_label = title_label
        self.model_counts = {
            str(name): max(0, int(count))
            for name, count in dict(model_counts or {}).items()
        }
        self.color = const.P1_COLOR if player == 1 else const.P2_COLOR
        if len(ships_data) > PICKER_CAPACITY:
            raise ValueError(
                f"The ship picker supports {PICKER_CAPACITY} ships; "
                f"the catalog contains {len(ships_data)}"
            )

        max_modal_height = min(
            int(const.SCREEN_WIDTH * 0.70), int(const.SCREEN_HEIGHT * 0.82)
        )
        title_height = int(const.SCREEN_HEIGHT * 0.055)
        max_grid_side = min(
            int(const.SCREEN_WIDTH * 0.70) - 2 * PICKER_CELL_GAP,
            max_modal_height - title_height - 3 * PICKER_CELL_GAP,
        )
        cell_size = (
            max_grid_side - PICKER_CELL_GAP * (PICKER_COLS - 1)
        ) // PICKER_COLS
        grid_side = (
            cell_size * PICKER_COLS
            + PICKER_CELL_GAP * (PICKER_COLS - 1)
        )
        modal_width = grid_side + 2 * PICKER_CELL_GAP
        modal_height = grid_side + title_height + 3 * PICKER_CELL_GAP
        self.rect = pygame.Rect(
            0,
            0,
            modal_width + 2 * PICKER_BORDER_WIDTH,
            modal_height + 2 * PICKER_BORDER_WIDTH,
        )
        self.rect.center = (const.SCREEN_WIDTH // 2, const.SCREEN_HEIGHT // 2)
        self.content_rect = self.rect.inflate(
            -2 * PICKER_BORDER_WIDTH, -2 * PICKER_BORDER_WIDTH
        )

        self.title_rect = pygame.Rect(
            self.content_rect.left + PICKER_CELL_GAP,
            self.content_rect.top + PICKER_CELL_GAP,
            grid_side,
            title_height,
        )
        button_width = int(grid_side * 0.20)
        button_height = int(const.SCREEN_HEIGHT * 0.045)
        self.cancel_rect = pygame.Rect(
            self.title_rect.right - button_width,
            self.title_rect.centery - button_height // 2,
            button_width,
            button_height,
        )
        self.title_text_rect = self.title_rect.copy()
        self.title_text_rect.width = (
            self.cancel_rect.left - PICKER_CELL_GAP - self.title_text_rect.left
        )

        grid_rect = pygame.Rect(
            self.content_rect.left + PICKER_CELL_GAP,
            self.title_rect.bottom + PICKER_CELL_GAP,
            grid_side,
            grid_side,
        )
        self.cell_rects = []
        for index in range(PICKER_CAPACITY):
            row, col = divmod(index, PICKER_COLS)
            self.cell_rects.append(
                pygame.Rect(
                    grid_rect.left + col * (cell_size + PICKER_CELL_GAP),
                    grid_rect.top + row * (cell_size + PICKER_CELL_GAP),
                    cell_size,
                    cell_size,
                )
            )

        picker_sprites = fit_ship_sprites(sprites, cell_size)
        self.ships = [
            (
                name,
                getattr(definition, "ship_type", ""),
                definition.cost,
                picker_sprites[name],
            )
            for name, definition in ships_data.items()
        ]
        self.cost_font = pygame.font.SysFont(None, const.SHIP_CATALOG_COST_FONT_SIZE)

    def ship_at_pos(self, pos):
        for index, ship in enumerate(self.ships):
            if self.cell_rects[index].collidepoint(pos):
                return ship
        return None

    def _hovered_ship(self, pos):
        for index, ship in enumerate(self.ships):
            if self.cell_rects[index].collidepoint(pos):
                return ship, self.cell_rects[index]
        return None

    def _tooltip_rect(self, text_surface, mouse_pos, cell_rect, screen_rect):
        return ui.tooltip_rect(
            text_surface,
            mouse_pos,
            cell_rect,
            screen_rect,
        )

    def draw(self, screen, title_font, tooltip_font):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, PICKER_BOX_COLOR, self.content_rect)

        if self.title_label is not None:
            title_label = self.title_label
        elif self.slot_index is not None:
            title_label = f"Player {self.player}: Select a Ship"
        elif self.quantity is None:
            title_label = f"Player {self.player}: Select a Ship to Fill"
        else:
            title_label = f"Player {self.player}: Select a Ship ({self.quantity}x)"
        title = title_font.render(title_label, True, ui.WHITE)
        screen.blit(title, title.get_rect(center=self.title_text_rect.center))

        mouse_pos = pygame.mouse.get_pos()
        cancel_color = (
            ui.CAN_RED_HI
            if self.cancel_rect.collidepoint(mouse_pos)
            else ui.CAN_RED
        )
        pygame.draw.rect(screen, cancel_color, self.cancel_rect, border_radius=5)
        cancel_text = title_font.render("Cancel", True, ui.WHITE)
        screen.blit(cancel_text, ui.centered_text_rect(cancel_text, self.cancel_rect))

        for index, cell_rect in enumerate(self.cell_rects):
            pygame.draw.rect(screen, PICKER_CELL_COLOR, cell_rect)

            if index >= len(self.ships):
                continue
            name, _, cost, sprite = self.ships[index]
            screen.blit(sprite, sprite.get_rect(center=cell_rect.center))
            cost_surface = self.cost_font.render(
                str(cost), True, const.SHIP_CATALOG_COST_COLOR
            )
            cost_pos = (cell_rect.left + 2, cell_rect.top + 2)
            screen.blit(cost_surface, cost_pos)
            self._draw_ai_dots(
                screen,
                cell_rect,
                cost_surface.get_rect(topleft=cost_pos),
                self.model_counts.get(name, 0),
            )

        hovered = self._hovered_ship(mouse_pos)
        if hovered is not None:
            (name, ship_type, _, _), cell_rect = hovered
            pygame.draw.rect(screen, ui.WHITE, cell_rect, 1)
            label = ui.format_ship_tooltip(name, ship_type, include_cost=False)
            ui.draw_ship_tooltip(
                screen,
                tooltip_font,
                label,
                mouse_pos,
                cell_rect,
            )

    def _draw_ai_dots(self, screen, cell_rect, cost_rect, count):
        y = cost_rect.bottom + 2
        if count <= 0 or y + PICKER_AI_DOT_DIAMETER > cell_rect.bottom:
            return
        left = cell_rect.left + 2
        step = PICKER_AI_DOT_DIAMETER + PICKER_AI_DOT_GAP
        for index in range(count):
            dot_rect = pygame.Rect(
                left + index * step,
                y,
                PICKER_AI_DOT_DIAMETER,
                PICKER_AI_DOT_DIAMETER,
            )
            if dot_rect.right > cell_rect.right:
                break
            pygame.draw.ellipse(screen, PICKER_AI_DOT_COLOR, dot_rect)


def load_ships():
    """Return the authoritative typed catalog (compatibility entry point)."""
    return SHIP_DEFINITIONS


def load_ship_sprites(ships_data, resources=None):
    def fallback(_ship_name):
        surface = pygame.Surface(SELECTION_ICON_SIZE, pygame.SRCALPHA)
        surface.fill(ui.GREY)
        return surface

    return load_menu_ship_sprites(ships_data, resources=resources, fallback=fallback)


def load_ship_model_counts(ships_data, repository=None):
    repository = repository or TrainingModelRepository(
        const.DEFAULT_MODELS_PATH,
        const.MODELS_PATH,
    )
    return trained_model_counts_for_ships(repository, ships_data.keys())


def create_sprite_sets(ships_data):
    """Load picker source art and create fleet sprites at the fleet scale."""
    original_sprites = load_ship_sprites(ships_data)

    # Calculate fleet icon size first
    fleet = ui_box.Fleet(0, 0, ui.SELECTION_WIDTH, ui.FLEET_HEIGHT, "", (0, 0))
    fleet_size = fleet.icon_size[0]
    fleet_sprites = scale_ship_sprites(original_sprites, fleet_size, ships_data)

    return original_sprites, fleet_sprites


scale_sprites = scale_ship_sprites


def _player_fleet(fleet_panel: ui_box.Fleet, ai_setting) -> PlayerFleet:
    return PlayerFleet(
        fleet_panel.model.ship_slots,
        bool(ai_setting),
        FLEET_AI_MODE_BASELINE
        if ai_setting == FLEET_AI_MODE_BASELINE
        else FLEET_AI_MODE_TRAINED,
    )


def save_fleets(
    left_fleet: ui_box.Fleet, right_fleet: ui_box.Fleet, left_ai, right_ai
):
    """Save the current fleets and AI settings to fleets.json.

    Each AI setting is ``False`` (human), ``True`` (trained AI) or
    ``"baseline"``.
    """
    fleets = Fleets(
        _player_fleet(left_fleet, left_ai),
        _player_fleet(right_fleet, right_ai),
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
    return fleets.player1.ai_setting, fleets.player2.ai_setting


def run(screen: pygame.Surface, menu_sound_manager=None, audio_service=None):
    """Run the Pick Fleet module."""
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    font = pygame.font.SysFont(None, cost_FONT_SIZE)
    title_font = pygame.font.SysFont(None, TITLE_FONT_SIZE)
    picker_title_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.042))
    picker_tooltip_font = pygame.font.SysFont(None, PICKER_TOOLTIP_FONT_SIZE)
    background = ui.load_background(
        const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT
    )

    # Load ships data and sprites
    ships_data = load_ships()
    ship_model_counts = load_ship_model_counts(ships_data)

    # Scale sprites for selection and fleet views, maintaining proportions
    selection_sprites, fleet_sprites = create_sprite_sets(ships_data)

    # Create UI components

    left_column_start = int(0.033 * const.SCREEN_WIDTH)
    # The control row fills the space between the title and the fleets, tall
    # enough for the AI selector's label and slider; the fleets stay put.
    fleet_top = int(0.15 * const.SCREEN_HEIGHT)
    top_button_start = int(0.0875 * const.SCREEN_HEIGHT)
    top_button_height = fleet_top - ui.button_spaceV - top_button_start

    right_column_start = int(const.SCREEN_WIDTH // 2 + (0.016 * const.SCREEN_WIDTH))
    columns = {1: left_column_start, 2: right_column_start}
    control_rects = {
        player: fleet_control_rects(
            columns[player],
            top_button_start,
            ui.SELECTION_WIDTH,
            top_button_height,
            ui.button_spaceH,
        )
        for player in (1, 2)
    }
    fleets = ui_box.create_player_fleet_panels(
        columns,
        fleet_top,
        ui.SELECTION_WIDTH,
        ui.FLEET_HEIGHT,
        FLEET_ICON_SIZE,
    )
    ai_toggles = {
        player: AIModeToggleButton(
            *control_rects[player]["ai"],
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
            *control_rects[player]["one_of_each"],
            "1 of Each",
            lambda player=player: create_one_of_each(player),
            bg_color=(*const.P1_COLOR, 75) if player == 1 else (*const.P2_COLOR, 75),
            hover_color=(
                (*const.P1_COLOR, 255) if player == 1 else (*const.P2_COLOR, 255)
            ),
        )
        for player in (1, 2)
    }

    ship_picker = None

    def open_bulk_picker(player, quantity):
        nonlocal ship_picker
        ship_picker = ShipPickerModal(
            player,
            None,
            ships_data,
            selection_sprites,
            quantity=quantity,
            model_counts=ship_model_counts,
        )

    bulk_buttons = {
        quantity: {
            player: ui_button.Button(
                *control_rects[player][label.lower()],
                label,
                lambda player=player, quantity=quantity: open_bulk_picker(
                    player, quantity
                ),
                bg_color=(
                    (*const.P1_COLOR, 75)
                    if player == 1
                    else (*const.P2_COLOR, 75)
                ),
                hover_color=(
                    (*const.P1_COLOR, 255)
                    if player == 1
                    else (*const.P2_COLOR, 255)
                ),
            )
            for player in (1, 2)
        }
        for label, quantity in (("2x", 2), ("4x", 4), ("Fill", None))
    }
    clear_buttons = {
        player: ui_button.Button(
            *control_rects[player]["clear"],
            "Clear",
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
        ai_toggles[player].set_setting(ai)

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

            try:
                pick_ship.run(
                    screen=screen,
                    menu_sound_manager=menu_sound_manager,
                    audio_service=audio_service,
                    player1_ai=ai_toggles[1].value,
                    player2_ai=ai_toggles[2].value,
                )
            finally:
                clock.reset()

    running = True

    control_groups = (
        ai_toggles,
        one_of_each_buttons,
        *bulk_buttons.values(),
        clear_buttons,
    )

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
        clock.tick()

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
                            name, _, cost, _ = selected
                            fleet = fleets[ship_picker.player]
                            if ship_picker.slot_index is not None:
                                added_count = int(
                                    fleet.set_ship_at_slot(
                                        ship_picker.slot_index,
                                        fleet_sprites[name],
                                        name,
                                        cost,
                                    )
                                )
                            else:
                                added_count = fleet.add_ships_after_last(
                                    fleet_sprites[name],
                                    name,
                                    cost,
                                    ship_picker.quantity,
                                )
                            if added_count:
                                print(
                                    f"Added {added_count} {name} ship(s) "
                                    f"in {fleet.title}"
                                )
                                if menu_sound_manager:
                                    menu_sound_manager.play_sound("menu")
                            ship_picker = None
                continue

            for controls in control_groups:
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
                                model_counts=ship_model_counts,
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

        controls_enabled = ship_picker is None
        for controls in control_groups:
            for control in controls.values():
                control.enabled = controls_enabled
        confirm_button.enabled = controls_enabled
        cancel_button.enabled = controls_enabled

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

        for controls in control_groups:
            for control in controls.values():
                control.draw(screen, font)
        for fleet in fleets.values():
            fleet.draw(screen, font)
        confirm_button.draw(screen, font)
        cancel_button.draw(screen, font)

        if controls_enabled:
            mouse_pos = pygame.mouse.get_pos()
            for fleet in fleets.values():
                hovered = fleet.occupied_slot_at_pos(mouse_pos)
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
                    picker_tooltip_font,
                    label,
                    mouse_pos,
                    fleet.slot_rect(slot_index),
                )
                break

        if ship_picker is not None:
            shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, MODAL_SHADE_ALPHA))
            screen.blit(shade, (0, 0))
            for player, fleet in fleets.items():
                color = const.P1_COLOR if player == 1 else const.P2_COLOR
                dark_border = tuple(int(channel * 0.35) for channel in color)
                pygame.draw.rect(screen, dark_border, fleet.rect, 3)
            ship_picker.draw(screen, picker_title_font, picker_tooltip_font)

        pygame.display.flip()
