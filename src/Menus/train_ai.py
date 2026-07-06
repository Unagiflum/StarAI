"""UI shell for configuring future AI training sessions."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pygame

import src.const as const
from src.Menus.pick_fleet import (
    MODAL_SHADE_ALPHA,
    PICKER_TOOLTIP_FONT_SIZE,
    ShipPickerModal,
)
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.UI import ui, ui_button, ui_slider
from src.UI.ship_sprites import fit_ship_sprites, load_menu_ship_sprites
from src.frame_timing import PresentationClock


REWARD_VALUES = tuple(
    [-10.24, -5.12, -2.56, -1.28, -0.64, -0.32, -0.16, -0.08, -0.04, -0.02, -0.01]
    + [0.0]
    + [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
)

REWARD_LABELS = (
    "Point at enemy",
    "Get in weapon range",
    "Spawn an object",
    "Harm enemy",
    "Harm enemy object",
    "Harm self",
    "Harmed by enemy",
    "Enemy loses crew",
    "Gain crew",
    "Gain battery",
    "Get to high speed",
    "Lose crew",
    "Lose battery",
    "Die",
    "Kill enemy",
)

MOVEMENT_BEHAVIORS = (
    "Move forward continuously",
    "Hold A1 continuously",
    "Hold A2 continuously",
)

TURNING_BEHAVIORS = (
    ("No turning", "none"),
    ("Face trainee", "face_trainee"),
    ("Face away from trainee", "face_away"),
    ("Turn right continuously", "turn_right"),
    ("Turn left continuously", "turn_left"),
)

ROUNDS_PER_EPOCH_VALUES = (1, 2, 5, 10, 20, 50)
MATCH_TIME_LIMIT_VALUES = (240, 480, 1200, 2400, 4800, 12000)
LEARNING_RATE_VALUES = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)
EPSILON_VALUES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.1, 0.2, 0.5)
HIDDEN_LAYER_SIZE_VALUES = (30, 100, 300, 1000, 3000)
HIDDEN_LAYER_COUNT_VALUES = (1, 2, 4, 8, 16)

CONTROL_WIDTH = const.SCREEN_WIDTH - const.SCREEN_HEIGHT
TAB_MARGIN = 8
TAB_GAP = 8
TAB_HEIGHT = 48
TAB_COLOR = (155, 0, 105, 75)
TAB_COLOR_HI = (155, 0, 105, 255)
TAB_HEADER_COLOR = (100, 100, 100)
CONTENT_TOP = TAB_MARGIN + TAB_HEIGHT + TAB_GAP
DISPLAY_TOP = 614
FOOTER_CONTROL_HEIGHT = 46
ACTION_TOP = 668
HUD_TOP = 722
HUD_BOTTOM_MARGIN = 10
CONTENT_BOTTOM = DISPLAY_TOP - TAB_GAP
CONTENT_VIEW_HEIGHT = CONTENT_BOTTOM - CONTENT_TOP


@dataclass
class TrainingUIState:
    active_tab: str = "trainee"
    selected_ship: str | None = None
    selected_slot: int = 1
    slot_labels: list[str] = field(default_factory=lambda: ["", "", "", ""])
    rewards: dict[str, float] = field(
        default_factory=lambda: {label: 0.0 for label in REWARD_LABELS}
    )
    opponent_mode: str = "simple"
    movement_behaviors: set[str] = field(default_factory=set)
    turning_behavior: str = "none"
    rounds_per_epoch: int = 10
    match_time_limit: int = 2400
    learning_rate: float = 0.001
    epsilon: float = 0.1
    hidden_layer_size: int = 100
    hidden_layer_count: int = 2
    display_on: bool = False
    running: bool = False

    @property
    def simple_behavior_controls_enabled(self):
        return self.opponent_mode == "simple"


@dataclass(frozen=True)
class TrainingLayout:
    control_rect: pygame.Rect
    arena_rect: pygame.Rect
    content_rect: pygame.Rect
    hud_rects: tuple[pygame.Rect, pygame.Rect]


def training_layout():
    hud_gap = 8
    hud_left = TAB_MARGIN
    hud_width = (CONTROL_WIDTH - 2 * TAB_MARGIN - hud_gap) // 2
    hud_height = const.SCREEN_HEIGHT - HUD_TOP - HUD_BOTTOM_MARGIN
    return TrainingLayout(
        control_rect=pygame.Rect(0, 0, CONTROL_WIDTH, const.SCREEN_HEIGHT),
        arena_rect=pygame.Rect(
            CONTROL_WIDTH, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT
        ),
        content_rect=pygame.Rect(
            0, CONTENT_TOP, CONTROL_WIDTH, CONTENT_VIEW_HEIGHT
        ),
        hud_rects=(
            pygame.Rect(hud_left, HUD_TOP, hud_width, hud_height),
            pygame.Rect(
                hud_left + hud_width + hud_gap,
                HUD_TOP,
                hud_width,
                hud_height,
            ),
        ),
    )


class TabButton(ui_button.Button):
    def __init__(self, x, y, width, height, text, callback):
        super().__init__(
            x, y, width, height, text, callback,
            bg_color=(*const.TAB_BUTTON_COLOR, const.TAB_BUTTON_NORMAL_ALPHA), 
            hover_color=(*const.TAB_BUTTON_COLOR, const.TAB_BUTTON_HOVER_ALPHA)
        )
        self.active = False

    def draw(self, surface, font, mouse_pos=None):
        if not self.enabled:
            color = (*ui.DARK_GREY, 255)
        else:
            if mouse_pos is None:
                mouse_pos = pygame.mouse.get_pos()
            color = (
                self.hover_color if self.rect.collidepoint(mouse_pos) and not self.active else self.bg_color
            )
            if self.active:
                color = (*const.TAB_BUTTON_COLOR, const.TAB_BUTTON_SELECTED_ALPHA)

        button_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        
        # Fill button with color
        pygame.draw.rect(
            button_surface, color, button_surface.get_rect(), 
            border_top_left_radius=5, border_top_right_radius=5
        )

        # Draw black border on all sides
        pygame.draw.rect(
            button_surface, const.TAB_BUTTON_BORDER_COLOR, button_surface.get_rect(), width=2,
            border_top_left_radius=5, border_top_right_radius=5
        )

        # Remove the bottom 2 pixels of the black border (replace with color)
        # We start at x=2 and width is width-4 to preserve the left and right borders
        button_surface.fill(color, pygame.Rect(2, self.rect.height - 2, self.rect.width - 4, 2))

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=button_surface.get_rect().center)
        button_surface.blit(text_surf, text_rect)

        surface.blit(button_surface, self.rect)


def largest_fitting_font(texts, max_width, max_height=36, maximum=36, minimum=16):
    """Return the largest system font fitting every supplied label."""
    for size in range(maximum, minimum - 1, -1):
        font = pygame.font.SysFont(None, size)
        if all(
            font.size(text)[0] <= max_width and font.get_linesize() <= max_height
            for text in texts
        ):
            return font
    return pygame.font.SysFont(None, minimum)


def _format_reward(value):
    return "0.00" if value == 0 else f"{value:+.2f}"


class RewardSlider:
    """Compact discrete slider with its label and value on one row."""

    def __init__(self, rect, label, value=0.0):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.values = REWARD_VALUES
        self.value = value
        self.dragging = False
        self.label_width = 278
        self.value_width = 70
        self.line_rect = pygame.Rect(
            self.rect.x + self.label_width,
            self.rect.centery - 2,
            self.rect.width - self.label_width - self.value_width - 8,
            4,
        )

    @property
    def handle_x(self):
        index = self.values.index(self.value)
        return self.line_rect.x + round(
            index * self.line_rect.width / (len(self.values) - 1)
        )

    def set_from_x(self, x):
        ratio = (x - self.line_rect.left) / max(1, self.line_rect.width)
        index = round(ratio * (len(self.values) - 1))
        index = max(0, min(len(self.values) - 1, index))
        self.value = self.values[index]

    def handle_event(self, event, sound_manager=None):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit_rect = self.line_rect.inflate(0, 20)
            if hit_rect.collidepoint(event.pos):
                if sound_manager:
                    sound_manager.play_sound("menu")
                self.dragging = True
                self.set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_from_x(event.pos[0])

    def draw(self, surface, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        row = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        row.fill(ui.SLIDER_BG_HI if hovered else ui.SLIDER_BG)
        surface.blit(row, self.rect)

        label = font.render(self.label, True, ui.WHITE)
        surface.blit(
            label,
            label.get_rect(midleft=(self.rect.left + 8, self.rect.centery)),
        )
        pygame.draw.rect(surface, ui.SLIDER_LINE, self.line_rect)
        pygame.draw.circle(
            surface, ui.HANDLE_COLOR, (self.handle_x, self.line_rect.centery), 7
        )
        value = font.render(_format_reward(self.value), True, ui.WHITE)
        surface.blit(
            value,
            value.get_rect(midright=(self.rect.right - 8, self.rect.centery)),
        )


class TextField:
    """Small single-line editor used for optional AI-slot descriptions."""

    def __init__(self, rect, text="", max_length=24):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.max_length = max_length
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode and event.unicode.isprintable():
                if len(self.text) < self.max_length:
                    self.text += event.unicode

    def draw(self, surface, font):
        pygame.draw.rect(surface, ui.BLACK, self.rect)
        pygame.draw.rect(
            surface,
            ui.BRIGHT_GREEN if self.active else ui.LIGHT_GREY,
            self.rect,
            2,
        )
        text = font.render(self.text, True, ui.WHITE)
        clip = self.rect.inflate(-12, -4)
        surface.set_clip(clip)
        text_rect = text.get_rect(midleft=(self.rect.left + 6, self.rect.centery))
        surface.blit(text, text_rect)
        if self.active and pygame.time.get_ticks() % 1000 < 500:
            cursor_x = text_rect.right + 2
            pygame.draw.line(surface, ui.WHITE, (cursor_x, self.rect.centery - font.get_linesize() // 2 + 2), (cursor_x, self.rect.centery + font.get_linesize() // 2 - 2), 2)
        surface.set_clip(None)


def _translated_event(event, viewport, scroll_y):
    if not hasattr(event, "pos"):
        return event
    attributes = dict(event.dict)
    attributes["pos"] = (
        event.pos[0] - viewport.x,
        event.pos[1] - viewport.y + scroll_y,
    )
    return pygame.event.Event(event.type, attributes)


def _draw_scrollbar(screen, viewport, content_height, scroll_y):
    if content_height <= viewport.height:
        return
    track = pygame.Rect(viewport.right - 8, viewport.top, 8, viewport.height)
    thumb_height = max(36, viewport.height * viewport.height // content_height)
    max_scroll = content_height - viewport.height
    thumb_y = track.top + round(
        scroll_y * (track.height - thumb_height) / max_scroll
    )
    pygame.draw.rect(screen, ui.DARK_GREY, track)
    pygame.draw.rect(
        screen, ui.LIGHT_GREY, (track.x, thumb_y, track.width, thumb_height)
    )


def _draw_arena_placeholder(screen, rect, state, font):
    pygame.draw.rect(screen, ui.BLACK, rect)
    pygame.draw.rect(screen, ui.GREY, rect, 2)
    if state.display_on:
        lines = ("Battle display", "Training visualization is not implemented yet")
    else:
        lines = ("Training statistics", "Round and opponent details will appear here")
    y = rect.centery - font.get_linesize()
    for line in lines:
        text = font.render(line, True, ui.LIGHT_GREY)
        screen.blit(text, text.get_rect(center=(rect.centerx, y)))
        y += font.get_linesize() + 8


def _draw_hud_placeholders(screen, hud_rects, font):
    for rect, label, color in zip(
        hud_rects,
        ("Trainee HUD", "Opponent HUD"),
        (const.P1_COLOR, const.P2_COLOR),
    ):
        pygame.draw.rect(screen, ui.BLACK, rect)
        pygame.draw.rect(screen, color, rect, 2)
        text = font.render(label, True, color)
        screen.blit(text, text.get_rect(center=rect.center))


def _draw_group_panel(surface, rect, hovered=False, enabled=True):
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    if not enabled:
        panel.fill((*ui.DARK_GREY, 255))
    else:
        panel.fill(ui.SLIDER_BG_HI if hovered else ui.SLIDER_BG)
    surface.blit(panel, rect)
    pygame.draw.rect(surface, ui.BLACK, rect, 3)


def run(screen: pygame.Surface, menu_sound_manager=None, audio_service=None):
    """Show the AI-training configuration UI without starting training yet."""
    _ = audio_service
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    layout = training_layout()
    state = TrainingUIState()
    background = ui.load_background(
        const.MENU_BG_PATH, const.SCREEN_WIDTH, const.SCREEN_HEIGHT
    )

    body_font = largest_fitting_font(
        REWARD_LABELS,
        270,
        max_height=34,
        maximum=32,
    )
    opponent_font = largest_fitting_font(
        MOVEMENT_BEHAVIORS
        + tuple(x[0] for x in TURNING_BEHAVIORS)
        + (
            "Train against all existing AIs",
            "Train against simple behaviors",
        ),
        CONTROL_WIDTH - 80,
        max_height=30,
        maximum=28,
    )
    tab_font = largest_fitting_font(
        ("Trainee", "Opponent", "Rewards", "Regimen"),
        (CONTROL_WIDTH - 5 * TAB_MARGIN) // 4 - 16,
        max_height=34,
        maximum=32,
    )
    rewards_font = largest_fitting_font(
        REWARD_LABELS,
        270,
        max_height=26,
        maximum=24,
    )
    small_font = pygame.font.SysFont(None, 24)
    arena_font = pygame.font.SysFont(None, 32)
    picker_title_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.042))
    picker_tooltip_font = pygame.font.SysFont(None, PICKER_TOOLTIP_FONT_SIZE)

    def fallback_ship_sprite(_ship_name):
        surface = pygame.Surface((188, 188), pygame.SRCALPHA)
        surface.fill(ui.GREY)
        return surface

    source_sprites = load_menu_ship_sprites(
        SHIP_DEFINITIONS, fallback=fallback_ship_sprite
    )
    selector_sprites = fit_ship_sprites(source_sprites, 188)

    tab_width = (CONTROL_WIDTH - 2 * TAB_MARGIN - 3 * TAB_GAP) // 4
    tab_height = TAB_HEIGHT + TAB_GAP
    trainee_tab = TabButton(
        TAB_MARGIN,
        TAB_MARGIN,
        tab_width,
        tab_height,
        "Trainee",
        lambda: setattr(state, "active_tab", "trainee"),
    )
    opponent_tab = TabButton(
        TAB_MARGIN + tab_width + TAB_GAP,
        TAB_MARGIN,
        tab_width,
        tab_height,
        "Opponent",
        lambda: setattr(state, "active_tab", "opponent"),
    )
    rewards_tab = TabButton(
        TAB_MARGIN + 2 * (tab_width + TAB_GAP),
        TAB_MARGIN,
        tab_width,
        tab_height,
        "Rewards",
        lambda: setattr(state, "active_tab", "rewards"),
    )
    regimen_tab = TabButton(
        TAB_MARGIN + 3 * (tab_width + TAB_GAP),
        TAB_MARGIN,
        tab_width,
        tab_height,
        "Regimen",
        lambda: setattr(state, "active_tab", "regimen"),
    )

    ship_tile = pygame.Rect((CONTROL_WIDTH - 200) // 2, 48, 200, 200)
    slot_rows = tuple(
        pygame.Rect(16, 290 + index * 46, CONTROL_WIDTH - 32, 40)
        for index in range(4)
    )
    slot_fields = [
        TextField((row.x + 64, row.y + 3, row.width - 64 - 40, row.height - 6))
        for row in slot_rows
    ]
    delete_buttons = [
        ui_button.Button(
            row.right - 36, row.y + 3, 34, row.height - 6, "X",
            lambda: None,
            bg_color=ui.CAN_RED, hover_color=ui.CAN_RED_HI
        )
        for row in slot_rows
    ]
    trainee_content_height = max(ship_tile.bottom, slot_rows[-1].bottom) + 12
    trainee_scroll_y = 0

    rewards_top = 8
    reward_sliders = [
        RewardSlider(
            (12, rewards_top + index * 34, CONTROL_WIDTH - 24, 30), label
        )
        for index, label in enumerate(REWARD_LABELS)
    ]
    rewards_content_height = reward_sliders[-1].rect.bottom + 8
    rewards_scroll_y = 0
    selected_opponent_mode = [state.opponent_mode]
    opponent_mode_buttons = []
    opponent_panels = (
        pygame.Rect(12, 12, CONTROL_WIDTH - 24, 100),
        pygame.Rect(12, 122, CONTROL_WIDTH - 24, 140),
        pygame.Rect(12, 272, CONTROL_WIDTH - 24, 258),
    )

    def select_opponent_mode(value):
        state.opponent_mode = value
        selected_opponent_mode[0] = value
        for button, option in zip(opponent_mode_buttons, ("all", "simple")):
            button.selected = option == value

    opponent_mode_buttons.extend(
        (
            ui_button.RadioButton(
                20,
                18,
                CONTROL_WIDTH - 40,
                40,
                "Train against all existing AIs",
                lambda: select_opponent_mode("all"),
            ),
            ui_button.RadioButton(
                20,
                64,
                CONTROL_WIDTH - 40,
                40,
                "Train against simple behaviors",
                lambda: select_opponent_mode("simple"),
                selected=True,
            ),
        )
    )
    movement_checkboxes = [
        ui_button.Checkbox(
            20,
            128 + index * 42,
            CONTROL_WIDTH - 40,
            38,
            label,
        )
        for index, label in enumerate(MOVEMENT_BEHAVIORS)
    ]

    turning_buttons = []

    def select_turning(value):
        state.turning_behavior = value
        for button, (_, option) in zip(turning_buttons, TURNING_BEHAVIORS):
            button.selected = option == value

    turning_start = 280
    for index, (label, value) in enumerate(TURNING_BEHAVIORS):
        turning_buttons.append(
            ui_button.RadioButton(
                20,
                turning_start + index * 48,
                CONTROL_WIDTH - 40,
                42,
                label,
                lambda selected=value: select_turning(selected),
                selected=value == "none",
            )
        )

    grouped_controls = (
        *opponent_mode_buttons,
        *movement_checkboxes,
        *turning_buttons,
    )
    for control in grouped_controls:
        control.bg_color = (0, 0, 0, 0)
        control.hover_color = (45, 45, 45, 160)

    regimen_left = 16
    regimen_width = CONTROL_WIDTH - 32
    regimen_top = CONTENT_TOP + 14
    regimen_spacing = 84
    regimen_sliders = (
        ui_slider.Slider(
            regimen_left,
            regimen_top,
            regimen_width,
            ROUNDS_PER_EPOCH_VALUES[0],
            ROUNDS_PER_EPOCH_VALUES[-1],
            state.rounds_per_epoch,
            "Rounds per epoch",
            is_int=True,
            values=ROUNDS_PER_EPOCH_VALUES,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + regimen_spacing,
            regimen_width,
            MATCH_TIME_LIMIT_VALUES[0],
            MATCH_TIME_LIMIT_VALUES[-1],
            state.match_time_limit,
            "Match Time Limit (frames)",
            is_int=True,
            values=MATCH_TIME_LIMIT_VALUES,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 2 * regimen_spacing,
            regimen_width,
            LEARNING_RATE_VALUES[0],
            LEARNING_RATE_VALUES[-1],
            state.learning_rate,
            "Learning rate",
            step=0.0001,
            values=LEARNING_RATE_VALUES,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 3 * regimen_spacing,
            regimen_width,
            EPSILON_VALUES[0],
            EPSILON_VALUES[-1],
            state.epsilon,
            "Epsilon",
            step=0.0001,
            values=EPSILON_VALUES,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 4 * regimen_spacing,
            regimen_width,
            HIDDEN_LAYER_SIZE_VALUES[0],
            HIDDEN_LAYER_SIZE_VALUES[-1],
            state.hidden_layer_size,
            "Hidden layer size",
            is_int=True,
            values=HIDDEN_LAYER_SIZE_VALUES,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 5 * regimen_spacing,
            regimen_width,
            HIDDEN_LAYER_COUNT_VALUES[0],
            HIDDEN_LAYER_COUNT_VALUES[-1],
            state.hidden_layer_count,
            "Hidden layer count",
            is_int=True,
            values=HIDDEN_LAYER_COUNT_VALUES,
        ),
    )

    display_checkbox = ui_button.Checkbox(
        TAB_MARGIN,
        DISPLAY_TOP,
        CONTROL_WIDTH - 2 * TAB_MARGIN,
        FOOTER_CONTROL_HEIGHT,
        "Display On",
    )
    exited = [False]

    def toggle_training():
        state.running = not state.running

    action_gap = 10
    action_width = (CONTROL_WIDTH - 2 * TAB_MARGIN - action_gap) // 2
    start_stop_button = ui_button.Button(
        TAB_MARGIN,
        ACTION_TOP,
        action_width,
        FOOTER_CONTROL_HEIGHT,
        "Start",
        toggle_training,
        ui.OK_GREEN,
        ui.OK_GREEN_HI,
    )
    back_button = ui_button.Button(
        TAB_MARGIN + action_width + action_gap,
        ACTION_TOP,
        action_width,
        FOOTER_CONTROL_HEIGHT,
        "Back",
        lambda: exited.__setitem__(0, True),
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )
    ship_picker = None

    while not exited[0]:
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
                            state.selected_ship = selected[0]
                            if menu_sound_manager:
                                menu_sound_manager.play_sound("menu")
                            ship_picker = None
                continue

            trainee_tab.handle_event(event, menu_sound_manager)
            opponent_tab.handle_event(event, menu_sound_manager)
            rewards_tab.handle_event(event, menu_sound_manager)
            regimen_tab.handle_event(event, menu_sound_manager)
            display_checkbox.handle_event(event, menu_sound_manager)
            start_stop_button.handle_event(event, menu_sound_manager)
            back_button.handle_event(event, menu_sound_manager)

            if state.active_tab == "trainee":
                translated = _translated_event(
                    event, layout.content_rect, trainee_scroll_y
                )
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if layout.content_rect.collidepoint(event.pos):
                        if ship_tile.collidepoint(translated.pos):
                            ship_picker = ShipPickerModal(
                                1,
                                None,
                                SHIP_DEFINITIONS,
                                source_sprites,
                                title_label="Select Trainee Ship",
                            )
                            if menu_sound_manager:
                                menu_sound_manager.play_sound("menu")
                        for index, row in enumerate(slot_rows):
                            if row.collidepoint(translated.pos):
                                state.selected_slot = index + 1
                                break
                for field in slot_fields:
                    field.handle_event(translated)
                for delete_btn in delete_buttons:
                    delete_btn.handle_event(translated, menu_sound_manager)
            elif state.active_tab == "rewards":
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button in (4, 5)
                    and layout.content_rect.collidepoint(event.pos)
                ):
                    direction = -1 if event.button == 4 else 1
                    max_scroll = max(
                        0, rewards_content_height - layout.content_rect.height
                    )
                    rewards_scroll_y = max(
                        0,
                        min(max_scroll, rewards_scroll_y + direction * 54),
                    )
                    continue

                translated = _translated_event(
                    event, layout.content_rect, rewards_scroll_y
                )
                for slider in reward_sliders:
                    slider.handle_event(translated, menu_sound_manager)
            elif state.active_tab == "opponent":
                translated = _translated_event(event, layout.content_rect, 0)
                for button in opponent_mode_buttons:
                    button.handle_event(translated, menu_sound_manager)
                enabled = state.simple_behavior_controls_enabled
                for checkbox in movement_checkboxes:
                    checkbox.enabled = enabled
                    checkbox.handle_event(translated, menu_sound_manager)
                for button in turning_buttons:
                    button.enabled = enabled
                    button.handle_event(translated, menu_sound_manager)
            else:
                for slider in regimen_sliders:
                    slider.handle_event(event, menu_sound_manager)

        state.display_on = display_checkbox.value
        state.slot_labels[:] = [field.text for field in slot_fields]
        state.rewards.update(
            (slider.label, slider.value) for slider in reward_sliders
        )
        state.movement_behaviors = {
            label
            for label, checkbox in zip(MOVEMENT_BEHAVIORS, movement_checkboxes)
            if checkbox.value
        }
        state.rounds_per_epoch = int(regimen_sliders[0].value)
        state.match_time_limit = int(regimen_sliders[1].value)
        state.learning_rate = regimen_sliders[2].value
        state.epsilon = regimen_sliders[3].value
        state.hidden_layer_size = int(regimen_sliders[4].value)
        state.hidden_layer_count = int(regimen_sliders[5].value)
        controls_enabled = state.simple_behavior_controls_enabled
        for checkbox in movement_checkboxes:
            checkbox.enabled = controls_enabled
        for button in turning_buttons:
            button.enabled = controls_enabled

        start_stop_button.text = "Stop" if state.running else "Start"
        start_stop_button.bg_color = ui.CAN_RED if state.running else ui.OK_GREEN
        start_stop_button.hover_color = (
            ui.CAN_RED_HI if state.running else ui.OK_GREEN_HI
        )

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)
        _draw_arena_placeholder(screen, layout.arena_rect, state, arena_font)

        trainee_tab.active = state.active_tab == "trainee"
        opponent_tab.active = state.active_tab == "opponent"
        rewards_tab.active = state.active_tab == "rewards"
        regimen_tab.active = state.active_tab == "regimen"

        if state.active_tab == "trainee":
            content = pygame.Surface(
                (CONTROL_WIDTH, max(trainee_content_height, layout.content_rect.height)), pygame.SRCALPHA
            )
            content.fill((0, 0, 0, 155))
            heading = body_font.render("Trainee Ship", True, ui.WHITE)
            content.blit(heading, heading.get_rect(center=(ship_tile.centerx, 25)))
            pygame.draw.rect(content, const.SHIP_PANEL_BACKGROUND_COLOR, ship_tile)
            pygame.draw.rect(content, const.P1_COLOR, ship_tile, 3)
            if state.selected_ship is None:
                prompt = body_font.render("Select Ship", True, ui.LIGHT_GREY)
                content.blit(prompt, prompt.get_rect(center=ship_tile.center))
            else:
                sprite = selector_sprites[state.selected_ship]
                content.blit(sprite, sprite.get_rect(center=ship_tile.center))

            slot_heading = body_font.render("AI Slot", True, ui.WHITE)
            content.blit(slot_heading, (slot_rows[0].x, slot_rows[0].y - 30))
            
            mouse_pos = pygame.mouse.get_pos()
            content_mouse_pos = (
                mouse_pos[0] - layout.content_rect.x,
                mouse_pos[1] - layout.content_rect.y + trainee_scroll_y,
            )
            
            for index, (row, field) in enumerate(zip(slot_rows, slot_fields)):
                pygame.draw.rect(content, ui.SLIDER_BG, row)
                circle_center = (row.x + 18, row.centery)
                pygame.draw.circle(content, ui.WHITE, circle_center, 9, 2)
                if state.selected_slot == index + 1:
                    pygame.draw.circle(content, ui.BRIGHT_GREEN, circle_center, 5)
                number = body_font.render(str(index + 1), True, ui.WHITE)
                content.blit(
                    number,
                    number.get_rect(midleft=(row.x + 36, row.centery)),
                )
                field.draw(content, body_font)
                delete_buttons[index].draw(content, body_font, content_mouse_pos)

            source = pygame.Rect(
                0,
                trainee_scroll_y,
                layout.content_rect.width,
                layout.content_rect.height,
            )
            screen.blit(content, layout.content_rect, source)
        elif state.active_tab == "rewards":
            content = pygame.Surface(
                (CONTROL_WIDTH, max(rewards_content_height, layout.content_rect.height)), pygame.SRCALPHA
            )
            content.fill((0, 0, 0, 155))
            mouse_pos = pygame.mouse.get_pos()
            content_mouse_pos = (
                mouse_pos[0] - layout.content_rect.x,
                mouse_pos[1] - layout.content_rect.y + rewards_scroll_y,
            )

            for slider in reward_sliders:
                slider.draw(content, rewards_font, content_mouse_pos)

            source = pygame.Rect(
                0,
                rewards_scroll_y,
                layout.content_rect.width,
                layout.content_rect.height,
            )
            screen.blit(content, layout.content_rect, source)
            _draw_scrollbar(
                screen,
                layout.content_rect,
                rewards_content_height,
                rewards_scroll_y,
            )
        elif state.active_tab == "opponent":
            content = pygame.Surface(layout.content_rect.size, pygame.SRCALPHA)
            content.fill((0, 0, 0, 155))
            mouse_pos = pygame.mouse.get_pos()
            content_mouse_pos = (
                mouse_pos[0] - layout.content_rect.x,
                mouse_pos[1] - layout.content_rect.y,
            )
            for i, panel in enumerate(opponent_panels):
                hovered = panel.collidepoint(content_mouse_pos)
                enabled = not (i > 0 and not state.simple_behavior_controls_enabled)
                _draw_group_panel(content, panel, hovered, enabled)
            for button in opponent_mode_buttons:
                button.draw(content, opponent_font, content_mouse_pos)
            for checkbox in movement_checkboxes:
                checkbox.draw(content, opponent_font, content_mouse_pos)
            for button in turning_buttons:
                button.draw(content, opponent_font, content_mouse_pos)
            screen.blit(content, layout.content_rect)
        else:
            panel = pygame.Surface(layout.content_rect.size, pygame.SRCALPHA)
            panel.fill((0, 0, 0, 155))
            screen.blit(panel, layout.content_rect)
            for slider in regimen_sliders:
                slider.draw(screen, body_font)

        # Draw inactive tabs behind the content window border
        if not trainee_tab.active: trainee_tab.draw(screen, tab_font)
        if not opponent_tab.active: opponent_tab.draw(screen, tab_font)
        if not rewards_tab.active: rewards_tab.draw(screen, tab_font)
        if not regimen_tab.active: regimen_tab.draw(screen, tab_font)

        pygame.draw.rect(screen, ui.BLACK, layout.content_rect, 2)

        # Draw active tab in front to merge seamlessly
        if trainee_tab.active: trainee_tab.draw(screen, tab_font)
        if opponent_tab.active: opponent_tab.draw(screen, tab_font)
        if rewards_tab.active: rewards_tab.draw(screen, tab_font)
        if regimen_tab.active: regimen_tab.draw(screen, tab_font)

        display_checkbox.draw(screen, body_font)
        start_stop_button.draw(screen, body_font)
        back_button.draw(screen, body_font)
        _draw_hud_placeholders(screen, layout.hud_rects, small_font)

        if (
            state.active_tab == "trainee"
            and state.selected_ship is not None
            and ship_picker is None
        ):
            visible_tile = ship_tile.move(
                layout.content_rect.x,
                layout.content_rect.y - trainee_scroll_y,
            )
            if (
                layout.content_rect.contains(visible_tile)
                and visible_tile.collidepoint(pygame.mouse.get_pos())
            ):
                definition = SHIP_DEFINITIONS[state.selected_ship]
                label = ui.format_ship_tooltip(
                    state.selected_ship,
                    definition.ship_type,
                    include_cost=False,
                )
                ui.draw_ship_tooltip(
                    screen,
                    picker_tooltip_font,
                    label,
                    pygame.mouse.get_pos(),
                    visible_tile,
                )

        if ship_picker is not None:
            shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, MODAL_SHADE_ALPHA))
            screen.blit(shade, (0, 0))
            ship_picker.draw(screen, picker_title_font, picker_tooltip_font)

        pygame.display.flip()
