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
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_BUNDLED,
    SLOT_EMPTY,
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    metadata_from_state,
)


REWARD_VALUES = tuple(
    [-10.24, -5.12, -2.56, -1.28, -0.64, -0.32, -0.16, -0.08, -0.04, -0.02, -0.01]
    + [0.0]
    + [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
)

REWARD_LABELS = (
    "Point A1 at enemy",
    "Get in A1 weapon range",
    "Spawn A1 object",
    "Point A2 at enemy",
    "Get in A2 weapon range",
    "Spawn A2 object",
    "Get to high speed",
    "Enemy loses crew",
    "Debuff enemy",
    "Kill enemy object",
    "Kill enemy",
    "Gain crew",
    "Gain battery",
    "Lose crew",
    "Lose battery",
    "Battery at zero",
    "Get debuffed",
    "Die",
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

REPLAY_BUFFER_SIZE_VALUES = (1000, 2000, 5000, 10000, 20000, 50000)
ROUNDS_PER_BATCH_VALUES = (1, 2, 5, 10, 20, 50)
PREDICTION_WINDOW_VALUES = (24, 48, 120, 240)
MATCH_TIME_LIMIT_VALUES = (240, 480, 1200, 2400, 4800, 12000)
LEARNING_RATE_VALUES = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)
EPSILON_VALUES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.1, 0.2, 0.5)
HIDDEN_LAYER_SIZE_VALUES = (32, 64, 128, 256, 512, 1024, 2048)
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
    rounds_per_batch: int = 10
    prediction_window: int = 120
    match_time_limit: int = 2400
    learning_rate: float = 0.001
    epsilon: float = 0.1
    hidden_layer_size: int = 128
    hidden_layer_count: int = 2
    replay_buffer_size: int = 10000
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


def _wrap_text(text, font, max_width):
    lines = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_notice(screen, notice, font):
    rendered = font.render(notice.text, True, ui.WHITE)
    padding = 14
    rect = rendered.get_rect()
    rect.width += padding * 2
    rect.height += padding
    rect.center = (const.SCREEN_WIDTH // 2, const.SCREEN_HEIGHT - 68)
    alpha = int(220 * min(1.0, max(0.0, notice.remaining_seconds / 0.75)))
    surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(surface, (0, 0, 0, alpha), surface.get_rect(), border_radius=5)
    pygame.draw.rect(surface, (*ui.LIGHT_GREY, alpha), surface.get_rect(), 1, border_radius=5)
    surface.blit(rendered, rendered.get_rect(center=surface.get_rect().center))
    screen.blit(surface, rect)


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
        self.enabled = True
        self.text_color = ui.WHITE

    def handle_event(self, event):
        if not self.enabled:
            self.active = False
            return
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
            ui.BRIGHT_GREEN if self.active and self.enabled else ui.LIGHT_GREY,
            self.rect,
            2,
        )
        text = font.render(self.text, True, self.text_color)
        clip = self.rect.inflate(-12, -4)
        surface.set_clip(clip)
        text_rect = text.get_rect(midleft=(self.rect.left + 6, self.rect.centery))
        surface.blit(text, text_rect)
        if self.active and self.enabled and pygame.time.get_ticks() % 1000 < 500:
            cursor_x = text_rect.right + 2
            pygame.draw.line(surface, ui.WHITE, (cursor_x, self.rect.centery - font.get_linesize() // 2 + 2), (cursor_x, self.rect.centery + font.get_linesize() // 2 - 2), 2)
        surface.set_clip(None)


@dataclass
class TrainingNotice:
    text: str
    remaining_seconds: float = 2.5


class ConfirmationPrompt:
    def __init__(self, text, on_confirm):
        self.text = text
        self.on_confirm = on_confirm
        width = min(640, const.SCREEN_WIDTH - 160)
        height = 210
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (const.SCREEN_WIDTH // 2, const.SCREEN_HEIGHT // 2)
        button_width = 170
        button_height = 48
        gap = 18
        top = self.rect.bottom - 68
        self.yes_button = ui_button.Button(
            self.rect.centerx - button_width - gap // 2,
            top,
            button_width,
            button_height,
            "Yes",
            self.confirm,
            ui.OK_GREEN,
            ui.OK_GREEN_HI,
        )
        self.no_button = ui_button.Button(
            self.rect.centerx + gap // 2,
            top,
            button_width,
            button_height,
            "No",
            self.cancel,
            ui.CAN_RED,
            ui.CAN_RED_HI,
        )
        self.done = False

    def confirm(self):
        self.on_confirm()
        self.done = True

    def cancel(self):
        self.done = True

    def handle_event(self, event, sound_manager=None):
        self.yes_button.handle_event(event, sound_manager)
        self.no_button.handle_event(event, sound_manager)

    def draw(self, screen, font, button_font):
        shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, MODAL_SHADE_ALPHA))
        screen.blit(shade, (0, 0))
        pygame.draw.rect(screen, ui.BLACK, self.rect)
        pygame.draw.rect(screen, ui.WHITE, self.rect, 2)
        lines = _wrap_text(self.text, font, self.rect.width - 40)
        y = self.rect.top + 34
        for line in lines:
            rendered = font.render(line, True, ui.WHITE)
            screen.blit(rendered, rendered.get_rect(center=(self.rect.centerx, y)))
            y += font.get_linesize()
        self.yes_button.draw(screen, button_font)
        self.no_button.draw(screen, button_font)


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
    model_repository = TrainingModelRepository(
        const.DEFAULT_MODELS_PATH,
        const.MODELS_PATH,
    )
    slot_models = [
        TrainingModelSlot("", slot, SLOT_EMPTY)
        for slot in range(1, MODEL_SLOT_COUNT + 1)
    ]
    confirmation_prompt = [None]
    notice = [None]
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
    available_height = CONTENT_VIEW_HEIGHT - 16
    step = min(34, (available_height - 30) // max(1, len(REWARD_LABELS) - 1)) if len(REWARD_LABELS) > 1 else 34
    rewards_font = largest_fitting_font(
        REWARD_LABELS,
        270,
        max_height=min(26, step),
        maximum=min(24, step - 4),
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
            (12, rewards_top + index * step, CONTROL_WIDTH - 24, min(30, step - 2)), label
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
    regimen_spacing = 64
    regimen_sliders = (
        ui_slider.Slider(
            regimen_left,
            regimen_top,
            regimen_width,
            REPLAY_BUFFER_SIZE_VALUES[0],
            REPLAY_BUFFER_SIZE_VALUES[-1],
            state.replay_buffer_size,
            "Replay Buffer Size",
            is_int=True,
            values=REPLAY_BUFFER_SIZE_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + regimen_spacing,
            regimen_width,
            ROUNDS_PER_BATCH_VALUES[0],
            ROUNDS_PER_BATCH_VALUES[-1],
            state.rounds_per_batch,
            "Rounds per batch",
            is_int=True,
            values=ROUNDS_PER_BATCH_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 2 * regimen_spacing,
            regimen_width,
            PREDICTION_WINDOW_VALUES[0],
            PREDICTION_WINDOW_VALUES[-1],
            state.prediction_window,
            "Prediction Window (Frames)",
            is_int=True,
            values=PREDICTION_WINDOW_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 3 * regimen_spacing,
            regimen_width,
            MATCH_TIME_LIMIT_VALUES[0],
            MATCH_TIME_LIMIT_VALUES[-1],
            state.match_time_limit,
            "Match Time Limit (frames)",
            is_int=True,
            values=MATCH_TIME_LIMIT_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 4 * regimen_spacing,
            regimen_width,
            LEARNING_RATE_VALUES[0],
            LEARNING_RATE_VALUES[-1],
            state.learning_rate,
            "Learning rate",
            step=0.0001,
            values=LEARNING_RATE_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 5 * regimen_spacing,
            regimen_width,
            EPSILON_VALUES[0],
            EPSILON_VALUES[-1],
            state.epsilon,
            "Epsilon",
            step=0.0001,
            values=EPSILON_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 6 * regimen_spacing,
            regimen_width,
            HIDDEN_LAYER_SIZE_VALUES[0],
            HIDDEN_LAYER_SIZE_VALUES[-1],
            state.hidden_layer_size,
            "Hidden layer size",
            is_int=True,
            values=HIDDEN_LAYER_SIZE_VALUES,
            height=58,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 7 * regimen_spacing,
            regimen_width,
            HIDDEN_LAYER_COUNT_VALUES[0],
            HIDDEN_LAYER_COUNT_VALUES[-1],
            state.hidden_layer_count,
            "Hidden layer count",
            is_int=True,
            values=HIDDEN_LAYER_COUNT_VALUES,
            height=58,
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

    def architecture_metadata():
        return {
            "hidden_layer_size": state.hidden_layer_size,
            "hidden_layer_count": state.hidden_layer_count,
        }

    def training_metadata():
        return {
            "opponent": {
                "mode": state.opponent_mode,
                "movement_behaviors": sorted(state.movement_behaviors),
                "turning_behavior": state.turning_behavior,
            },
            "rewards": dict(state.rewards),
            "regimen": {
                "replay_buffer_size": state.replay_buffer_size,
                "rounds_per_batch": state.rounds_per_batch,
                "prediction_window": state.prediction_window,
                "match_time_limit": state.match_time_limit,
                "learning_rate": state.learning_rate,
                "epsilon": state.epsilon,
            },
        }

    def selected_model_slot():
        if state.selected_ship is None:
            return None
        return slot_models[state.selected_slot - 1]

    def show_notice(text):
        notice[0] = TrainingNotice(text)

    def refresh_slot_controls():
        if state.selected_ship is None:
            for field, delete_button in zip(slot_fields, delete_buttons):
                field.text = ""
                field.enabled = False
                field.text_color = ui.GREY
                delete_button.enabled = False
            return

        slot_models[:] = model_repository.slots_for_ship(state.selected_ship)
        for field, delete_button, model_slot in zip(slot_fields, delete_buttons, slot_models):
            field.text = model_slot.description
            if model_slot.source == SLOT_BUNDLED:
                field.enabled = False
                field.text_color = (80, 160, 255)
                delete_button.enabled = False
            elif model_slot.source == SLOT_USER:
                field.enabled = True
                field.text_color = ui.BRIGHT_GREEN
                delete_button.enabled = True
            else:
                field.enabled = True
                field.text_color = ui.WHITE
                delete_button.enabled = False

    def update_field_colors():
        if state.selected_ship is None:
            return
        for field, model_slot in zip(slot_fields, slot_models):
            if model_slot.source == SLOT_BUNDLED:
                field.text = "Default"
                field.text_color = (80, 160, 255)
            elif model_slot.source == SLOT_USER:
                field.text_color = (
                    ui.BRIGHT_GREEN
                    if field.text == model_slot.description
                    else ui.CAN_RED
                )
            else:
                field.text_color = ui.CAN_RED if field.text else ui.WHITE

    def set_selected_ship(ship):
        state.selected_ship = ship
        state.selected_slot = 1
        refresh_slot_controls()

    def clear_selected_ship():
        state.selected_ship = None
        state.selected_slot = 1
        slot_models[:] = [
            TrainingModelSlot("", slot, SLOT_EMPTY)
            for slot in range(1, MODEL_SLOT_COUNT + 1)
        ]
        refresh_slot_controls()

    def persist_selected_model():
        model_slot = selected_model_slot()
        if state.selected_ship is None or model_slot is None or model_slot.is_bundled:
            return
        metadata = metadata_from_state(
            ship=state.selected_ship,
            slot=state.selected_slot,
            description=slot_fields[state.selected_slot - 1].text,
            architecture=architecture_metadata(),
            training=training_metadata(),
        )
        model_repository.create_or_update_user_model(metadata)
        refresh_slot_controls()

    def changed_training_groups(old_training, new_training):
        return [
            name
            for name in ("opponent", "rewards", "regimen")
            if old_training.get(name) != new_training.get(name)
        ]

    def describe_model(model_slot):
        description = slot_fields[model_slot.slot - 1].text
        suffix = f" ({description})" if description else ""
        return f"{model_slot.ship} Model {model_slot.slot:02d}{suffix}"

    def request_delete(slot):
        model_slot = slot_models[slot - 1]
        if not model_slot.is_user:
            return

        def delete_model():
            model_repository.delete_user_model(model_slot.ship, model_slot.slot)
            refresh_slot_controls()
            show_notice(f"Deleted {describe_model(model_slot)}")

        confirmation_prompt[0] = ConfirmationPrompt(
            f"Do you want to delete {describe_model(model_slot)}?",
            delete_model,
        )

    def start_selected_model():
        model_slot = selected_model_slot()
        if state.selected_ship is None or model_slot is None or model_slot.is_bundled:
            return

        new_architecture = architecture_metadata()
        new_training = training_metadata()
        current_description = slot_fields[state.selected_slot - 1].text

        if model_slot.source == SLOT_EMPTY:
            persist_selected_model()
            show_notice(f"Created {describe_model(selected_model_slot())}")
            return

        metadata = model_slot.metadata if isinstance(model_slot.metadata, dict) else {}
        old_architecture = metadata.get("architecture", {})
        old_training = metadata.get("training", {})
        old_description = metadata.get("description", model_slot.description)

        if old_architecture and old_architecture != new_architecture:
            confirmation_prompt[0] = ConfirmationPrompt(
                f"Do you want to overwrite {describe_model(model_slot)}?",
                lambda: (persist_selected_model(), show_notice(f"Updated {describe_model(model_slot)}")),
            )
            return

        changed_groups = (
            changed_training_groups(old_training, new_training)
            if old_training
            else []
        )
        if changed_groups:
            if len(changed_groups) == 1:
                changed_summary = changed_groups[0]
            else:
                changed_summary = ", ".join(changed_groups[:-1]) + f" and {changed_groups[-1]}"
            confirmation_prompt[0] = ConfirmationPrompt(
                f"Do you want to run {describe_model(model_slot)} with new {changed_summary} settings?",
                lambda: (persist_selected_model(), show_notice(f"Updated {describe_model(model_slot)}")),
            )
            return

        if current_description != old_description:
            persist_selected_model()
            show_notice(
                f'Model description of {state.selected_ship} {state.selected_slot:02d} '
                f'changed from "{old_description}" to "{current_description}"'
            )
            return

        persist_selected_model()
        show_notice(f"No changes for {describe_model(model_slot)}")

    action_gap = 10
    action_width = (CONTROL_WIDTH - 2 * TAB_MARGIN - action_gap) // 2
    start_stop_button = ui_button.Button(
        TAB_MARGIN,
        ACTION_TOP,
        action_width,
        FOOTER_CONTROL_HEIGHT,
        "Start",
        start_selected_model,
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
    for index, delete_button in enumerate(delete_buttons):
        delete_button.callback = lambda slot=index + 1: request_delete(slot)
    refresh_slot_controls()

    while not exited[0]:
        elapsed_seconds = clock.tick()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if confirmation_prompt[0] is not None:
                confirmation_prompt[0].handle_event(event, menu_sound_manager)
                if confirmation_prompt[0].done:
                    confirmation_prompt[0] = None
                continue

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
                            set_selected_ship(selected[0])
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
                            if state.selected_ship is None:
                                ship_picker = ShipPickerModal(
                                    1,
                                    None,
                                    SHIP_DEFINITIONS,
                                    source_sprites,
                                    title_label="Select Trainee Ship",
                                )
                            else:
                                clear_selected_ship()
                            if menu_sound_manager:
                                menu_sound_manager.play_sound("menu")
                        for index, row in enumerate(slot_rows):
                            if state.selected_ship is not None and row.collidepoint(translated.pos):
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
        state.replay_buffer_size = int(regimen_sliders[0].value)
        state.rounds_per_batch = int(regimen_sliders[1].value)
        state.prediction_window = int(regimen_sliders[2].value)
        state.match_time_limit = int(regimen_sliders[3].value)
        state.learning_rate = regimen_sliders[4].value
        state.epsilon = regimen_sliders[5].value
        state.hidden_layer_size = int(regimen_sliders[6].value)
        state.hidden_layer_count = int(regimen_sliders[7].value)
        controls_enabled = state.simple_behavior_controls_enabled
        for checkbox in movement_checkboxes:
            checkbox.enabled = controls_enabled
        for button in turning_buttons:
            button.enabled = controls_enabled

        update_field_colors()
        selected_slot = selected_model_slot()
        start_stop_button.enabled = (
            state.selected_ship is not None
            and selected_slot is not None
            and not selected_slot.is_bundled
        )
        start_stop_button.text = "Start"
        start_stop_button.bg_color = ui.OK_GREEN
        start_stop_button.hover_color = ui.OK_GREEN_HI

        if notice[0] is not None:
            notice[0].remaining_seconds -= elapsed_seconds
            if notice[0].remaining_seconds <= 0:
                notice[0] = None

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
                enabled = state.selected_ship is not None
                pygame.draw.rect(content, ui.SLIDER_BG if enabled else ui.DARK_GREY, row)
                circle_center = (row.x + 18, row.centery)
                circle_color = ui.WHITE if enabled else ui.GREY
                pygame.draw.circle(content, circle_color, circle_center, 9, 2)
                if enabled and state.selected_slot == index + 1:
                    pygame.draw.circle(content, ui.BRIGHT_GREEN, circle_center, 5)
                number = body_font.render(str(index + 1), True, circle_color)
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

        if notice[0] is not None:
            _draw_notice(screen, notice[0], small_font)

        if confirmation_prompt[0] is not None:
            confirmation_prompt[0].draw(screen, arena_font, body_font)

        pygame.display.flip()
