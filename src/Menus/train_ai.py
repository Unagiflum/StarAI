"""UI shell for configuring future AI training sessions."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pygame

import src.const as const
from src.Battle.battle_draw import (
    BattleDrawController,
    BattleDrawLayout,
    BattleDrawOptions,
    DisplayStarField,
    HUD_BOTTOM_PADDING,
    MARINE_REGION_HEIGHT,
    VIEWPORT_SIZE,
)
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
    model_architecture_metadata,
)
from src.training.orchestration import TrainingOrchestrationConfig
from src.training.rewards import LEGACY_REWARD_ALIASES, REWARD_COMPONENTS
from src.training.session import (
    TrainingSession,
    TrainingSessionError,
    validate_model_metadata,
)


REWARD_VALUES = tuple(
    [-40.96, -20.48, -10.24, -5.12, -2.56, -1.28, -0.64, -0.32, -0.16, -0.08, -0.04, -0.02, -0.01]
    + [0.0]
    + [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96]
)

REWARD_LABELS = REWARD_COMPONENTS

SIMPLE_ACTIVITY_VALUES = tuple(float(value) for value in range(0, 101, 5))

REPLAY_BUFFER_SIZE_VALUES = (1000, 2000, 5000, 10000, 20000, 50000)
ROUNDS_PER_BATCH_VALUES = (1, 2, 5, 10, 20, 50)
BATCH_GROUPING_VALUES = (50, 100, 250, 500, 1000)
MATCH_TIME_LIMIT_VALUES = (240, 480, 1200, 2400, 4800, 12000)
MINIBATCH_SIZE_VALUES = (16, 32, 64, 128, 256)
REPLAY_UPDATES_PER_BATCH_VALUES = (100, 200, 500, 1000, 2000)
LEARNING_RATE_VALUES = (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01)
EPSILON_VALUES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.1, 0.2, 0.5)
GAMMA_VALUES = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
HIDDEN_LAYER_SIZE_VALUES = (32, 64, 128, 256, 512, 1024, 2048)
HIDDEN_LAYER_COUNT_VALUES = (1, 2, 4, 8, 16)
REGIMEN_REPLAY_BUFFER_INDEX = 0
REGIMEN_ROUNDS_PER_BATCH_INDEX = 1
REGIMEN_BATCH_GROUPING_INDEX = 2
REGIMEN_MATCH_TIME_LIMIT_INDEX = 3
REGIMEN_MINIBATCH_SIZE_INDEX = 4
REGIMEN_REPLAY_UPDATES_INDEX = 5
REGIMEN_LEARNING_RATE_INDEX = 6
REGIMEN_EPSILON_INDEX = 7
REGIMEN_GAMMA_INDEX = 8
REGIMEN_HIDDEN_LAYER_SIZE_INDEX = 9
REGIMEN_HIDDEN_LAYER_COUNT_INDEX = 10
CURRENT_BATCH_STATE_WIDTH = len("stopped (stopping)")
CURRENT_BATCH_BATCH_WIDTH = 6
CURRENT_BATCH_ROUND_WIDTH = 4
CURRENT_BATCH_REPLAY_WIDTH = 6
CURRENT_BATCH_RETURN_WIDTH = 9
CURRENT_BATCH_LOSS_WIDTH = 10
CURRENT_BATCH_REWARD_NAME_WIDTH = max((len(label) for label in REWARD_LABELS), default=0)
CURRENT_BATCH_REWARD_VALUE_WIDTH = 8

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
TRAINING_HUD_HEIGHT = MARINE_REGION_HEIGHT + VIEWPORT_SIZE + HUD_BOTTOM_PADDING
HUD_TOP = const.SCREEN_HEIGHT - TRAINING_HUD_HEIGHT
HUD_BOTTOM_MARGIN = 0
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
    forward_activity: float = 0.0
    a1_activity: float = 0.0
    a2_activity: float = 0.0
    face_opponent_activity: float = 0.0
    rounds_per_batch: int = 10
    batch_grouping: int = 250
    match_time_limit: int = 2400
    learning_rate: float = 0.001
    epsilon: float = 0.1
    gamma: float = 0.99
    minibatch_size: int = 32
    replay_updates_per_batch: int = 100
    hidden_layer_size: int = 128
    hidden_layer_count: int = 2
    replay_buffer_size: int = 10000
    display_on: bool = False
    running: bool = False
    loaded_ship: str | None = None
    loaded_slot: int | None = None
    loaded_architecture: dict | None = None
    loaded_training: dict | None = None

    @property
    def simple_behavior_controls_enabled(self):
        return not self.running


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


def _set_slider_value(slider, value):
    if slider.values is not None and value not in slider.values:
        return False
    slider.value = value
    if hasattr(slider, "value_to_position"):
        slider.handle_x = slider.value_to_position(value)
    return True


def _set_checkbox_value(checkbox, value):
    checkbox.is_checked = bool(value)


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
        self.enabled = True
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
        if not self.enabled:
            self.dragging = False
            return
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
        row.fill(ui.SLIDER_BG_HI if hovered and self.enabled else ui.SLIDER_BG)
        surface.blit(row, self.rect)

        label_color = ui.WHITE if self.enabled else ui.GREY
        label = font.render(self.label, True, label_color)
        surface.blit(
            label,
            label.get_rect(midleft=(self.rect.left + 8, self.rect.centery)),
        )
        pygame.draw.rect(surface, ui.SLIDER_LINE, self.line_rect)
        pygame.draw.circle(
            surface, ui.HANDLE_COLOR if self.enabled else ui.GREY, (self.handle_x, self.line_rect.centery), 7
        )
        value = font.render(_format_reward(self.value), True, label_color)
        surface.blit(
            value,
            value.get_rect(midright=(self.rect.right - 8, self.rect.centery)),
        )


class TextField:
    """Small single-line editor used for AI-slot descriptions."""

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


class TrainingBatchLogBox:
    """Scrollable selectable text view for completed-batch summaries."""

    def __init__(self):
        self.lines: tuple[str, ...] = ()
        self.scroll_line = 0
        self.visible_count = 1
        self.dragging = False
        self.selection_anchor: int | None = None
        self.selection_focus: int | None = None

    def set_lines(self, lines):
        old_max = self._max_scroll_line()
        was_at_bottom = self.scroll_line >= old_max
        self.lines = tuple(lines)
        if was_at_bottom:
            self.scroll_line = self._max_scroll_line()
        else:
            self._clamp_scroll_line()

    @property
    def selected_text(self):
        if self.selection_anchor is None or self.selection_focus is None:
            return ""
        first = min(self.selection_anchor, self.selection_focus)
        last = max(self.selection_anchor, self.selection_focus)
        return "\n".join(self.lines[first:last + 1])

    def handle_event(self, event, rect, font):
        if event.type == pygame.MOUSEWHEEL:
            mouse_pos = getattr(event, "pos", pygame.mouse.get_pos())
            if rect.collidepoint(mouse_pos):
                self._update_visible_count(rect, font)
                self._scroll_lines(-event.y * 3)
            return

        if event.type == pygame.MOUSEBUTTONDOWN and rect.collidepoint(event.pos):
            if event.button in (4, 5):
                direction = -1 if event.button == 4 else 1
                self._update_visible_count(rect, font)
                self._scroll_lines(direction * 3)
            elif event.button == 1:
                line_index = self._line_at_pos(event.pos, rect, font)
                if line_index is not None:
                    self.dragging = True
                    self.selection_anchor = line_index
                    self.selection_focus = line_index
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            line_index = self._line_at_pos(event.pos, rect, font)
            if line_index is not None:
                self.selection_focus = line_index
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
            modifiers = pygame.key.get_mods()
            if modifiers & (pygame.KMOD_CTRL | pygame.KMOD_META):
                self._copy_selected_text()

    def draw(self, surface, rect, font):
        pygame.draw.rect(surface, ui.BLACK, rect)
        pygame.draw.rect(surface, ui.GREY, rect, 2)
        line_height = font.get_linesize()
        self._update_visible_count(rect, font)
        start = self.scroll_line
        selected = self._selected_range()
        y = rect.top + 8
        clip = rect.inflate(-10, -8)
        surface.set_clip(clip)
        for index, line in enumerate(self.lines[start:start + self.visible_count], start=start):
            if selected is not None and selected[0] <= index <= selected[1]:
                highlight = pygame.Rect(rect.left + 6, y, rect.width - 12, line_height)
                pygame.draw.rect(surface, (55, 80, 120), highlight)
            rendered = font.render(line, True, ui.WHITE)
            surface.blit(rendered, (rect.left + 8, y))
            y += line_height
        surface.set_clip(None)
        _draw_scrollbar(surface, rect, max(len(self.lines), self.visible_count) * line_height, start * line_height)

    def _line_at_pos(self, pos, rect, font):
        if not self.lines:
            return None
        self._update_visible_count(rect, font)
        line_height = font.get_linesize()
        start = self.scroll_line
        offset = (pos[1] - rect.top - 8) // line_height
        if offset < 0:
            return None
        return max(0, min(len(self.lines) - 1, start + int(offset)))

    def _update_visible_count(self, rect, font):
        line_height = font.get_linesize()
        self.visible_count = max(1, (rect.height - 16) // line_height)
        self._clamp_scroll_line()

    def _max_scroll_line(self):
        return max(0, len(self.lines) - self.visible_count)

    def _clamp_scroll_line(self):
        self.scroll_line = max(0, min(self.scroll_line, self._max_scroll_line()))

    def _scroll_lines(self, amount):
        self.scroll_line += amount
        self._clamp_scroll_line()

    def _selected_range(self):
        if self.selection_anchor is None or self.selection_focus is None:
            return None
        return (
            min(self.selection_anchor, self.selection_focus),
            max(self.selection_anchor, self.selection_focus),
        )

    def _copy_selected_text(self):
        text = self.selected_text
        if not text:
            return
        try:
            pygame.scrap.init()
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        except pygame.error:
            pass


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


def _draw_training_status(screen, rect, status, font, small_font):
    pygame.draw.rect(screen, ui.BLACK, rect)
    pygame.draw.rect(screen, ui.GREY, rect, 2)
    lines = (
        "Training running" if status.running else "Training stopped",
        f"Batch {status.completed_batches + 1} | Round {status.current_round}/{status.total_rounds}",
        f"Opponent: {status.current_opponent or '-'}",
        f"Replay: {status.replay_size}",
        f"Return: {status.weighted_total_return:.2f}",
        f"Loss: {status.recent_loss:.4f}" if status.recent_loss is not None else "Loss: -",
    )
    y = rect.top + 40
    for index, line in enumerate(lines):
        rendered = (font if index == 0 else small_font).render(line, True, ui.WHITE)
        screen.blit(rendered, rendered.get_rect(midtop=(rect.centerx, y)))
        y += rendered.get_height() + 14
    component_lines = [
        f"{name}: {status.component_totals.get(name, 0.0):.4f}"
        for name in REWARD_LABELS
    ]
    for line in component_lines:
        rendered = small_font.render(line, True, ui.LIGHT_GREY)
        screen.blit(rendered, rendered.get_rect(midtop=(rect.centerx, y)))
        y += rendered.get_height() + 6


def _current_batch_console_lines(status):
    state_label = "running" if status.running else "stopped"
    if status.stopping:
        state_label += " (stopping)"
    total_round_width = max(
        CURRENT_BATCH_ROUND_WIDTH,
        len(str(status.current_round)),
        len(str(status.total_rounds)),
    )
    lines = [
        "Current batch",
        f"{'State:':<10}{state_label:<{CURRENT_BATCH_STATE_WIDTH}}",
        f"{'Batch:':<10}{status.completed_batches + 1:>{CURRENT_BATCH_BATCH_WIDTH}d}",
        f"{'Round:':<10}{status.current_round:>{total_round_width}d}/{status.total_rounds:>{total_round_width}d}",
        f"{'Opponent:':<10}{status.current_opponent or '-':<{CURRENT_BATCH_REWARD_NAME_WIDTH}}",
        f"{'Replay:':<10}{status.replay_size:>{CURRENT_BATCH_REPLAY_WIDTH}d}",
        f"{'Return:':<10}{status.weighted_total_return:>{CURRENT_BATCH_RETURN_WIDTH}.2f}",
        f"{'Loss:':<10}{status.recent_loss:>{CURRENT_BATCH_LOSS_WIDTH}.4f}"
        if status.recent_loss is not None
        else f"{'Loss:':<10}{'-':>{CURRENT_BATCH_LOSS_WIDTH}}",
        "",
    ]
    
    ship_name = status.previous_opponent or "-"
    col2_header = f" {ship_name[:10]:>10} "
    
    if status.batch_component_totals:
        col3_header = f" Batch {status.completed_batches}"
    else:
        col3_header = " Batch -"
    col3_width = max(9, len(col3_header))
    
    lines.append(f"{'Reward components':<{CURRENT_BATCH_REWARD_NAME_WIDTH}}|{col2_header}|{col3_header:>{col3_width}}")
    
    for name in REWARD_LABELS:
        col1 = f"{name:<{CURRENT_BATCH_REWARD_NAME_WIDTH}}"
        val2 = status.component_totals.get(name, 0.0)
        col2 = f"{val2:>11.4f} "
        
        if status.batch_component_totals:
            val3 = status.batch_component_totals.get(name, 0.0)
            col3 = f"{val3:>{col3_width}.4f}"
        else:
            col3 = f"{'-':>{col3_width}}"
            
        lines.append(f"{col1}|{col2}|{col3}")
        
    return tuple(lines)


def _display_off_console_lines(status, log_lines):
    lines = []
    if log_lines:
        lines.append("Completed batches")
        lines.extend(log_lines)
    elif status is None:
        lines.append("Completed batch summaries will appear here.")
    elif status.running:
        lines.append("Waiting for the first completed batch summary...")
    else:
        lines.append("No completed batch summaries yet.")
    if status is not None:
        lines.extend(("", *_current_batch_console_lines(status)))
    return tuple(lines)


def _training_battle_view_args(status):
    battle_view = status.battle_view if status is not None else None
    if not battle_view:
        return {
            "game_objects": (),
            "border_color": ui.GREY,
            "camera_targets": (),
            "entry_state": None,
            "frame_id": 0,
            "original_ships": (),
        }
    return battle_view


def _draw_training_battle(
    screen,
    rect,
    status,
    star_field_renderer,
    battle_draw_controller=None,
):
    battle_view = _training_battle_view_args(status)
    controller = battle_draw_controller or BattleDrawController()
    controller.draw(
        screen,
        battle_view["game_objects"],
        BattleDrawLayout(
            arena_rect=pygame.Rect(rect),
            player1_hud_rect=None,
            player2_hud_rect=None,
        ),
        battle_view["border_color"],
        star_field_renderer,
        camera_targets=battle_view.get("camera_targets"),
        entry_state=battle_view.get("entry_state"),
        frame_id=battle_view.get("frame_id", 0),
        original_ships=battle_view.get("original_ships"),
        options=BattleDrawOptions(draw_huds=False),
    )


def _draw_training_huds(
    screen,
    hud_rects,
    status,
    star_field_renderer,
    battle_draw_controller=None,
):
    battle_view = _training_battle_view_args(status)
    controller = battle_draw_controller or BattleDrawController()
    controller.draw(
        screen,
        battle_view["game_objects"],
        BattleDrawLayout(
            arena_rect=pygame.Rect(0, 0, 0, 0),
            player1_hud_rect=pygame.Rect(hud_rects[0]),
            player2_hud_rect=pygame.Rect(hud_rects[1]),
        ),
        battle_view["border_color"],
        star_field_renderer,
        camera_targets=battle_view.get("camera_targets"),
        entry_state=battle_view.get("entry_state"),
        frame_id=battle_view.get("frame_id", 0),
        original_ships=battle_view.get("original_ships"),
        options=BattleDrawOptions(draw_arena=False),
    )


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


def training_config_from_state(state: TrainingUIState) -> TrainingOrchestrationConfig:
    return TrainingOrchestrationConfig(
        trainee_ship=str(state.selected_ship),
        reward_weights=dict(state.rewards),
        opponent_mode=state.opponent_mode,
        forward_activity=state.forward_activity,
        a1_activity=state.a1_activity,
        a2_activity=state.a2_activity,
        face_opponent_activity=state.face_opponent_activity,
        rounds_per_batch=state.rounds_per_batch,
        gamma=state.gamma,
        match_time_limit=state.match_time_limit,
        replay_capacity=state.replay_buffer_size,
        learning_rate=state.learning_rate,
        epsilon=state.epsilon,
        hidden_layer_width=state.hidden_layer_size,
        hidden_layer_count=state.hidden_layer_count,
        minibatch_size=state.minibatch_size,
        replay_updates_per_batch=state.replay_updates_per_batch,
        display_on=state.display_on,
    )


def _progress_for_model_update(existing_metadata, progress=None, *, reset_checkpoint=False):
    if progress is not None:
        return dict(progress)
    if reset_checkpoint:
        return {"completed_batches": 0}
    if isinstance(existing_metadata, dict):
        existing_progress = existing_metadata.get("progress", {})
        if isinstance(existing_progress, dict):
            return dict(existing_progress)
    return None


def run(screen: pygame.Surface, menu_sound_manager=None, audio_service=None):
    """Show the AI-training configuration UI without starting training yet."""
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
        (
            "Include AI opponents",
            "Simple opponents only",
            "Forward Activity: 100.0",
            "A1 Activity: 100.0",
            "A2 Activity: 100.0",
            "Face opponent: 100.0",
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
    regimen_font = largest_fitting_font(
        (
            "Replay Buffer Size: 50000",
            "Rounds per batch: 50",
            "Batch grouping: 1000",
            "Match Time Limit (frames): 12000",
            "Minibatch size: 256",
            "Updates per minibatch: 2000",
            "Learning rate: 0.0100",
            "Epsilon: 0.5000",
            "Gamma: 0.999",
            "Hidden layer size: 2048",
            "Hidden layer count: 16",
        ),
        CONTROL_WIDTH - 64,
        max_height=22,
        maximum=21,
    )
    small_font = pygame.font.SysFont(None, 24)
    arena_font = pygame.font.SysFont(None, 32)
    log_font = pygame.font.SysFont("Consolas", 11)
    picker_title_font = pygame.font.SysFont(None, int(const.SCREEN_HEIGHT * 0.042))
    picker_tooltip_font = pygame.font.SysFont(None, PICKER_TOOLTIP_FONT_SIZE)
    training_battle_renderer = DisplayStarField()
    training_battle_controller = BattleDrawController()

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
    load_button_rect = pygame.Rect(
        16,
        slot_rows[-1].bottom + 10,
        CONTROL_WIDTH - 32,
        42,
    )
    trainee_content_height = max(ship_tile.bottom, load_button_rect.bottom) + 12
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
        pygame.Rect(12, 122, CONTROL_WIDTH - 24, 222),
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
                "Include AI opponents",
                lambda: select_opponent_mode("all"),
            ),
            ui_button.RadioButton(
                20,
                64,
                CONTROL_WIDTH - 40,
                40,
                "Simple opponents only",
                lambda: select_opponent_mode("simple"),
                selected=True,
            ),
        )
    )
    simple_activity_sliders = (
        ui_slider.Slider(
            20,
            132,
            CONTROL_WIDTH - 40,
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.forward_activity,
            "Forward Activity",
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            height=44,
        ),
        ui_slider.Slider(
            20,
            182,
            CONTROL_WIDTH - 40,
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.a1_activity,
            "A1 Activity",
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            height=44,
        ),
        ui_slider.Slider(
            20,
            232,
            CONTROL_WIDTH - 40,
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.a2_activity,
            "A2 Activity",
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            height=44,
        ),
        ui_slider.Slider(
            20,
            282,
            CONTROL_WIDTH - 40,
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.face_opponent_activity,
            "Face opponent",
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            height=44,
        )
    )

    grouped_controls = (
        *opponent_mode_buttons,
        *simple_activity_sliders,
    )
    for control in grouped_controls:
        control.bg_color = (0, 0, 0, 0)
        control.hover_color = (45, 45, 45, 160)

    regimen_left = 16
    regimen_width = CONTROL_WIDTH - 32
    regimen_top = CONTENT_TOP + 14
    regimen_spacing = 44
    regimen_height = 38
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
            height=regimen_height,
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
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 2 * regimen_spacing,
            regimen_width,
            BATCH_GROUPING_VALUES[0],
            BATCH_GROUPING_VALUES[-1],
            state.batch_grouping,
            "Batch grouping",
            is_int=True,
            values=BATCH_GROUPING_VALUES,
            height=regimen_height,
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
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 4 * regimen_spacing,
            regimen_width,
            MINIBATCH_SIZE_VALUES[0],
            MINIBATCH_SIZE_VALUES[-1],
            state.minibatch_size,
            "Minibatch size",
            is_int=True,
            values=MINIBATCH_SIZE_VALUES,
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 5 * regimen_spacing,
            regimen_width,
            REPLAY_UPDATES_PER_BATCH_VALUES[0],
            REPLAY_UPDATES_PER_BATCH_VALUES[-1],
            state.replay_updates_per_batch,
            "Updates per minibatch",
            is_int=True,
            values=REPLAY_UPDATES_PER_BATCH_VALUES,
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 6 * regimen_spacing,
            regimen_width,
            LEARNING_RATE_VALUES[0],
            LEARNING_RATE_VALUES[-1],
            state.learning_rate,
            "Learning rate",
            step=0.0001,
            values=LEARNING_RATE_VALUES,
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 7 * regimen_spacing,
            regimen_width,
            EPSILON_VALUES[0],
            EPSILON_VALUES[-1],
            state.epsilon,
            "Epsilon",
            step=0.0001,
            values=EPSILON_VALUES,
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 8 * regimen_spacing,
            regimen_width,
            GAMMA_VALUES[0],
            GAMMA_VALUES[-1],
            state.gamma,
            "Gamma",
            step=0.001,
            values=GAMMA_VALUES,
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 9 * regimen_spacing,
            regimen_width,
            HIDDEN_LAYER_SIZE_VALUES[0],
            HIDDEN_LAYER_SIZE_VALUES[-1],
            state.hidden_layer_size,
            "Hidden layer size",
            is_int=True,
            values=HIDDEN_LAYER_SIZE_VALUES,
            height=regimen_height,
        ),
        ui_slider.Slider(
            regimen_left,
            regimen_top + 10 * regimen_spacing,
            regimen_width,
            HIDDEN_LAYER_COUNT_VALUES[0],
            HIDDEN_LAYER_COUNT_VALUES[-1],
            state.hidden_layer_count,
            "Hidden layer count",
            is_int=True,
            values=HIDDEN_LAYER_COUNT_VALUES,
            height=regimen_height,
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
    training_session = [None]
    last_session_running = [False]
    batch_log_box = TrainingBatchLogBox()

    def sync_state_from_ui():
        state.display_on = display_checkbox.value
        state.slot_labels[:] = [field.text for field in slot_fields]
        state.rewards.update(
            (slider.label, slider.value) for slider in reward_sliders
        )
        state.forward_activity = simple_activity_sliders[0].value
        state.a1_activity = simple_activity_sliders[1].value
        state.a2_activity = simple_activity_sliders[2].value
        state.face_opponent_activity = simple_activity_sliders[3].value
        state.replay_buffer_size = int(
            regimen_sliders[REGIMEN_REPLAY_BUFFER_INDEX].value
        )
        state.rounds_per_batch = int(
            regimen_sliders[REGIMEN_ROUNDS_PER_BATCH_INDEX].value
        )
        state.batch_grouping = int(
            regimen_sliders[REGIMEN_BATCH_GROUPING_INDEX].value
        )
        state.match_time_limit = int(
            regimen_sliders[REGIMEN_MATCH_TIME_LIMIT_INDEX].value
        )
        state.minibatch_size = int(
            regimen_sliders[REGIMEN_MINIBATCH_SIZE_INDEX].value
        )
        state.replay_updates_per_batch = int(
            regimen_sliders[REGIMEN_REPLAY_UPDATES_INDEX].value
        )
        state.learning_rate = regimen_sliders[REGIMEN_LEARNING_RATE_INDEX].value
        state.epsilon = regimen_sliders[REGIMEN_EPSILON_INDEX].value
        state.gamma = regimen_sliders[REGIMEN_GAMMA_INDEX].value
        state.hidden_layer_size = int(
            regimen_sliders[REGIMEN_HIDDEN_LAYER_SIZE_INDEX].value
        )
        state.hidden_layer_count = int(
            regimen_sliders[REGIMEN_HIDDEN_LAYER_COUNT_INDEX].value
        )

    def architecture_metadata():
        return model_architecture_metadata(
            state.hidden_layer_size,
            state.hidden_layer_count,
        )

    def training_metadata():
        return {
            "opponent": {
                "mode": state.opponent_mode,
                "forward_activity": state.forward_activity,
                "a1_activity": state.a1_activity,
                "a2_activity": state.a2_activity,
                "face_opponent_activity": state.face_opponent_activity,
            },
            "rewards": dict(state.rewards),
            "regimen": {
                "replay_buffer_size": state.replay_buffer_size,
                "rounds_per_batch": state.rounds_per_batch,
                "batch_grouping": state.batch_grouping,
                "match_time_limit": state.match_time_limit,
                "minibatch_size": state.minibatch_size,
                "replay_updates_per_batch": state.replay_updates_per_batch,
                "learning_rate": state.learning_rate,
                "epsilon": state.epsilon,
                "gamma": state.gamma,
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
                field.enabled = not state.running
                field.text_color = ui.BRIGHT_GREEN
                delete_button.enabled = not state.running
            else:
                field.enabled = not state.running
                field.text_color = ui.WHITE
                delete_button.enabled = False

    def update_field_colors():
        if state.selected_ship is None:
            return
            
        current_arch = architecture_metadata()
        current_training = training_metadata()
        
        for index, (field, model_slot) in enumerate(zip(slot_fields, slot_models)):
            slot_number = index + 1
            is_selected = slot_number == state.selected_slot
            is_loaded = (
                state.loaded_ship == state.selected_ship 
                and state.loaded_slot == slot_number
            )

            if model_slot.source == SLOT_BUNDLED:
                field.text = "Default"
                field.text_color = (80, 160, 255)
            elif model_slot.source == SLOT_USER:
                if is_selected:
                    settings_match = False
                    if is_loaded and isinstance(model_slot.metadata, dict):
                        saved_arch = model_slot.metadata.get("architecture", {})
                        saved_training = model_slot.metadata.get("training", {})
                        settings_match = (
                            saved_arch == current_arch
                            and saved_training == current_training
                            and field.text == model_slot.description
                        )
                    field.text_color = (
                        ui.BRIGHT_GREEN if (is_loaded and settings_match) else ui.CAN_RED
                    )
                else:
                    field.text_color = ui.WHITE
            else:
                if is_selected:
                    field.text_color = ui.CAN_RED
                else:
                    field.text_color = ui.WHITE

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

    def persist_selected_model(progress=None, *, reset_checkpoint=False):
        model_slot = selected_model_slot()
        if state.selected_ship is None or model_slot is None or model_slot.is_bundled:
            return None
        description = slot_fields[state.selected_slot - 1].text.strip()
        existing_metadata = (
            model_slot.metadata if isinstance(model_slot.metadata, dict) else {}
        )
        metadata = metadata_from_state(
            ship=state.selected_ship,
            slot=state.selected_slot,
            description=description,
            architecture=architecture_metadata(),
            training=training_metadata(),
            progress=_progress_for_model_update(
                existing_metadata,
                progress,
                reset_checkpoint=reset_checkpoint,
            ),
        )
        updated_slot = model_repository.create_or_update_user_model(metadata)
        if reset_checkpoint and updated_slot.pth_path is not None:
            updated_slot.pth_path.write_bytes(b"")
            
        state.loaded_ship = state.selected_ship
        state.loaded_slot = state.selected_slot
        state.loaded_architecture = architecture_metadata()
        state.loaded_training = training_metadata()
        
        refresh_slot_controls()
        return updated_slot

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

    def load_selected_model_conditions():
        model_slot = selected_model_slot()
        if model_slot is None or not model_slot.is_user:
            return
        metadata = model_slot.metadata if isinstance(model_slot.metadata, dict) else {}
        if not metadata:
            show_notice("Selected AI has no saved conditions")
            return

        skipped = []
        architecture = metadata.get("architecture", {})
        training = metadata.get("training", {})

        if isinstance(training, dict):
            opponent = training.get("opponent", {})
            if isinstance(opponent, dict):
                mode = opponent.get("mode")
                if mode in {"all", "simple"}:
                    select_opponent_mode(mode)
                for key, slider in (
                    ("forward_activity", simple_activity_sliders[0]),
                    ("a1_activity", simple_activity_sliders[1]),
                    ("a2_activity", simple_activity_sliders[2]),
                    ("face_opponent_activity", simple_activity_sliders[3]),
                ):
                    if key not in opponent:
                        continue
                    try:
                        value = float(opponent[key])
                    except (TypeError, ValueError):
                        skipped.append(slider.label)
                        continue
                    if not _set_slider_value(slider, value):
                        skipped.append(slider.label)

            rewards = training.get("rewards", {})
            if isinstance(rewards, dict):
                for slider in reward_sliders:
                    reward_key = slider.label
                    if reward_key not in rewards:
                        legacy_key = LEGACY_REWARD_ALIASES.get(slider.label)
                        if legacy_key not in rewards:
                            continue
                        reward_key = legacy_key
                    if reward_key not in rewards:
                        continue
                    try:
                        value = float(rewards[reward_key])
                    except (TypeError, ValueError):
                        skipped.append(slider.label)
                        continue
                    if not _set_slider_value(slider, value):
                        skipped.append(slider.label)

            regimen = training.get("regimen", {})
            if isinstance(regimen, dict):
                regimen_fields = (
                    (
                        "replay_buffer_size",
                        regimen_sliders[REGIMEN_REPLAY_BUFFER_INDEX],
                        int,
                    ),
                    (
                        "rounds_per_batch",
                        regimen_sliders[REGIMEN_ROUNDS_PER_BATCH_INDEX],
                        int,
                    ),
                    (
                        "batch_grouping",
                        regimen_sliders[REGIMEN_BATCH_GROUPING_INDEX],
                        int,
                    ),
                    (
                        "match_time_limit",
                        regimen_sliders[REGIMEN_MATCH_TIME_LIMIT_INDEX],
                        int,
                    ),
                    (
                        "minibatch_size",
                        regimen_sliders[REGIMEN_MINIBATCH_SIZE_INDEX],
                        int,
                    ),
                    (
                        "replay_updates_per_batch",
                        regimen_sliders[REGIMEN_REPLAY_UPDATES_INDEX],
                        int,
                    ),
                    (
                        "learning_rate",
                        regimen_sliders[REGIMEN_LEARNING_RATE_INDEX],
                        float,
                    ),
                    ("epsilon", regimen_sliders[REGIMEN_EPSILON_INDEX], float),
                    ("gamma", regimen_sliders[REGIMEN_GAMMA_INDEX], float),
                )
                for key, slider, caster in regimen_fields:
                    if key not in regimen:
                        continue
                    try:
                        value = caster(regimen[key])
                    except (TypeError, ValueError):
                        skipped.append(key.replace("_", " "))
                        continue
                    if not _set_slider_value(slider, value):
                        skipped.append(key.replace("_", " "))

        if isinstance(architecture, dict):
            architecture_fields = (
                (
                    architecture.get(
                        "hidden_layer_width",
                        architecture.get("hidden_layer_size"),
                    ),
                    regimen_sliders[REGIMEN_HIDDEN_LAYER_SIZE_INDEX],
                    "hidden layer size",
                ),
                (
                    architecture.get("hidden_layer_count"),
                    regimen_sliders[REGIMEN_HIDDEN_LAYER_COUNT_INDEX],
                    "hidden layer count",
                ),
            )
            for raw_value, slider, label in architecture_fields:
                if raw_value is None:
                    continue
                try:
                    value = int(raw_value)
                except (TypeError, ValueError):
                    skipped.append(label)
                    continue
                if not _set_slider_value(slider, value):
                    skipped.append(label)

        if skipped:
            show_notice(f"Loaded AI; skipped unsupported {skipped[0]}")
        else:
            show_notice(f"Loaded {describe_model(model_slot)} conditions")

        sync_state_from_ui()
        state.loaded_ship = state.selected_ship
        state.loaded_slot = state.selected_slot
        state.loaded_architecture = architecture_metadata()
        state.loaded_training = training_metadata()

    def clear_session_continuity():
        training_session[0] = None
        last_session_running[0] = False

    def session_continuity_for(model_slot):
        active_session = training_session[0]
        if (
            active_session is None
            or active_session.slot.ship != model_slot.ship
            or active_session.slot.slot != model_slot.slot
        ):
            return (), ()
        return active_session.history, active_session.log_lines

    def request_back():
        if training_session[0] is not None and training_session[0].status.running:
            if state.display_on:
                display_checkbox.is_checked = False
                state.display_on = False
                training_session[0].set_display_on(False)
            training_session[0].request_stop()
            show_notice("Training pausing; current batch will be abandoned")
        else:
            exited[0] = True

    def begin_training():
        model_slot = selected_model_slot()
        if state.selected_ship is None or model_slot is None or model_slot.is_bundled:
            return
        if model_slot.source == SLOT_EMPTY:
            model_slot = persist_selected_model()
            if model_slot is None:
                return

        metadata = model_slot.metadata if isinstance(model_slot.metadata, dict) else {}
        if not metadata:
            metadata = metadata_from_state(
                ship=state.selected_ship,
                slot=state.selected_slot,
                description=slot_fields[state.selected_slot - 1].text.strip(),
                architecture=architecture_metadata(),
                training=training_metadata(),
            )
        report = validate_model_metadata(metadata, architecture=architecture_metadata())
        if report.errors:
            show_notice(report.errors[0])
            return
        if report.warnings:
            show_notice(report.warnings[0])

        try:
            initial_history, initial_log_lines = session_continuity_for(model_slot)
            session = TrainingSession(
                repository=model_repository,
                slot=model_slot,
                metadata=metadata,
                config=training_config_from_state(state),
                batch_grouping=state.batch_grouping,
                audio_service=audio_service,
                initial_history=initial_history,
                initial_log_lines=initial_log_lines,
            )
            training_session[0] = session
            state.running = True
            session.start()
            show_notice(f"Training {describe_model(model_slot)}")
        except (TrainingSessionError, RuntimeError, ValueError) as exc:
            show_notice(str(exc))

    def start_selected_model():
        if training_session[0] is not None and training_session[0].status.running:
            if state.display_on:
                display_checkbox.is_checked = False
                state.display_on = False
                training_session[0].set_display_on(False)
            training_session[0].request_stop()
            show_notice("Training pausing; current batch will be abandoned")
            return

        model_slot = selected_model_slot()
        if state.selected_ship is None or model_slot is None or model_slot.is_bundled:
            return

        new_architecture = architecture_metadata()
        new_training = training_metadata()
        current_description = slot_fields[state.selected_slot - 1].text.strip()

        if model_slot.source == SLOT_EMPTY:
            if not current_description:
                show_notice("Enter a model description before creating a new AI")
                return
            persist_selected_model()
            refresh_slot_controls()
            begin_training()
            return

        metadata = model_slot.metadata if isinstance(model_slot.metadata, dict) else {}
        old_architecture = metadata.get("architecture", {})
        old_training = metadata.get("training", {})
        old_description = metadata.get("description", model_slot.description)

        if old_architecture and old_architecture != new_architecture:
            confirmation_prompt[0] = ConfirmationPrompt(
                f"Do you want to overwrite {describe_model(model_slot)}?",
                lambda: (
                    persist_selected_model(reset_checkpoint=True),
                    clear_session_continuity(),
                    begin_training(),
                ),
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
                lambda: (persist_selected_model(), begin_training()),
            )
            return

        if current_description != old_description:
            persist_selected_model()
            begin_training()
            return

        begin_training()

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
        request_back,
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )
    load_button = ui_button.Button(
        load_button_rect.x,
        load_button_rect.y,
        load_button_rect.width,
        load_button_rect.height,
        "Load",
        load_selected_model_conditions,
        ui.MENU_BUTTON_COLOR,
        ui.MENU_BUTTON_COLOR_HI,
    )
    load_button.enabled = False
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
            if not display_checkbox.value:
                batch_log_box.handle_event(event, layout.arena_rect, log_font)

            if state.active_tab == "trainee":
                translated = _translated_event(
                    event, layout.content_rect, trainee_scroll_y
                )
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if layout.content_rect.collidepoint(event.pos):
                        if ship_tile.collidepoint(translated.pos) and not state.running:
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
                            if state.selected_ship is not None and row.collidepoint(translated.pos) and not state.running:
                                state.selected_slot = index + 1
                                break
                for field in slot_fields:
                    field.handle_event(translated)
                for delete_btn in delete_buttons:
                    delete_btn.handle_event(translated, menu_sound_manager)
                load_button.handle_event(translated, menu_sound_manager)
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
                for slider in simple_activity_sliders:
                    slider.enabled = enabled
                    slider.handle_event(translated, menu_sound_manager)
            else:
                for slider in regimen_sliders:
                    slider.handle_event(event, menu_sound_manager)

        sync_state_from_ui()
        controls_enabled = state.simple_behavior_controls_enabled
        for slider in simple_activity_sliders:
            slider.enabled = controls_enabled

        for btn in opponent_mode_buttons:
            btn.enabled = not state.running
        for slider in reward_sliders:
            slider.enabled = not state.running
        for slider in regimen_sliders:
            slider.enabled = not state.running

        active_session = training_session[0]
        session_status = active_session.status if active_session is not None else None
        if active_session is not None:
            if session_status.running:
                active_session.set_display_on(state.display_on)
            batch_log_box.set_lines(
                _display_off_console_lines(session_status, active_session.log_lines)
            )
            state.running = session_status.running
            if not last_session_running[0] and session_status.running:
                refresh_slot_controls()
            if last_session_running[0] and not session_status.running:
                display_checkbox.is_checked = False
                state.display_on = False
                active_session.set_display_on(False)
                if session_status.error:
                    show_notice(session_status.error)
                else:
                    show_notice("Training stopped")
                refresh_slot_controls()
            last_session_running[0] = session_status.running

        update_field_colors()
        selected_slot = selected_model_slot()
        back_button.enabled = not state.running
        start_stop_button.enabled = (
            state.selected_ship is not None
            and selected_slot is not None
            and not selected_slot.is_bundled
            and (
                (session_status is not None and session_status.running)
                or bool(slot_fields[state.selected_slot - 1].text.strip())
            )
        )

        is_currently_loaded = (
            state.loaded_ship == state.selected_ship
            and state.loaded_slot == state.selected_slot
            and state.loaded_architecture == architecture_metadata()
            and state.loaded_training == training_metadata()
        )

        if is_currently_loaded and state.selected_ship is not None:
            load_button.text = f"{state.selected_ship}-{state.selected_slot:02d} Loaded"
            load_button.enabled = False
        else:
            load_button.text = "Load"
            load_button.enabled = selected_slot is not None and selected_slot.is_user and not state.running

        if session_status is not None and session_status.running:
            start_stop_button.text = "Stopping" if session_status.stopping else "Stop"
            start_stop_button.bg_color = (*ui.CAN_RED[:3], const.TAB_BUTTON_HOVER_ALPHA)
            start_stop_button.hover_color = ui.CAN_RED_HI
            
            display_checkbox.bg_color = (*ui.MENU_BUTTON_COLOR[:3], const.TAB_BUTTON_HOVER_ALPHA)
            display_checkbox.hover_color = ui.MENU_BUTTON_COLOR_HI
        else:
            start_stop_button.text = "Start"
            start_stop_button.bg_color = ui.OK_GREEN
            start_stop_button.hover_color = ui.OK_GREEN_HI
            
            display_checkbox.bg_color = ui.MENU_BUTTON_COLOR
            display_checkbox.hover_color = ui.MENU_BUTTON_COLOR_HI

        if notice[0] is not None:
            notice[0].remaining_seconds -= elapsed_seconds
            if notice[0].remaining_seconds <= 0:
                notice[0] = None

        if background:
            screen.blit(background, (0, 0))
        else:
            screen.fill(ui.BG_COLOR)
        if session_status is not None:
            if state.display_on:
                _draw_training_battle(
                    screen,
                    layout.arena_rect,
                    session_status,
                    training_battle_renderer,
                    training_battle_controller,
                )
            else:
                batch_log_box.draw(screen, layout.arena_rect, log_font)
        else:
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
            pygame.draw.rect(content, const.P1_COLOR if not state.running else ui.DARK_GREY, ship_tile, 3)
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
                enabled = state.selected_ship is not None and not state.running
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

            load_button.draw(content, body_font, content_mouse_pos)

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
            for slider in simple_activity_sliders:
                slider.draw(content, opponent_font)
            screen.blit(content, layout.content_rect)
        else:
            panel = pygame.Surface(layout.content_rect.size, pygame.SRCALPHA)
            panel.fill((0, 0, 0, 155))
            screen.blit(panel, layout.content_rect)
            for slider in regimen_sliders:
                slider.draw(screen, regimen_font)

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

        if state.running:
            pygame.draw.rect(screen, ui.BLACK, display_checkbox.rect, 2, border_radius=5)
            pygame.draw.rect(screen, ui.BLACK, start_stop_button.rect, 2, border_radius=5)
        if state.display_on and session_status is not None:
            _draw_training_huds(
                screen,
                layout.hud_rects,
                session_status,
                training_battle_renderer,
                training_battle_controller,
            )
        else:
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
