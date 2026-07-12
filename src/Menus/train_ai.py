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
from src.UI import ui, ui_button
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
AI_OPPONENT_PERCENT_VALUES = SIMPLE_ACTIVITY_VALUES

REPLAY_BUFFER_SIZE_VALUES = tuple(range(5000, 250001, 5000))
ROUNDS_PER_BATCH_VALUES = tuple(range(1, 51, 1))
BATCH_GROUPING_VALUES = tuple(range(25, 1001, 25))
MATCH_TIME_LIMIT_VALUES = tuple(range(240, 12001, 240))
MINIBATCH_SIZE_VALUES = (16, 32, 64, 128, 256, 512)
REPLAY_UPDATES_PER_BATCH_VALUES = tuple(range(100, 5001, 100))
LEARNING_RATE_VALUES = (0.00001, 0.00003, 0.00010, 0.00030, 0.00100, 0.00300, 0.01000)
EPSILON_VALUES = tuple(round(i * 0.025, 3) for i in range(41))
EPSILON_DECAY_VALUES = tuple(round(0.950 + i * 0.001, 3) for i in range(51))
EPSILON_FRAME_SPAN_VALUES = tuple(range(1, 49))
GAMMA_VALUES = tuple(round(0.950 + i * 0.001, 3) for i in range(51))
HIDDEN_LAYER_SIZE_VALUES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
HIDDEN_LAYER_COUNT_VALUES = tuple(range(1, 9, 1))
REGIMEN_MATCH_TIME_LIMIT_INDEX = 0
REGIMEN_ROUNDS_PER_BATCH_INDEX = 1
REGIMEN_BATCH_GROUPING_INDEX = 2
REGIMEN_REPLAY_BUFFER_INDEX = 3
REGIMEN_MINIBATCH_SIZE_INDEX = 4
REGIMEN_REPLAY_UPDATES_INDEX = 5
REGIMEN_LEARNING_RATE_INDEX = 6
REGIMEN_STARTING_EPSILON_INDEX = 7
REGIMEN_EPSILON_DECAY_INDEX = 8
REGIMEN_EPSILON_FRAME_SPAN_INDEX = 9
REGIMEN_GAMMA_INDEX = 10
REGIMEN_HIDDEN_LAYER_SIZE_INDEX = 11
REGIMEN_HIDDEN_LAYER_COUNT_INDEX = 12
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
INSTANCE_TOP = 8
INSTANCE_CONTROL_HEIGHT = 30
INSTANCE_GAP = 8
INSTANCE_DROPDOWN_VISIBLE_ROWS = 8
INSTANCE_SEPARATOR_HEIGHT = 4
TRAINING_INSTANCE_SOFT_MAX = 4
TRAINING_INSTANCE_SUPPORTED_MAX = 25
INSTANCE_POSITION_WIDTH = 56
INSTANCE_RUNNING_WIDTH = 42
INSTANCE_CLOSE_WIDTH = 60
INSTANCE_ADD_WIDTH = 50
INSTANCE_DROPDOWN_COLOR = (0, 70, 75, 235)
INSTANCE_DROPDOWN_HOVER_COLOR = (0, 100, 110, 255)
UI_TOP_MARGIN = INSTANCE_TOP + INSTANCE_CONTROL_HEIGHT + 12
TAB_GAP = 8
TAB_HEIGHT = 32
TAB_COLOR = (155, 0, 105, 75)
TAB_COLOR_HI = (155, 0, 105, 255)
TAB_HEADER_COLOR = (100, 100, 100)
CONTENT_TOP = UI_TOP_MARGIN + TAB_HEIGHT + TAB_GAP
FOOTER_CONTROL_HEIGHT = 36
ACTION_TOP = 678
DISPLAY_TOP = ACTION_TOP
TRAINING_HUD_HEIGHT = MARINE_REGION_HEIGHT + VIEWPORT_SIZE + HUD_BOTTOM_PADDING
HUD_TOP = const.SCREEN_HEIGHT - TRAINING_HUD_HEIGHT
HUD_BOTTOM_MARGIN = 0
CONTENT_BOTTOM = ACTION_TOP - TAB_GAP
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
    ai_opponent_chance: float = 0.0
    forward_activity: float = 0.0
    a1_activity: float = 0.0
    a2_activity: float = 0.0
    face_opponent_activity: float = 0.0
    rounds_per_batch: int = 1
    batch_grouping: int = 50
    match_time_limit: int = 1200
    learning_rate: float = 0.00010
    starting_epsilon: float = 0.100
    current_epsilon: float = 0.100
    epsilon_decay: float = 0.998
    epsilon_frame_span: int = 8
    gamma: float = 0.990
    minibatch_size: int = 64
    replay_updates_per_batch: int = 500
    hidden_layer_size: int = 256
    hidden_layer_count: int = 2
    replay_buffer_size: int = 50000
    display_on: bool = False
    running: bool = False
    loaded_ship: str | None = None
    loaded_slot: int | None = None
    loaded_architecture: dict | None = None
    loaded_training: dict | None = None

    @property
    def simple_behavior_controls_enabled(self):
        return not self.running

    @property
    def epsilon(self):
        return self.starting_epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.starting_epsilon = float(value)
        self.current_epsilon = float(value)


@dataclass
class TrainingInstance:
    instance_id: int
    label: str
    state: TrainingUIState
    session: TrainingSession | None = None
    pending_removal: bool = False
    last_running: bool = False
    writer_key: tuple[str, int] | None = None


class TrainingInstanceManager:
    def __init__(
        self,
        *,
        soft_max=TRAINING_INSTANCE_SOFT_MAX,
        supported_max=TRAINING_INSTANCE_SUPPORTED_MAX,
    ):
        self.soft_max = int(soft_max)
        self.supported_max = int(supported_max)
        self._next_instance_id = 2
        self._writer_reservations: dict[tuple[str, int], int] = {}
        self.instances = [
            TrainingInstance(
                instance_id=1,
                label="Instance 1",
                state=TrainingUIState(),
            )
        ]
        self.active_instance_id = 1

    @property
    def active_instance(self):
        for instance in self.instances:
            if instance.instance_id == self.active_instance_id:
                return instance
        raise RuntimeError("Active training instance is missing")

    @property
    def active_index(self):
        for index, instance in enumerate(self.instances):
            if instance.instance_id == self.active_instance_id:
                return index
        raise RuntimeError("Active training instance is missing")

    @property
    def active_state(self):
        return self.active_instance.state

    @property
    def active_session(self):
        return self.active_instance.session

    def set_active_session(self, session):
        self.active_instance.session = session

    def clear_active_session_continuity(self):
        instance = self.active_instance
        self.release_writer(instance)
        instance.session = None
        instance.last_running = False

    def add_instance(self):
        if not self.can_add_instance():
            raise ValueError(
                f"Only {self.supported_max} training instances are supported"
            )
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        self.disable_display(self.active_instance)
        instance = TrainingInstance(
            instance_id=instance_id,
            label=f"Instance {instance_id}",
            state=TrainingUIState(),
        )
        self.instances.append(instance)
        self.active_instance_id = instance_id
        return instance

    def can_add_instance(self):
        return len(self.instances) < self.supported_max

    def add_requires_confirmation(self):
        return len(self.instances) >= self.soft_max

    def select_instance(self, instance_id):
        previous = self.active_instance
        for instance in self.instances:
            if instance.instance_id == instance_id:
                if instance.instance_id != self.active_instance_id:
                    self.disable_display(previous)
                    self.disable_display(instance)
                self.active_instance_id = instance_id
                return instance
        raise ValueError(f"Unknown training instance {instance_id}")

    def _insert_new_instance(self):
        instance_id = self._next_instance_id
        self._next_instance_id += 1
        instance = TrainingInstance(
            instance_id=instance_id,
            label=f"Instance {instance_id}",
            state=TrainingUIState(),
        )
        self.instances.append(instance)
        return instance

    def remove_active_stopped_instance(self):
        if len(self.instances) <= 1:
            return False
        index = self.active_index
        instance = self.instances[index]
        if self.is_running_or_stopping(instance):
            return False
        self.disable_display(instance)
        self.release_writer(instance)
        self.instances.pop(index)
        next_index = min(index, len(self.instances) - 1)
        self.active_instance_id = self.instances[next_index].instance_id
        self.disable_display(self.active_instance)
        return True

    def request_close_active_instance(self):
        index = self.active_index
        instance = self.instances[index]
        if self.is_running_or_stopping(instance):
            self.disable_display(instance)
            self.request_stop(instance)
            instance.pending_removal = True
            if len(self.instances) == 1:
                replacement = self._insert_new_instance()
                self.active_instance_id = replacement.instance_id
            else:
                next_index = (index + 1) % len(self.instances)
                if self.instances[next_index] is instance:
                    next_index = 0
                self.active_instance_id = self.instances[next_index].instance_id
                self.disable_display(self.active_instance)
            return "pending"
        if self.remove_active_stopped_instance():
            return "removed"
        return "last"

    def active_position_text(self):
        width = max(2, len(str(len(self.instances))))
        return f"{self.active_index + 1:0{width}d}/{len(self.instances):0{width}d}"

    def running_count(self):
        return len(self.running_instances())

    def running_count_text(self):
        return f"{self.running_count():02d}>"

    def status_for(self, instance):
        return instance.session.status if instance.session is not None else None

    def is_running_or_stopping(self, instance):
        status = self.status_for(instance)
        return bool(
            getattr(status, "running", False)
            or getattr(status, "stopping", False)
        )

    def running_instances(self):
        return [
            instance
            for instance in self.instances
            if self.is_running_or_stopping(instance)
        ]

    def non_active_running_instances(self):
        return [
            instance
            for instance in self.running_instances()
            if instance.instance_id != self.active_instance_id
        ]

    def back_action(self):
        if self.non_active_running_instances():
            return "stop_all"
        if self.is_running_or_stopping(self.active_instance):
            return "active_running"
        return "exit"

    def background_instances_running(self):
        return bool(self.non_active_running_instances())

    def any_instance_running(self):
        return bool(self.running_instances())

    def disable_display(self, instance):
        instance.state.display_on = False
        if instance.session is not None:
            instance.session.set_display_on(False)

    def set_active_display(self, enabled):
        active_instance = self.active_instance
        if enabled:
            for instance in self.instances:
                if instance is not active_instance:
                    self.disable_display(instance)
            active_instance.state.display_on = True
            if active_instance.session is not None:
                active_instance.session.set_display_on(True)
        else:
            self.disable_display(active_instance)

    def request_stop(self, instance):
        if instance.session is not None:
            instance.session.request_stop()

    def request_stop_active(self):
        self.disable_display(self.active_instance)
        self.request_stop(self.active_instance)

    def request_stop_all_running(self):
        for instance in self.running_instances():
            self.disable_display(instance)
            self.request_stop(instance)

    def reserve_writer(self, instance, ship, slot):
        self.release_stopped_writers()
        key = (ship, int(slot))
        owner_id = self._writer_reservations.get(key)
        if owner_id is not None and owner_id != instance.instance_id:
            return False
        self._writer_reservations[key] = instance.instance_id
        instance.writer_key = key
        return True

    def writer_owner(self, ship, slot):
        return self._writer_reservations.get((ship, int(slot)))

    def release_writer(self, instance):
        key = instance.writer_key
        if key is not None and self._writer_reservations.get(key) == instance.instance_id:
            del self._writer_reservations[key]
        instance.writer_key = None

    def release_stopped_writers(self):
        for instance in self.instances:
            if instance.writer_key is not None and not self.is_running_or_stopping(instance):
                self.release_writer(instance)

    def cleanup_stopped_pending_removals(self):
        removed_active = False
        kept = []
        for instance in self.instances:
            if instance.pending_removal and not self.is_running_or_stopping(instance):
                self.release_writer(instance)
                removed_active = removed_active or instance.instance_id == self.active_instance_id
                continue
            kept.append(instance)
        if not kept:
            kept.append(self._insert_new_instance())
        self.instances = kept
        if removed_active or not any(
            instance.instance_id == self.active_instance_id
            for instance in self.instances
        ):
            self.active_instance_id = self.instances[0].instance_id
            self.disable_display(self.active_instance)
        self.release_stopped_writers()


def _instance_status_text(instance):
    status = instance.session.status if instance.session is not None else None
    if status is None:
        return "Stopped"
    if getattr(status, "error", None):
        return "Error"
    if getattr(status, "stopping", False):
        return "Stopping"
    if getattr(status, "running", False):
        return "Running"
    return "Stopped"


def _instance_model_text(instance):
    state = instance.state
    if state.selected_ship is None or state.selected_slot is None:
        return "-------------"
    return f"{state.selected_ship}-{state.selected_slot:02d}"


def _instance_row_parts(position, instance):
    status = _instance_status_text(instance)
    return f"{position:02d}] {_instance_model_text(instance):>13} ", status


def _instance_status_color(status):
    if status == "Running":
        return ui.BRIGHT_GREEN
    if status in {"Stopped", "Error"}:
        return (255, 80, 80)
    return (255, 255, 0)


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


class InstanceDropdown:
    def __init__(self, rect, manager, callback):
        self.rect = pygame.Rect(rect)
        self.manager = manager
        self.callback = callback
        self.expanded = False
        self.scroll_index = 0
        self.row_height = INSTANCE_CONTROL_HEIGHT
        self.max_visible_rows = INSTANCE_DROPDOWN_VISIBLE_ROWS

    def list_rect(self):
        visible_rows = min(self.max_visible_rows, len(self.manager.instances))
        return pygame.Rect(
            self.rect.x,
            self.rect.bottom + 4,
            self.rect.width,
            self.row_height * visible_rows,
        )

    def _visible_instances(self):
        max_scroll = max(0, len(self.manager.instances) - self.max_visible_rows)
        self.scroll_index = max(0, min(self.scroll_index, max_scroll))
        end = self.scroll_index + self.max_visible_rows
        return self.manager.instances[self.scroll_index:end]

    def _select_at_pos(self, pos):
        list_rect = self.list_rect()
        if not list_rect.collidepoint(pos):
            return False
        row = (pos[1] - list_rect.y) // self.row_height
        index = self.scroll_index + row
        if 0 <= index < len(self.manager.instances):
            self.callback(self.manager.instances[index].instance_id)
            self.expanded = False
            return True
        return False

    def handle_event(self, event, sound_manager=None):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                if sound_manager:
                    sound_manager.play_sound("menu")
                return True
            if self.expanded:
                selected = self._select_at_pos(event.pos)
                if selected and sound_manager:
                    sound_manager.play_sound("menu")
                if not selected:
                    self.expanded = False
                return selected
        elif (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button in (4, 5)
            and self.expanded
            and self.list_rect().collidepoint(event.pos)
        ):
            direction = -1 if event.button == 4 else 1
            max_scroll = max(0, len(self.manager.instances) - self.max_visible_rows)
            self.scroll_index = max(0, min(max_scroll, self.scroll_index + direction))
            return True
        elif (
            event.type == pygame.MOUSEWHEEL
            and self.expanded
            and self.list_rect().collidepoint(getattr(event, "pos", pygame.mouse.get_pos()))
        ):
            max_scroll = max(0, len(self.manager.instances) - self.max_visible_rows)
            self.scroll_index = max(0, min(max_scroll, self.scroll_index - event.y))
            return True
        return False

    def draw(self, surface, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        bg_color = (
            INSTANCE_DROPDOWN_HOVER_COLOR
            if self.rect.collidepoint(mouse_pos)
            else INSTANCE_DROPDOWN_COLOR
        )
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=5)
        pygame.draw.rect(surface, ui.WHITE, self.rect, 3, border_radius=5)
        active_position = self.manager.active_index + 1
        prefix, status = _instance_row_parts(
            active_position,
            self.manager.active_instance,
        )
        self._draw_row_text(surface, font, self.rect.inflate(-6, -4), prefix, status)
        arrow = font.render("^" if self.expanded else "v", True, ui.WHITE)
        surface.blit(arrow, arrow.get_rect(midright=(self.rect.right - 8, self.rect.centery)))

        if not self.expanded:
            return

        list_rect = self.list_rect()
        pygame.draw.rect(surface, ui.BLACK, list_rect)
        pygame.draw.rect(surface, ui.WHITE, list_rect, 3)
        old_clip = surface.get_clip()
        inner_rect = list_rect.inflate(-6, -6)
        surface.set_clip(inner_rect.clip(old_clip))
        for offset, instance in enumerate(self._visible_instances()):
            index = self.scroll_index + offset
            row = pygame.Rect(
                inner_rect.x,
                inner_rect.y + offset * self.row_height,
                inner_rect.width,
                self.row_height,
            )
            if instance.instance_id == self.manager.active_instance_id:
                pygame.draw.rect(surface, (55, 80, 120), row.inflate(-2, -2))
            elif row.collidepoint(mouse_pos):
                pygame.draw.rect(surface, ui.DARK_GREY, row.inflate(-2, -2))
            prefix, status = _instance_row_parts(index + 1, instance)
            self._draw_row_text(surface, font, row, prefix, status)
        surface.set_clip(old_clip)

        if len(self.manager.instances) > self.max_visible_rows:
            track = pygame.Rect(list_rect.right - 7, list_rect.y + 5, 3, list_rect.height - 10)
            pygame.draw.rect(surface, ui.DARK_GREY, track)
            ratio = self.max_visible_rows / len(self.manager.instances)
            thumb_h = max(12, int(track.height * ratio))
            max_scroll = len(self.manager.instances) - self.max_visible_rows
            thumb_y = track.y
            if max_scroll:
                thumb_y += int((track.height - thumb_h) * (self.scroll_index / max_scroll))
            pygame.draw.rect(surface, ui.LIGHT_GREY, pygame.Rect(track.x, thumb_y, track.width, thumb_h))

    def _draw_row_text(self, surface, font, rect, prefix, status):
        text_clip = rect.inflate(-16, 0)
        old_clip = surface.get_clip()
        surface.set_clip(text_clip.clip(old_clip))
        prefix_surf = font.render(prefix, True, ui.WHITE)
        x = rect.x + 8
        surface.blit(prefix_surf, prefix_surf.get_rect(midleft=(x, rect.centery)))
        status_surf = font.render(status, True, _instance_status_color(status))
        surface.blit(
            status_surf,
            status_surf.get_rect(midleft=(x + prefix_surf.get_width(), rect.centery)),
        )
        surface.set_clip(old_clip)


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


def _format_percent(value):
    return f"{int(value)}%"


def _format_short_count(value):
    value = int(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for suffix, divisor in (("M", 1_000_000), ("k", 1_000)):
        if value >= divisor:
            scaled = value / divisor
            if scaled.is_integer():
                return f"{sign}{int(scaled)}{suffix}"
            return f"{sign}{scaled:.1f}{suffix}"
    return f"{sign}{value}"


def _normalized_training_settings(training):
    if not isinstance(training, dict):
        return training
    normalized = {}
    for group, settings in training.items():
        if isinstance(settings, dict):
            normalized[group] = dict(settings)
        else:
            normalized[group] = settings
    regimen = normalized.get("regimen")
    if isinstance(regimen, dict):
        if "starting_epsilon" not in regimen and "epsilon" in regimen:
            regimen["starting_epsilon"] = regimen["epsilon"]
        regimen.pop("current_epsilon", None)
        regimen.pop("epsilon", None)
    return normalized


def _training_settings_match(first, second):
    return _normalized_training_settings(first) == _normalized_training_settings(second)


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


class SliderRow:
    """Training screen slider with one-line text and a fixed track region."""

    LABEL_SLIDER_VALUE = "label-slider-value"
    LABEL_VALUE_SLIDER = "label-value-slider"
    VALUE_COLOR = (255, 255, 0)

    def __init__(
        self,
        rect,
        label,
        min_val,
        max_val,
        value,
        *,
        values=None,
        is_int=False,
        step=1,
        decimal_places=None,
        value_suffix="",
        value_formatter=None,
        layout=LABEL_SLIDER_VALUE,
        label_width=278,
        value_width=70,
        slider_width=None,
        track_height=4,
        handle_radius=7,
        bg_color=ui.SLIDER_BG,
        hover_color=ui.SLIDER_BG_HI,
    ):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.values = tuple(values) if values is not None else None
        if self.values is not None:
            if len(self.values) < 2 or value not in self.values:
                raise ValueError("Slider values must contain the starting value")
        self.is_int = is_int
        self.step = step
        if decimal_places is not None:
            self.decimal_places = decimal_places
        else:
            self.decimal_places = (
                abs(len(str(step).split(".")[-1]))
                if "." in str(step) and "e" not in str(step)
                else 0
            )
        self.value_suffix = value_suffix
        self.value_formatter = value_formatter
        self.layout = layout
        self.label_width = label_width
        self.value_width = value_width
        self.slider_width = slider_width
        self.track_height = track_height
        self.handle_radius = handle_radius
        self.bg_color = (*bg_color, 255) if len(bg_color) == 3 else bg_color
        self.hover_color = (*hover_color, 255) if len(hover_color) == 3 else hover_color
        self.dragging = False
        self.enabled = True
        self.is_hovered = False
        self.line_rect = self._line_rect()
        self.handle_x = self.value_to_position(self.value)

    def _line_rect(self):
        if self.layout == self.LABEL_VALUE_SLIDER:
            slider_width = self.slider_width or max(
                1, self.rect.width - self.label_width - self.value_width - 8
            )
            x = self.rect.right - slider_width - 8
            width = slider_width
        else:
            x = self.rect.x + self.label_width
            width = self.rect.width - self.label_width - self.value_width - 8
        return pygame.Rect(
            x,
            self.rect.centery - self.track_height // 2,
            max(1, width),
            self.track_height,
        )

    def value_to_position(self, value):
        if self.values is not None:
            ratio = self.values.index(value) / (len(self.values) - 1)
        else:
            ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.line_rect.x + round(ratio * self.line_rect.width)

    def position_to_value(self, pos_x):
        ratio = (pos_x - self.line_rect.left) / max(1, self.line_rect.width)
        if self.values is not None:
            index = round(ratio * (len(self.values) - 1))
            index = max(0, min(len(self.values) - 1, index))
            return self.values[index]
        value = self.min_val + ratio * (self.max_val - self.min_val)
        value = round(value / self.step) * self.step
        return max(self.min_val, min(self.max_val, value))

    def set_from_x(self, x):
        x = max(self.line_rect.left, min(self.line_rect.right, x))
        self.value = self.position_to_value(x)
        self.handle_x = self.value_to_position(self.value)

    def adjust_value(self, increment):
        if self.values is not None:
            index = self.values.index(self.value) + (1 if increment else -1)
            index = max(0, min(len(self.values) - 1, index))
            self.value = self.values[index]
        else:
            delta = self.step if increment else -self.step
            self.value = max(self.min_val, min(self.max_val, self.value + delta))
        self.handle_x = self.value_to_position(self.value)

    def get_handle_rect(self):
        return pygame.Rect(
            self.handle_x - self.handle_radius,
            self.line_rect.centery - self.handle_radius,
            self.handle_radius * 2,
            self.handle_radius * 2,
        )

    def handle_event(self, event, sound_manager=None):
        if not self.enabled:
            self.dragging = False
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.line_rect.inflate(0, 20).collidepoint(event.pos):
                    if sound_manager:
                        sound_manager.play_sound("menu")
                    self.dragging = True
                    self.set_from_x(event.pos[0])
            elif self.is_hovered and event.button in (4, 5):
                if sound_manager:
                    sound_manager.play_sound("menu")
                self.adjust_value(event.button == 4)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            if self.dragging:
                self.set_from_x(event.pos[0])

    def format_value(self):
        if self.value_formatter is not None:
            return f"{self.value_formatter(self.value)}{self.value_suffix}"
        if self.is_int:
            return f"{int(self.value)}{self.value_suffix}"
        return f"{self.value:.{self.decimal_places}f}{self.value_suffix}"

    def draw(self, surface, font, mouse_pos=None):
        if mouse_pos is None:
            mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos) or self.is_hovered
        row = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        if not self.enabled:
            row.fill((*ui.DARK_GREY, 255))
        else:
            row.fill(self.hover_color if hovered else self.bg_color)
        surface.blit(row, self.rect)

        label_color = ui.WHITE if self.enabled else ui.GREY
        value_color = self.VALUE_COLOR
        label_text = f"{self.label}: " if self.layout == self.LABEL_VALUE_SLIDER else self.label
        label = font.render(label_text, True, label_color)
        label_rect = label.get_rect(midleft=(self.rect.left + 8, self.rect.centery))
        old_clip = None
        if self.layout == self.LABEL_VALUE_SLIDER:
            old_clip = surface.get_clip()
            text_clip = pygame.Rect(
                self.rect.left + 8,
                self.rect.top,
                max(1, self.line_rect.left - self.rect.left - 16),
                self.rect.height,
            )
            surface.set_clip(text_clip.clip(old_clip))
        surface.blit(label, label_rect)

        value = font.render(self.format_value(), True, value_color)
        if self.layout == self.LABEL_VALUE_SLIDER:
            value_rect = value.get_rect(
                midleft=(min(label_rect.right + 2, self.line_rect.left - 8), self.rect.centery)
            )
        else:
            value_rect = value.get_rect(midright=(self.rect.right - 8, self.rect.centery))
        surface.blit(value, value_rect)
        if old_clip is not None:
            surface.set_clip(old_clip)

        pygame.draw.rect(surface, ui.SLIDER_LINE, self.line_rect)
        pygame.draw.circle(
            surface,
            ui.HANDLE_COLOR if self.enabled else ui.GREY,
            (self.handle_x, self.line_rect.centery),
            self.handle_radius,
        )


class RewardSlider(SliderRow):
    """Compatibility wrapper for tests and older training UI callers."""

    def __init__(self, rect, label, value=0.0):
        super().__init__(
            rect,
            label,
            REWARD_VALUES[0],
            REWARD_VALUES[-1],
            value,
            values=REWARD_VALUES,
            value_formatter=_format_reward,
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
    title = getattr(status, "display_message", "") or (
        "Training running" if status.running else "Training stopped"
    )
    lines = (
        title,
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
    status_font=None,
    status_small_font=None,
):
    if not getattr(status, "battle_view", None):
        status_font = status_font or pygame.font.SysFont(None, 36)
        status_small_font = status_small_font or pygame.font.SysFont(None, 24)
        _draw_training_status(screen, rect, status, status_font, status_small_font)
        return

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
        ai_opponent_chance=state.ai_opponent_chance,
        forward_activity=state.forward_activity,
        a1_activity=state.a1_activity,
        a2_activity=state.a2_activity,
        face_opponent_activity=state.face_opponent_activity,
        rounds_per_batch=state.rounds_per_batch,
        gamma=state.gamma,
        match_time_limit=state.match_time_limit,
        replay_capacity=state.replay_buffer_size,
        learning_rate=state.learning_rate,
        starting_epsilon=state.starting_epsilon,
        epsilon=state.current_epsilon,
        epsilon_decay=state.epsilon_decay,
        epsilon_frame_span=state.epsilon_frame_span,
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


def _epsilon_for_model_update(starting_epsilon, current_epsilon, *, reset_checkpoint=False):
    return float(starting_epsilon if reset_checkpoint else current_epsilon)


def _clear_reset_model_artifacts(model_slot):
    if model_slot.pth_path is not None:
        model_slot.pth_path.write_bytes(b"")
        csv_path = model_slot.pth_path.with_suffix(".csv")
        try:
            csv_path.unlink()
        except FileNotFoundError:
            pass


def run(screen: pygame.Surface, menu_sound_manager=None, audio_service=None):
    """Show the AI-training configuration UI without starting training yet."""
    clock = PresentationClock(const.FPS, const.VIDEO_FPS_MULTIPLIER)
    layout = training_layout()
    instance_manager = TrainingInstanceManager()
    state = instance_manager.active_state
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
            "Simple vs. AI",
            "Forward Activity",
            "A1 Activity",
            "A2 Activity",
            "Face opponent",
        ),
        142,
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
            "Match frame limit: 12000",
            "Rounds per batch: 50",
            "Batch grouping: 1000",
            "Replay size, batch=15M: 250k (1250MB)",
            "Minibatch size: 512",
            "Gradient steps, UTD=999.9: 5000",
            "Learning rate: 0.01000",
            "Starting Epsilon: 1.000",
            "Epsilon decay: 1.000",
            "Epsilon frame span: 48",
            "Gamma: 1.000",
            "Hidden layer size: 4096",
            "Hidden layer count: 8",
        ),
        320,
        max_height=24,
        maximum=22,
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
        UI_TOP_MARGIN,
        tab_width,
        tab_height,
        "Trainee",
        lambda: setattr(state, "active_tab", "trainee"),
    )
    opponent_tab = TabButton(
        TAB_MARGIN + tab_width + TAB_GAP,
        UI_TOP_MARGIN,
        tab_width,
        tab_height,
        "Opponent",
        lambda: setattr(state, "active_tab", "opponent"),
    )
    rewards_tab = TabButton(
        TAB_MARGIN + 2 * (tab_width + TAB_GAP),
        UI_TOP_MARGIN,
        tab_width,
        tab_height,
        "Rewards",
        lambda: setattr(state, "active_tab", "rewards"),
    )
    regimen_tab = TabButton(
        TAB_MARGIN + 3 * (tab_width + TAB_GAP),
        UI_TOP_MARGIN,
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
        SliderRow(
            (12, rewards_top + index * step, CONTROL_WIDTH - 24, min(30, step - 2)),
            label,
            REWARD_VALUES[0],
            REWARD_VALUES[-1],
            0.0,
            values=REWARD_VALUES,
            value_formatter=_format_reward,
            label_width=278,
            value_width=70,
        )
        for index, label in enumerate(REWARD_LABELS)
    ]
    rewards_content_height = reward_sliders[-1].rect.bottom + 8
    rewards_scroll_y = 0
    opponent_panels = (
        pygame.Rect(12, 12, CONTROL_WIDTH - 24, 70),
        pygame.Rect(12, 92, CONTROL_WIDTH - 24, 222),
    )
    opponent_label_width = 170
    opponent_value_width = 58
    ai_opponent_slider = SliderRow(
        (20, 22, CONTROL_WIDTH - 40, 44),
        "Simple vs. AI",
        AI_OPPONENT_PERCENT_VALUES[0],
        AI_OPPONENT_PERCENT_VALUES[-1],
        state.ai_opponent_chance,
        is_int=True,
        step=5.0,
        values=AI_OPPONENT_PERCENT_VALUES,
        value_formatter=_format_percent,
        label_width=opponent_label_width,
        value_width=opponent_value_width,
    )
    simple_activity_sliders = (
        SliderRow(
            (20, 102, CONTROL_WIDTH - 40, 44),
            "Forward Activity",
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.forward_activity,
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            value_formatter=_format_percent,
            label_width=opponent_label_width,
            value_width=opponent_value_width,
        ),
        SliderRow(
            (20, 152, CONTROL_WIDTH - 40, 44),
            "A1 Activity",
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.a1_activity,
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            value_formatter=_format_percent,
            label_width=opponent_label_width,
            value_width=opponent_value_width,
        ),
        SliderRow(
            (20, 202, CONTROL_WIDTH - 40, 44),
            "A2 Activity",
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.a2_activity,
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            value_formatter=_format_percent,
            label_width=opponent_label_width,
            value_width=opponent_value_width,
        ),
        SliderRow(
            (20, 252, CONTROL_WIDTH - 40, 44),
            "Face opponent",
            SIMPLE_ACTIVITY_VALUES[0],
            SIMPLE_ACTIVITY_VALUES[-1],
            state.face_opponent_activity,
            step=5.0,
            values=SIMPLE_ACTIVITY_VALUES,
            value_formatter=_format_percent,
            label_width=opponent_label_width,
            value_width=opponent_value_width,
        )
    )

    grouped_controls = (
        ai_opponent_slider,
        *simple_activity_sliders,
    )
    for control in grouped_controls:
        control.bg_color = (0, 0, 0, 0)
        control.hover_color = (45, 45, 45, 160)

    regimen_left = 16
    regimen_width = CONTROL_WIDTH - 32
    regimen_top = CONTENT_TOP + 14
    regimen_spacing = 38
    regimen_height = 36
    regimen_slider_width = 204
    regimen_layout = SliderRow.LABEL_VALUE_SLIDER
    regimen_sliders = (
        SliderRow(
            (regimen_left, regimen_top, regimen_width, regimen_height),
            "Match frame limit",
            MATCH_TIME_LIMIT_VALUES[0],
            MATCH_TIME_LIMIT_VALUES[-1],
            state.match_time_limit,
            is_int=True,
            values=MATCH_TIME_LIMIT_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + regimen_spacing, regimen_width, regimen_height),
            "Rounds per batch",
            ROUNDS_PER_BATCH_VALUES[0],
            ROUNDS_PER_BATCH_VALUES[-1],
            state.rounds_per_batch,
            is_int=True,
            values=ROUNDS_PER_BATCH_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 2 * regimen_spacing, regimen_width, regimen_height),
            "Batch grouping",
            BATCH_GROUPING_VALUES[0],
            BATCH_GROUPING_VALUES[-1],
            state.batch_grouping,
            is_int=True,
            values=BATCH_GROUPING_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 3 * regimen_spacing, regimen_width, regimen_height),
            "Replay size, batch=30k",
            REPLAY_BUFFER_SIZE_VALUES[0],
            REPLAY_BUFFER_SIZE_VALUES[-1],
            state.replay_buffer_size,
            is_int=True,
            values=REPLAY_BUFFER_SIZE_VALUES,
            value_formatter=_format_short_count,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 4 * regimen_spacing, regimen_width, regimen_height),
            "Minibatch size",
            MINIBATCH_SIZE_VALUES[0],
            MINIBATCH_SIZE_VALUES[-1],
            state.minibatch_size,
            is_int=True,
            values=MINIBATCH_SIZE_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 5 * regimen_spacing, regimen_width, regimen_height),
            "Gradient steps, UTD=1.1",
            REPLAY_UPDATES_PER_BATCH_VALUES[0],
            REPLAY_UPDATES_PER_BATCH_VALUES[-1],
            state.replay_updates_per_batch,
            is_int=True,
            values=REPLAY_UPDATES_PER_BATCH_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 6 * regimen_spacing, regimen_width, regimen_height),
            "Learning rate",
            LEARNING_RATE_VALUES[0],
            LEARNING_RATE_VALUES[-1],
            state.learning_rate,
            step=0.00001,
            values=LEARNING_RATE_VALUES,
            decimal_places=5,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 7 * regimen_spacing, regimen_width, regimen_height),
            "Starting Epsilon",
            EPSILON_VALUES[0],
            EPSILON_VALUES[-1],
            state.starting_epsilon,
            step=0.025,
            values=EPSILON_VALUES,
            decimal_places=3,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 8 * regimen_spacing, regimen_width, regimen_height),
            "Epsilon decay",
            EPSILON_DECAY_VALUES[0],
            EPSILON_DECAY_VALUES[-1],
            state.epsilon_decay,
            step=0.001,
            values=EPSILON_DECAY_VALUES,
            decimal_places=3,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 9 * regimen_spacing, regimen_width, regimen_height),
            "Epsilon frame span",
            EPSILON_FRAME_SPAN_VALUES[0],
            EPSILON_FRAME_SPAN_VALUES[-1],
            state.epsilon_frame_span,
            is_int=True,
            values=EPSILON_FRAME_SPAN_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 10 * regimen_spacing, regimen_width, regimen_height),
            "Gamma",
            GAMMA_VALUES[0],
            GAMMA_VALUES[-1],
            state.gamma,
            step=0.001,
            values=GAMMA_VALUES,
            decimal_places=3,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 11 * regimen_spacing, regimen_width, regimen_height),
            "Hidden layer size",
            HIDDEN_LAYER_SIZE_VALUES[0],
            HIDDEN_LAYER_SIZE_VALUES[-1],
            state.hidden_layer_size,
            is_int=True,
            values=HIDDEN_LAYER_SIZE_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
        SliderRow(
            (regimen_left, regimen_top + 12 * regimen_spacing, regimen_width, regimen_height),
            "Hidden layer count",
            HIDDEN_LAYER_COUNT_VALUES[0],
            HIDDEN_LAYER_COUNT_VALUES[-1],
            state.hidden_layer_count,
            is_int=True,
            values=HIDDEN_LAYER_COUNT_VALUES,
            layout=regimen_layout,
            slider_width=regimen_slider_width,
        ),
    )

    display_checkbox = None  # Instantiated later with other action buttons
    exited = [False]
    stopping_background_instances = [False]
    last_starting_epsilon_slider_value = [state.starting_epsilon]
    batch_log_box = TrainingBatchLogBox()

    def sync_state_from_ui():
        instance_manager.set_active_display(display_checkbox.value)
        state.slot_labels[:] = [field.text for field in slot_fields]
        state.rewards.update(
            (slider.label, slider.value) for slider in reward_sliders
        )
        state.ai_opponent_chance = ai_opponent_slider.value
        state.opponent_mode = "all" if state.ai_opponent_chance > 0 else "simple"
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
        starting_epsilon = regimen_sliders[
            REGIMEN_STARTING_EPSILON_INDEX
        ].value
        if starting_epsilon != last_starting_epsilon_slider_value[0]:
            state.starting_epsilon = starting_epsilon
            state.current_epsilon = starting_epsilon
            last_starting_epsilon_slider_value[0] = starting_epsilon
            active_session = instance_manager.active_session
            if active_session is not None:
                active_session.set_starting_epsilon(starting_epsilon)
        else:
            state.starting_epsilon = starting_epsilon
        state.epsilon_decay = regimen_sliders[REGIMEN_EPSILON_DECAY_INDEX].value
        state.epsilon_frame_span = int(
            regimen_sliders[REGIMEN_EPSILON_FRAME_SPAN_INDEX].value
        )
        state.gamma = regimen_sliders[REGIMEN_GAMMA_INDEX].value
        state.hidden_layer_size = int(
            regimen_sliders[REGIMEN_HIDDEN_LAYER_SIZE_INDEX].value
        )
        state.hidden_layer_count = int(
            regimen_sliders[REGIMEN_HIDDEN_LAYER_COUNT_INDEX].value
        )

    def apply_state_to_ui():
        nonlocal trainee_scroll_y, rewards_scroll_y
        display_checkbox.is_checked = state.display_on
        refresh_slot_controls(load_labels=False)
        for slider in reward_sliders:
            _set_slider_value(slider, state.rewards.get(slider.label, 0.0))
        _set_slider_value(ai_opponent_slider, state.ai_opponent_chance)
        _set_slider_value(simple_activity_sliders[0], state.forward_activity)
        _set_slider_value(simple_activity_sliders[1], state.a1_activity)
        _set_slider_value(simple_activity_sliders[2], state.a2_activity)
        _set_slider_value(simple_activity_sliders[3], state.face_opponent_activity)
        _set_slider_value(
            regimen_sliders[REGIMEN_REPLAY_BUFFER_INDEX],
            state.replay_buffer_size,
        )
        _set_slider_value(
            regimen_sliders[REGIMEN_ROUNDS_PER_BATCH_INDEX],
            state.rounds_per_batch,
        )
        _set_slider_value(
            regimen_sliders[REGIMEN_BATCH_GROUPING_INDEX],
            state.batch_grouping,
        )
        _set_slider_value(
            regimen_sliders[REGIMEN_MATCH_TIME_LIMIT_INDEX],
            state.match_time_limit,
        )
        _set_slider_value(
            regimen_sliders[REGIMEN_MINIBATCH_SIZE_INDEX],
            state.minibatch_size,
        )
        _set_slider_value(
            regimen_sliders[REGIMEN_REPLAY_UPDATES_INDEX],
            state.replay_updates_per_batch,
        )
        _set_slider_value(regimen_sliders[REGIMEN_LEARNING_RATE_INDEX], state.learning_rate)
        _set_slider_value(
            regimen_sliders[REGIMEN_STARTING_EPSILON_INDEX],
            state.starting_epsilon,
        )
        _set_slider_value(regimen_sliders[REGIMEN_EPSILON_DECAY_INDEX], state.epsilon_decay)
        _set_slider_value(
            regimen_sliders[REGIMEN_EPSILON_FRAME_SPAN_INDEX],
            state.epsilon_frame_span,
        )
        _set_slider_value(regimen_sliders[REGIMEN_GAMMA_INDEX], state.gamma)
        _set_slider_value(
            regimen_sliders[REGIMEN_HIDDEN_LAYER_SIZE_INDEX],
            state.hidden_layer_size,
        )
        _set_slider_value(
            regimen_sliders[REGIMEN_HIDDEN_LAYER_COUNT_INDEX],
            state.hidden_layer_count,
        )
        last_starting_epsilon_slider_value[0] = state.starting_epsilon
        trainee_scroll_y = 0
        rewards_scroll_y = 0
        batch_log_box.set_lines(())

    def architecture_metadata():
        return model_architecture_metadata(
            state.hidden_layer_size,
            state.hidden_layer_count,
        )

    def training_metadata(*, reset_checkpoint=False):
        current_epsilon = _epsilon_for_model_update(
            state.starting_epsilon,
            state.current_epsilon,
            reset_checkpoint=reset_checkpoint,
        )
        return {
            "opponent": {
                "mode": state.opponent_mode,
                "ai_opponent_chance": state.ai_opponent_chance,
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
                "starting_epsilon": state.starting_epsilon,
                "current_epsilon": current_epsilon,
                "epsilon": current_epsilon,
                "epsilon_decay": state.epsilon_decay,
                "epsilon_frame_span": state.epsilon_frame_span,
                "gamma": state.gamma,
            },
        }

    def selected_model_slot():
        if state.selected_ship is None:
            return None
        return slot_models[state.selected_slot - 1]

    def show_notice(text):
        notice[0] = TrainingNotice(text)

    def select_training_instance(instance_id):
        nonlocal state
        if instance_id == instance_manager.active_instance_id:
            return
        sync_state_from_ui()
        instance_manager.select_instance(instance_id)
        state = instance_manager.active_state
        apply_state_to_ui()
        refresh_slot_controls(load_labels=False)

    def add_training_instance():
        nonlocal state
        sync_state_from_ui()
        if not instance_manager.can_add_instance():
            show_notice(
                f"Only {instance_manager.supported_max} training instances are supported"
            )
            return

        def create_instance():
            nonlocal state
            instance = instance_manager.add_instance()
            state = instance.state
            apply_state_to_ui()
            show_notice(f"Added {instance.label}")

        if instance_manager.add_requires_confirmation():
            confirmation_prompt[0] = ConfirmationPrompt(
                (
                    f"You already have {len(instance_manager.instances)} training "
                    "instances. Add another?"
                ),
                create_instance,
            )
            return

        create_instance()

    def close_active_training_instance():
        nonlocal state
        sync_state_from_ui()
        result = instance_manager.request_close_active_instance()
        if result == "last":
            show_notice("At least one training instance must remain")
            return
        if result == "pending":
            show_notice("Closing instance after training stops")
        else:
            show_notice("Closed training instance")
        state = instance_manager.active_state
        apply_state_to_ui()

    def refresh_slot_controls(*, load_labels=True):
        if state.selected_ship is None:
            for index, (field, delete_button) in enumerate(zip(slot_fields, delete_buttons)):
                field.text = "" if load_labels else state.slot_labels[index]
                field.enabled = False
                field.text_color = ui.GREY
                delete_button.enabled = False
            state.slot_labels[:] = [field.text for field in slot_fields]
            return

        slot_models[:] = model_repository.slots_for_ship(state.selected_ship)
        for index, (field, delete_button, model_slot) in enumerate(zip(slot_fields, delete_buttons, slot_models)):
            field.text = model_slot.description if load_labels else state.slot_labels[index]
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
        state.slot_labels[:] = [field.text for field in slot_fields]

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
                            and _training_settings_match(saved_training, current_training)
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
        updated_training = training_metadata(reset_checkpoint=reset_checkpoint)
        metadata = metadata_from_state(
            ship=state.selected_ship,
            slot=state.selected_slot,
            description=description,
            architecture=architecture_metadata(),
            training=updated_training,
            progress=_progress_for_model_update(
                existing_metadata,
                progress,
                reset_checkpoint=reset_checkpoint,
            ),
        )
        updated_slot = model_repository.create_or_update_user_model(metadata)
        if reset_checkpoint:
            _clear_reset_model_artifacts(updated_slot)
            state.current_epsilon = _epsilon_for_model_update(
                state.starting_epsilon,
                state.current_epsilon,
                reset_checkpoint=True,
            )
            
        state.loaded_ship = state.selected_ship
        state.loaded_slot = state.selected_slot
        state.loaded_architecture = architecture_metadata()
        state.loaded_training = updated_training
        
        refresh_slot_controls()
        return updated_slot

    def changed_training_groups(old_training, new_training):
        return [
            name
            for name in ("opponent", "rewards", "regimen")
            if not _training_settings_match(
                {name: old_training.get(name)},
                {name: new_training.get(name)},
            )
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
                if "ai_opponent_chance" in opponent:
                    try:
                        value = float(opponent["ai_opponent_chance"])
                    except (TypeError, ValueError):
                        skipped.append(ai_opponent_slider.label)
                    else:
                        if not _set_slider_value(ai_opponent_slider, value):
                            skipped.append(ai_opponent_slider.label)
                else:
                    mode = opponent.get("mode")
                    if mode in {"all", "simple"}:
                        _set_slider_value(
                            ai_opponent_slider,
                            100.0 if mode == "all" else 0.0,
                        )
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
                if "current_epsilon" in regimen:
                    try:
                        float(regimen["current_epsilon"])
                    except (TypeError, ValueError):
                        skipped.append("current epsilon")
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
                    (
                        "epsilon_decay",
                        regimen_sliders[REGIMEN_EPSILON_DECAY_INDEX],
                        float,
                    ),
                    (
                        "epsilon_frame_span",
                        regimen_sliders[REGIMEN_EPSILON_FRAME_SPAN_INDEX],
                        int,
                    ),
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
                starting_epsilon_key = (
                    "starting_epsilon"
                    if "starting_epsilon" in regimen
                    else "epsilon"
                )
                if starting_epsilon_key in regimen:
                    try:
                        value = float(regimen[starting_epsilon_key])
                    except (TypeError, ValueError):
                        skipped.append("starting epsilon")
                    else:
                        if not _set_slider_value(
                            regimen_sliders[REGIMEN_STARTING_EPSILON_INDEX],
                            value,
                        ):
                            skipped.append("starting epsilon")

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
        if isinstance(training, dict):
            regimen = training.get("regimen", {})
            if isinstance(regimen, dict):
                try:
                    state.current_epsilon = float(
                        regimen.get("current_epsilon", state.starting_epsilon)
                    )
                except (TypeError, ValueError):
                    state.current_epsilon = state.starting_epsilon
        last_starting_epsilon_slider_value[0] = state.starting_epsilon
        state.loaded_ship = state.selected_ship
        state.loaded_slot = state.selected_slot
        state.loaded_architecture = architecture_metadata()
        state.loaded_training = training_metadata()

    def clear_session_continuity():
        instance_manager.clear_active_session_continuity()

    def session_continuity_for(model_slot):
        active_session = instance_manager.active_session
        if (
            active_session is None
            or active_session.slot.ship != model_slot.ship
            or active_session.slot.slot != model_slot.slot
        ):
            return (), ()
        return active_session.history, active_session.log_lines

    def request_back():
        def stop_all_running_instances():
            instance_manager.request_stop_all_running()
            stopping_background_instances[0] = True
            display_checkbox.is_checked = False
            show_notice("Stopping background instances")

        def leave_training_screen():
            exited[0] = True

        action = instance_manager.back_action()
        if action == "stop_all":
            confirmation_prompt[0] = ConfirmationPrompt(
                "Do you want to stop all running training instances?",
                stop_all_running_instances,
            )
        elif action == "exit":
            confirmation_prompt[0] = ConfirmationPrompt(
                "Do you want to leave AI training?",
                leave_training_screen,
            )

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

        active_instance = instance_manager.active_instance
        if not instance_manager.reserve_writer(
            active_instance,
            model_slot.ship,
            model_slot.slot,
        ):
            show_notice(
                f"{model_slot.ship}-{model_slot.slot:02d} is already training"
            )
            return

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
            instance_manager.set_active_session(session)
            state.running = True
            session.start()
            show_notice(f"Training {describe_model(model_slot)}")
        except (TrainingSessionError, RuntimeError, ValueError) as exc:
            instance_manager.release_writer(active_instance)
            show_notice(str(exc))

    def start_selected_model():
        active_session = instance_manager.active_session
        if instance_manager.is_running_or_stopping(instance_manager.active_instance):
            instance_manager.request_stop_active()
            display_checkbox.is_checked = False
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
    action_width = (CONTROL_WIDTH - 2 * TAB_MARGIN - 2 * action_gap) // 3
    display_checkbox = ui_button.Checkbox(
        TAB_MARGIN,
        ACTION_TOP,
        action_width,
        FOOTER_CONTROL_HEIGHT,
        "Display On",
    )
    start_stop_button = ui_button.Button(
        TAB_MARGIN + action_width + action_gap,
        ACTION_TOP,
        action_width,
        FOOTER_CONTROL_HEIGHT,
        "Start",
        start_selected_model,
        ui.OK_GREEN,
        ui.OK_GREEN_HI,
    )
    back_button = ui_button.Button(
        TAB_MARGIN + 2 * (action_width + action_gap),
        ACTION_TOP,
        action_width,
        FOOTER_CONTROL_HEIGHT,
        "Back",
        request_back,
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )
    instance_indicator_rect = pygame.Rect(
        TAB_MARGIN,
        INSTANCE_TOP,
        INSTANCE_POSITION_WIDTH,
        INSTANCE_CONTROL_HEIGHT,
    )
    running_indicator_rect = pygame.Rect(
        instance_indicator_rect.right + INSTANCE_GAP,
        INSTANCE_TOP,
        INSTANCE_RUNNING_WIDTH,
        INSTANCE_CONTROL_HEIGHT,
    )
    instance_dropdown_rect = pygame.Rect(
        running_indicator_rect.right + INSTANCE_GAP,
        INSTANCE_TOP,
        CONTROL_WIDTH
        - 2 * TAB_MARGIN
        - instance_indicator_rect.width
        - running_indicator_rect.width
        - INSTANCE_CLOSE_WIDTH
        - INSTANCE_ADD_WIDTH
        - 4 * INSTANCE_GAP,
        INSTANCE_CONTROL_HEIGHT,
    )
    close_instance_button = ui_button.Button(
        instance_dropdown_rect.right + INSTANCE_GAP,
        INSTANCE_TOP,
        INSTANCE_CLOSE_WIDTH,
        INSTANCE_CONTROL_HEIGHT,
        "Close",
        close_active_training_instance,
        ui.CAN_RED,
        ui.CAN_RED_HI,
    )
    add_instance_button = ui_button.Button(
        close_instance_button.rect.right + INSTANCE_GAP,
        INSTANCE_TOP,
        INSTANCE_ADD_WIDTH,
        INSTANCE_CONTROL_HEIGHT,
        "Add",
        add_training_instance,
        ui.OK_GREEN,
        ui.OK_GREEN_HI,
    )
    instance_dropdown = InstanceDropdown(
        instance_dropdown_rect,
        instance_manager,
        select_training_instance,
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

            if instance_dropdown.handle_event(event, menu_sound_manager):
                continue
            close_instance_button.handle_event(event, menu_sound_manager)
            add_instance_button.handle_event(event, menu_sound_manager)

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
                enabled = not state.running
                ai_opponent_slider.enabled = enabled
                ai_opponent_slider.handle_event(translated, menu_sound_manager)
                enabled = state.simple_behavior_controls_enabled
                for slider in simple_activity_sliders:
                    slider.enabled = enabled
                    slider.handle_event(translated, menu_sound_manager)
            else:
                for slider in regimen_sliders:
                    slider.handle_event(event, menu_sound_manager)

        sync_state_from_ui()
        controls_enabled = state.simple_behavior_controls_enabled
        ai_opponent_slider.enabled = not state.running
        for slider in simple_activity_sliders:
            slider.enabled = controls_enabled

        for slider in reward_sliders:
            slider.enabled = not state.running
        for slider in regimen_sliders:
            slider.enabled = not state.running
            
        max_batch_frames = state.rounds_per_batch * state.match_time_limit * len(SHIP_DEFINITIONS)
        regimen_sliders[
            REGIMEN_REPLAY_BUFFER_INDEX
        ].label = f"Replay size, batch={_format_short_count(max_batch_frames)}"
        buffer_size_mb = int((state.replay_buffer_size / 1000) * 5)
        regimen_sliders[REGIMEN_REPLAY_BUFFER_INDEX].value_suffix = f" ({buffer_size_mb}MB)"
        if max_batch_frames > 0:
            utd = (state.minibatch_size * state.replay_updates_per_batch) / max_batch_frames
            regimen_sliders[
                REGIMEN_REPLAY_UPDATES_INDEX
            ].label = f"Gradient steps, UTD={utd:.1f}"
        else:
            regimen_sliders[REGIMEN_REPLAY_UPDATES_INDEX].label = "Gradient steps"

        previous_active_id = instance_manager.active_instance_id
        for instance in list(instance_manager.instances):
            session = instance.session
            status = session.status if session is not None else None
            if session is None or status is None:
                instance.state.running = False
                instance.last_running = False
                continue

            instance.state.running = status.running
            if instance.instance_id == instance_manager.active_instance_id:
                state.current_epsilon = float(
                    getattr(status, "current_epsilon", state.current_epsilon)
                )
                batch_log_box.set_lines(
                    _display_off_console_lines(status, session.log_lines)
                )
                if not instance.last_running and status.running:
                    refresh_slot_controls()
                if instance.last_running and not status.running:
                    display_checkbox.is_checked = False
                    instance_manager.disable_display(instance)
                    if status.error:
                        show_notice(status.error)
                    else:
                        show_notice("Training stopped")
                    refresh_slot_controls()

            if instance.last_running and not status.running:
                instance_manager.disable_display(instance)
                instance_manager.release_writer(instance)
            instance.last_running = status.running

        instance_manager.cleanup_stopped_pending_removals()
        if instance_manager.active_instance_id != previous_active_id:
            state = instance_manager.active_state
            apply_state_to_ui()

        active_instance = instance_manager.active_instance
        active_session = active_instance.session
        session_status = active_session.status if active_session is not None else None
        background_running = instance_manager.background_instances_running()
        active_running = instance_manager.is_running_or_stopping(active_instance)
        any_running = instance_manager.any_instance_running()

        if stopping_background_instances[0]:
            if background_running:
                if notice[0] is None or notice[0].text == "Stopping background instances":
                    show_notice("Stopping background instances")
            elif not any_running:
                stopping_background_instances[0] = False

        update_field_colors()
        selected_slot = selected_model_slot()
        back_button.text = "Stop All" if background_running else "Back"
        back_button.enabled = background_running or not active_running
        if background_running:
            back_button.bg_color = (*ui.CAN_RED[:3], const.TAB_BUTTON_HOVER_ALPHA)
            back_button.hover_color = ui.CAN_RED_HI
        else:
            back_button.bg_color = ui.CAN_RED
            back_button.hover_color = ui.CAN_RED_HI
        close_instance_button.enabled = (
            len(instance_manager.instances) > 1
            or instance_manager.is_running_or_stopping(active_instance)
        )
        add_instance_button.enabled = instance_manager.can_add_instance()
        start_stop_button.enabled = (
            state.selected_ship is not None
            and selected_slot is not None
            and not selected_slot.is_bundled
            and (
                (
                    session_status is not None
                    and (session_status.running or session_status.stopping)
                )
                or bool(slot_fields[state.selected_slot - 1].text.strip())
            )
        )

        is_currently_loaded = (
            state.loaded_ship == state.selected_ship
            and state.loaded_slot == state.selected_slot
            and state.loaded_architecture == architecture_metadata()
            and _training_settings_match(state.loaded_training, training_metadata())
        )

        if is_currently_loaded and state.selected_ship is not None:
            load_button.text = f"{state.selected_ship}-{state.selected_slot:02d} Loaded"
            load_button.enabled = False
        else:
            load_button.text = "Load"
            load_button.enabled = selected_slot is not None and selected_slot.is_user and not state.running

        if session_status is not None and (
            session_status.running or session_status.stopping
        ):
            start_stop_button.text = "Stop"
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
                    arena_font,
                    small_font,
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
                enabled = not state.running
                _draw_group_panel(content, panel, hovered, enabled)
            ai_opponent_slider.draw(content, opponent_font)
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

        pygame.draw.rect(screen, ui.BLACK, instance_indicator_rect, border_radius=5)
        pygame.draw.rect(screen, ui.LIGHT_GREY, instance_indicator_rect, 1, border_radius=5)
        indicator_text = small_font.render(
            instance_manager.active_position_text(),
            True,
            ui.WHITE,
        )
        screen.blit(
            indicator_text,
            indicator_text.get_rect(center=instance_indicator_rect.center),
        )
        running_color = (
            ui.BRIGHT_GREEN if instance_manager.running_count() > 0 else ui.LIGHT_GREY
        )
        pygame.draw.rect(screen, ui.BLACK, running_indicator_rect, border_radius=5)
        pygame.draw.rect(screen, running_color, running_indicator_rect, 1, border_radius=5)
        running_text = small_font.render(
            instance_manager.running_count_text(),
            True,
            running_color,
        )
        screen.blit(
            running_text,
            running_text.get_rect(center=running_indicator_rect.center),
        )
        instance_dropdown.draw(screen, small_font)
        close_instance_button.draw(screen, small_font)
        add_instance_button.draw(screen, small_font)
        separator_y = INSTANCE_TOP + INSTANCE_CONTROL_HEIGHT + 5
        pygame.draw.line(
            screen,
            ui.WHITE,
            (TAB_MARGIN, separator_y),
            (CONTROL_WIDTH - TAB_MARGIN, separator_y),
            INSTANCE_SEPARATOR_HEIGHT,
        )

        display_checkbox.draw(screen, body_font)
        start_stop_button.draw(screen, body_font)
        back_button.draw(screen, body_font)

        if state.running:
            pygame.draw.rect(screen, ui.BLACK, display_checkbox.rect, 2, border_radius=5)
            pygame.draw.rect(screen, ui.BLACK, start_stop_button.rect, 2, border_radius=5)
        if background_running:
            pygame.draw.rect(screen, ui.BLACK, back_button.rect, 2, border_radius=5)
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
