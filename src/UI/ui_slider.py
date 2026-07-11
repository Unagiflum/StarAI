import pygame
from . import ui
import src.const as Const


class Slider:
    def __init__(
        self,
        x,
        y,
        width,
        min_val,
        max_val,
        start_val,
        label,
        is_int=False,
        step=1,
        bg_color=ui.SLIDER_BG,
        hover_color=ui.SLIDER_BG_HI,
        values=None,
        height=None,
        decimal_places=None,
        value_suffix="",
    ):
        self.bg_rect_height = height if height is not None else int(0.07 * Const.SCREEN_HEIGHT)
        self.rect = pygame.Rect(x, y, width, self.bg_rect_height)
        self.line_width = int(self.bg_rect_height / 15)
        self.padding_x = int(self.bg_rect_height / 5)
        self.padding_y = int(self.bg_rect_height / 5)
        self.handle_radius = int(self.bg_rect_height / 8)
        self.handle_offset = int(self.bg_rect_height * 0.75)
        self.bg_rect = pygame.Rect(
            self.rect.x, self.rect.y, self.rect.width, self.bg_rect_height
        )
        self.line_rect = pygame.Rect(
            self.rect.x + self.padding_x,
            self.rect.y + self.handle_offset - self.line_width,
            self.rect.width - 2 * self.padding_x,
            2 * self.line_width,
        )

        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.label = label
        self.is_int = is_int
        self.step = step
        self.values = tuple(values) if values is not None else None
        if self.values is not None:
            if len(self.values) < 2 or start_val not in self.values:
                raise ValueError("Slider values must contain the starting value")

        self.dragging = False
        self.bg_color = (*bg_color, 255) if len(bg_color) == 3 else bg_color
        self.hover_color = (*hover_color, 255) if len(hover_color) == 3 else hover_color
        self.is_hovered = False
        self.enabled = True
        if decimal_places is not None:
            self.decimal_places = decimal_places
        else:
            self.decimal_places = (
                abs(len(str(self.step).split(".")[-1])) if "." in str(self.step) and "e" not in str(self.step) else 0
            )
        self.value_suffix = value_suffix
        self.handle_x = self.value_to_position(self.value)

    def value_to_position(self, value):
        if self.values is not None:
            ratio = self.values.index(value) / (len(self.values) - 1)
            return self.line_rect.x + int(ratio * self.line_rect.width)
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.line_rect.x + int(ratio * self.line_rect.width)

    def position_to_value(self, pos_x):
        ratio = (pos_x - self.line_rect.x) / self.line_rect.width
        if self.values is not None:
            index = round(ratio * (len(self.values) - 1))
            index = max(0, min(len(self.values) - 1, index))
            return self.values[index]
        value = self.min_val + ratio * (self.max_val - self.min_val)
        value = round(value / self.step) * self.step
        return max(self.min_val, min(self.max_val, value))

    def adjust_value(self, increment):
        if self.values is not None:
            index = self.values.index(self.value) + (1 if increment else -1)
            index = max(0, min(len(self.values) - 1, index))
            self.value = self.values[index]
            self.handle_x = self.value_to_position(self.value)
            return
        new_value = self.value + (self.step if increment else -self.step)
        self.value = max(self.min_val, min(self.max_val, new_value))
        self.handle_x = self.value_to_position(self.value)

    def handle_event(self, event, sound_manager=None):
        if not self.enabled:
            self.dragging = False
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                handle_rect = self.get_handle_rect()
                if handle_rect.collidepoint(event.pos):
                    if sound_manager:
                        sound_manager.play_sound("menu")
                    self.dragging = True
                elif self.line_rect.collidepoint(event.pos):
                    if sound_manager:
                        sound_manager.play_sound("menu")
                    self.value = self.position_to_value(event.pos[0])
                    self.handle_x = self.value_to_position(self.value)
            elif self.is_hovered:  # Mouse wheel
                if event.button == 4:  # Scroll up
                    if sound_manager:
                        sound_manager.play_sound("menu")
                    self.adjust_value(True)
                elif event.button == 5:  # Scroll down
                    if sound_manager:
                        sound_manager.play_sound("menu")
                    self.adjust_value(False)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            self.is_hovered = self.bg_rect.collidepoint(mouse_pos)

            if self.dragging:
                mouse_x = max(
                    self.rect.x, min(self.rect.x + self.rect.width, event.pos[0])
                )
                self.value = self.position_to_value(mouse_x)
                self.handle_x = self.value_to_position(self.value)

    def get_handle_rect(self):
        return pygame.Rect(
            self.handle_x - self.handle_radius,
            self.rect.y - self.handle_radius + self.handle_offset,
            self.handle_radius * 2,
            self.handle_radius * 2,
        )

    def draw(self, surface, font):
        # Create background surface with alpha
        bg_surface = pygame.Surface(self.bg_rect.size, pygame.SRCALPHA)
        if not self.enabled:
            bg_color = (*ui.DARK_GREY, 255)
        else:
            bg_color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(bg_surface, bg_color, bg_surface.get_rect(), border_radius=5)
        surface.blit(bg_surface, self.bg_rect.topleft)

        # Draw slider line and handle
        pygame.draw.rect(surface, ui.SLIDER_LINE, self.line_rect)
        pygame.draw.circle(
            surface,
            ui.HANDLE_COLOR if self.enabled else ui.GREY,
            (self.handle_x, self.rect.y + self.handle_offset),
            self.handle_radius,
        )

        # Draw label
        label_text = f"{self.label}: {self.format_value()}"
        label_color = ui.WHITE if self.enabled else ui.GREY
        label_surf = font.render(label_text, True, label_color)
        label_rect = label_surf.get_rect(
            topleft=(self.line_rect.x, self.rect.y + self.padding_y)
        )
        surface.blit(label_surf, label_rect)

    def format_value(self):
        if self.is_int:
            return f"{int(self.value)}{self.value_suffix}"
        return f"{self.value:.{self.decimal_places}f}{self.value_suffix}"
