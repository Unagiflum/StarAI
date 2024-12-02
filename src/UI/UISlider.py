import pygame
from . import UI

class Slider:
    def __init__(self, x, y, width, min_val, max_val, start_val, label, is_int=False, step=1,
                 bg_color=UI.SLIDER_BG, hover_color=UI.SLIDER_BG_HI):
        self.rect = pygame.Rect(x, y, width, int(0.02 * UI.SCREEN_HEIGHT))
        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.label = label
        self.is_int = is_int
        self.step = step
        self.handle_x = self.value_to_position(self.value)
        self.handle_radius = int(0.015 * UI.SCREEN_HEIGHT)
        self.dragging = False
        self.bg_color = (*bg_color, 255) if len(bg_color) == 3 else bg_color
        self.hover_color = (*hover_color, 255) if len(hover_color) == 3 else hover_color
        self.is_hovered = False
        self.line_rect = pygame.Rect(self.rect.x, self.rect.y+int(0.003 * UI.SCREEN_HEIGHT),
                                   self.rect.width, int(0.015 * UI.SCREEN_HEIGHT))
        self.decimal_places = abs(len(str(self.step).split('.')[-1])) if '.' in str(self.step) else 0

    def value_to_position(self, value):
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)

    def position_to_value(self, pos_x):
        ratio = (pos_x - self.rect.x) / self.rect.width
        value = self.min_val + ratio * (self.max_val - self.min_val)
        value = round(value / self.step) * self.step
        return max(self.min_val, min(self.max_val, value))

    def adjust_value(self, increment):
        new_value = self.value + (self.step if increment else -self.step)
        self.value = max(self.min_val, min(self.max_val, new_value))
        self.handle_x = self.value_to_position(self.value)

    def handle_event(self, event, sound_manager=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                handle_rect = self.get_handle_rect()
                if handle_rect.collidepoint(event.pos):
                    if sound_manager:
                        sound_manager.play_sound('menu')
                    self.dragging = True
                elif self.line_rect.collidepoint(event.pos):
                    if sound_manager:
                        sound_manager.play_sound('menu')
                    self.value = self.position_to_value(event.pos[0])
                    self.handle_x = self.value_to_position(self.value)
            elif self.is_hovered:  # Mouse wheel
                if event.button == 4:  # Scroll up
                    if sound_manager:
                        sound_manager.play_sound('menu')
                    self.adjust_value(True)
                elif event.button == 5:  # Scroll down
                    if sound_manager:
                        sound_manager.play_sound('menu')
                    self.adjust_value(False)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            padding_x = int(0.02 * UI.SCREEN_WIDTH)
            padding_y = int(0.02 * UI.SCREEN_HEIGHT)
            bg_rect = pygame.Rect(
                self.rect.x - padding_x,
                self.rect.y - int(0.04 * UI.SCREEN_HEIGHT),
                self.rect.width + 2 * padding_x,
                0.08 * UI.SCREEN_HEIGHT
            )
            self.is_hovered = bg_rect.collidepoint(mouse_pos)

            if self.dragging:
                mouse_x = max(self.rect.x, min(self.rect.x + self.rect.width, event.pos[0]))
                self.value = self.position_to_value(mouse_x)
                self.handle_x = self.value_to_position(self.value)

    def get_handle_rect(self):
        return pygame.Rect(
            self.handle_x - self.handle_radius,
            self.rect.y - self.handle_radius + int(0.01 * UI.SCREEN_HEIGHT),
            self.handle_radius * 2,
            self.handle_radius * 2
        )

    def draw(self, surface, font):
        padding_x = int(0.02 * UI.SCREEN_WIDTH)
        padding_y = int(0.02 * UI.SCREEN_HEIGHT)
        bg_x = self.rect.x - padding_x
        bg_y = self.rect.y - int(0.04 * UI.SCREEN_HEIGHT)
        bg_width = self.rect.width + 2 * padding_x
        bg_height = 0.08 * UI.SCREEN_HEIGHT

        bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
        bg_color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(bg_surface, bg_color, bg_surface.get_rect(), border_radius=5)
        surface.blit(bg_surface, (bg_x, bg_y))

        pygame.draw.rect(surface, UI.SLIDER_LINE, self.line_rect)
        pygame.draw.circle(surface, UI.HANDLE_COLOR, (self.handle_x, self.rect.y + int(0.01 * UI.SCREEN_HEIGHT)),
                           self.handle_radius)

        label_text = f"{self.label}: {self.format_value()}"
        label_surf = font.render(label_text, True, UI.WHITE)
        label_rect = label_surf.get_rect(topleft=(self.rect.x, self.rect.y - int(0.03 * UI.SCREEN_HEIGHT)))
        surface.blit(label_surf, label_rect)

    def format_value(self):
        if self.is_int:
            return f"{int(self.value)}"
        return f"{self.value:.{self.decimal_places}f}"