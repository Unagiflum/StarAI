import pygame
from . import UI

class Button:
    def __init__(self, x, y, width, height, text, callback, bg_color=UI.GREY, hover_color=UI.LIGHT_GREY, text_color=UI.WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.bg_color = (*bg_color, 255) if len(bg_color) == 3 else bg_color
        self.hover_color = (*hover_color, 255) if len(hover_color) == 3 else hover_color
        self.text_color = text_color
        self.enabled = True

    def handle_event(self, event, sound_manager=None):
        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if UI.sound_manager:
                    UI.sound_manager.play_sound('menu')
                self.callback()

    def draw(self, surface, font):
        if not self.enabled:
            color = (*UI.DARK_GREY, 255)
        else:
            mouse_pos = pygame.mouse.get_pos()
            color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.bg_color

        button_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(button_surface, color, button_surface.get_rect(), border_radius=5)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=button_surface.get_rect().center)
        button_surface.blit(text_surf, text_rect)

        surface.blit(button_surface, self.rect)

class ToggleButton(Button):
    def __init__(self, x, y, width, height, text, callback=None, initial_state=False,
                 bg_color=UI.MENU_BUTTON_COLOR, active_color=UI.BRIGHT_GREEN, text_color=UI.WHITE, hover_color=UI.MENU_BUTTON_COLOR_HI):
        super().__init__(x, y, width, height, text, callback, bg_color, hover_color, text_color)
        self.is_on = initial_state
        self.active_color = (*active_color, 255) if len(active_color) == 3 else active_color

        self.switch_width = int(0.02 * UI.SCREEN_WIDTH)
        self.switch_height = int(0.02 * UI.SCREEN_WIDTH)
        self.switch_rect = pygame.Rect(
            self.rect.right - self.switch_width - int(0.005 * UI.SCREEN_WIDTH),
            self.rect.y + int(0.005 * UI.SCREEN_WIDTH),
            self.switch_width,
            self.switch_height
        )

        self.button_width = self.switch_width // 2
        self.button_height = self.switch_height

    def draw(self, surface, font):
        # Create background surface with alpha
        button_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)

        if not self.enabled:
            color = (*UI.DARK_GREY, 255)
        else:
            mouse_pos = pygame.mouse.get_pos()
            color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.bg_color

        pygame.draw.rect(button_surface, color, button_surface.get_rect(), border_radius=5)
        surface.blit(button_surface, self.rect)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        surface.blit(text_surf, text_rect)

        switch_color = self.active_color if self.is_on else (*UI.DARK_RED, 255) if len(UI.DARK_RED) == 3 else UI.DARK_RED
        switch_surface = pygame.Surface((self.switch_rect.width, self.switch_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(switch_surface, switch_color, switch_surface.get_rect(), border_radius=self.switch_height // 2)
        surface.blit(switch_surface, self.switch_rect)

        button_x = (self.switch_rect.right - self.button_width - 2
                    if self.is_on
                    else self.switch_rect.x + 2)
        button_rect = pygame.Rect(
            0,
            0,
            self.button_width - 2,
            self.button_height - 4
        )
        button_surface = pygame.Surface((button_rect.width, button_rect.height), pygame.SRCALPHA)
        button_color = self.active_color if self.is_on else (*UI.DARK_RED, 255) if len(UI.DARK_RED) == 3 else UI.DARK_RED
        pygame.draw.rect(button_surface, button_color, button_rect, border_radius=self.button_height // 2)
        surface.blit(button_surface, (button_x, self.switch_rect.y + 2))

    def handle_event(self, event, sound_manager=None):
        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if UI.sound_manager:
                    UI.sound_manager.play_sound('menu')
                self.is_on = not self.is_on
                if self.callback:
                    self.callback(self.is_on)

    @property
    def value(self):
        return self.is_on

class KeyBinding(Button):
    def __init__(self, x, y, width, height, label, default_key, callback=None,
                 bg_color=UI.DARK_GREY, hover_color=UI.DARK_GREEN, text_color=UI.WHITE):
        super().__init__(x, y, width, height, text=default_key.upper(), callback=callback,
                         bg_color=bg_color, hover_color=hover_color, text_color=text_color)
        self.label = label
        self.default_key = default_key
        self.key = default_key
        self.waiting_for_key = False

    def handle_event(self, event, sound_manager=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if sound_manager:
                    sound_manager.play_sound('menu')
                self.waiting_for_key = True
        elif event.type == pygame.KEYDOWN and self.waiting_for_key:
            if sound_manager:
                sound_manager.play_sound('menu')
            self.key = pygame.key.name(event.key)
            self.text = self.key.upper()
            self.waiting_for_key = False
            if self.callback:
                self.callback(self.label, self.key)

    def draw(self, surface, font):
        # Draw the label
        label_font = pygame.font.SysFont(None, int(0.03*UI.SCREEN_HEIGHT))
        label_surf = label_font.render(self.label, True, UI.WHITE)
        label_rect = label_surf.get_rect(midright=(self.rect.x - int(0.01*UI.SCREEN_WIDTH), self.rect.y + self.rect.height // 2))
        surface.blit(label_surf, label_rect)

        # Determine button color based on hover state
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            color = self.hover_color
        else:
            color = self.bg_color

        # Draw the button background
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, UI.WHITE, self.rect, 2, border_radius=5)  # Border

        # Decide which text to display
        if self.waiting_for_key:
            display_text = "Press a key..."
            text_color = UI.RED  # Change text color for prompt visibility
        else:
            display_text = self.key.upper()
            text_color = self.text_color

        # Render the appropriate text
        key_surf = font.render(display_text, True, text_color)
        key_rect = key_surf.get_rect(center=self.rect.center)
        surface.blit(key_surf, key_rect)
