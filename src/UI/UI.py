import pygame

SCREEN_HEIGHT = 960
SCREEN_WIDTH = int(SCREEN_HEIGHT*1.25)
FPS = 30

button_spaceH = int(SCREEN_WIDTH * 0.005)
button_spaceV = int(SCREEN_HEIGHT * 0.00625)

ok_button_width = int(0.150 * SCREEN_WIDTH)
ok_button_height = int(0.05 * SCREEN_HEIGHT)
ok_button_left = SCREEN_WIDTH // 2 - ok_button_width - int(0.0167*SCREEN_WIDTH)
can_button_left = SCREEN_WIDTH // 2 + int(0.0167*SCREEN_WIDTH)
ok_button_top = SCREEN_HEIGHT-ok_button_height-button_spaceV*4
# Colors
WHITE = (255, 255, 255)
LIGHT_GREY = (170, 170, 170)
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
BLACK = (0, 0, 0)

ORANGE = (255, 100, 0)
RED = (255,0,0)
DARK_RED = (50, 0, 0)

BRIGHT_GREEN = (150,255,100)
DARK_GREEN = (0, 50, 0)

OK_GREEN = (0, 155, 0)
OK_GREEN_HI = (100, 200, 100)
CAN_RED = (155, 0, 0)
CAN_RED_HI = (200, 100, 100)

MENU_BUTTON_COLOR = (0, 125, 125)
MENU_BUTTON_COLOR_HI = (0,175, 175)

MAIN_BUTTON_COLOR = (0, 0, 0, 125)
MAIN_BUTTON_COLOR_HI = (0, 0, 0, 255)

HANDLE_COLOR = (0, 0, 0)

BG_COLOR = (0, 0, 20)

def load_background(path, screen_width, screen_height):
    """Load and scale the background image to fit the screen."""
    try:
        background = pygame.image.load(path)
        return pygame.transform.scale(background, (screen_width, screen_height))
    except pygame.error as e:
        print(f"Could not load background image: {e}")
        return None


class Button:
    def __init__(self, x, y, width, height, text, callback, bg_color=GREY, hover_color=LIGHT_GREY, text_color=WHITE):
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
                if sound_manager:
                    sound_manager.play_sound('menu')
                self.callback()

    def draw(self, surface, font):
        if not self.enabled:
            color = (*DARK_GREY, 255)
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
                 bg_color=MENU_BUTTON_COLOR, active_color=BRIGHT_GREEN, text_color=WHITE, hover_color=MENU_BUTTON_COLOR_HI):
        super().__init__(x, y, width, height, text, callback, bg_color, hover_color, text_color)
        self.is_on = initial_state
        self.active_color = active_color

        # Toggle switch dimensions
        self.switch_width = int(0.02*SCREEN_WIDTH)
        self.switch_height = int(0.02*SCREEN_WIDTH)
        self.switch_rect = pygame.Rect(
            self.rect.right - self.switch_width - int(0.005*SCREEN_WIDTH),
            self.rect.y + int(0.005*SCREEN_WIDTH),
            self.switch_width,
            self.switch_height
        )

        # Sliding button dimensions
        self.button_width = self.switch_width // 2
        self.button_height = self.switch_height

    def handle_event(self, event, sound_manager=None):
        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if sound_manager:
                    sound_manager.play_sound('menu')
                self.is_on = not self.is_on
                if self.callback:
                    self.callback(self.is_on)

    def draw(self, surface, font):
        # Determine background color based on state and hover
        if not self.enabled:
            color = DARK_GREY
        else:
            mouse_pos = pygame.mouse.get_pos()
            color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.bg_color

        # Draw the button background
        pygame.draw.rect(surface, color, self.rect, border_radius=5)

        # Draw the label aligned to the left
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        surface.blit(text_surf, text_rect)

        # Draw toggle switch background
        switch_color = self.active_color if self.is_on else DARK_RED
        pygame.draw.rect(surface, switch_color, self.switch_rect, border_radius=self.switch_height // 2)

        # Draw sliding button
        button_x = (self.switch_rect.right - self.button_width - 2
                    if self.is_on
                    else self.switch_rect.x + 2)
        button_rect = pygame.Rect(
            button_x,
            self.switch_rect.y + 2,
            self.button_width - 2,
            self.button_height - 4
        )
        button_color = self.active_color if self.is_on else DARK_RED
        pygame.draw.rect(surface, button_color, button_rect, border_radius=self.button_height // 2)

    @property
    def value(self):
        return self.is_on

class KeyBinding(Button):
    def __init__(self, x, y, width, height, label, default_key, callback=None,
                 bg_color=DARK_GREY, hover_color=DARK_GREEN, text_color=WHITE):
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
        label_font = pygame.font.SysFont(None, int(0.03*SCREEN_HEIGHT))
        label_surf = label_font.render(self.label, True, WHITE)
        label_rect = label_surf.get_rect(midright=(self.rect.x - int(0.01*SCREEN_WIDTH), self.rect.y + self.rect.height // 2))
        surface.blit(label_surf, label_rect)

        # Determine button color based on hover state
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            color = self.hover_color
        else:
            color = self.bg_color

        # Draw the button background
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=5)  # Border

        # Decide which text to display
        if self.waiting_for_key:
            display_text = "Press a key..."
            text_color = RED  # Change text color for prompt visibility
        else:
            display_text = self.key.upper()
            text_color = self.text_color

        # Render the appropriate text
        key_surf = font.render(display_text, True, text_color)
        key_rect = key_surf.get_rect(center=self.rect.center)
        surface.blit(key_surf, key_rect)

class Slider:
    def __init__(self, x, y, width, min_val, max_val, start_val, label, is_int=False, step=1):
        self.rect = pygame.Rect(x, y, width, int(0.02*SCREEN_HEIGHT))
        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.label = label
        self.is_int = is_int
        self.step = step
        self.handle_x = self.value_to_position(self.value)
        self.handle_radius = int(0.015*SCREEN_HEIGHT)
        self.dragging = False

    def value_to_position(self, value):
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)

    def position_to_value(self, pos_x):
        ratio = (pos_x - self.rect.x) / self.rect.width
        value = self.min_val + ratio * (self.max_val - self.min_val)
        if self.is_int:
            value = round(value / self.step) * self.step
        return max(self.min_val, min(self.max_val, value))

    def handle_event(self, event, sound_manager=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.get_handle_rect().collidepoint(event.pos):
                if sound_manager:
                    sound_manager.play_sound('menu')
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x = max(self.rect.x, min(self.rect.x + self.rect.width, event.pos[0]))
                self.value = self.position_to_value(mouse_x)
                self.handle_x = self.value_to_position(self.value)

    def get_handle_rect(self):
        return pygame.Rect(
            self.handle_x - self.handle_radius,
            self.rect.y - self.handle_radius + int(0.01*SCREEN_HEIGHT),
            self.handle_radius * 2,
            self.handle_radius * 2
        )

    def draw(self, surface, font):
        padding_x = int(0.02*SCREEN_WIDTH)
        padding_y = int(0.02*SCREEN_HEIGHT)
        bg_x = self.rect.x - padding_x
        bg_y = self.rect.y - int(0.04*SCREEN_HEIGHT)
        bg_width = self.rect.width + 2 * padding_x
        bg_height = 0.08*SCREEN_HEIGHT

        bg_rect = pygame.Rect(bg_x, bg_y, bg_width, bg_height)
        pygame.draw.rect(surface, DARK_GREY, bg_rect, border_radius=5)

        pygame.draw.line(surface, WHITE, (self.rect.x, self.rect.y + int(0.01*SCREEN_HEIGHT)),
                         (self.rect.x + self.rect.width, self.rect.y + int(0.01*SCREEN_HEIGHT)), 5)
        pygame.draw.circle(surface, HANDLE_COLOR, (self.handle_x, self.rect.y + int(0.01*SCREEN_HEIGHT)), self.handle_radius)

        label_text = f"{self.label}: {self.format_value()}"
        label_surf = font.render(label_text, True, WHITE)
        label_rect = label_surf.get_rect(topleft=(self.rect.x, self.rect.y - int(0.03*SCREEN_HEIGHT)))
        surface.blit(label_surf, label_rect)

    def format_value(self):
        if self.is_int:
            return f"{int(self.value)}"
        return f"{self.value:.4f}"


def draw_title(screen, text, font_size=40, y_pos=50):
    """Utility function to draw a centered title with consistent styling"""
    font = pygame.font.SysFont(None, font_size)
    title_surf = font.render(text, True, WHITE)
    title_rect = title_surf.get_rect(center=(screen.get_width() // 2, y_pos))
    screen.blit(title_surf, title_rect)


class SoundManager:
    def __init__(self):
        """Initialize the sound manager and create empty sound dictionaries."""
        # Ensure pygame mixer is initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        self.sounds = {}
        self.music = {}

        # Set default volume levels
        self.sound_volume = 1.0
        self.music_volume = 0.5

    def load_sounds(self):
        """Load all sound effects from the UI directory."""
        sound_files = {
            'menu': 'UI/menu.wav',
            # Add more sound mappings here as needed
        }

        for sound_name, path in sound_files.items():
            try:
                sound = pygame.mixer.Sound(path)
                sound.set_volume(self.sound_volume)
                self.sounds[sound_name] = sound
            except pygame.error as e:
                print(f"Could not load sound '{sound_name}' from {path}: {e}")

    def play_sound(self, sound_name):
        """Play a sound effect by its name."""
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
        else:
            print(f"Warning: Sound '{sound_name}' not found")

    def set_sound_volume(self, volume):
        """Set volume for all sound effects (0.0 to 1.0)."""
        self.sound_volume = max(0.0, min(1.0, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.sound_volume)

    def set_music_volume(self, volume):
        """Set volume for music (0.0 to 1.0)."""
        self.music_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.music_volume)

    def play_music(self, music_name, loops=-1):
        """Play background music."""
        if music_name in self.music:
            pygame.mixer.music.load(self.music[music_name])
            pygame.mixer.music.set_volume(self.music_volume)
            pygame.mixer.music.play(loops)

    def stop_music(self):
        """Stop currently playing music."""
        pygame.mixer.music.stop()

    def pause_music(self):
        """Pause currently playing music."""
        pygame.mixer.music.pause()

    def unpause_music(self):
        """Unpause currently playing music."""
        pygame.mixer.music.unpause()

# Create a global instance of the sound manager
sound_manager = SoundManager()
