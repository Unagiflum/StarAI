import pygame
import src.const as Const
from src.resources import default_assets
from src.audio import compatibility_audio_service

button_spaceH = int(Const.SCREEN_WIDTH * 0.005)
button_spaceV = int(Const.SCREEN_HEIGHT * 0.00625)

ok_button_width = int(0.150 * Const.SCREEN_WIDTH)
ok_button_height = int(0.05 * Const.SCREEN_HEIGHT)
ok_button_left = (
    Const.SCREEN_WIDTH // 2 - ok_button_width - int(0.0167 * Const.SCREEN_WIDTH)
)
can_button_left = Const.SCREEN_WIDTH // 2 + int(0.0167 * Const.SCREEN_WIDTH)
ok_button_top = Const.SCREEN_HEIGHT - ok_button_height - button_spaceV * 4

SELECTION_WIDTH = int(0.448 * Const.SCREEN_WIDTH)
SELECTION_HEIGHT = int(0.35 * Const.SCREEN_HEIGHT)
FLEET_HEIGHT = int(0.753 * Const.SCREEN_HEIGHT)

# Colors
WHITE = (255, 255, 255)
LIGHT_GREY = (170, 170, 170)
GREY = (100, 100, 100)

DARK_GREY = (50, 50, 50)
BLACK = (0, 0, 0)

ORANGE = (255, 100, 0)
RED = (255, 0, 0)
DARK_RED = (50, 0, 0)

BRIGHT_GREEN = (150, 255, 100)
DARK_GREEN = (0, 50, 0)

OK_GREEN = (0, 155, 0, 75)
OK_GREEN_HI = (0, 155, 0, 255)
CAN_RED = (155, 0, 0, 75)
CAN_RED_HI = (155, 0, 0, 255)

MENU_BUTTON_COLOR = (0, 175, 175, 75)
MENU_BUTTON_COLOR_HI = (0, 175, 175, 255)

DISABLED_BUTTON = (100, 100, 100, 75)

MAIN_BUTTON_COLOR = (0, 0, 0, 125)
MAIN_BUTTON_COLOR_HI = (0, 0, 0, 255)

SLIDER_BG = (50, 50, 100, 100)
SLIDER_BG_HI = (50, 50, 100, 255)
SLIDER_LINE = (100, 100, 100)

HANDLE_COLOR = (255, 0, 0)
BG_COLOR = (0, 0, 20)


def load_background(path, screen_width, screen_height, resources=None):
    """Load and scale the background image to fit the screen."""
    try:
        return (resources or default_assets()).background(
            path, (screen_width, screen_height)
        )
    except pygame.error as e:
        print(f"Could not load background image: {e}")
        return None


def draw_title(screen, text, font_size=40, y_pos=50):
    """Utility function to draw a centered title with consistent styling"""
    font = pygame.font.SysFont(None, font_size)
    title_surf = font.render(text, True, WHITE)
    title_rect = title_surf.get_rect(center=(screen.get_width() // 2, y_pos))
    screen.blit(title_surf, title_rect)


def format_ship_tooltip(name, ship_type, cost=None, *, include_cost=True):
    """Return the shared ship tooltip label."""
    label = f"{name} {ship_type}" if ship_type else name
    return f"{label}: {cost}" if include_cost and cost is not None else label


def tooltip_rect(
    text_surface,
    mouse_pos,
    anchor_rect,
    screen_rect,
    padding=Const.SHIP_TOOLTIP_PADDING,
    offset=Const.SHIP_TOOLTIP_VERTICAL_OFFSET,
):
    """Position a tooltip at a fixed distance below the cursor."""
    _ = anchor_rect  # Retained for compatibility with existing callers.
    rect = pygame.Rect(
        0,
        0,
        text_surface.get_width() + 2 * padding[0],
        text_surface.get_height() + 2 * padding[1],
    )
    rect.midtop = (mouse_pos[0], mouse_pos[1] + offset)
    rect.clamp_ip(screen_rect)
    return rect


def centered_text_rect(text_surface, container_rect):
    """Center the visible glyphs instead of the font surface's line box."""
    bounds = text_surface.get_bounding_rect()
    return text_surface.get_rect(
        x=container_rect.centerx - bounds.centerx,
        y=container_rect.centery - bounds.centery,
    )


def draw_ship_tooltip(screen, font, label, mouse_pos, anchor_rect):
    """Draw a consistently styled ship tooltip and return its rectangle."""
    text_surface = font.render(label, True, Const.SHIP_TOOLTIP_TEXT_COLOR)
    rect = tooltip_rect(
        text_surface,
        mouse_pos,
        anchor_rect,
        screen.get_rect(),
    )
    tooltip_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    surface_rect = tooltip_surface.get_rect()
    pygame.draw.rect(
        tooltip_surface,
        (*Const.SHIP_TOOLTIP_BACKGROUND_COLOR, Const.SHIP_TOOLTIP_ALPHA),
        surface_rect,
        border_radius=Const.SHIP_TOOLTIP_BORDER_RADIUS,
    )
    pygame.draw.rect(
        tooltip_surface,
        Const.SHIP_TOOLTIP_BORDER_COLOR,
        surface_rect,
        Const.SHIP_TOOLTIP_BORDER_WIDTH,
        border_radius=Const.SHIP_TOOLTIP_BORDER_RADIUS,
    )
    tooltip_surface.blit(
        text_surface,
        centered_text_rect(text_surface, surface_rect),
    )
    screen.blit(tooltip_surface, rect)
    return rect


class SoundManager:
    def __init__(self, enabled=True, resources=None, audio_service=None):
        self.resources = resources or default_assets()
        self.audio_service = (
            audio_service
            if audio_service is not None
            else compatibility_audio_service(enabled, self.resources)
        )
        self.enabled = self.audio_service.enabled
        self.sounds = {}
        self.volume = 1.0

    def load_sounds(self):
        if not self.enabled:
            return
        sound_files = {
            "menu": Const.MENU_WAV_PATH,
        }
        for sound_name, path in sound_files.items():
            try:
                sound = self.audio_service.load_effect(path, self.volume)
                if sound is None:
                    continue
                self.sounds[sound_name] = sound
            except pygame.error as e:
                print(f"Could not load sound '{sound_name}' from {path}: {e}")

    def play_sound(self, sound_name):
        if not self.enabled:
            return
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
        else:
            print(f"Warning: Sound '{sound_name}' not found")

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.volume)
