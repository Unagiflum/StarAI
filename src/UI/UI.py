import pygame
import src.Const as Const

SCREEN_HEIGHT = 960
SCREEN_WIDTH = int(SCREEN_HEIGHT*1.3)
FPS = 30

button_spaceH = int(SCREEN_WIDTH * 0.005)
button_spaceV = int(SCREEN_HEIGHT * 0.00625)

ok_button_width = int(0.150 * SCREEN_WIDTH)
ok_button_height = int(0.05 * SCREEN_HEIGHT)
ok_button_left = SCREEN_WIDTH // 2 - ok_button_width - int(0.0167*SCREEN_WIDTH)
can_button_left = SCREEN_WIDTH // 2 + int(0.0167*SCREEN_WIDTH)
ok_button_top = SCREEN_HEIGHT-ok_button_height-button_spaceV*4

FLEET_ICON_SIZE = (int(SCREEN_WIDTH*0.060), int(SCREEN_WIDTH*0.060))
SELECTION_WIDTH = int(0.45 * SCREEN_WIDTH)
SELECTION_HEIGHT = int(.35 * SCREEN_HEIGHT)
FLEET_HEIGHT = int(.40 * SCREEN_HEIGHT)

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
SLIDER_LINE = (100,100,100)

HANDLE_COLOR = (255, 0, 0)
BG_COLOR = (0, 0, 20)

DEFAULT_KEYS = {
    "Player 1: Left": pygame.K_a,
    "Player 1: Right": pygame.K_d,
    "Player 1: Forward": pygame.K_w,
    "Player 1: Action 1": pygame.K_TAB,
    "Player 1: Action 2": pygame.K_BACKQUOTE,
    "Player 2: Left": pygame.K_LEFT,
    "Player 2: Right": pygame.K_RIGHT,
    "Player 2: Forward": pygame.K_UP,
    "Player 2: Action 1": pygame.K_RCTRL,
    "Player 2: Action 2": pygame.K_RSHIFT,
}

def load_background(path, screen_width, screen_height):
    """Load and scale the background image to fit the screen."""
    try:
        background = pygame.image.load(path)
        return pygame.transform.scale(background, (screen_width, screen_height))
    except pygame.error as e:
        print(f"Could not load background image: {e}")
        return None

def draw_title(screen, text, font_size=40, y_pos=50):
    """Utility function to draw a centered title with consistent styling"""
    font = pygame.font.SysFont(None, font_size)
    title_surf = font.render(text, True, WHITE)
    title_rect = title_surf.get_rect(center=(screen.get_width() // 2, y_pos))
    screen.blit(title_surf, title_rect)

class SoundManager:
    def __init__(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self.sounds = {}
        self.volume = 1.0

    def load_sounds(self):
        sound_files = {
            'menu': Const.MENU_WAV_PATH,
        }
        for sound_name, path in sound_files.items():
            try:
                sound = pygame.mixer.Sound(path)
                sound.set_volume(self.volume)
                self.sounds[sound_name] = sound
            except pygame.error as e:
                print(f"Could not load sound '{sound_name}' from {path}: {e}")

    def play_sound(self, sound_name):
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
        else:
            print(f"Warning: Sound '{sound_name}' not found")

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
        for sound in self.sounds.values():
            sound.set_volume(self.volume)

sound_manager = SoundManager()
