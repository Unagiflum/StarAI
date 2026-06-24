# Constants for game mechanics
import pygame
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent


def source_path(path):
    """Resolve a path stored relative to the source/resource directory."""
    path = Path(path)
    return path if path.is_absolute() else SOURCE_ROOT / path

# Player Colors
P1_COLOR = (200, 50, 50)
P2_COLOR = (50, 200, 50)

# Screen
SCREEN_HEIGHT = 960
SCREEN_WIDTH = int(SCREEN_HEIGHT*1.6)
SCREEN_LEFT = int((SCREEN_WIDTH - SCREEN_HEIGHT)/2)
FPS = 30
INPUT_REPEAT_DELAY_FRAMES = 3
# Aftermath timing is simulation-owned rather than derived from audio playback.
# shipdies.wav is about 50 frames and the eight-frame explosion animation lasts
# 16 frames, so two seconds leaves a small buffer after both complete.  The
# longest current victory ditty is about 156 frames; six seconds leaves another
# small buffer before ship selection.
POST_DEATH_EFFECT_FRAMES = FPS * 2
VICTORY_DITTY_VIEW_FRAMES = FPS * 6

# Compatibility names retained for callers that import the older constants.
POST_DEATH_ANIMATION_VIEW_FRAMES = POST_DEATH_EFFECT_FRAMES
POST_DEATH_CONTROL_FRAMES = (
    POST_DEATH_EFFECT_FRAMES + VICTORY_DITTY_VIEW_FRAMES
)

#Arena
ARENA_SIZE = 6000
MIN_SHIP_SEPARATION = ARENA_SIZE // 4
CENTER_BUFFER = ARENA_SIZE // 4
PLANET_POSITION = [ARENA_SIZE/2, ARENA_SIZE/2]
MAX_ZOOM = 1.0

#Game Physics
SPEED_SCALE = 0.8
GRAVITY_MULTIPLIER = 0.5
GRAVITY_RANGE = CENTER_BUFFER // 2
TURN_WAIT_SCALE = 2
THRUST_WAIT_SCALE = 2
ACTION_WAIT_SCALE = 2
RECHARGE_DELAY_SCALE = 2
SPEED_LIMIT = 150
MAX_GRAV_WHIP = 100

#Stars
STAR_COUNT = int(ARENA_SIZE*ARENA_SIZE/64000)
STAR_WEIGHTS = [45,45,10,0,0] # Smallest to largest
STAR_DEPTHS = 3
PLANET_WEIGHTS = [25,25,25,25] # Gas, Ice, Life, Rocky
STAR_ALPHA = 200

#Asteroids
ASTEROID_COUNT = 5 # Asteroid placing function will get stuck if you have too many
ASTEROID_PATH = source_path("Objects/Space/Asteroid")
ASTEROID_SPEED = 30

#Ships
SHIP_DIRECTIONS = 16
TURN_ANGLE = 360 / SHIP_DIRECTIONS
FLEET_ICON_SIZE = (int(SCREEN_HEIGHT*0.078), int(SCREEN_HEIGHT*0.078))
SELECTION_ICON_SIZE = (int(SCREEN_HEIGHT*0.09755), int(SCREEN_HEIGHT*0.0975))
MAX_SHIP_SIZE = 200
SHIP_COLS = 8
SHIP_ROWS = 4
SOUND_EFFECT_VOLUME = 0.4

# Battle entry animation.
ENTRY_TRAIL_SILHOUETTES = 12
ENTRY_TRAIL_SPACING = MAX_SHIP_SIZE + 5
ENTRY_TRAIL_STAGGER_FRAMES = 2
ENTRY_TRAIL_FADE_FRAMES = 12
PKUNK_REBIRTH_PAUSE_FRAMES = POST_DEATH_EFFECT_FRAMES

#Projectiles
PROJ_LIFE_SCALE = 1
PROJ_SPEED_SCALE = 1.25
PROJ_GAP = 5

#File Paths
GAME_JSON_PATH = source_path("Config/game_settings.json")
TRAINING_JSON_PATH = source_path("Config/train_settings.json")
FLEETS_JSON_PATH = source_path("Config/fleets.json")

SHIPS_JSON_PATH = source_path("Objects/Ships/space_ships.json")
ABILITIES_JSON_PATH = source_path("Objects/Ships/abilities.json")
PLANETS_JSON_PATH = source_path("Objects/Space/planets.json")
STARS_JSON_PATH = source_path("Objects/Space/stars.json")

BATTLE_MUSIC_PATH = source_path("Battle/Resources/battle.ogg")
BATTLE_MUSIC_VOLUME = 0.2

MAIN_BG_PATH = source_path("UI/Resources/Main.png")
MENU_BG_PATH = source_path("UI/Resources/Menu.png")
MENU_WAV_PATH = source_path("UI/Resources/Menu.wav")

# Default settings
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

DEFAULT_TRAINING = {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "number_of_hidden_layers": 3,
    "layer_size": 128,
    "batch_size": 64,
}
