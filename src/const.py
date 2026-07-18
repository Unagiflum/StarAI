# Constants for game mechanics
import os
import pygame
from pathlib import Path
import shutil
import sys

SOURCE_ROOT = (
    Path(sys._MEIPASS) / "src"
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
    else Path(__file__).resolve().parent
)
PROJECT_ROOT = SOURCE_ROOT.parent


def _default_user_data_root():
    """Return a per-user writable directory without adding a dependency."""
    override = os.environ.get("STARAI_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "StarAI"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "StarAI"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "StarAI"


USER_DATA_ROOT = _default_user_data_root()


def source_path(path):
    """Resolve a path stored relative to the source/resource directory."""
    path = Path(path)
    return path if path.is_absolute() else SOURCE_ROOT / path


def cooldown_frames(wait_value):
    """Convert UQM gap frames to engine cooldown frames."""
    return round(wait_value) + 1


# Player Colors
P1_COLOR = (50, 200, 200)
P2_COLOR = (200, 50, 200)

# Tab Button Colors
TAB_BUTTON_COLOR = (120, 30, 90)
TAB_BUTTON_NORMAL_ALPHA = 75
TAB_BUTTON_HOVER_ALPHA = 150
TAB_BUTTON_SELECTED_ALPHA = 255
TAB_BUTTON_BORDER_COLOR = (0, 0, 0)

# HUD Colors
HUD_PANEL_BG = (70, 70, 70)
HUD_PANEL_ALPHA = 150
HUD_BAR_BG = (0, 0, 0)
HUD_BAR_BORDER = (50, 50, 50)
HUD_HP_COLOR = (0, 255, 0)
HUD_NONSENTIENT_HP_COLOR = (128, 128, 128)
HUD_ENERGY_COLOR = (255, 0, 0)
HUD_VIEWPORT_BORDER = HUD_BAR_BORDER

# Screen
SCREEN_HEIGHT = 960
SCREEN_WIDTH = int(SCREEN_HEIGHT * 1.6)
SCREEN_LEFT = int((SCREEN_WIDTH - SCREEN_HEIGHT) / 2)
FPS = 24
INPUT_REPEAT_DELAY_FRAMES = 3
COORDINATED_TRAINING_YIELD_INTERVAL_FRAMES = 4
COORDINATED_TRAINING_YIELD_SECONDS = 0 #0.00025
# Enables detailed training timing in the coordinator and CPU workers and
# writes the associated CSV output. Keep disabled for normal training runs.
TRAINING_TIMING_ENABLED = False

# Pre-battle countdown. COUNT_DOWN_TIME is the wall-clock duration, in seconds,
# for which each number is displayed.
COUNT_DOWN_STEPS = 3
COUNT_DOWN_TIME = 0.5
AI_SHIP_SELECTION_DELAY_SECONDS = 0.5

# Aftermath timing is simulation-owned rather than derived from audio playback.
# shipdies.wav is about 50 frames and the eight-frame explosion animation lasts
# 16 frames, so two seconds leaves a small buffer after both complete.  The
# longest current victory ditty is about 156 frames; six seconds leaves another
# small buffer before ship selection.
POST_DEATH_EFFECT_FRAMES = FPS * 2
VICTORY_DITTY_VIEW_FRAMES = FPS * 6

# Compatibility names retained for callers that import the older constants.
POST_DEATH_ANIMATION_VIEW_FRAMES = POST_DEATH_EFFECT_FRAMES
POST_DEATH_CONTROL_FRAMES = POST_DEATH_EFFECT_FRAMES + VICTORY_DITTY_VIEW_FRAMES

# Graphics Settings

# Fleet and ship-catalog colors
SHIP_PANEL_BACKGROUND_COLOR = (0, 10, 0)
SHIP_BOX_BACKGROUND_COLOR = (0, 0, 0)

# Arena
ARENA_SIZE = 8000
SHIP_SPAWN_SEPARATION = 2000
OBJECT_SPAWN_SEPARATION = 1000
VUX_OBJECT_SPAWN_CLEARANCE = 500
VUX_SPAWN_SEARCH_STEP = 25
VUX_SPAWN_ANGULAR_SPACING = 50
TRAINING_VUX_CLOSE_START_CHANCE = 1.0 / 3.0
PLANET_POSITION = [ARENA_SIZE / 2, ARENA_SIZE / 2]
MAX_ZOOM = 1.0

# Game Physics
SPEED_SCALE = 1.0
GRAVITY_MULTIPLIER = 1.0
GRAVITY_RANGE = 1020
SPEED_LIMIT = 200
MAX_GRAV_WHIP = 72
PLANET_IMPACT_DAMAGE_PERCENT = 0.25
# "Enemy dies" reward attribution. These remain independent from K/D stats
# and from the incremental crew-loss reward factors below.
PLANET_ENEMY_DEATH_REWARD_FACTOR = 0.0
DRUUGE_A2_ENEMY_DEATH_REWARD_FACTOR = 0.0
SHOFIXTI_A2_ENEMY_DEATH_REWARD_FACTOR = 0.0

# Incremental "Reduce enemy crew" reward attribution.
PLANET_CREW_LOSS_REWARD_FACTOR = 0.0
DRUUGE_A2_CREW_LOSS_REWARD_FACTOR = 0.0
SHOFIXTI_A2_CREW_LOSS_REWARD_FACTOR = 0.0

# Stars
STAR_COUNT = int(ARENA_SIZE * ARENA_SIZE / 100000)
STAR_WEIGHTS = [45, 45, 5, 0, 0]  # Smallest to largest
STAR_DEPTHS = 3
PLANET_WEIGHTS = [25, 25, 25, 25]  # Gas, Ice, Life, Rocky
STAR_ALPHA = 200

# Asteroids. ASTEROID_COUNT is updated from persisted game settings at startup.
ASTEROID_COUNT = 5
ASTEROID_PATH = source_path("Objects/Space/Asteroid")
ASTEROID_MIN_SPEED = 16
ASTEROID_MAX_SPEED = 44
ASTEROID_SPEED_STEP = 4
ASTEROID_MASS = 3

# Ships
ASSET_SPRITE_DIRECTIONS = 16
DIRECTIONS_MULTIPLIER = 1
SHIP_DIRECTIONS = ASSET_SPRITE_DIRECTIONS * DIRECTIONS_MULTIPLIER
TURN_ANGLE = 360 / SHIP_DIRECTIONS

VIDEO_FPS_MULTIPLIER = 5
VIDEO_FPS = FPS * VIDEO_FPS_MULTIPLIER

# Total number of sprite images per directional object (gameplay + video interpolation).
TOTAL_SPRITE_DIRECTIONS = SHIP_DIRECTIONS * VIDEO_FPS_MULTIPLIER
TOTAL_SPRITE_STEP = 360 / TOTAL_SPRITE_DIRECTIONS


def _recompute_direction_constants():
    """Recompute gameplay and rendered direction counts from their multipliers."""
    global SHIP_DIRECTIONS, TURN_ANGLE
    global TOTAL_SPRITE_DIRECTIONS, TOTAL_SPRITE_STEP

    SHIP_DIRECTIONS = ASSET_SPRITE_DIRECTIONS * DIRECTIONS_MULTIPLIER
    TURN_ANGLE = 360 / SHIP_DIRECTIONS
    TOTAL_SPRITE_DIRECTIONS = SHIP_DIRECTIONS * VIDEO_FPS_MULTIPLIER
    TOTAL_SPRITE_STEP = 360 / TOTAL_SPRITE_DIRECTIONS


def heading_to_sprite_index(heading):
    """Convert a gameplay heading to a sprite/mask array index."""
    return (heading * VIDEO_FPS_MULTIPLIER) % TOTAL_SPRITE_DIRECTIONS


FLEET_ICON_SIZE = (int(SCREEN_HEIGHT * 0.078), int(SCREEN_HEIGHT * 0.078))
SELECTION_ICON_SIZE = (int(SCREEN_HEIGHT * 0.09755), int(SCREEN_HEIGHT * 0.0975))

# Ship-name tooltips shared by fleet selection, the ship catalog, and battle
# ship selection. The text color is also used for the border.
SHIP_TOOLTIP_FONT_SIZE = int(SCREEN_HEIGHT * 0.042)
SHIP_TOOLTIP_TEXT_COLOR = (0, 0, 0)
SHIP_TOOLTIP_BACKGROUND_COLOR = (255, 255, 200)
SHIP_TOOLTIP_BORDER_COLOR = (25, 25, 25)
SHIP_TOOLTIP_ALPHA = 220
SHIP_TOOLTIP_BORDER_WIDTH = 1
SHIP_TOOLTIP_BORDER_RADIUS = 6
SHIP_TOOLTIP_VERTICAL_OFFSET = SELECTION_ICON_SIZE[1] // 2
SHIP_TOOLTIP_PADDING = (8, 5)
SHIP_CATALOG_COST_FONT_SIZE = 30
SHIP_CATALOG_COST_COLOR = (255, 255, 255)

MAX_SHIP_SIZE = 200
SHIP_COLS = 7
SHIP_ROWS = 7
SOUND_EFFECT_VOLUME = 0.4
SHIP_CROSSHAIRS = "always"
SHOW_PLANET_GRAVITY_MARKER = True

# Battle entry animation.
ENTRY_TRAIL_SILHOUETTES = 12
ENTRY_TRAIL_SPACING = MAX_SHIP_SIZE + 5
ENTRY_TRAIL_STAGGER_FRAMES = 2
ENTRY_TRAIL_FADE_FRAMES = 12
PKUNK_REBIRTH_PAUSE_FRAMES = POST_DEATH_EFFECT_FRAMES

# Projectiles
PROJ_GAP = 5

# File Paths
DEFAULT_GAME_JSON_PATH = source_path("Config/game_settings.json")
DEFAULT_DISPLAY_JSON_PATH = source_path("Config/display_settings.json")
DEFAULT_FLEETS_JSON_PATH = source_path("Config/fleets.json")
DEFAULT_MODELS_PATH = source_path("Models")

GAME_JSON_PATH = USER_DATA_ROOT / "game_settings.json"
DISPLAY_JSON_PATH = USER_DATA_ROOT / "display_settings.json"
FLEETS_JSON_PATH = USER_DATA_ROOT / "fleets.json"
MODELS_PATH = USER_DATA_ROOT / "models"

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

DEFAULT_DISPLAY = {
    "video_frame_rate": 120,
    "ship_crosshairs": "always",
    "show_planet_gravity_marker": True,
}

DEFAULT_GAMEPLAY = {
    "asteroid_count": 5,
    "ship_directions": 16,
    "repeat_key_delay": 3,
}


def apply_game_settings(settings):
    """Apply game settings that currently affect runtime behavior."""
    global ASTEROID_COUNT, DIRECTIONS_MULTIPLIER, INPUT_REPEAT_DELAY_FRAMES

    previous_multiplier = DIRECTIONS_MULTIPLIER
    ASTEROID_COUNT = settings.asteroid_count
    DIRECTIONS_MULTIPLIER = settings.ship_directions // ASSET_SPRITE_DIRECTIONS
    INPUT_REPEAT_DELAY_FRAMES = settings.repeat_key_delay
    _recompute_direction_constants()
    return DIRECTIONS_MULTIPLIER != previous_multiplier


def apply_display_settings(settings):
    """Apply validated display settings to runtime rendering constants."""
    global VIDEO_FPS_MULTIPLIER, VIDEO_FPS
    global SHIP_CROSSHAIRS, SHOW_PLANET_GRAVITY_MARKER

    VIDEO_FPS_MULTIPLIER = settings.video_frame_rate // FPS
    VIDEO_FPS = settings.video_frame_rate
    _recompute_direction_constants()
    SHIP_CROSSHAIRS = settings.ship_crosshairs
    SHOW_PLANET_GRAVITY_MARKER = settings.show_planet_gravity_marker


def initialize_user_data(user_data_root=None):
    """Seed missing user configuration from the bundled defaults.

    Existing files are intentionally left untouched so installing a new game
    build cannot reset player settings or saved fleets.
    """
    root = Path(user_data_root) if user_data_root is not None else USER_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "game_settings": root / "game_settings.json",
        "display_settings": root / "display_settings.json",
        "fleets": root / "fleets.json",
        "models": root / "models",
    }
    defaults = {
        "game_settings": DEFAULT_GAME_JSON_PATH,
        "display_settings": DEFAULT_DISPLAY_JSON_PATH,
        "fleets": DEFAULT_FLEETS_JSON_PATH,
    }
    for name, destination in paths.items():
        if name == "models":
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if destination.exists():
            continue
        try:
            with defaults[name].open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
        except FileExistsError:
            # Another process initialized the same file first.
            pass
    return paths
