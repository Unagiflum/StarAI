# Constants for game mechanics
import pygame
from pathlib import Path

# Screen
SCREEN_HEIGHT = 960
SCREEN_WIDTH = int(SCREEN_HEIGHT*1.3)
FPS = 30

#Arena
ARENA_SIZE = 8000
MIN_SHIP_SEPARATION = ARENA_SIZE // 4
CENTER_BUFFER = ARENA_SIZE // 4
PLANET_POSITION = [ARENA_SIZE/2, ARENA_SIZE/2]
MAX_ZOOM = 1.0
STAR_COUNT = int(ARENA_SIZE*ARENA_SIZE/64000)
STAR_WEIGHTS = [45,45,10,0,0] # Smallest to largest
STAR_DEPTHS = 3
PLANET_WEIGHTS = [25,25,25,25] # Gas, Ice, Life, Rocky
STAR_ALPHA = 200

#Game Physics
GRAVITY_MULTIPLIER = 1.0
GRAVITY_RANGE = CENTER_BUFFER // 2
SPEED_SCALE = 0.8
TURN_WAIT_SCALE = 2
THRUST_WAIT_SCALE = 2
SPEED_LIMIT = 200*SPEED_SCALE

#File Paths
GAME_JSON_PATH = Path("Config/Gamesettings.json")
TRAINING_JSON_PATH = Path("Config/Trainingsettings.json")
FLEETS_JSON_PATH = Path("Config/Fleets.json") #os.path.join("Config","Fleets.json")

SHIPS_JSON_PATH = Path("Objects/Ships/Ships.json")
PLANETS_JSON_PATH = Path("Objects/Space/planets.json")
STARS_JSON_PATH = Path("Objects/Space/stars.json")

BATTLE_MUSIC_PATH = Path("Battle/Resources/battle.ogg")
BATTLE_MUSIC_VOLUME = 0.4

MAIN_BG_PATH = Path("UI/Resources/Main.png")
MENU_BG_PATH = Path("UI/Resources/Menu.png")
MENU_WAV_PATH = Path("UI/Resources/Menu.wav")

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