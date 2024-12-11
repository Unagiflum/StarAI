# Constants for game mechanics
import os

ARENA_SIZE = 3000
MIN_SHIP_SEPARATION = ARENA_SIZE // 4
CENTER_BUFFER = ARENA_SIZE // 4

SPEED_SCALE = 1.0
TURN_WAIT_SCALE = 2
THRUST_WAIT_SCALE = 2

GRAVITY_MULTIPLIER = 1000
STAR_COUNT = int(ARENA_SIZE*ARENA_SIZE*0.00001)

SPEED_LIMIT = 500

STAR_WEIGHTS = [45,45,10,0,0]
PLANET_WEIGHTS = [25,25,25,25]

GAME_JSON_PATH = os.path.join("Config","Gamesettings.json")
TRAINING_JSON_PATH = os.path.join("Config","Trainingsettings.json")
FLEETS_JSON_PATH = os.path.join("Config","Fleets.json")

SHIPS_JSON_PATH = os.path.join("Objects", "Ships", "Ships.json")
PLANETS_JSON_PATH = os.path.join("Objects", "Space", "planets.json")
STARS_JSON_PATH = os.path.join("Objects", "Space", "stars.json")

MAIN_BG_PATH = os.path.join("UI","Main.png")
MENU_BG_PATH = os.path.join("UI","Menu.png")
MENU_WAV_PATH = os.path.join("UI","Menu.wav")

DEFAULT_TRAINING = {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "number_of_hidden_layers": 3,
    "layer_size": 128,
    "batch_size": 64,
}