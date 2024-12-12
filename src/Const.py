# Constants for game mechanics
from pathlib import Path

ARENA_SIZE = 3000
MIN_SHIP_SEPARATION = ARENA_SIZE // 4
CENTER_BUFFER = ARENA_SIZE // 4

SPEED_SCALE = 1.0
TURN_WAIT_SCALE = 2
THRUST_WAIT_SCALE = 2

GRAVITY_MULTIPLIER = 1000
STAR_COUNT = int(ARENA_SIZE*ARENA_SIZE*0.00001)

SPEED_LIMIT = 500

STAR_WEIGHTS = [45,45,10,0,0] # Smallest to largest
PLANET_WEIGHTS = [25,25,25,25] # Gas, Ice, Life, Rocky
STAR_ALPHA = 175

GAME_JSON_PATH = Path("Config/Gamesettings.json")
TRAINING_JSON_PATH = Path("Config/Trainingsettings.json")
FLEETS_JSON_PATH = Path("Config/Fleets.json") #os.path.join("Config","Fleets.json")

SHIPS_JSON_PATH = Path("Objects/Ships/Ships.json")
PLANETS_JSON_PATH = Path("Objects/Space/planets.json")
STARS_JSON_PATH = Path("Objects/Space/stars.json")

MAIN_BG_PATH = Path("UI/Resources/Main.png")
MENU_BG_PATH = Path("UI/Resources/Menu.png")
MENU_WAV_PATH = Path("UI/Resources/Menu.wav")

DEFAULT_TRAINING = {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "number_of_hidden_layers": 3,
    "layer_size": 128,
    "batch_size": 64,
}