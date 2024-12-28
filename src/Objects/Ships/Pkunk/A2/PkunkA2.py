from src.Objects.Ships.ability import Ability, ABILITIES_DATA
from pathlib import Path
import pygame, random


class PkunkA2(Ability):
    _insults = {}
    _initialized = False

    @classmethod
    def _initialize_sounds(cls, file_path):
        if not cls._initialized:
            cls._insults["PkunkA2"] = []
            i = 0
            while True:
                try:
                    sound_path = Path(__file__).parent / f"PkunkA2{str(i).zfill(2)}.wav"
                    sound = pygame.mixer.Sound(str(sound_path))
                    cls._insults["PkunkA2"].append(sound)
                    i += 1
                except pygame.error:
                    break

            if not cls._insults["PkunkA2"]:
                cls._insults["PkunkA2"] = None
            cls._initialized = True

    def __init__(self, parent, angle_offset=0):
        super().__init__("PkunkA2", parent)
        ability_data = ABILITIES_DATA["PkunkA2"]
        self.file_path = ability_data.get("file_path", "")
        self._initialize_sounds(self.file_path)

    def play_random_insult(self):
        if self._insults["PkunkA2"] and isinstance(self._insults["PkunkA2"], list):
            random_sound = random.choice(self._insults["PkunkA2"])
            random_sound.play()