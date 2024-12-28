from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import pygame
from pathlib import Path
import random
import glob
import src.const as const


class PkunkA2(Ability):
    _insults = []

    def __init__(self, parent):
        super().__init__("PkunkA2", parent)
        ability_data = ABILITIES_DATA["PkunkA2"]
        self.ENERGY_GAIN = ability_data.get("ENERGY_GAIN", 2)
        self.file_path = ability_data.get("file_path")
        sound_dir = Path(ability_data['file_path'])
        pattern = str(sound_dir / "PkunkA2[0-9][0-9].wav")

        for sound_path in glob.glob(pattern):
            try:
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(const.SOUND_EFFECT_VOLUME)
                self._insults.append(sound)
            except pygame.error:
                continue

    def play_insult(self):
        if self._insults:
            random.choice(self._insults).play()