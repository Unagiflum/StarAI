from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const


class PkunkA2(Ability):
    def __init__(self, parent):
        super().__init__("PkunkA2", parent)
        ability_data = ABILITIES_DATA["PkunkA2"]
        self.ENERGY_GAIN = ability_data.get("ENERGY_GAIN", 2)
        self.file_path = ability_data.get("file_path")
        sound_dir = const.source_path(ability_data["file_path"])
        self.insults = tuple(
            sound
            for index in range(14)
            if (
                sound := self.audio_service.load_effect(
                    sound_dir / f"PkunkA2{index:02d}.wav",
                    const.SOUND_EFFECT_VOLUME,
                )
            )
            is not None
        )

    def play_insult(self):
        if self.insults:
            self.rng.choice(self.insults).play()
