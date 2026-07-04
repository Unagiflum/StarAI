from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const


class PkunkA2(Ability):
    def __init__(self, parent):
        super().__init__("PkunkA2", parent)
        ability_data = ABILITIES_DATA["PkunkA2"]
        self.ENERGY_GAIN = ability_data.get("energy_gain", 2)
        self.file_path = ability_data.get("file_path")
        sound_dir = const.source_path(ability_data["file_path"])
        loaded_insults = tuple(
            (index, sound)
            for index in range(14)
            if (
                sound := self.audio_service.load_effect(
                    sound_dir / f"PkunkA2{index:02d}.wav",
                    const.SOUND_EFFECT_VOLUME,
                )
            )
            is not None
        )
        self.insult_indices = tuple(index for index, _ in loaded_insults)
        self.insults = tuple(sound for _, sound in loaded_insults)

    def play_insult(self):
        choices = [
            (index, sound)
            for index, sound in zip(self.insult_indices, self.insults)
            if index != self.parent.last_insult_index
        ]
        if not choices:
            choices = list(zip(self.insult_indices, self.insults))
        if choices:
            index, sound = self.rng.choice(choices)
            self.parent.last_insult_index = index
            sound.play()
