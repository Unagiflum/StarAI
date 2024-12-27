from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class DruugeA2(Ability):
    def __init__(self, parent):
        super().__init__("DruugeA2", parent)
        ability_data = ABILITIES_DATA["DruugeA2"]
        self.ENERGY_GAIN = ability_data.get("ENERGY_GAIN", 16)
