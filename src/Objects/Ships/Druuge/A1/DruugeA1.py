from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class DruugeA1(Ability):
    def __init__(self, parent):
        super().__init__("DruugeA1", parent)
        ability_data = ABILITIES_DATA["DruugeA1"]
