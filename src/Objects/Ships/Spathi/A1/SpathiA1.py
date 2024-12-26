from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class SpathiA1(Ability):
    def __init__(self, parent):
        super().__init__("SpathiA1", parent)
        ability_data = ABILITIES_DATA["SpathiA1"]
