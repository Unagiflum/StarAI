from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class SpathiA2(Ability):
    def __init__(self, parent):
        super().__init__("SpathiA2", parent)
        ability_data = ABILITIES_DATA["SpathiA2"]
