from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class PkunkA1(Ability):
    def __init__(self, parent):
        super().__init__("PkunkA1", parent)
        ability_data = ABILITIES_DATA["PkunkA1"]
