from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class ZoqFotA1(Ability):
    def __init__(self, parent):
        super().__init__("ZoqFotA1", parent)
        ability_data = ABILITIES_DATA["ZoqFotA1"]
