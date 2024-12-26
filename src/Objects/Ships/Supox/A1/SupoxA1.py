from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class SupoxA1(Ability):
    def __init__(self, parent):
        super().__init__("SupoxA1", parent)
        ability_data = ABILITIES_DATA["SupoxA1"]
