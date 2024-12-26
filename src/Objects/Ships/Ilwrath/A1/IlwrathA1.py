from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class IlwrathA1(Ability):
    def __init__(self, parent):
        super().__init__("IlwrathA1", parent)
        ability_data = ABILITIES_DATA["IlwrathA1"]
