from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class ShofixtiA1(Ability):
    def __init__(self, parent):
        super().__init__("ShofixtiA1", parent)
        ability_data = ABILITIES_DATA["ShofixtiA1"]
