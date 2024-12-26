from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class YehatA1(Ability):
    def __init__(self, parent):
        super().__init__("YehatA1", parent)
        ability_data = ABILITIES_DATA["YehatA1"]
