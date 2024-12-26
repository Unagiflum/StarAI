from src.Objects.Ships.ability import Ability, ABILITIES_DATA


class EarthlingA1(Ability):
    def __init__(self, parent):
        super().__init__("EarthlingA1", parent)
        ability_data = ABILITIES_DATA["EarthlingA1"]
