from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class ThraddashA1(Ability):
    def __init__(self, parent):
        super().__init__("ThraddashA1", parent)
        ability_data = ABILITIES_DATA["ThraddashA1"]
