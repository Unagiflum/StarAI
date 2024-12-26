from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class ThraddashA2(Ability):
    def __init__(self, parent):
        super().__init__("ThraddashA2", parent)
        ability_data = ABILITIES_DATA["ThraddashA2"]
