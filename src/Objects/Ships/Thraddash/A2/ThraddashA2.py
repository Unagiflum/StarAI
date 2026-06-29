from src.Objects.Ships.ability import Ability, ABILITIES_DATA


class ThraddashA2(Ability):
    def __init__(self, parent):
        super().__init__("ThraddashA2", parent)
        ability_data = ABILITIES_DATA["ThraddashA2"]
        self.REUNK_THRUST = ability_data.get("REUNK_THRUST", 72)
        self.REUNK_INCREMENT = ability_data.get("REUNK_INCREMENT", 12)
        self.place_self()

    def place_self(self):
        self.launch_from_gun()
