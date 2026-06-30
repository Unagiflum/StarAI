from src.Objects.Ships.ability import Ability, ABILITIES_DATA


class ZoqFotPikA1(Ability):
    def __init__(self, parent):
        super().__init__("ZoqFotPikA1", parent)
        ability_data = ABILITIES_DATA["ZoqFotPikA1"]
        self.SPREAD_ANGLE = ability_data.get("SPREAD_ANGLE", 3)
        self.place_self()

    def place_self(self):
        direction = self.rng.randint(-1, 1)
        angle_offset = direction * self.SPREAD_ANGLE
        configured_direction = self.configured_gun()[1]
        self.launch_from_gun(
            relative_direction=configured_direction + angle_offset
        )
