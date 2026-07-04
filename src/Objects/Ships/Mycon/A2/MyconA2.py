from src.Objects.Ships.ability import Ability, ABILITIES_DATA


class MyconA2(Ability):
    def __init__(self, parent):
        super().__init__("MyconA2", parent)
        ability_data = ABILITIES_DATA["MyconA2"]
        self.HP_GAIN = ability_data.get("hp_gain", 4)
