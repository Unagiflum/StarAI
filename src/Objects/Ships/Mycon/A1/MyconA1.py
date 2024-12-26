from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class MyconA1(Ability):
    def __init__(self, parent):
        super().__init__("MyconA1", parent)
        ability_data = ABILITIES_DATA["MyconA1"]
