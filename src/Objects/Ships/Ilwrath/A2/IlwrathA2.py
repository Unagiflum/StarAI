from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math


class IlwrathA2(Ability):
    def __init__(self, parent):
        super().__init__("IlwrathA2", parent)
        ability_data = ABILITIES_DATA["IlwrathA2"]
        # self.place_self()

    # def place_self(self):
    #   pass
