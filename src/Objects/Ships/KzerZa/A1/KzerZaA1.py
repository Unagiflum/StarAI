from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class KzerZaA1(Ability):
    def __init__(self, parent):
        super().__init__("KzerZaA1", parent)
        ability_data = ABILITIES_DATA["KzerZaA1"]
