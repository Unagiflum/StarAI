from src.Objects.Ships.ability import Ability, ABILITIES_DATA

class ArilouA2(Ability):
    def __init__(self, parent):
        super().__init__("ArilouA2", parent)
        ability_data = ABILITIES_DATA["ArilouA2"]
        self.place_self()

    def place_self(self):
        self.position = self.parent.position.copy()
        self.heading = 0
        self.rotation = 0
        self.velocity = [0, 0]


