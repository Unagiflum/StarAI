from src.Objects.Ships.ability import Ability


class ThraddashA1(Ability):
    def __init__(self, parent):
        super().__init__("ThraddashA1", parent)
        self.place_self()

    def place_self(self):
        self.launch_from_gun()
