from src.Objects.Ships.ability import Ability


class SpathiA1(Ability):
    def __init__(self, parent):
        super().__init__("SpathiA1", parent)
        self.place_self()

    def place_self(self):
        self.launch_from_gun()
