from src.Objects.Ships.ability import Ability


class SpathiA2(Ability):
    def __init__(self, parent):
        super().__init__("SpathiA2", parent)
        self.place_self()

    def place_self(self):
        self.launch_from_gun()
