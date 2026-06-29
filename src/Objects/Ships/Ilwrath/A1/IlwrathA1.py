from src.Objects.Ships.ability import Ability


class IlwrathA1(Ability):
    def __init__(self, parent):
        super().__init__("IlwrathA1", parent)
        self.place_self()

    def place_self(self):
        self.launch_from_gun()
