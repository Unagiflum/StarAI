from src.Objects.Ships.ability import Ability


class UmgahA2(Ability):
    def __init__(self, parent):
        super().__init__("UmgahA2", parent)
        self.can_move = False
        self.can_die = False
        self.can_collide = False
