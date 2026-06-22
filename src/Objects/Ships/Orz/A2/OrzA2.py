import src.const as const
from src.Objects.Ships.ability import Ability


class OrzA2(Ability):
    """Persistent visual and direction state for the Orz turret."""

    def __init__(self, parent):
        super().__init__("OrzA2", parent)
        self.relative_heading = 0
        self.can_move = False
        self.can_collide = False
        self.can_die = False
        self.can_expire = False

    @property
    def absolute_heading(self):
        return (
            self.parent.heading + self.relative_heading
        ) % const.SHIP_DIRECTIONS

    def reset(self):
        self.relative_heading = 0

    def turn(self, direction):
        self.relative_heading = (
            self.relative_heading + direction
        ) % const.SHIP_DIRECTIONS

    def get_sprite(self):
        self.heading = self.absolute_heading
        return self.sprites[self.heading]

    def update(self):
        # The controller is owned and rendered by the parent ship rather than
        # participating as an independent battle object.
        return False

    def draw(self, screen, scale_factor, translation):
        return None
