import src.const as const
from src.Objects.Ships.ability import Ability


class OrzA1(Ability):
    """Directional howitzer round fired along the Orz turret heading."""

    def __init__(self, parent):
        super().__init__("OrzA1", parent)
        self.place_self()

    def place_self(self):
        self.heading = self.parent.turret_heading
        self.rotation = self.heading * const.TURN_ANGLE
        self.launch_from_gun(
            gun_rotation=self.rotation,
            launch_direction=self.rotation,
        )
