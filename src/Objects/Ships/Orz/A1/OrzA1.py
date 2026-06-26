import math

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
        angle_rad = math.radians(self.rotation)
        spawn_distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2
        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance,
        ]
        self.velocity = [
            math.sin(angle_rad) * self.speed
            + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle_rad) * self.speed
            + self.parent.velocity[1] * self.parent_vel,
        ]
