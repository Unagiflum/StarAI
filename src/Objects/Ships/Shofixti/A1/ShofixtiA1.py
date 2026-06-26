from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math


class ShofixtiA1(Ability):
    def __init__(self, parent):
        super().__init__("ShofixtiA1", parent)
        ability_data = ABILITIES_DATA["ShofixtiA1"]
        self.place_self()

    def place_self(self):
        angle_rad = math.radians(self.parent.rotation)
        spawn_distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2
        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance,
        ]
        self.heading = self.parent.heading
        self.rotation = self.parent.rotation
        angle_rad = math.radians(self.rotation)
        self.velocity = [
            math.sin(angle_rad) * self.speed
            + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle_rad) * self.speed
            + self.parent.velocity[1] * self.parent_vel,
        ]
