from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math


class YehatA1(Ability):
    def __init__(self, parent, offset=0):
        super().__init__("YehatA1", parent)
        ability_data = ABILITIES_DATA["YehatA1"]
        self.place_self(offset)

    def place_self(self, offset):
        angle_rad = math.radians(self.parent.rotation)
        spawn_distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2
        offset_x = math.cos(angle_rad) * offset
        offset_y = math.sin(angle_rad) * offset

        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance + offset_x,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance + offset_y,
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
