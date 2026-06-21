from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math

class ZoqFotA1(Ability):
    def __init__(self, parent):
        super().__init__("ZoqFotA1", parent)
        ability_data = ABILITIES_DATA["ZoqFotA1"]
        self.SPREAD_ANGLE = ability_data.get("SPREAD_ANGLE", 3)
        self.place_self()

    def place_self(self):

        direction = self.rng.randint(-1, 1)
        angle_offset = direction * self.SPREAD_ANGLE
        angle_rad = math.radians(self.parent.rotation + angle_offset)

        spawn_distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2

        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance
        ]

        self.heading = 0
        self.rotation = self.parent.rotation + angle_offset

        angle_rad = math.radians(self.rotation)
        self.velocity = [
            math.sin(angle_rad) * self.speed + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle_rad) * self.speed + self.parent.velocity[1] * self.parent_vel
        ]
