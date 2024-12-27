from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math

class KohrAhA2(Ability):
    def __init__(self, parent, angle_offset = 0):
        super().__init__("KohrAhA2", parent)
        ability_data = ABILITIES_DATA["KohrAhA2"]
        self.DMG_TO_PROJ = ability_data.get("DMG_TO_PROJ", 50)
        self.place_self(angle_offset)

    def place_self(self, angle_offset):
        angle_rad = math.radians(self.parent.rotation + angle_offset)
        spawn_distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2

        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance
        ]

        self.heading = 0
        self.rotation = 0

        self.velocity = [
            math.sin(angle_rad) * self.speed + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle_rad) * self.speed + self.parent.velocity[1] * self.parent_vel
        ]
