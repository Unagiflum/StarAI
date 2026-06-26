from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math


class ThraddashA2(Ability):
    def __init__(self, parent):
        super().__init__("ThraddashA2", parent)
        ability_data = ABILITIES_DATA["ThraddashA2"]
        self.REUNK_THRUST = ability_data.get("REUNK_THRUST", 72)
        self.REUNK_INCREMENT = ability_data.get("REUNK_INCREMENT", 12)
        self.place_self()

    def place_self(self, angle_offset=180):
        angle_rad = math.radians(self.parent.rotation + angle_offset)
        spawn_distance = const.PROJ_GAP + (self.size[1] + self.parent.size[1]) / 2

        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance,
        ]

        self.heading = (
            int(self.parent.heading + angle_offset / const.TURN_ANGLE)
            % const.SHIP_DIRECTIONS
        )
        self.rotation = self.heading * const.TURN_ANGLE

        self.velocity = [
            math.sin(angle_rad) * self.speed
            + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle_rad) * self.speed
            + self.parent.velocity[1] * self.parent_vel,
        ]
