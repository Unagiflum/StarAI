from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math


class UtwigA1(Ability):
    def __init__(self, parent, x_offset=0, y_offset=0):
        super().__init__("UtwigA1", parent)
        self.place_self(x_offset, y_offset)

    def place_self(self, x_offset, y_offset):
        angle_rad = math.radians(self.parent.rotation)
        spawn_distance = const.PROJ_GAP + self.size[1] / 2 + y_offset

        rotated_x_offset = math.cos(angle_rad) * x_offset
        rotated_y_offset = math.sin(angle_rad) * x_offset

        self.position = [
            self.parent.position[0]
            + math.sin(angle_rad) * spawn_distance
            + rotated_x_offset,
            self.parent.position[1]
            - math.cos(angle_rad) * spawn_distance
            + rotated_y_offset,
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

    @classmethod
    def create_parallel_projectiles(cls, ship):
        data = ABILITIES_DATA["UtwigA1"]
        gun_locations = data.gun_locations

        projectiles = []
        if not gun_locations:
            return projectiles

        ship_width, ship_height = ship.size
        center_x, center_y = ship_width / 2, ship_height / 2

        for loc in gun_locations:
            x, y = loc
            x_offset = x - center_x
            y_offset = center_y - y
            projectiles.append(cls(ship, x_offset=x_offset, y_offset=y_offset))

        return projectiles
