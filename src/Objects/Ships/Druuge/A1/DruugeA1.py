from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math


class DruugeA1(Ability):
    def __init__(self, parent):
        super().__init__("DruugeA1", parent)
        ability_data = ABILITIES_DATA["DruugeA1"]
        self.MAX_RECOIL = ability_data.get("MAX_RECOIL", 96)
        self.RECOIL_INCREMENT = ability_data.get("RECOIL_INCREMENT", 24)
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

    def on_ship_impact(self, ship):
        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed <= 0:
            return

        momentum = max(0.1, self.parent.mass) * self.RECOIL_INCREMENT
        ship_mass = max(0.1, ship.mass)
        ship.add_impulse(
            self.velocity[0] / speed * momentum / ship_mass,
            self.velocity[1] / speed * momentum / ship_mass,
        )
