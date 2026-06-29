from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import math


class DruugeA1(Ability):
    def __init__(self, parent):
        super().__init__("DruugeA1", parent)
        ability_data = ABILITIES_DATA["DruugeA1"]
        self.MAX_RECOIL = ability_data.get("MAX_RECOIL", 96)
        self.RECOIL_INCREMENT = ability_data.get("RECOIL_INCREMENT", 24)
        self.place_self()

    def place_self(self):
        self.launch_from_gun()

    def on_ship_impact(self, ship):
        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed <= 0:
            return

        abs_angle_rad = math.atan2(self.velocity[0], -self.velocity[1])
        abs_angle_deg = math.degrees(abs_angle_rad)
        relative_angle = abs_angle_deg - ship.rotation
        ship.apply_thrust(self.MAX_RECOIL, self.RECOIL_INCREMENT, relative_angle, False)
