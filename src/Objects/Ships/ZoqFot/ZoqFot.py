from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ZoqFot.A1.ZoqFotA1 import ZoqFotA1
import src.const as const
import math
import random


class ZoqFot(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        self.spread_angle = 3  # Degrees of spread for off-center shots

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            # Randomly choose direction: -1 for left, 0 for center, 1 for right
            direction = random.randint(-1, 1)
            angle_offset = direction * self.spread_angle

            angle_rad = math.radians(self.rotation + angle_offset)
            projectile = ZoqFotA1(self)

            spawn_distance = const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
            projectile.position = [
                self.position[0] + math.sin(angle_rad) * spawn_distance,
                self.position[1] - math.cos(angle_rad) * spawn_distance
            ]
            projectile.heading = 0
            projectile.rotation = angle_offset

            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False