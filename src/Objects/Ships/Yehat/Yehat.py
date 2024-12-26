from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Yehat.A1.YehatA1 import YehatA1
import src.const as const
import math


class Yehat(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)
            side_offset = self.size[0] / 2

            projectiles = []
            for offset in [-side_offset, side_offset]:
                projectile = YehatA1(self)

                spawn_distance = const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
                offset_x = math.cos(angle_rad) * offset
                offset_y = math.sin(angle_rad) * offset

                projectile.position = [
                    self.position[0] + math.sin(angle_rad) * spawn_distance + offset_x,
                    self.position[1] - math.cos(angle_rad) * spawn_distance + offset_y
                ]
                projectile.heading = self.heading
                projectile.rotation = self.rotation
                projectile.velocity = [
                    math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                    -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
                ]

                if projectile.launch_sound: projectile.launch_sound.play()
                projectiles.append(projectile)

            return projectiles
        return None


    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False