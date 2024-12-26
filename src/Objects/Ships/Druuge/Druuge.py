from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Druuge.A1.DruugeA1 import DruugeA1
from src.Objects.Ships.Druuge.A2.DruugeA2 import DruugeA2
import src.const as const
import math

MAX_RECOIL = 96
RECOIL_INCREMENT = MAX_RECOIL / 4
A2_ENERGY = 16

class Druuge(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = DruugeA1(self)

            spawn_distance = const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
            projectile.position = [
                self.position[0] + math.sin(angle_rad) * spawn_distance,
                self.position[1] - math.cos(angle_rad) * spawn_distance
            ]
            projectile.heading = self.heading
            projectile.rotation = self.rotation

            angle_rad = math.radians(self.rotation)
            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]

            self.apply_thrust(MAX_RECOIL,RECOIL_INCREMENT,180, True)

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action2(self):
        if self.can_action2() and self.current_energy < self.max_energy and self.current_hp > 1:
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            projectile = DruugeA2(self)
            projectile.position = self.position.copy()
            projectile.velocity = [0, 0]
            self.current_energy = min(self.max_energy, self.current_energy + A2_ENERGY)
            self.current_hp -= 1

            if projectile.launch_sound: projectile.launch_sound.play()
            return None
        return None

    def perform_action3(self):
        return None, False