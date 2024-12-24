from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Ships.Spathi.A1.SpathiA1 import SpathiA1
from src.Objects.Ships.Spathi.A2.SpathiA2 import SpathiA2
import src.Const as Const
import math


class Spathi(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = SpathiA1(self)

            spawn_distance = Const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
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

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * Const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = SpathiA2(self)

            spawn_distance = Const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
            projectile.position = [
                self.position[0] - math.sin(angle_rad) * spawn_distance,
                self.position[1] + math.cos(angle_rad) * spawn_distance
            ]
            projectile.heading = int(self.heading + Const.SHIP_DIRECTIONS/2) % Const.SHIP_DIRECTIONS
            projectile.rotation = (self.rotation + 180) % 360
            angle_rad = math.radians(self.rotation)
            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action3(self):
        return None, False