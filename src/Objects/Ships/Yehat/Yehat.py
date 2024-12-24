from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const
import math


class Yehat(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)
            side_offset = self.size[0] / 2

            projectiles = []
            for offset in [-side_offset, side_offset]:
                projectile = Projectile("YehatA1", self)

                spawn_distance = Const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
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
            self.action2_timer = int(self.a2_wait * Const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False