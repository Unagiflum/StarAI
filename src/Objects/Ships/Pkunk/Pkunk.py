from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const
import math


class Pkunk(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)

            projectiles = []
            spawn_distance = Const.PROJ_GAP + (self.size[1] + Projectile("PkunkA1", self).size[1]) / 2

            for angle_offset in [-90, 0, 90]:
                angle_rad = math.radians(self.rotation + angle_offset)
                h_mult = 1.0
                if angle_offset != 0:
                    h_mult *= self.size[0]/self.size[1]
                projectile = Projectile("PkunkA1", self)
                projectile.position = [
                    self.position[0] + math.sin(angle_rad) * spawn_distance * h_mult,
                    self.position[1] - math.cos(angle_rad) * spawn_distance * h_mult
                ]
                if projectile.omnidirectional:
                    projectile.heading = 0
                else:
                    projectile.heading = int(self.heading + angle_offset // Const.TURN_ANGLE) % Const.SHIP_DIRECTIONS
                projectile.rotation = projectile.heading * Const.TURN_ANGLE
                projectile.velocity = [
                    math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                    -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
                ]

                projectiles.append(projectile)
            if projectile.launch_sound: projectile.launch_sound.play()
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