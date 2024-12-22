from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const
import math


class Pkunk(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        if Pkunk.shared_sprites is None:
            Pkunk.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                Pkunk.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

        self.sprites = Pkunk.shared_sprites

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)

            projectiles = []
            spawn_distance = (self.size[1] + Projectile("PkunkA1", self).size[1]) / 2

            for angle_offset in [-90, 0, 90]:
                angle_rad = math.radians(self.rotation + angle_offset)

                projectile = Projectile("PkunkA1", self)
                projectile.position = [
                    self.position[0] + math.sin(angle_rad) * spawn_distance,
                    self.position[1] - math.cos(angle_rad) * spawn_distance
                ]

                projectile.heading = (self.heading + angle_offset // Const.TURN_ANGLE) % Const.SHIP_DIRECTIONS
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