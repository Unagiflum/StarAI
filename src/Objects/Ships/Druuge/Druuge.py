from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const
import math

MAX_RECOIL = 96
INC_RECOIL = MAX_RECOIL / 4

class Druuge(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        if Druuge.shared_sprites is None:
            Druuge.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                Druuge.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

        self.sprites = Druuge.shared_sprites

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = Projectile("DruugeA1", self)

            spawn_distance = (self.size[1] + projectile.size[1]) / 2  # Ship height + projectile height
            projectile.position = [
                self.position[0] + math.sin(angle_rad) * spawn_distance,
                self.position[1] - math.cos(angle_rad) * spawn_distance
            ]
            projectile.heading = self.heading
            projectile.rotation = self.rotation
            projectile.opponent = self.opponent
            angle_rad = math.radians(self.rotation)
            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]
            recoil_target = [math.sin(angle_rad)*MAX_RECOIL, -math.cos(angle_rad)*MAX_RECOIL]
            diff_vector = [recoil_target[0] - self.velocity[0], recoil_target[1] - self.velocity[1]]
            diff_magnitude = math.sqrt(diff_vector[0]**2 + diff_vector[1]**2)
            if diff_magnitude > 0:
                if diff_magnitude < INC_RECOIL:
                    self.velocity[0] = -diff_vector[0]
                    self.velocity[1] = -diff_vector[1]
                else:
                    self.velocity[0] -= diff_vector[0]*INC_RECOIL/diff_magnitude
                    self.velocity[1] -= diff_vector[1]*INC_RECOIL/diff_magnitude


            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * Const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False