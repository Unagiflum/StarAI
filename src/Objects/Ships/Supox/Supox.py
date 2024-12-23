from src.Objects.Ships.SpaceShip import SpaceShip
from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const
import math


class Supox(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        if Supox.shared_sprites is None:
            Supox.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                base_sprite = pygame.image.load(str(sprite_path)).convert_alpha()
                scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                Supox.shared_sprites.append(scaled_sprite)

        self.sprites = Supox.shared_sprites

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = Projectile("SupoxA1", self)

            spawn_distance = (self.size[1] + projectile.size[1]) / 2  # Ship height + projectile height
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
            return None
        return None

    def perform_action3(self):
        return None, False