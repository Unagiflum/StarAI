#Arilou
from src.Objects.Ships.SpaceShip import SpaceShip
import pygame
import src.Const as Const

class Arilou(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        if Arilou.shared_sprites is None:
            Arilou.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                base_sprite = pygame.image.load(str(sprite_path)).convert_alpha()
                scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                Arilou.shared_sprites.append(scaled_sprite)

        self.sprites = Arilou.shared_sprites

    def perform_action1(self):
        if self.can_action1():
            print("Action 1", self.current_energy, self.a1_cost)
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action2(self):
        if self.can_action2():
            print("Action 2", self.current_energy, self.a2_cost)
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * Const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False