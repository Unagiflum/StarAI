#KzerZa
from src.Objects.Ships.SpaceShip import SpaceShip
import src.Objects.Ships.KzerZa.KzerZaA1 as A1
import src.Objects.Ships.KzerZa.KzerZaA2 as A2
import src.Objects.Ships.KzerZa.KzerZaA3 as A3
import pygame
import src.Const as Const

class KzerZa(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        # Load shared sprites if not already loaded
        if KzerZa.shared_sprites is None:
            KzerZa.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):  # Assuming 16 directions for the sprite
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                KzerZa.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

        # Use the shared sprites for this instance
        self.sprites = KzerZa.shared_sprites

    def perform_action1(self):
        return A1.action(self)

    def perform_action2(self):
        return A2.action(self)
		
    def perform_action3(self):
        return A3.action(self)