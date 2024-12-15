#Thraddash
from src.Objects.Ships.SpaceShip import SpaceShip
import src.Objects.Ships.Thraddash.ThraddashA1 as A1
import src.Objects.Ships.Thraddash.ThraddashA2 as A2
import src.Objects.Ships.Thraddash.ThraddashA3 as A3
import pygame
import src.Const as Const

class Thraddash(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        # Load shared sprites if not already loaded
        if Thraddash.shared_sprites is None:
            Thraddash.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):  # Assuming 16 directions for the sprite
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                Thraddash.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

        # Use the shared sprites for this instance
        self.sprites = Thraddash.shared_sprites

    def perform_action1(self):
        return A1.action(self)

    def perform_action2(self):
        return A2.action(self)
		
    def perform_action3(self):
        return A3.action(self)