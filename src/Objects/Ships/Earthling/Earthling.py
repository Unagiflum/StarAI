#Earthling
from src.Objects.Ships.SpaceShip import SpaceShip
import src.Objects.Ships.Earthling.EarthlingA1 as A1
import src.Objects.Ships.Earthling.EarthlingA2 as A2
import src.Objects.Ships.Earthling.EarthlingA3 as A3
import pygame
import src.Const as Const

class Earthling(SpaceShip):
    shared_sprites = None

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

        # Load shared sprites if not already loaded
        if Earthling.shared_sprites is None:
            Earthling.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):  # Assuming 16 directions for the sprite
                sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                Earthling.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

        # Use the shared sprites for this instance
        self.sprites = Earthling.shared_sprites

    def perform_action1(self):
        print("Action1")
        return A1.action(self)

    def perform_action2(self):
        print("Action2")
        return A2.action(self)

    #def perform_action3(self):
    #    print("Action3")
    #    return A3.action(self), True