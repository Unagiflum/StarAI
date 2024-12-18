from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const

class EarthlingA1(Projectile):
    shared_sprites = None

    def __init__(self, parent):
        super().__init__("EarthlingA1", parent)

        if EarthlingA1.shared_sprites is None:
            EarthlingA1.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'EarthlingA1{i:02d}.png')
                EarthlingA1.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

        self.sprites = EarthlingA1.shared_sprites