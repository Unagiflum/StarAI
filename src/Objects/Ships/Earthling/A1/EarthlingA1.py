from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const

class EarthlingA1(Projectile):
    shared_sprites = None
    death_anim = None
    launch_sound = None

    def __init__(self, parent):
        super().__init__("EarthlingA1", parent)

        if EarthlingA1.shared_sprites is None:
            # Load sprites
            EarthlingA1.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'EarthlingA1{i:02d}.png')
                EarthlingA1.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

            # Load death animation if it exists
            if self.death_anim > 0:
                EarthlingA1.death_anim = []
                for i in range(self.death_anim):
                    try:
                        death_path = self.sprite_location.joinpath(f'EarthlingA1die{i:02d}.png')
                        EarthlingA1.death_anim.append(pygame.image.load(str(death_path)).convert_alpha())
                    except pygame.error:
                        break

            # Load sound if it exists
            try:
                sound_path = self.sprite_location.joinpath('EarthlingA1.wav')
                EarthlingA1.launch_sound = pygame.mixer.Sound(str(sound_path))
            except pygame.error:
                EarthlingA1.launch_sound = None