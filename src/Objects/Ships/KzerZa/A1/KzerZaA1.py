from src.Objects.Ships.Projectile import Projectile
import pygame
import src.Const as Const

class KzerZaA1(Projectile):
    shared_sprites = None
    death_anim = None
    launch_sound = None

    def __init__(self, parent):
        super().__init__("KzerZaA1", parent)

        if KzerZaA1.shared_sprites is None:
            # Load sprites
            KzerZaA1.shared_sprites = []
            for i in range(Const.SHIP_DIRECTIONS):
                sprite_path = self.sprite_location.joinpath(f'{self.name}{i:02d}.png')
                KzerZaA1.shared_sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

            # Load death animation if it exists
            if self.death_anim > 0:
                KzerZaA1.death_anim = []
                for i in range(self.death_anim):
                    try:
                        death_path = self.sprite_location.joinpath(f'{self.name}die{i:02d}.png')
                        KzerZaA1.death_anim.append(pygame.image.load(str(death_path)).convert_alpha())
                    except pygame.error:
                        break

            # Load sound if it exists
            try:
                sound_path = self.sprite_location.joinpath(f'{self.name}.wav')
                KzerZaA1.launch_sound = pygame.mixer.Sound(str(sound_path))
            except pygame.error:
                KzerZaA1.launch_sound = None