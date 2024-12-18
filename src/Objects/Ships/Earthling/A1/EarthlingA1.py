import pygame
from pathlib import Path


class EarthlingA1:
    sprites = []
    death_sprites = []

    @classmethod
    def load_sprites(cls, path, num_directions, num_death_frames):
        # Load directional sprites
        for i in range(num_directions):
            sprite_path = path / f"EarthlingA1{i:02d}.png"
            cls.sprites.append(pygame.image.load(sprite_path))

        # Load death animation sprites if they exist
        if num_death_frames > 0:
            for i in range(num_death_frames):
                death_path = path / f"EarthlingA1die{i:02d}.png"
                cls.death_sprites.append(pygame.image.load(death_path))

    def __init__(self, path, num_directions, num_death_frames):
        if not self.sprites:
            self.load_sprites(Path(path), num_directions, num_death_frames)