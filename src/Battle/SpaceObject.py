import pygame
import os

class SpaceObject:
    def __init__(self, x, y, sprite_dir, base_filename, frame_count):
        self.x = x
        self.y = y
        self.sprite_dir = sprite_dir
        self.base_filename = base_filename
        self.frame_count = frame_count
        self.current_frame = 0
        self.sprites = []
        self.masks = []
        self.size = None
        self.initialize()

    def initialize(self):
        for i in range(self.frame_count):
            path = os.path.join(self.sprite_dir, f"{self.base_filename}{i:02d}.png")
            sprite = pygame.image.load(path).convert_alpha()
            if not self.size:
                self.size = sprite.get_size()
            self.sprites.append(sprite)
            self.masks.append(pygame.mask.from_surface(sprite))

    def set_frame(self, frame_num):
        self.current_frame = frame_num % self.frame_count

    def check_collision(self, other):
        offset = (int(other.x - self.x), int(other.y - self.y))
        return self.masks[self.current_frame].overlap(other.masks[other.current_frame], offset)
