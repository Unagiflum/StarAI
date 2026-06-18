import math
from pathlib import Path

import pygame

import src.const as const
from src.Objects.object import Object


BATTLE_ASSET_PATH = Path("Objects/Battle")


class BattleEffect(Object):
    _blast_sprites = None
    _boom_sounds = {}

    def __init__(self, position, frames, frame_delay=2, scale=1.0):
        first_frame = frames[0] if frames else None
        size = [0, 0]
        if first_frame:
            size = [
                max(1, int(first_frame.get_width() * scale)),
                max(1, int(first_frame.get_height() * scale)),
            ]

        super().__init__(
            name="BattleEffect",
            sprite_location=None,
            size=size,
        )
        self.position = list(position)
        self.frames = frames
        self.frame_delay = max(1, frame_delay)
        self.scale = scale
        self.current_frame = 0
        self.frame_timer = self.frame_delay
        self.can_collide = False
        self.can_expire = True

    @classmethod
    def from_animation(cls, position, frames, frame_delay=2, scale=1.0,
                       direction_vector=None, align_edge=False):
        if align_edge and frames:
            position = cls._edge_aligned_position(position, frames[0], scale, direction_vector)
        return cls(position, frames, frame_delay, scale)

    @classmethod
    def from_blast(cls, position, direction_vector, damage, align_edge=False):
        if cls._blast_sprites is None:
            cls._blast_sprites = [
                pygame.image.load(str(BATTLE_ASSET_PATH / f"blast-{i:03d}.png")).convert_alpha()
                for i in range(8)
            ]

        index = cls._blast_index(direction_vector)
        scale = min(1.0, max(0.35, max(1, damage) / 6))
        if align_edge:
            position = cls._edge_aligned_position(
                position,
                cls._blast_sprites[index],
                scale,
                direction_vector
            )
        return cls(position, [cls._blast_sprites[index]], frame_delay=4, scale=scale)

    @classmethod
    def play_boom(cls, damage):
        sound_name = cls._boom_name(damage)
        if sound_name not in cls._boom_sounds:
            try:
                sound = pygame.mixer.Sound(str(BATTLE_ASSET_PATH / sound_name))
                sound.set_volume(const.SOUND_EFFECT_VOLUME)
                cls._boom_sounds[sound_name] = sound
            except pygame.error:
                cls._boom_sounds[sound_name] = None

        sound = cls._boom_sounds[sound_name]
        if sound:
            sound.play()

    @staticmethod
    def _boom_name(damage):
        damage = max(1, int(math.ceil(damage)))
        if damage == 1:
            return "boom1.wav"
        if damage <= 3:
            return "boom2.wav"
        if damage <= 5:
            return "boom4.wav"
        return "boom6.wav"

    @staticmethod
    def _blast_index(direction_vector):
        dx, dy = direction_vector
        if dx == 0 and dy == 0:
            return 0
        angle = math.degrees(math.atan2(dx, -dy))
        return round(angle / 45) % 8

    @staticmethod
    def _edge_aligned_position(contact_position, sprite, scale, direction_vector):
        dx, dy = direction_vector or (0, 0)
        length = math.hypot(dx, dy)
        if length == 0:
            return contact_position

        radius = max(sprite.get_width(), sprite.get_height()) * scale / 2
        return [
            (contact_position[0] + dx / length * radius) % const.ARENA_SIZE,
            (contact_position[1] + dy / length * radius) % const.ARENA_SIZE,
        ]

    def update(self):
        if not self.frames:
            return False

        self.frame_timer -= 1
        if self.frame_timer > 0:
            return True

        self.frame_timer = self.frame_delay
        self.current_frame += 1
        return self.current_frame < len(self.frames)

    def draw(self, screen, scale_factor, translation):
        if not self.frames or self.current_frame >= len(self.frames):
            return

        sprite = self.frames[self.current_frame]
        scaled_width = max(1, int(sprite.get_width() * scale_factor * self.scale))
        scaled_height = max(1, int(sprite.get_height() * scale_factor * self.scale))
        scaled_sprite = pygame.transform.smoothscale(sprite, (scaled_width, scaled_height))
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                if (-scaled_rect.width <= pos_x <= const.SCREEN_HEIGHT + scaled_rect.width and
                        -scaled_rect.height <= pos_y <= const.SCREEN_HEIGHT + scaled_rect.height):
                    screen.blit(scaled_sprite, (
                        const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2,
                    ))
