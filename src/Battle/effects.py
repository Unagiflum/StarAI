import math

import pygame

import src.const as const
from src.Objects.object import Object
from src.resources import active_asset_manager, default_assets
from src.audio import active_audio_service, compatibility_audio_service


BATTLE_ASSET_PATH = const.source_path("Objects/Battle")


class BattleEffect(Object):
    sound_enabled = True
    resources = default_assets()

    @classmethod
    def _resources(cls):
        return active_asset_manager() or cls.resources

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
        self.previous_position = self.position.copy()
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
    def from_blast(cls, position, direction_vector, damage):
        blast_sprites = cls._resources().animation(
            "battle-blasts",
            tuple(BATTLE_ASSET_PATH / f"blast-{i:03d}.png" for i in range(8)),
        )

        index = cls._blast_index(direction_vector)
        scale = cls._blast_scale(damage)
        return cls(position, [blast_sprites[index]], frame_delay=4, scale=scale)

    @classmethod
    def ship_explosion(cls, position, frame_delay=2, scale=1.0):
        ship_explosion_sprites = cls._resources().animation(
            "ship-explosions",
            tuple(
                BATTLE_ASSET_PATH / f"explosion-{index:03d}.png"
                for index in range(8)
            ),
        )

        return cls(position, ship_explosion_sprites, frame_delay=frame_delay, scale=scale)

    @classmethod
    def play_ship_death(cls):
        audio = active_audio_service() or compatibility_audio_service(
            cls.sound_enabled, cls.resources
        )
        seconds = audio.play_effect(
            BATTLE_ASSET_PATH / "shipdies.wav",
            const.SOUND_EFFECT_VOLUME,
        )
        return int(math.ceil(seconds * const.FPS))

    @classmethod
    def play_boom(cls, damage):
        sound_name = cls._boom_name(damage)
        audio = active_audio_service() or compatibility_audio_service(
            cls.sound_enabled, cls.resources
        )
        audio.play_effect(
            BATTLE_ASSET_PATH / sound_name,
            const.SOUND_EFFECT_VOLUME,
        )

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
    def _blast_scale(damage):
        damage = max(1, int(math.ceil(damage)))
        if damage == 1:
            return 0.7
        if damage <= 3:
            return 0.8
        if damage <= 5:
            return 0.9
        return 1.0

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

        nx = dx / length
        ny = dy / length
        mask = pygame.mask.from_surface(sprite)
        bounds = mask.get_bounding_rects()
        center_x = sprite.get_width() / 2
        center_y = sprite.get_height() / 2
        inward_projection = None
        for rect in bounds:
            for y in range(rect.top, rect.bottom):
                for x in range(rect.left, rect.right):
                    if not mask.get_at((x, y)):
                        continue
                    projection = (
                        (x - center_x) * nx + (y - center_y) * ny
                    )
                    if inward_projection is None or projection < inward_projection:
                        inward_projection = projection

        if inward_projection is None:
            inward_projection = -(
                abs(nx) * sprite.get_width()
                + abs(ny) * sprite.get_height()
            ) / 2
        offset = -inward_projection * scale
        return [
            (contact_position[0] + nx * offset) % const.ARENA_SIZE,
            (contact_position[1] + ny * offset) % const.ARENA_SIZE,
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

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        if not self.frames or self.current_frame >= len(self.frames):
            return

        sprite = self.frames[self.current_frame]
        scaled_width = max(1, int(sprite.get_width() * scale_factor * self.scale))
        scaled_height = max(1, int(sprite.get_height() * scale_factor * self.scale))
        scaled_sprite = pygame.transform.smoothscale(sprite, (scaled_width, scaled_height))
        scaled_rect = scaled_sprite.get_rect()

        from src.Battle.interpolation import interpolated_position
        pos = interpolated_position(self, interp_t)
        screen_x = int((pos[0] + translation[0]) * scale_factor)
        screen_y = int((pos[1] + translation[1]) * scale_factor)

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
