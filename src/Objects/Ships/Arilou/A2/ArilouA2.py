from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import pygame


class ArilouA2(Ability):
    def __init__(self, parent):
        super().__init__("ArilouA2", parent)
        ability_data = ABILITIES_DATA["ArilouA2"]
        self.place_self()

    def place_self(self):
        self.position = self.parent.position.copy()
        self.heading = 0
        self.rotation = 0
        self.velocity = [0, 0]

    def get_sprite(self, interp_t=0.0):
        sprite = super().get_sprite(interp_t)
        cache = getattr(self, "_circular_sprite_cache", None)
        if cache is None:
            cache = self._circular_sprite_cache = {}
        key = id(sprite)
        if key not in cache:
            clipped = sprite.copy()
            alpha_mask = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            center = (sprite.get_width() // 2, sprite.get_height() // 2)
            pygame.draw.circle(
                alpha_mask,
                (255, 255, 255, 255),
                center,
                min(sprite.get_size()) // 2,
            )
            clipped.blit(alpha_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            cache[key] = clipped
        return cache[key]
