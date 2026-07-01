import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.toroidal import wrapped_delta


class ChmmrA2(Ability):
    def __init__(self, parent):
        super().__init__("ChmmrA2", parent)
        definition = ABILITY_DEFINITIONS["ChmmrA2"]
        self.BASE_SPEED = definition.base_speed
        self.SILHOUETTE_COUNT = definition.silhouette_count
        self.SILHOUETTE_COLORS = definition.silhouette_colors
        self.SILHOUETTE_DIST = definition.silhouette_distances
        self.target = parent.opponent
        self.position = self.configured_gun_position()

    def _target_is_visible(self):
        return bool(
            self.target
            and self.target.currently_alive
            and self.target.current_hp > 0
            and not getattr(self.target, "cloaked", False)
            and getattr(self.target, "trackable", True)
        )

    def update(self):
        if not self.currently_alive:
            return False
        self.previous_position = self.position.copy()
        self.position = self.configured_gun_position()
        target = self.target
        if self._target_is_visible() and getattr(target, "inertia", False):
            delta = wrapped_delta(target.position, self.parent.position)
            distance = math.hypot(*delta)
            mass = getattr(target, "mass", 0)
            if distance > 0 and mass > 0:
                impulse = self.BASE_SPEED / mass
                target.add_impulse(
                    delta[0] / distance * impulse,
                    delta[1] / distance * impulse,
                )
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    @staticmethod
    def _silhouette(sprite, color, target):
        cache = getattr(target, "_chmmr_silhouette_cache", None)
        if cache is None:
            cache = {}
            target._chmmr_silhouette_cache = cache
        key = (id(sprite), color)
        if key not in cache:
            mask = pygame.mask.from_surface(sprite)
            cache[key] = mask.to_surface(
                setcolor=(*color, 255),
                unsetcolor=(0, 0, 0, 0),
            )
        return cache[key]

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        if not self._target_is_visible():
            return

        from src.Battle.interpolation import interpolated_position

        target_position = interpolated_position(self.target, interp_t)
        parent_position = interpolated_position(self.parent, interp_t)
        delta = wrapped_delta(target_position, parent_position)
        distance = math.hypot(*delta)
        if distance == 0:
            return
        direction = (delta[0] / distance, delta[1] / distance)
        sprite = self.target.set_sprite(interp_t)

        entries = zip(self.SILHOUETTE_COLORS, self.SILHOUETTE_DIST)
        for color, offset in reversed(tuple(entries)):
            silhouette = self._silhouette(sprite, color, self.target)
            scaled = pygame.transform.smoothscale_by(silhouette, scale_factor)
            rect = scaled.get_rect()
            position = [
                target_position[0] + direction[0] * offset,
                target_position[1] + direction[1] * offset,
            ]
            screen_x = int((position[0] + translation[0]) * scale_factor)
            screen_y = int((position[1] + translation[1]) * scale_factor)
            for wrap_x in (-1, 0, 1):
                for wrap_y in (-1, 0, 1):
                    x = screen_x + wrap_x * const.ARENA_SIZE * scale_factor
                    y = screen_y + wrap_y * const.ARENA_SIZE * scale_factor
                    if (
                        -rect.width <= x <= const.SCREEN_HEIGHT + rect.width
                        and -rect.height <= y <= const.SCREEN_HEIGHT + rect.height
                    ):
                        screen.blit(
                            scaled,
                            (
                                const.SCREEN_LEFT + x - rect.width // 2,
                                y - rect.height // 2,
                            ),
                        )
