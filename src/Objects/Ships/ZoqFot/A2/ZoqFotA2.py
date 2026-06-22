import math

import pygame

import src.const as const
from src.Battle.collision_geometry import collision_info, objects_overlap
from src.collision_capabilities import AreaDamageCapabilities, CollisionRole
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class ZoqFotA2(Ability):
    """A parent-mounted, sprite-shaped area attack that retracts toward the ship."""

    persistent_area_damage = True
    plays_area_impact_sound = True

    def __init__(self, parent):
        super().__init__("ZoqFotA2", parent)
        definition = ABILITY_DEFINITIONS["ZoqFotA2"]
        self.sprite_scale_x = definition.sprite_scale_x
        self.sprite_scale_y = definition.sprite_scale_y
        self._source_sprites = tuple(
            self._scale_directional_sprite(sprite, heading)
            for heading, sprite in enumerate(self.sprites)
        )
        self._source_masks = tuple(
            pygame.mask.from_surface(sprite) for sprite in self._source_sprites
        )
        self._damaged_targets = set()
        self._age = 0
        self._duration = max(1, int(self.life_time * const.PROJ_LIFE_SCALE))
        self.base_offset = definition.offset
        self.velocity = [0.0, 0.0]
        self.can_move = False
        self.can_die = False
        self.can_expire = True
        self.area_damage_pending = parent.in_battle
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=True,
            vulnerable=False,
        )
        self._current_mask = None
        self._sync_to_parent(1.0)

    def _scale_directional_sprite(self, sprite, heading):
        scale_x = self.sprite_scale_x
        scale_y = self.sprite_scale_y
        if scale_x == scale_y:
            return pygame.transform.smoothscale_by(sprite, scale_x)

        angle = heading * const.TURN_ANGLE
        local_sprite = pygame.transform.rotate(sprite, angle)
        local_bounds = local_sprite.get_bounding_rect(min_alpha=1)
        if local_bounds.width and local_bounds.height:
            local_sprite = local_sprite.subsurface(local_bounds).copy()
        local_size = (
            max(1, round(local_sprite.get_width() * scale_x)),
            max(1, round(local_sprite.get_height() * scale_y)),
        )
        local_sprite = pygame.transform.smoothscale(local_sprite, local_size)
        scaled_sprite = pygame.transform.rotate(local_sprite, -angle)
        scaled_bounds = scaled_sprite.get_bounding_rect(min_alpha=1)
        if scaled_bounds.width and scaled_bounds.height:
            scaled_sprite = scaled_sprite.subsurface(scaled_bounds).copy()
        return scaled_sprite

    @staticmethod
    def _projection_bounds(mask, heading):
        width, height = mask.get_size()
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2
        angle = math.radians(heading * const.TURN_ANGLE)
        forward_x = math.sin(angle)
        forward_y = -math.cos(angle)
        projections = (
            (x - center_x) * forward_x + (y - center_y) * forward_y
            for y in range(height)
            for x in range(width)
            if mask.get_at((x, y))
        )
        projections = tuple(projections)
        return (min(projections), max(projections)) if projections else (0.0, 0.0)

    def _retracted_sprite(self, heading, ratio):
        source_sprite = self._source_sprites[heading]
        source_mask = self._source_masks[heading]
        if ratio >= 1.0:
            return source_sprite, source_mask

        visible_mask = source_mask.copy()
        minimum, maximum = self._projection_bounds(source_mask, heading)
        cutoff = minimum + (maximum - minimum) * max(0.0, ratio)
        width, height = source_mask.get_size()
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2
        angle = math.radians(heading * const.TURN_ANGLE)
        forward_x = math.sin(angle)
        forward_y = -math.cos(angle)
        for y in range(height):
            for x in range(width):
                projection = (
                    (x - center_x) * forward_x
                    + (y - center_y) * forward_y
                )
                if projection > cutoff:
                    visible_mask.set_at((x, y), 0)

        sprite = source_sprite.copy()
        alpha_mask = visible_mask.to_surface(
            setcolor=(255, 255, 255, 255),
            unsetcolor=(255, 255, 255, 0),
        )
        sprite.blit(alpha_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return sprite, visible_mask

    def _sync_to_parent(self, retraction_ratio):
        self.heading = self.parent.heading % const.SHIP_DIRECTIONS
        self.rotation = self.parent.rotation
        sprite, self._current_mask = self._retracted_sprite(
            self.heading, retraction_ratio
        )
        displayed_sprites = list(self._source_sprites)
        displayed_sprites[self.heading] = sprite
        self.sprites = tuple(displayed_sprites)
        self.size = list(sprite.get_size())

        parent_mask = self.parent.get_collision_mask()
        parent_forward = self._projection_bounds(parent_mask, self.heading)[1]
        effect_rear = self._projection_bounds(
            self._source_masks[self.heading], self.heading
        )[0]
        base_distance = (
            parent_forward + const.PROJ_GAP
        ) * self.base_offset
        distance = base_distance - effect_rear
        angle = math.radians(self.rotation)
        self.position = [
            (self.parent.position[0] + math.sin(angle) * distance)
            % const.ARENA_SIZE,
            (self.parent.position[1] - math.cos(angle) * distance)
            % const.ARENA_SIZE,
        ]

    def update(self):
        if not self.currently_alive:
            return False
        if self._age >= self._duration:
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        ratio = 1.0 - self._age / self._duration
        self._sync_to_parent(ratio)
        self._age += 1
        self.area_damage_pending = self.parent.in_battle
        return True

    def area_damage_for_target(self, target):
        if target in self._damaged_targets:
            return 0

        role = target.collision_capabilities.role
        if role == CollisionRole.ASTEROID:
            pass
        elif role not in (
            CollisionRole.SHIP,
            CollisionRole.PROJECTILE,
            CollisionRole.FIGHTER,
        ) or target.player == self.player:
            return 0

        _, _, overlap = collision_info(self, target)
        if not objects_overlap(self, target, overlap):
            return 0
        return self.current_damage

    def record_area_damage_hit(self, target):
        self._damaged_targets.add(target)

    def get_collision_mask(self):
        return self._current_mask
