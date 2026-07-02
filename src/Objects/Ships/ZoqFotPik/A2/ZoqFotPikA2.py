import math
from dataclasses import replace

import pygame

import src.const as const
from src.Battle.collision_geometry import (
    collision_info,
    estimated_impact,
    objects_overlap,
)
from src.Battle.effects import BattleEffect
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionRole,
    ProjectileContactPolicy,
)
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class ZoqFotPikA2(Ability):
    """A parent-mounted, sprite-shaped area attack that retracts toward the ship."""

    render_priority = 1

    def __init__(self, parent):
        super().__init__("ZoqFotPikA2", parent)
        definition = ABILITY_DEFINITIONS["ZoqFotPikA2"]
        self.advancing_frames = definition.advancing_frames or 5
        self.retracting_frames = definition.retracting_frames or 5
        self.area_width = definition.area_width or 12
        self.area_length = definition.area_length or 100
        self.base_offset = definition.offset
        self._damaged_targets = set()
        self._age = 0
        self._duration = self.advancing_frames + self.retracting_frames
        self.velocity = [0.0, 0.0]
        self.can_move = False
        self.can_die = True
        self.can_expire = True
        self.area_damage_pending = parent.in_battle
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=True,
            vulnerable=True,
            persistent=True,
            plays_impact_sound=True,
        )
        self.laser_target_capabilities = replace(
            self.laser_target_capabilities,
            targetable=True,
            vulnerable=True,
            blocks_lasers=True,
        )
        self._blocking_target = None
        self._current_sprite = None
        self._current_mask = None

        self._shape_cache = {}

        self._sync_to_parent(0)
        self._age = 1

    def _generate_shape(self, length, visual_idx):
        cache_key = (length, visual_idx)
        if cache_key not in self._shape_cache:
            base_surf = pygame.Surface((self.area_width, length), pygame.SRCALPHA)
            dark_red = (139, 0, 0)
            pygame.draw.ellipse(base_surf, dark_red, (0, 0, self.area_width, length))
            pygame.draw.rect(
                base_surf, dark_red, (0, length // 2, self.area_width, length // 2)
            )

            red = (255, 0, 0)
            inner_width = max(2, self.area_width - 2)
            inner_length = max(4, length - 2)
            pygame.draw.ellipse(base_surf, red, (1, 1, inner_width, inner_length))
            pygame.draw.rect(
                base_surf, red, (1, length // 2, inner_width, (length // 2) - 1)
            )

            angle = visual_idx * const.TOTAL_SPRITE_STEP
            sprite = pygame.transform.rotate(base_surf, -angle)
            mask = pygame.mask.from_surface(sprite)
            self._shape_cache[cache_key] = (sprite, mask)
        return self._shape_cache[cache_key]

    def _sync_to_parent(self, age_frame):
        relative_direction = self.configured_gun()[1] or 0
        self.rotation = (self.parent.rotation + relative_direction) % 360
        self.heading = round(self.rotation / const.TURN_ANGLE) % const.SHIP_DIRECTIONS

        if age_frame < self.advancing_frames:
            scale_factor = (age_frame + 1) / max(1, self.advancing_frames)
        else:
            scale_factor = 1.0 - (age_frame - self.advancing_frames + 1) / max(
                1, self.retracting_frames
            )
        scale_factor = max(0.01, min(1.0, scale_factor))

        current_length = max(1, round(self.area_length * scale_factor))

        visual_idx = const.heading_to_sprite_index(self.heading)
        self._current_sprite, self._current_mask = self._generate_shape(
            current_length, visual_idx
        )
        self.size = list(self._current_sprite.get_size())

        self.current_length = current_length
        self.position = self._mounted_position(
            self.parent.position, self.parent.rotation
        )

    def _mounted_position(self, parent_position, parent_rotation):
        relative_direction = self.configured_gun()[1] or 0
        rotation = (parent_rotation + relative_direction) % 360
        base_position = self.configured_gun_position(
            rotation=parent_rotation,
            position=parent_position,
        )
        distance = self.current_length / 2.0
        angle = math.radians(rotation)
        return [
            (base_position[0] + math.sin(angle) * distance) % const.ARENA_SIZE,
            (base_position[1] - math.cos(angle) * distance) % const.ARENA_SIZE,
        ]

    def update(self):
        if not self.currently_alive:
            return False
        if self._age >= self._duration:
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        self._sync_to_parent(self._age)
        self._age += 1
        self.area_damage_pending = self.parent.in_battle
        return True

    def area_damage_for_target(self, target, distance):
        if target in self._damaged_targets:
            return 0
        if getattr(target, "player", None) == self.player:
            return 0

        _, _, overlap = collision_info(self, target)
        if not objects_overlap(self, target, overlap):
            return 0
        return self.current_damage

    def on_area_damage_hit(self, target, damage):
        self._damaged_targets.add(target)
        capabilities = getattr(
            target,
            "special_object_collision_capabilities",
            None,
        )
        if (
            capabilities is not None
            and capabilities.projectile_contact_policy
            is ProjectileContactPolicy.FRAGILE
        ):
            return

        self._blocking_target = target
        self.current_hp = 0
        self.currently_alive = False
        self.area_damage_pending = False

    def append_area_hit_effects(self, target, effects, delta, distance, damage):
        if target is not self._blocking_target:
            return
        contact, normal = estimated_impact(self, target)
        effects.append(BattleEffect.from_blast(contact, normal, damage))

    def get_collision_mask(self):
        return self._current_mask

    def get_sprite(self, interp_t=0.0):
        from src.Battle.interpolation import interpolated_sprite_index

        visual_idx = interpolated_sprite_index(self.parent, interp_t)
        sprite, _ = self._generate_shape(getattr(self, "current_length", 1), visual_idx)
        return sprite

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import (
            interpolated_position,
            interpolated_sprite_index,
        )

        parent_pos = interpolated_position(self.parent, interp_t)
        visual_idx = interpolated_sprite_index(self.parent, interp_t)
        visual_heading = visual_idx / const.VIDEO_FPS_MULTIPLIER
        parent_rotation = visual_heading * const.TURN_ANGLE
        visual_pos = self._mounted_position(parent_pos, parent_rotation)

        sprite = self.get_sprite(interp_t)
        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int((visual_pos[0] + translation[0]) * scale_factor)
        screen_y = int((visual_pos[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                if (
                    -scaled_rect.width
                    <= pos_x
                    <= const.SCREEN_HEIGHT + scaled_rect.width
                    and -scaled_rect.height
                    <= pos_y
                    <= const.SCREEN_HEIGHT + scaled_rect.height
                ):
                    screen.blit(
                        scaled_sprite,
                        (
                            const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                            pos_y - scaled_rect.height // 2,
                        ),
                    )
