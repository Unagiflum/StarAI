import pygame

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.launch_geometry import (
    absolute_direction,
    direction_vector,
    gun_world_position,
    mask_projection_bounds,
)


class MelnormeA1(Ability):
    def __init__(self, parent):
        super().__init__("MelnormeA1", parent)
        definition = ABILITY_DEFINITIONS[self.name]
        self.gun_levels = definition.gun_levels
        self.gun_level_mult = definition.gun_level_mult
        self.gun_level_timer = definition.gun_level_timer
        self.level = 0
        self.level_timer = 0
        self.held = True
        self._restore_held_hp = False
        self.animation_timer = 2

        resource_dir = const.source_path(definition.file_path)
        self.level_sprites = tuple(
            self.resources.animation(
                f"{self.name}-level-{level}",
                tuple(
                    resource_dir / f"{self.name}0{level}_{frame:02d}.png"
                    for frame in range(self.frames)
                ),
            )
            for level in range(self.gun_levels)
        )
        self.level_masks = tuple(
            tuple(pygame.mask.from_surface(sprite) for sprite in sprites)
            for sprites in self.level_sprites
        )
        self.reference_mask = self.level_masks[self.gun_levels - 1][2]
        self._apply_level(reset_animation=True)
        self._place_held(reset_previous=True)
        self.parent.held_a1 = self

    def _level_value(self, base):
        return base * self.gun_level_mult**self.level

    def _apply_level(self, *, reset_animation=False):
        definition = ABILITY_DEFINITIONS[self.name]
        self.start_hp = self._level_value(definition.start_hp[0])
        self.current_hp = self.start_hp
        self.current_damage = self._level_value(definition.damage[0])
        if reset_animation:
            self.current_frame = 0
            self.animation_timer = 2
        self.size = list(
            self.level_sprites[self.level][self.current_frame].get_size()
        )

    def _place_held(self, *, reset_previous=False):
        gun_location, relative_direction = self.configured_gun()
        direction = absolute_direction(self.parent, relative_direction)
        muzzle = gun_world_position(self.parent, gun_location)
        rear_projection, _ = mask_projection_bounds(self.reference_mask, 0)
        forward_x, forward_y = direction_vector(direction)
        distance = const.PROJ_GAP * 2 - rear_projection
        position = [
            (muzzle[0] + forward_x * distance) % const.ARENA_SIZE,
            (muzzle[1] + forward_y * distance) % const.ARENA_SIZE,
        ]
        self.position = position
        if reset_previous:
            self.previous_position = position.copy()
        self.rotation = direction
        self.heading = 0
        self.velocity = list(self.parent.velocity)

    def release(self):
        if not self.held or not self.is_alive():
            return
        self.held = False
        if getattr(self.parent, "held_a1", None) is self:
            self.parent.held_a1 = None
        direction = self.rotation
        forward_x, forward_y = direction_vector(direction)
        self.velocity = [
            forward_x * self.speed + self.parent.velocity[0] * self.parent_vel,
            forward_y * self.speed + self.parent.velocity[1] * self.parent_vel,
        ]
        self.expiration_timer = self._duration

    def _advance_animation(self):
        self.animation_timer -= 1
        if self.animation_timer <= 0:
            self.current_frame = (self.current_frame + 1) % self.frames
            self.animation_timer = 1
            self.size = list(
                self.level_sprites[self.level][self.current_frame].get_size()
            )

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        if self.held:
            if self._restore_held_hp:
                self.current_hp = self.start_hp
                self._restore_held_hp = False
            self._place_held()
            self.level_timer += 1
            if (
                self.level < self.gun_levels - 1
                and self.level_timer >= self.gun_level_timer
            ):
                self.level += 1
                self.level_timer = 0
                self._apply_level(reset_animation=True)
        else:
            self.update_physics()
            self.expiration_timer -= 1

        self._advance_animation()
        return (
            self.currently_alive
            and self.current_hp > 0
            and (self.held or self.expiration_timer > 0)
        )

    def get_sprite(self, interp_t=0.0):
        return self.level_sprites[self.level][self.current_frame]

    def get_collision_mask(self):
        return self.level_masks[self.level][self.current_frame]

    def set_hp(self, new_hp):
        super().set_hp(new_hp)
        if self.held and self.current_hp > 0:
            self._restore_held_hp = True

    def handle_projectile_contact(self, projectile):
        if not self.held:
            return False

        capabilities = getattr(
            projectile, "special_object_collision_capabilities", None
        )
        incoming_damage = (
            projectile.current_damage
            if capabilities is None or capabilities.damages_projectiles
            else 0
        )
        self.set_hp(self.current_hp - incoming_damage)

        if self.special_object_collision_capabilities.damages_projectiles:
            setter = getattr(projectile, "set_hp", None)
            new_hp = projectile.current_hp - self.current_damage
            if setter is not None:
                setter(new_hp)
            else:
                projectile.current_hp = max(0, new_hp)
        return True

    def on_destroyed(self):
        if not self.held:
            return
        cost = self.parent.a1_cost
        if self.parent.current_hp > 0 and self.parent.current_energy >= cost:
            self.parent.current_energy -= cost
            self.level = 0
            self.level_timer = 0
            self._apply_level(reset_animation=True)
            self.currently_alive = True
            self._destruction_finalized = False
            self._restore_held_hp = False
            self.expiration_timer = self._duration
            self._place_held(reset_previous=True)
            self.parent.held_a1 = self
            return

        self.held = False
        if getattr(self.parent, "held_a1", None) is self:
            self.parent.held_a1 = None

    def stop_and_track(self):
        self.release()
