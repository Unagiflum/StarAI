import pygame

import src.const as const
from src.Battle.collision_geometry import collision_info, objects_overlap, radius
from src.Battle.effects import BattleEffect
from src.Battle.interpolation import interpolated_sprite_index
from src.collision_capabilities import AreaDamageCapabilities, LaserTargetCapabilities
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.launch_geometry import (
    absolute_direction,
    anchored_sprite_position,
)


class UmgahA1(Ability):
    """Parent-mounted, mask-shaped antimatter cone."""

    render_priority = -1

    def __init__(self, parent):
        super().__init__("UmgahA1", parent)
        self.current_frame = parent.take_a1_animation_frame()
        self.position, self.rotation, self.heading = self._mount_transform()
        self.previous_position = self.position.copy()
        _, relative_direction = self.configured_gun()
        heading_offset = round(relative_direction / const.TURN_ANGLE)
        self.previous_heading = (
            parent.previous_heading + heading_offset
        ) % const.SHIP_DIRECTIONS
        self.velocity = [0.0, 0.0]
        self.can_move = False
        self.can_die = False
        self.can_expire = True
        self.area_damage_pending = parent.in_battle
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=False,
            vulnerable=False,
        )
        self.laser_target_capabilities = LaserTargetCapabilities(
            targetable=True,
            vulnerable=True,
            blocks_lasers=True,
        )
        self._age = 0

    def update(self):
        if not self.currently_alive or self._age >= self.life_time:
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        self.previous_heading = self.heading
        self.position, self.rotation, self.heading = self._mount_transform()
        self.area_damage_pending = self.parent.in_battle
        self._age += 1
        return True

    def _mount_transform(self):
        gun_location, relative_direction = self.configured_gun()
        direction = absolute_direction(self.parent, relative_direction)
        position = anchored_sprite_position(
            self.parent,
            gun_location,
            relative_direction,
            self.anchor_offsets[self.current_frame],
        )
        heading = round(direction / const.TURN_ANGLE) % const.SHIP_DIRECTIONS
        return position, direction, heading

    def area_damage_for_target(self, target, distance):
        _, _, overlap = collision_info(self, target)
        if not objects_overlap(self, target, overlap):
            return 0
        return self.current_damage

    def on_planet_area_hit(self, planet, effects, delta, distance, damage):
        if distance > 0:
            normal = [-delta[0] / distance, -delta[1] / distance]
            contact = [
                (planet.position[0] + normal[0] * radius(planet))
                % const.ARENA_SIZE,
                (planet.position[1] + normal[1] * radius(planet))
                % const.ARENA_SIZE,
            ]
        else:
            normal = [0.0, -1.0]
            contact = list(planet.position)
        effects.append(BattleEffect.from_blast(contact, normal, damage, planet))

    def get_sprite(self, interp_t=0.0):
        return self.sprites[self.current_frame][
            interpolated_sprite_index(self, interp_t)
        ]

    def get_collision_mask(self):
        return self.masks[self.current_frame][
            const.heading_to_sprite_index(self.heading)
        ]

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        sprite = self.get_sprite(interp_t)
        from src.Battle.interpolation import interpolated_position

        position = interpolated_position(self, interp_t)
        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
        scaled_rect = scaled_sprite.get_rect()
        screen_x = int((position[0] + translation[0]) * scale_factor)
        screen_y = int((position[1] + translation[1]) * scale_factor)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor
                if (
                    -scaled_rect.width <= pos_x <= const.SCREEN_HEIGHT + scaled_rect.width
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
