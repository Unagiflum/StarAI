from dataclasses import replace
import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability, SPECIAL_OBJECT_AREA_IMMUNITIES
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.collision_capabilities import ProjectileContactPolicy
from src.toroidal import wrapped_delta, wrapped_distance


class ChmmrSatellite(Ability):
    def __init__(self, parent, orbit_index=0):
        super().__init__("ChmmrSatellite", parent)
        definition = SHIP_DEFINITIONS[parent.name]
        self.orbit_index = orbit_index
        self.orbit_phase = orbit_index * definition.satellite_period / definition.satellite_count
        self.current_hp = definition.satellite_hp
        self.start_hp = definition.satellite_hp
        self.speed = definition.satellite_speed
        self.orbit_period = definition.satellite_period
        self.orbit_distance = definition.satellite_distance
        self.laser_range = definition.satellite_laser_range
        self.laser_wait = definition.satellite_laser_wait
        self.laser_timer = 0
        self.animation_frame = 0
        self._spawned_objects = []
        self.expiration_timer = float("inf")

        directory = const.source_path("Objects/Ships/Chmmr")
        self._satellite_frames = self.resources.animation(
            "ChmmrSatellite",
            tuple(directory / f"ChmmrSat{index:02d}.png" for index in range(8)),
            interpolated=True,
        )
        base_frames = self._satellite_frames[:: const.VIDEO_FPS_MULTIPLIER]
        self.masks = tuple(pygame.mask.from_surface(frame) for frame in base_frames)
        self.size = list(base_frames[0].get_size())

        self.area_damage_capabilities = replace(
            self.area_damage_capabilities,
            immune_to_sources=SPECIAL_OBJECT_AREA_IMMUNITIES,
        )
        self.special_object_collision_capabilities = replace(
            self.special_object_collision_capabilities,
            collides_with_planets=False,
            collides_with_asteroids=False,
            damages_asteroids=False,
            collides_with_projectiles=True,
            damages_projectiles=True,
            collides_with_enemy_ships=False,
            collides_with_friendly_ships=False,
            collides_with_fighters=False,
            projectile_contact_policy=(
                ProjectileContactPolicy.TAKE_DAMAGE_AND_DESTROY_PROJECTILE
            ),
        )
        self._place_at_orbit_target()
        self.previous_position = self.position.copy()

    def _orbit_target(self):
        angle = 2 * math.pi * (self.orbit_phase / self.orbit_period)
        return [
            (self.parent.position[0] + math.sin(angle) * self.orbit_distance)
            % const.ARENA_SIZE,
            (self.parent.position[1] - math.cos(angle) * self.orbit_distance)
            % const.ARENA_SIZE,
        ]

    def _place_at_orbit_target(self):
        self.position = self._orbit_target()

    def _move_toward_orbit_target(self):
        delta = wrapped_delta(self.position, self._orbit_target())
        distance = math.hypot(*delta)
        if distance == 0:
            return
        step = min(self.speed, distance)
        self.position[0] = (self.position[0] + delta[0] / distance * step) % const.ARENA_SIZE
        self.position[1] = (self.position[1] + delta[1] / distance * step) % const.ARENA_SIZE

    def should_collide_with_projectile_like(self, other):
        return (
            getattr(other, "type", None) == "projectile"
            and getattr(other, "player", self.player) != self.player
        )

    def _valid_laser_target(self, target):
        if (
            target is None
            or getattr(target, "player", self.player) == self.player
            or not getattr(target, "currently_alive", False)
            or getattr(target, "current_hp", 0) <= 0
            or getattr(target, "name", None) == "SyreenCrew"
            or getattr(target, "cloaked", False)
            or not getattr(target, "trackable", True)
            or wrapped_distance(self.position, target.position) > self.laser_range
        ):
            return False
        physics = getattr(target, "physical_collision_capabilities", None)
        if physics is not None and physics.is_intangible:
            return False
        capabilities = getattr(target, "laser_target_capabilities", None)
        return capabilities is None or capabilities.vulnerable

    def _select_laser_target(self):
        candidates = []
        if self._valid_laser_target(self.opponent):
            candidates.append(self.opponent)
        candidates.extend(
            target
            for target in self.enemy_objects
            if self._valid_laser_target(target)
        )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda target: (
                target.current_hp,
                wrapped_distance(self.position, target.position),
            ),
        )

    def update(self):
        if (
            not self.currently_alive
            or self.current_hp <= 0
            or not self.parent.currently_alive
            or self.parent.current_hp <= 0
        ):
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        self.orbit_phase = (self.orbit_phase + 1) % self.orbit_period
        self._move_toward_orbit_target()
        self.animation_frame = (self.animation_frame + 1) % 8

        if self.laser_timer > 0:
            self.laser_timer -= 1
        else:
            target = self._select_laser_target()
            if target is not None:
                from src.Objects.Ships.Chmmr.A3.ChmmrSatelliteLaser import (
                    ChmmrSatelliteLaser,
                )

                self._spawned_objects.append(ChmmrSatelliteLaser(self, target))
                self.laser_timer = self.laser_wait
        return True

    def drain_spawned_objects(self):
        spawned, self._spawned_objects = self._spawned_objects, []
        return spawned

    def get_collision_mask(self):
        return self.masks[self.animation_frame]

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import interpolated_position

        subframe = min(
            const.VIDEO_FPS_MULTIPLIER - 1,
            max(0, int(interp_t * const.VIDEO_FPS_MULTIPLIER)),
        )
        frame_index = self.animation_frame * const.VIDEO_FPS_MULTIPLIER + subframe
        sprite = self._satellite_frames[frame_index % len(self._satellite_frames)]
        scaled = pygame.transform.smoothscale_by(sprite, scale_factor)
        rect = scaled.get_rect()
        position = interpolated_position(self, interp_t)
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
                        (const.SCREEN_LEFT + x - rect.width // 2, y - rect.height // 2),
                    )
