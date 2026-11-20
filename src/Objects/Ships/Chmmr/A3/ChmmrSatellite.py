from dataclasses import replace
import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability, SPECIAL_OBJECT_AREA_IMMUNITIES
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.collision_capabilities import ProjectileContactPolicy
from src.toroidal import wrapped_delta, wrapped_distance


# HP bar drawn above each satellite: white at full HP, then green at 9 HP,
# yellow at 5 and red at 1, blended in between.
HP_BAR_FULL_COLOR = (255, 255, 255)
HP_BAR_COLOR_STOPS = ((1, (255, 0, 0)), (5, (255, 255, 0)), (9, (0, 255, 0)))
HP_BAR_BACK_COLOR = (60, 60, 60)
HP_BAR_HEIGHT = 3
HP_BAR_GAP = 3


def hp_bar_appearance(hp, max_hp):
    """Return the HP bar's filled fraction and colour for ``hp`` of ``max_hp``."""

    fraction = max(0.0, min(1.0, hp / max_hp)) if max_hp > 0 else 0.0
    if hp >= max_hp:
        return fraction, HP_BAR_FULL_COLOR
    low_hp, low_color = HP_BAR_COLOR_STOPS[0]
    if hp <= low_hp:
        return fraction, low_color
    for high_hp, high_color in HP_BAR_COLOR_STOPS[1:]:
        if hp <= high_hp:
            blend = (hp - low_hp) / (high_hp - low_hp)
            color = tuple(
                round(low + (high - low) * blend)
                for low, high in zip(low_color, high_color)
            )
            return fraction, color
        low_hp, low_color = high_hp, high_color
    return fraction, low_color


class ChmmrSatellite(Ability):
    def __init__(
        self,
        parent,
        orbit_index=0,
        *,
        orbit_direction=1,
        starting_hp=None,
    ):
        super().__init__("ChmmrSatellite", parent)
        definition = SHIP_DEFINITIONS[parent.name]
        laser_definition = ABILITY_DEFINITIONS["ChmmrSatelliteLaser"]
        self.orbit_index = orbit_index
        self.orbit_direction = -1 if orbit_direction < 0 else 1
        self.orbit_phase = orbit_index * definition.satellite_period / definition.satellite_count
        self.current_hp = (
            definition.satellite_hp
            if starting_hp is None
            else max(1, min(definition.satellite_hp, int(starting_hp)))
        )
        self.start_hp = definition.satellite_hp
        self.speed = definition.satellite_speed
        self.orbit_period = definition.satellite_period
        self.orbit_distance = definition.satellite_distance
        self.laser_range = laser_definition.range
        self.laser_wait = laser_definition.weapon_wait
        self.laser_timer = 0
        self.animation_frame = 0
        # Frames updated, and the frame each ramming ship last touched it.
        self._frames_updated = 0
        self._last_ram_frames = {}
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

    def should_collide_with_ship(self, ship):
        """Ships pass through satellites, except an enemy Androsynth blazer,
        which rams them."""

        return (
            getattr(ship, "player", self.player) != self.player
            and bool(getattr(ship, "is_blazer", False))
        )

    def ram_contact_allowed(self, ship):
        """One ram per contact: a ship touching this satellite last frame
        as well deals no further damage until the contact ends."""

        frame = self._frames_updated
        previous = self._last_ram_frames.get(id(ship))
        self._last_ram_frames[id(ship)] = frame
        return previous is None or previous < frame - 1

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
        ):
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        self._frames_updated += 1
        self.orbit_phase = (
            self.orbit_phase + self.orbit_direction
        ) % self.orbit_period
        self._move_toward_orbit_target()
        self.animation_frame = (
            self.animation_frame + self.orbit_direction
        ) % 8

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
        if self.orbit_direction > 0 or subframe == 0:
            frame_index = (
                self.animation_frame * const.VIDEO_FPS_MULTIPLIER + subframe
            )
        else:
            previous_frame = (self.animation_frame - 1) % 8
            frame_index = (
                previous_frame * const.VIDEO_FPS_MULTIPLIER
                + const.VIDEO_FPS_MULTIPLIER
                - subframe
            )
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
                    left = const.SCREEN_LEFT + x - rect.width // 2
                    top = y - rect.height // 2
                    screen.blit(scaled, (left, top))
                    self._draw_hp_bar(screen, left, top, rect.width, scale_factor)

    def _draw_hp_bar(self, screen, left, sprite_top, width, scale_factor):
        """Draw the HP bar centred above the satellite sprite."""

        fraction, color = hp_bar_appearance(self.current_hp, self.start_hp)
        height = max(2, round(HP_BAR_HEIGHT * scale_factor))
        top = sprite_top - max(1, round(HP_BAR_GAP * scale_factor)) - height
        pygame.draw.rect(screen, HP_BAR_BACK_COLOR, (left, top, width, height))
        filled = round(width * fraction)
        if filled > 0:
            pygame.draw.rect(screen, color, (left, top, filled, height))
