from dataclasses import replace
import math

import src.const as const
from src.Objects.Ships.ability import (
    Ability,
    ABILITIES_DATA,
    SPECIAL_OBJECT_AREA_IMMUNITIES,
)
from src.collision_capabilities import (
    ProjectileContactPolicy,
    SameTypeContactPolicy,
    SpecialObjectCollisionCapabilities,
)
from src.toroidal import wrapped_delta


class SyreenCrew(Ability):
    survives_parent_cleanup = True

    def __init__(self, parent, position=None, origin_ship=None):
        super().__init__("SyreenCrew", parent)
        self.area_damage_capabilities = replace(
            self.area_damage_capabilities,
            immune_to_sources=SPECIAL_OBJECT_AREA_IMMUNITIES,
        )
        self.position = list(parent.position if position is None else position)
        self.origin_ship = origin_ship
        definition = ABILITIES_DATA["SyreenCrew"]
        self.expiration_timer = definition.life_time
        self.speed = definition.speed
        self.turn_wait = 0  # Can change direction instantly

        # Crew is rendered as an anti-aliased circle with configurable radius and colors.
        radius = definition.radius if definition.radius is not None else 2
        self.size = [radius * 2, radius * 2]
        self.colors = definition.colors or ((0, 255, 0, 255),)
        self.current_color_index = 0
        self.damages = list(definition.damage)
        self.current_damage = definition.damage[0]
        self.hp_array = list(definition.start_hp)
        self.current_hp = definition.start_hp[0]
        self.gravity_velocity = [0.0, 0.0]

        if getattr(self.parent, "planet", None):
            self.set_planet(self.parent.planet)

        self.special_object_collision_capabilities = SpecialObjectCollisionCapabilities(
            collides_with_planets=True,
            collides_with_asteroids=True,
            damages_asteroids=False,
            collides_with_projectiles=True,
            damages_projectiles=False,
            collides_with_enemy_ships=True,
            collides_with_friendly_ships=True,
            collides_with_fighters=True,
            bounces_off_same_type=True,
            projectile_contact_policy=ProjectileContactPolicy.FRAGILE,
            same_type_contact_policy=SameTypeContactPolicy.BOUNCE,
        )
        self.physical_collision_capabilities = replace(
            self.physical_collision_capabilities,
            is_fragile=True,
        )

    def handle_ship_contact(self, ship, normal):
        # Either ship, if it collides with the crew, will recover them and gain hit points.
        # "Either ship can recover the crew even if at full health, but doing so will not increase crew at that point."
        if ship.current_hp < ship.max_hp:
            ship.current_hp += 1
        self.current_hp = 0
        self.currently_alive = False

        from src.audio import active_audio_service

        audio = active_audio_service()
        if audio:
            audio.play_effect(
                const.source_path("Objects/Ships/Syreen/A2/SyreenA2Pickup.wav"),
                const.SOUND_EFFECT_VOLUME,
            )

        return True

    def can_recover_with_parent(self):
        return self.parent is not None

    def recover_with_parent(self):
        self.handle_ship_contact(self.parent, None)

    def handle_projectile_contact(self, projectile):
        if getattr(projectile, "projectile_name", None) == self.projectile_name:
            return False
        # Destroyed on contact with projectiles (except bounces off same type handled by responses)
        self.current_hp = 0
        self.currently_alive = False
        return True

    def _tracking_ship(self):
        if self.parent and self.parent.currently_alive and self.parent.current_hp > 0:
            return self.parent

        if (
            self.origin_ship
            and self.origin_ship.currently_alive
            and self.origin_ship.current_hp > 0
            and not getattr(self.origin_ship, "cloaked", False)
        ):
            return self.origin_ship

        return None

    def _homing_velocity(self, position, target_position):
        if target_position is None:
            return [0.0, 0.0]

        dx, dy = wrapped_delta(position, target_position)
        target_angle = math.degrees(math.atan2(dx, -dy)) % 360
        target_angle = (round(target_angle / 45.0) * 45.0) % 360
        angle_rad = math.radians(target_angle)
        return [
            math.sin(angle_rad) * self.speed,
            -math.cos(angle_rad) * self.speed,
        ]

    def _gravity_acceleration(self, position):
        if not self.planet:
            return [0.0, 0.0]

        p_dx, p_dy = wrapped_delta(position, self.planet.position)
        distance = math.hypot(p_dx, p_dy)
        if not (self.planet.diameter / 2 <= distance <= const.GRAVITY_RANGE):
            return [0.0, 0.0]

        gravity_force = const.GRAVITY_MULTIPLIER * self.planet.gravity
        return [
            gravity_force * p_dx / distance,
            gravity_force * p_dy / distance,
        ]

    @staticmethod
    def _limited_velocity(velocity):
        speed = math.hypot(*velocity)
        if speed <= const.SPEED_LIMIT:
            return list(velocity)
        scale = const.SPEED_LIMIT / speed
        return [velocity[0] * scale, velocity[1] * scale]

    def _advance_motion(
        self,
        position,
        gravity_velocity,
        target_position,
        impulse=(0.0, 0.0),
    ):
        """Advance one unhindered frame without mutating the live object."""
        acceleration = self._gravity_acceleration(position)
        next_gravity_velocity = [
            gravity_velocity[0] + acceleration[0],
            gravity_velocity[1] + acceleration[1],
        ]
        homing_velocity = self._homing_velocity(position, target_position)

        base_velocity = self._limited_velocity(
            [
                homing_velocity[0] + next_gravity_velocity[0],
                homing_velocity[1] + next_gravity_velocity[1],
            ]
        )
        # Preserve the limited gravity component, not the inertialess homing command.
        next_gravity_velocity = [
            base_velocity[0] - homing_velocity[0],
            base_velocity[1] - homing_velocity[1],
        ]
        velocity = self._limited_velocity(
            [base_velocity[0] + impulse[0], base_velocity[1] + impulse[1]]
        )
        next_position = [
            (position[0] + velocity[0] * const.SPEED_SCALE) % const.ARENA_SIZE,
            (position[1] + velocity[1] * const.SPEED_SCALE) % const.ARENA_SIZE,
        ]
        return next_position, velocity, next_gravity_velocity

    def update_physics(self):
        tracking_ship = self._tracking_ship()
        target_position = tracking_ship.position if tracking_ship else None
        self.position, self.velocity, self.gravity_velocity = self._advance_motion(
            self.position,
            self.gravity_velocity,
            target_position,
            self.accumulated_impulses,
        )
        self.accumulated_impulses = [0.0, 0.0]
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)

    def predict_unhindered_trajectory(self, frames=60):
        position = list(self.position)
        gravity_velocity = list(self.gravity_velocity)
        tracking_ship = self._tracking_ship()
        target_trajectory = (
            tracking_ship.predict_unhindered_trajectory(frames)
            if tracking_ship
            else []
        )
        expiration = self.expiration_timer if self.can_expire else float("inf")
        impulse = list(self.accumulated_impulses)
        trajectory = []

        for frame in range(frames):
            if frame >= expiration:
                break
            target_position = (
                target_trajectory[frame]
                if frame < len(target_trajectory)
                else (tracking_ship.position if tracking_ship else None)
            )
            position, _, gravity_velocity = self._advance_motion(
                position,
                gravity_velocity,
                target_position,
                impulse if frame == 0 else (0.0, 0.0),
            )
            trajectory.append(list(position))

        return trajectory

    def update(self):
        if self.current_hp <= 0:
            self.currently_alive = False
            return False
        return super().update()

    def get_sprite(self, interp_t=0.0):
        colors = getattr(self, "colors", ((0, 255, 0, 255),))
        color_index = getattr(self, "current_color_index", 0)
        if not hasattr(self, "_cached_sprites"):
            import pygame
            import pygame.gfxdraw

            radius = int(self.size[0] / 2)
            diameter = radius * 2
            sprites = []
            for color in colors:
                sprite = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
                pygame.draw.circle(sprite, color, (radius, radius), radius)
                pygame.gfxdraw.aacircle(sprite, radius, radius, radius, color)
                sprites.append(sprite)
            self._cached_sprites = tuple(sprites)
        return self._cached_sprites[color_index]

    def get_collision_mask(self):
        if not hasattr(self, "_cached_mask"):
            import pygame

            self._cached_mask = pygame.mask.from_surface(self.get_sprite())
        return self._cached_mask
