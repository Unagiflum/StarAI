import math

import src.const as const
from src.Battle.collision_geometry import collision_info
from src.Battle.collision_physics import bounce_off_static_body, elastic_bounce
from src.collision_capabilities import (
    PhysicalCollisionCapabilities,
    SpecialObjectCollisionCapabilities,
)
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.toroidal import wrapped_delta


class ChenjesuA2(Ability):
    AREA_IMMUNITIES = {"SyreenA2", "SlylandroA2"}

    def __init__(self, parent):
        super().__init__("ChenjesuA2", parent)
        definition = ABILITY_DEFINITIONS[self.name]
        self.max_count = definition.max_count
        self.jitter = definition.jitter
        self.drain = definition.drain
        self.avoid_angle = definition.avoid_angle
        self.mass = definition.mass
        self.expiration_timer = float("inf")
        self.physical_collision_capabilities = PhysicalCollisionCapabilities(
            is_solid=True,
            is_projectile=True,
        )
        self.special_object_collision_capabilities = (
            SpecialObjectCollisionCapabilities(
                collides_with_planets=True,
                collides_with_asteroids=True,
                damages_asteroids=False,
                collides_with_projectiles=True,
                damages_projectiles=True,
                collides_with_enemy_ships=True,
                collides_with_friendly_ships=True,
                collides_with_fighters=True,
                bounces_off_same_type=True,
                bounces_off_ships_without_damage=True,
            )
        )
        directory = const.source_path(definition.file_path)
        self.hit_sound = self.audio_service.load_effect(
            directory / "ChenjesuA2hit.wav",
            const.SOUND_EFFECT_VOLUME,
        )
        self.death_sound = self.audio_service.load_effect(
            directory / "ChenjesuA2end.wav",
            const.SOUND_EFFECT_VOLUME,
        )
        self._death_sound_played = False
        self.launch_from_gun(inherit_parent_velocity=False)

    def update(self):
        if not self.currently_alive or self.current_hp <= 0:
            return False
        if not self._parent_alive():
            self.current_hp = 0
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        target = self._live_trackable_opponent()
        base_velocity = self._target_velocity(target)
        jitter_angle = self.rng.uniform(0.0, 360.0)
        jitter_x, jitter_y = self._direction_vector(jitter_angle, self.jitter)
        self.velocity = [
            base_velocity[0] + jitter_x,
            base_velocity[1] + jitter_y,
        ]
        self.apply_speed_limit()
        self.position[0] = (
            self.position[0] + self.velocity[0] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + self.velocity[1] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        if self.velocity != [0.0, 0.0]:
            self.rotation = math.degrees(
                math.atan2(self.velocity[0], -self.velocity[1])
            ) % 360
        return True

    def _target_velocity(self, target):
        if target is None:
            return [0.0, 0.0]

        target_to_self = wrapped_delta(target.position, self.position)
        radial_angle = self._vector_angle(*target_to_self)
        front_offset = self._signed_angle(radial_angle - target.rotation)
        if abs(front_offset) <= self.avoid_angle:
            candidates = (
                (radial_angle - 90.0) % 360,
                (radial_angle + 90.0) % 360,
            )
            move_angle = max(
                candidates,
                key=lambda angle: abs(self._signed_angle(angle - target.rotation)),
            )
            return list(self._direction_vector(move_angle, self.speed))

        dx, dy = wrapped_delta(self.position, target.position)
        distance = math.hypot(dx, dy)
        if distance == 0:
            return [0.0, 0.0]
        return [dx / distance * self.speed, dy / distance * self.speed]

    def should_collide_with_ship(self, ship):
        return True

    def should_collide_with_projectile_like(self, other):
        if getattr(other, "type", None) != "special_object":
            return True
        return getattr(other, "name", None) in {"ChenjesuA2", "OrzA3"}

    def should_take_area_damage_from(self, source):
        return getattr(source, "name", None) not in self.AREA_IMMUNITIES

    def handle_ship_contact(self, ship, normal=None):
        self._bounce_with(ship, normal)
        if ship.player != self.player:
            ship.current_energy = max(0, ship.current_energy - self.drain)
            if self.hit_sound:
                self.hit_sound.play()
        return True

    def handle_asteroid_contact(self, asteroid, normal=None):
        self._bounce_with(asteroid, normal)
        return True

    def handle_planet_contact(self, planet, outward_normal, overlap):
        bounce_off_static_body(self, planet, outward_normal, overlap)
        self.set_hp(self.current_hp - 1)
        # Returning false on a lethal impact lets the common collision path
        # create the configured animation and invoke on_destroyed.
        return self.current_hp > 0

    def handle_projectile_contact(self, projectile):
        if projectile.name == "ChenjesuA2":
            return False
        if projectile.name == "OrzA3":
            self._bounce_with(projectile)
            return True

        self.set_hp(self.current_hp - projectile.current_damage)
        projectile.set_hp(0)
        return True

    def on_destroyed(self):
        if not self._death_sound_played and self.death_sound:
            self.death_sound.play()
        self._death_sound_played = True

    def _bounce_with(self, other, normal=None):
        calculated_normal, distance, overlap = collision_info(self, other)
        elastic_bounce(
            self,
            other,
            calculated_normal if normal is None else normal,
            distance,
            overlap,
        )

    def _parent_alive(self):
        return self.parent.currently_alive and self.parent.current_hp > 0

    @staticmethod
    def _direction_vector(angle, magnitude):
        radians = math.radians(angle)
        return math.sin(radians) * magnitude, -math.cos(radians) * magnitude

    @staticmethod
    def _vector_angle(dx, dy):
        return math.degrees(math.atan2(dx, -dy)) % 360

    @staticmethod
    def _signed_angle(angle):
        return (angle + 180.0) % 360.0 - 180.0
