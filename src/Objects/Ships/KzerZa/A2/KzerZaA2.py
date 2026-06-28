import math
import random

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability, ABILITIES_DATA
from src.Objects.Ships.KzerZa.A2.KzerZaA2Laser import KzerZaA2Laser
from src.toroidal import wrapped_delta


class KzerZaA2(Ability):
    LAUNCHING = "launching"
    ATTACKING = "attacking"
    RETURNING = "returning"

    def __init__(self, parent, launch_angle=0, formation_index=0):
        super().__init__("KzerZaA2", parent)
        data = ABILITIES_DATA["KzerZaA2"]
        fighter_sounds = self._load_fighter_sounds(data["file_path"])
        self.launch_sound = fighter_sounds["launch"]
        self.laser_sound = fighter_sounds["laser"]
        self.return_sound = fighter_sounds["return"]

        self.one_way_flight = data["one_way_flight"]
        self.launch_time = data.get("launch_time", 0)
        self.mass = data.get("mass", 0)
        self.weapon_wait = data["weapon_wait"]
        self.laser_offset = data["offset"]
        self.laser_range = data["range"]
        self.track_directions = data["track_directions"]

        self.position = parent.position.copy()
        self.previous_position = self.position.copy()
        self.rotation = (parent.rotation + launch_angle) % 360
        self._set_velocity_for_angle(self.rotation, self.speed)
        self.mode = self.LAUNCHING
        self.launch_timer = self.launch_time
        self.attack_elapsed = 0
        self.weapon_timer = 0
        self.formation_index = formation_index
        self.spawned_objects = []
        self.planet_avoidance = None
        self.jitter_angle_toggle = random.choice([True, False])
        self.jitter_dist_toggle = random.choice([True, False])

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.expiration_timer -= 1
        if self.expiration_timer <= 0 or self.current_hp <= 0:
            self.currently_alive = False
            return False

        if self.mode == self.LAUNCHING:
            if self.planet_avoidance is not None:
                angle = math.radians(self.rotation)
                dest = [
                    self.position[0] + math.sin(angle) * 1000,
                    self.position[1] - math.cos(angle) * 1000,
                ]
                self._move_around_planet(dest)
            else:
                self.update_physics()

            self.launch_timer -= 1
            if self.launch_timer <= 0:
                target = self._live_trackable_opponent()
                self.mode = self.ATTACKING if target else self.RETURNING
            return self.currently_alive

        self.attack_elapsed += 1
        if random.random() < 0.20:
            if random.choice([True, False]):
                self.jitter_angle_toggle = not self.jitter_angle_toggle
            else:
                self.jitter_dist_toggle = not self.jitter_dist_toggle

        target = self._live_trackable_opponent()
        parent_alive = self._parent_alive()

        if not parent_alive:
            if target:
                self.mode = self.ATTACKING
            else:
                self.velocity = [0.0, 0.0]
                return True
        elif self.attack_elapsed >= 2 * self.one_way_flight:
            self.mode = self.RETURNING
        elif target:
            self.mode = self.ATTACKING
        else:
            self.mode = self.RETURNING

        destination = self._destination(target)
        if destination is None:
            self.velocity = [0.0, 0.0]
            return True

        if self.planet_avoidance is not None:
            self._move_around_planet(destination)
        else:
            self._move_toward(destination)

        if self.mode == self.ATTACKING and target and self._is_at_position(destination):
            self._update_weapon(target)
        return True

    def _destination(self, target):
        if self.mode == self.RETURNING:
            return self.parent.position.copy() if self._parent_alive() else None
        if target is None:
            return None

        flank_destinations = [self._attack_position(target, side) for side in (90, 270)]
        destination = min(
            flank_destinations,
            key=lambda position: sum(
                component * component
                for component in wrapped_delta(self.position, position)
            ),
        )

        # SpecialObjects may share a flank. Fan them out around its center instead of
        # assigning every other special_object to the far side of the target.
        spread_slot = self.formation_index % 13
        spread_index = (spread_slot + 1) // 2
        spread = spread_index * (5 if spread_slot % 2 else -5)
        if spread == 0:
            return destination

        side = 90 if destination is flank_destinations[0] else 270
        return self._attack_position(target, side + spread)

    def _attack_position(self, target, angle_offset):
        jitter_angle = 5 if self.jitter_angle_toggle else 0
        angle = math.radians((target.rotation + angle_offset + jitter_angle) % 360)
        dist = self.laser_range - (5 if self.jitter_dist_toggle else 0)
        return [
            (target.position[0] + math.sin(angle) * dist)
            % const.ARENA_SIZE,
            (target.position[1] - math.cos(angle) * dist)
            % const.ARENA_SIZE,
        ]

    def _move_toward(self, destination):
        dx, dy = wrapped_delta(self.position, destination)
        distance = math.hypot(dx, dy)
        if distance <= 0:
            self.velocity = [0.0, 0.0]
            return

        velocity_speed = min(self.speed, distance / const.SPEED_SCALE)
        self.velocity = [dx / distance * velocity_speed, dy / distance * velocity_speed]
        self._update_rotation_from_vector(dx, dy)
        self.position[0] = (
            self.position[0] + self.velocity[0] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + self.velocity[1] * const.SPEED_SCALE
        ) % const.ARENA_SIZE

    def _is_at_position(self, destination):
        dx, dy = wrapped_delta(self.position, destination)
        return math.hypot(dx, dy) <= 0.5

    def _move_around_planet(self, destination):
        planet, direction = self.planet_avoidance
        if not _segment_intersects_body(self.position, destination, planet, self.size):
            self.planet_avoidance = None
            self._move_toward(destination)
            return

        cx, cy = wrapped_delta(planet.position, self.position)
        dist = math.hypot(cx, cy)
        if dist > 0:
            normal = [cx / dist, cy / dist]
            base_tangent = [-normal[1], normal[0]]
        else:
            base_tangent = [1.0, 0.0]

        if direction is None:
            dx, dy = wrapped_delta(self.position, destination)
            if base_tangent[0] * dx + base_tangent[1] * dy < 0:
                direction = -1
            else:
                direction = 1
            self.planet_avoidance = (planet, direction)

        tangent = [base_tangent[0] * direction, base_tangent[1] * direction]

        self.velocity = [tangent[0] * self.speed, tangent[1] * self.speed]
        self._update_rotation_from_vector(*self.velocity)
        self.position[0] = (
            self.position[0] + self.velocity[0] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + self.velocity[1] * const.SPEED_SCALE
        ) % const.ARENA_SIZE

    def handle_planet_contact(self, planet, outward_normal, overlap):
        from src.Battle.collision_responses import separate_from_static_body
        separate_from_static_body(self, planet, outward_normal, overlap, extra_clearance=1.0)
        self.planet_avoidance = (planet, None)
        return True

    def handle_projectile_contact(self, projectile):
        self.set_hp(0)
        return True

    def _update_weapon(self, target):
        if self.weapon_timer <= 0:
            self.spawned_objects.append(KzerZaA2Laser(self, target))
            if self.laser_sound:
                self.laser_sound.play()
            self.weapon_timer = const.cooldown_frames(self.weapon_wait)
        else:
            self.weapon_timer -= 1

    def drain_spawned_objects(self):
        result = self.spawned_objects
        self.spawned_objects = []
        return result

    def can_recover_with_parent(self):
        return self.mode == self.RETURNING

    def on_opponent_lost(self, opponent):
        super().on_opponent_lost(opponent)
        if self._parent_alive() and self.mode != self.LAUNCHING:
            self.mode = self.RETURNING

    def recover_with_parent(self):
        if not self.currently_alive or not self._parent_alive():
            return
        self.parent.current_hp = min(self.parent.max_hp, self.parent.current_hp + 1)
        if self.return_sound:
            self.return_sound.play()
        self.current_hp = 0
        self.currently_alive = False

    def _load_fighter_sounds(self, file_path):
        sound_files = {
            "launch": "KzerZaA2Launch.wav",
            "laser": "KzerZaA2Laser.wav",
            "return": "KzerZaA2Return.wav",
        }
        return {
            sound_name: self.audio_service.load_effect(
                const.source_path(file_path) / filename,
                const.SOUND_EFFECT_VOLUME,
            )
            for sound_name, filename in sound_files.items()
        }

    def _parent_alive(self):
        return self.parent.currently_alive and self.parent.current_hp > 0

    def _set_velocity_for_angle(self, angle_degrees, speed):
        angle = math.radians(angle_degrees)
        self.velocity = [
            math.sin(angle) * speed + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle) * speed + self.parent.velocity[1] * self.parent_vel,
        ]
        self._update_rotation_from_vector(*self.velocity)

    def _update_rotation_from_vector(self, dx, dy):
        if dx == 0 and dy == 0:
            return
        self.rotation = math.degrees(math.atan2(dx, -dy)) % 360
        self.heading = round(self.rotation / const.TURN_ANGLE) % const.SHIP_DIRECTIONS


def _segment_intersects_body(start, end, body, fighter_size):
    dx, dy = wrapped_delta(start, end)
    cx, cy = wrapped_delta(start, body.position)
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return False
    ratio = max(0.0, min(1.0, (cx * dx + cy * dy) / length_squared))
    nearest_x = dx * ratio
    nearest_y = dy * ratio
    radius = body.diameter / 2 + max(fighter_size) / 2
    return math.hypot(cx - nearest_x, cy - nearest_y) < radius
