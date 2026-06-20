import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability, ABILITIES_DATA
from src.Objects.Ships.KzerZa.A2.KzerZaA2Laser import KzerZaA2Laser
from src.toroidal import wrapped_delta


class KzerZaA2(Ability):
    LAUNCHING = "launching"
    ATTACKING = "attacking"
    RETURNING = "returning"
    _fighter_sounds = {}

    def __init__(self, parent, launch_angle=0, formation_index=0):
        super().__init__("KzerZaA2", parent)
        data = ABILITIES_DATA["KzerZaA2"]
        self._load_fighter_sounds(data["file_path"])
        self.launch_sound = self._sound("launch")
        self.laser_sound = self._sound("laser")
        self.return_sound = self._sound("return")

        self.one_way_flight = data["one_way_flight"]
        self.launch_distance = data["launch_distance"]
        self.mass = data.get("mass", 0)
        self.weapon_wait = data["weapon_wait"]
        self.laser_offset = data["offset"]
        self.laser_range = data["range"]
        self.track_directions = data["track_directions"]

        self.laser_vulnerable = data.get("laser_vulnerable", True)
        self.collide_fighters = data.get("collide_fighters", True)
        self.collide_projectiles = data.get("collide_projectiles", True)
        self.damage_projectiles = data.get("damage_projectiles", True)
        self.collide_asteroids = data.get("collide_asteroids", True)
        self.damage_asteroids = data.get("damage_asteroids", True)
        self.collide_enemy_ships = data.get("collide_enemy_ships", True)
        self.collide_friendly_ships = data.get("collide_friendly_ships", False)
        self.collide_planets = data.get("collide_planets", True)

        self.position = parent.position.copy()
        self.previous_position = self.position.copy()
        self.rotation = (parent.rotation + launch_angle) % 360
        self._set_velocity_for_angle(self.rotation, self.speed)
        self.mode = self.LAUNCHING
        self.launch_travelled = 0.0
        self.attack_elapsed = 0
        self.weapon_timer = 0
        self.formation_index = formation_index
        self.spawned_objects = []
        self.planet_avoidance = None

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.expiration_timer -= 1
        if self.expiration_timer <= 0 or self.current_hp <= 0:
            self.currently_alive = False
            return False

        if self.mode == self.LAUNCHING:
            self._update_launch()
            return self.currently_alive

        self.attack_elapsed += 1
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

    def _update_launch(self):
        max_step = self.speed * const.SPEED_SCALE
        step = min(max_step, self.launch_distance - self.launch_travelled)
        angle = math.radians(self.rotation)
        self.position[0] = (self.position[0] + math.sin(angle) * step) % const.ARENA_SIZE
        self.position[1] = (self.position[1] - math.cos(angle) * step) % const.ARENA_SIZE
        self.launch_travelled += step

        if self.launch_travelled >= self.launch_distance:
            target = self._live_trackable_opponent()
            self.mode = self.ATTACKING if target else self.RETURNING

    def _destination(self, target):
        if self.mode == self.RETURNING:
            return self.parent.position.copy() if self._parent_alive() else None
        if target is None:
            return None

        flank_destinations = [
            self._attack_position(target, side)
            for side in (90, 270)
        ]
        destination = min(
            flank_destinations,
            key=lambda position: sum(
                component * component
                for component in wrapped_delta(self.position, position)
            ),
        )

        # Fighters may share a flank. Fan them out around its center instead of
        # assigning every other fighter to the far side of the target.
        spread_slot = self.formation_index % 13
        spread_index = (spread_slot + 1) // 2
        spread = spread_index * (5 if spread_slot % 2 else -5)
        if spread == 0:
            return destination

        side = 90 if destination is flank_destinations[0] else 270
        return self._attack_position(target, side + spread)

    def _attack_position(self, target, angle_offset):
        angle = math.radians((target.rotation + angle_offset) % 360)
        return [
            (target.position[0] + math.sin(angle) * self.laser_range) % const.ARENA_SIZE,
            (target.position[1] - math.cos(angle) * self.laser_range) % const.ARENA_SIZE,
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
        self.position[0] = (self.position[0] + self.velocity[0] * const.SPEED_SCALE) % const.ARENA_SIZE
        self.position[1] = (self.position[1] + self.velocity[1] * const.SPEED_SCALE) % const.ARENA_SIZE

    def _is_at_position(self, destination):
        dx, dy = wrapped_delta(self.position, destination)
        return math.hypot(dx, dy) <= 0.5

    def _move_around_planet(self, destination):
        planet, tangent = self.planet_avoidance
        if not _segment_intersects_body(self.position, destination, planet, self.size):
            self.planet_avoidance = None
            self._move_toward(destination)
            return

        dx, dy = wrapped_delta(self.position, destination)
        if tangent[0] * dx + tangent[1] * dy < 0:
            tangent = [-tangent[0], -tangent[1]]
            self.planet_avoidance = (planet, tangent)
        self.velocity = [tangent[0] * self.speed, tangent[1] * self.speed]
        self._update_rotation_from_vector(*self.velocity)
        self.position[0] = (self.position[0] + self.velocity[0] * const.SPEED_SCALE) % const.ARENA_SIZE
        self.position[1] = (self.position[1] + self.velocity[1] * const.SPEED_SCALE) % const.ARENA_SIZE

    def begin_planet_avoidance(self, planet, outward_normal):
        tangent = [-outward_normal[1], outward_normal[0]]
        self.planet_avoidance = (planet, tangent)

    def _update_weapon(self, target):
        if self.weapon_timer <= 0:
            self.spawned_objects.append(KzerZaA2Laser(self, target))
            if self.laser_sound:
                self.laser_sound.play()
            self.weapon_timer = int(self.weapon_wait * const.ACTION_WAIT_SCALE)
        else:
            self.weapon_timer -= 1

    def drain_spawned_objects(self):
        result = self.spawned_objects
        self.spawned_objects = []
        return result

    def recover(self):
        if not self.currently_alive or not self._parent_alive():
            return
        self.parent.current_hp = min(self.parent.max_hp, self.parent.current_hp + 1)
        if self.return_sound:
            self.return_sound.play()
        self.current_hp = 0
        self.currently_alive = False

    @classmethod
    def _load_fighter_sounds(cls, file_path):
        if not cls.sound_enabled:
            return
        sound_files = {
            "launch": "KzerZaA2Launch.wav",
            "laser": "KzerZaA2Laser.wav",
            "return": "KzerZaA2Return.wav",
        }
        for sound_name, filename in sound_files.items():
            if sound_name in cls._fighter_sounds:
                continue
            try:
                sound = pygame.mixer.Sound(str(const.source_path(file_path) / filename))
                sound.set_volume(const.SOUND_EFFECT_VOLUME)
                cls._fighter_sounds[sound_name] = sound
            except (pygame.error, FileNotFoundError):
                cls._fighter_sounds[sound_name] = None

    def _sound(self, sound_name):
        if not self.sound_enabled:
            return None
        return self._fighter_sounds.get(sound_name)

    def _live_trackable_opponent(self):
        if (
            self.opponent is not None and
            self.opponent.currently_alive and
            self.opponent.current_hp > 0 and
            self.opponent.trackable
        ):
            return self.opponent
        return None

    def _parent_alive(self):
        return self.parent.currently_alive and self.parent.current_hp > 0

    def _set_velocity_for_angle(self, angle_degrees, speed):
        angle = math.radians(angle_degrees)
        self.velocity = [math.sin(angle) * speed, -math.cos(angle) * speed]
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
