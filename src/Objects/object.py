import pygame
import math
import src.const as Const
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    LaserTargetCapabilities,
)
from src.toroidal import wrapped_delta

class Object:
    def __init__(self, name, sprite_location, size, sprite_scale=1.0):
        self.name = name
        self.sprite_location = sprite_location
        self.size = size
        self.sprite_scale = sprite_scale

        # Position and state
        self.position = [0.0, 0.0]
        self.previous_position = self.position.copy()
        self.currently_alive = True

        # Collision and expiration
        self.can_collide = True
        self.collision_capabilities = CollisionCapabilities()
        self.laser_target_capabilities = LaserTargetCapabilities()
        self.area_damage_capabilities = AreaDamageCapabilities()
        self.can_expire = False
        self.expiration_timer = 0

        # Behavior flags
        self.can_move = False
        self.can_die = False
        self.inertia = False

        # Sprite handling
        self.sprites = []
        self.image = None

    def drain_spawned_objects(self):
        """Return objects created during this object's most recent update."""
        return []

    def get_collision_mask(self):
        """Return the current pixel collision mask, if this object has one."""
        return None

    def is_alive(self):
        """Return whether this object participates as a living object."""
        return self.currently_alive

class ThrustMarker(Object):
    def __init__(self, x, y):
        super().__init__(
            name="ThrustMarker",
            sprite_location=None,
            size=[6, 6]
        )
        self.position = [x, y]
        self.life = Const.FPS/2
        self.can_collide = False
        self.can_expire = True
        self.expiration_timer = self.life

    def update(self):
        self.expiration_timer -= 1
        return self.expiration_timer > 0

    def get_color(self):
        fade_ratio = self.expiration_timer / 30
        start_color = (255, 255, 0)
        end_color = (100, 0, 0)
        r = int(start_color[0] * fade_ratio + end_color[0] * (1 - fade_ratio))
        g = int(start_color[1] * fade_ratio + end_color[1] * (1 - fade_ratio))
        b = int(start_color[2] * fade_ratio + end_color[2] * (1 - fade_ratio))
        return (r, g, b)

    def draw(self, screen, scale_factor, translation):
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (0 <= pos_x <= Const.SCREEN_HEIGHT and
                        0 <= pos_y <= Const.SCREEN_HEIGHT):
                    pygame.draw.circle(screen, self.get_color(), (Const.SCREEN_LEFT + pos_x, pos_y), 1.0 + 3.0*scale_factor)

class PlayerObject(Object):
    def __init__(self, name, sprite_location, size, player, sprite_scale):
        super().__init__(name, sprite_location, size, sprite_scale)

        # Player-specific attributes
        self.player = player

        # Movement and physics state
        self.velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self.heading = 0
        self.rotation = 0.0
        self.can_move = True


        # Battle state
        self.in_battle = False
        self.planet = None
        self.opponent = None
        self.friendly_objects = []
        self.enemy_objects = []
        self.asteroids = []

    def set_planet(self, planet):
        self.planet = planet



    def distance_to(self, obj):
        dx, dy = wrapped_delta(self.position, obj.position)
        distance = math.sqrt(dx * dx + dy * dy)
        return [dx, dy], distance

    def apply_verlet(self):
        gravity_impulse = self.get_gravity()
        acc0 = [gravity_impulse[0], gravity_impulse[1]]
        self.position[0] = (self.position[0] + (self.velocity[0]
                                                + 0.5 * acc0[0]) * Const.SPEED_SCALE) % Const.ARENA_SIZE
        self.position[1] = (self.position[1] + (self.velocity[1]
                                                + 0.5 * acc0[1]) * Const.SPEED_SCALE) % Const.ARENA_SIZE
        gravity_impulse = self.get_gravity()
        acc1 = [gravity_impulse[0], gravity_impulse[1]]
        self.velocity[0] += (acc0[0] + acc1[0]) * 0.5
        self.velocity[1] += (acc0[1] + acc1[1]) * 0.5

    def get_gravity(self):
        if not self.planet:
            return [0.0, 0.0]

        [dx, dy], distance = self.distance_to(self.planet)
        if distance < self.planet.diameter / 2 or distance > Const.GRAVITY_RANGE:
            return [0.0, 0.0]

        gravity_force = Const.GRAVITY_MULTIPLIER * self.planet.gravity
        return [
            gravity_force * dx / distance,
            gravity_force * dy / distance
        ]

    def apply_speed_limit(self):
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > Const.SPEED_LIMIT:
            scale = Const.SPEED_LIMIT / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale

    def add_impulse(self, dx, dy):
        if self.can_move:
            self.accumulated_impulses[0] += dx
            self.accumulated_impulses[1] += dy

    def get_thrust_marker_position(self, thrust_angle=0):
        angle_rad = math.radians(self.rotation + thrust_angle)
        offset = (self.size[1] / 2) + 6
        marker_x = self.position[0] - math.sin(angle_rad) * offset
        marker_y = self.position[1] + math.cos(angle_rad) * offset
        return marker_x, marker_y

    def apply_thrust(self, max_thrust, thrust_increment, angle, make_marker):
        angle_rad = math.radians(self.rotation + angle)
        thrust_direction = [math.sin(angle_rad), -math.cos(angle_rad)]

        if self.inertia:
            new_velocity = [
                self.velocity[0] + thrust_direction[0] * thrust_increment,
                self.velocity[1] + thrust_direction[1] * thrust_increment
            ]

            speed = math.sqrt(new_velocity[0] ** 2 + new_velocity[1] ** 2)
            scale = 1.0

            if self.planet:
                _, planet_distance = self.distance_to(self.planet)
            else:
                planet_distance = float('inf')

            if speed > max_thrust and planet_distance > Const.GRAVITY_RANGE:
                scale = max_thrust / speed
            if speed > Const.MAX_GRAV_WHIP:
                scale = Const.MAX_GRAV_WHIP / speed

            target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

            diff_vector = [target_velocity[0] - self.velocity[0], target_velocity[1] - self.velocity[1]]

            diff_magnitude = math.sqrt(diff_vector[0] ** 2 + diff_vector[1] ** 2)
            if diff_magnitude > thrust_increment:
                scale_diff = thrust_increment / diff_magnitude
                self.add_impulse(diff_vector[0] * scale_diff, diff_vector[1] * scale_diff)
            else:
                self.add_impulse(diff_vector[0], diff_vector[1])
        else:
            self.add_impulse(
                thrust_direction[0] * max_thrust,
                thrust_direction[1] * max_thrust
            )

        if make_marker:
            marker_x, marker_y = self.get_thrust_marker_position(angle)
            marker = ThrustMarker(marker_x, marker_y)
            return marker
        return None
