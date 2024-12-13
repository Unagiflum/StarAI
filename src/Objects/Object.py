import pygame
import math
import src.Const as Const
from src.UI import UI

class Object:
    def __init__(self, name, sprite_location, size, sprite_scale=1.0):
        self.name = name
        self.sprite_location = sprite_location
        self.size = size
        self.sprite_scale = sprite_scale

        # Position and state
        self.position = [0.0, 0.0]
        self.currently_alive = True

        # Collision and expiration
        self.can_collide = True
        self.can_expire = False
        self.expiration_timer = 0

        # Behavior flags
        self.can_move = False
        self.can_die = False

        # Sprite handling
        self.sprites = []
        self.image = None

class ThrustMarker(Object):
    def __init__(self, x, y):
        super().__init__(
            name="ThrustMarker",
            sprite_location=None,
            size=[6, 6]
        )
        self.position = [x, y]
        self.life = 20
        self.can_collide = False
        self.can_expire = True
        self.expiration_timer = self.life

    def update(self):
        self.expiration_timer -= 1
        return self.expiration_timer > 0

    def get_color(self):
        fade_ratio = self.expiration_timer / 30
        start_color = (255, 255, 0)
        end_color = (150, 0, 0)
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

                if (0 <= pos_x <= UI.SCREEN_HEIGHT and
                        0 <= pos_y <= UI.SCREEN_HEIGHT):
                    pygame.draw.circle(screen, self.get_color(), (pos_x, pos_y), 1 + 2*scale_factor)

class MovableObject(Object):
    def __init__(self, name, sprite_location, size, player, max_hp, start_hp, inertia, sprite_scale,
                 max_thrust, thrust_increment, thrust_wait, turn_wait, mass):
        super().__init__(name, sprite_location, size, sprite_scale)

        # Player attributes
        self.player = player
        self.max_hp = max_hp
        self.start_hp = start_hp
        self.current_hp = start_hp

        # Movement properties
        self.inertia = inertia
        self.mass = mass
        self.max_thrust = max_thrust
        self.thrust_increment = thrust_increment
        self.thrust_wait = thrust_wait
        self.turn_wait = turn_wait
        self.can_move = True
        self.leaves_trail = True

        # Physics state
        self.velocity = [0.0, 0.0]
        self.velocit0 = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self.heading = 0
        self.rotation = 0.0

        # Timers
        self.thrust_timer = 0
        self.turn_timer = 0

        # Battle state
        self.in_battle = False


    def add_impulse(self, dx, dy):
        if self.can_move:
            self.accumulated_impulses[0] += dx
            self.accumulated_impulses[1] += dy

    def update(self):
        self.update_physics()
        return True

    def update_physics(self):
        if self.can_move:
            self.velocit0[0] = self.velocity[0]
            self.velocit0[1] = self.velocity[1]

            if self.inertia:
                self.velocity[0] += self.accumulated_impulses[0]
                self.velocity[1] += self.accumulated_impulses[1]
            else:
                self.velocity = self.accumulated_impulses.copy()

            speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
            if speed > Const.SPEED_LIMIT:
                scale = Const.SPEED_LIMIT / speed
                self.velocity[0] *= scale
                self.velocity[1] *= scale

            self.accumulated_impulses = [0.0, 0.0]
            self.position[0] = (self.position[0] + Const.SPEED_SCALE * 0.5 * (
                    self.velocit0[0] + self.velocity[0])) % Const.ARENA_SIZE
            self.position[1] = (self.position[1] + Const.SPEED_SCALE * 0.5 * (
                    self.velocit0[1] + self.velocity[1])) % Const.ARENA_SIZE

    def apply_gravity(self, source_position, gravity_strength, min_distance=0):
        if not self.can_move or not self.inertia:
            return

        dx = source_position[0] - self.position[0]
        dy = source_position[1] - self.position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > min_distance:
            gravity_force = Const.GRAVITY_MULTIPLIER * gravity_strength / (distance * distance)
            self.add_impulse(
                gravity_force * dx / distance,
                gravity_force * dy / distance
            )

    def can_thrust(self):
        return self.thrust_timer == 0

    def can_turn(self):
        return self.turn_timer == 0

    def update_timers(self, forward_pressed: bool):
        if self.thrust_timer > 0:
            self.thrust_timer -= 1
        if self.turn_timer > 0:
            self.turn_timer -= 1
        if not self.inertia and self.thrust_timer == 0 and not forward_pressed:
            self.velocity = [0.0, 0.0]

    def apply_thrust(self):
        if self.can_thrust():
            angle_rad = math.radians(self.rotation)
            thrust_direction = [math.sin(angle_rad), -math.cos(angle_rad)]

            if self.inertia:
                new_velocity = [
                    self.velocity[0] + thrust_direction[0] * self.thrust_increment,
                    self.velocity[1] + thrust_direction[1] * self.thrust_increment
                ]

                speed = math.sqrt(new_velocity[0] ** 2 + new_velocity[1] ** 2)
                scale = 1.0
                if speed > self.max_thrust:
                    scale = self.max_thrust / speed
                target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

                diff_vector = [
                    target_velocity[0] - self.velocity[0],
                    target_velocity[1] - self.velocity[1]
                ]

                diff_magnitude = math.sqrt(diff_vector[0] ** 2 + diff_vector[1] ** 2)
                if diff_magnitude > 0:
                    scale = self.thrust_increment / diff_magnitude
                    self.add_impulse(diff_vector[0] * scale, diff_vector[1] * scale)
            else:
                self.add_impulse(
                    thrust_direction[0] * self.max_thrust,
                    thrust_direction[1] * self.max_thrust
                )

            self.thrust_timer = int(self.thrust_wait * Const.THRUST_WAIT_SCALE)
            if self.leaves_trail:
                marker_x, marker_y = self.get_thrust_marker_position()
                marker = ThrustMarker(marker_x, marker_y)
                return marker

        return None

    def get_thrust_marker_position(self):
        angle_rad = math.radians(self.rotation)
        offset = (self.size[1] / 2) + 6
        marker_x = self.position[0] - math.sin(angle_rad) * offset
        marker_y = self.position[1] + math.cos(angle_rad) * offset
        return marker_x, marker_y

    def turn_left(self):
        if self.can_turn():
            self.heading = (self.heading - 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.turn_wait * Const.TURN_WAIT_SCALE)

    def turn_right(self):
        if self.can_turn():
            self.heading = (self.heading + 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.turn_wait * Const.TURN_WAIT_SCALE)
