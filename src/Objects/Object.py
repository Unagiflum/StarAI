import math
import src.Const as Const

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
        self.can_move = True

    def add_impulse(self, dx, dy):
        if self.can_move:
            self.accumulated_impulses[0] += dx
            self.accumulated_impulses[1] += dy

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