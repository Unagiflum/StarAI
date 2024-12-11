import math
import src.Const as Const

class Object:
    def __init__(self, player_num, max_hp, start_hp, inertia, sprite_location, sprite_scale, size):
        # Intrinsic characteristics
        self.player = player_num
        self.max_hp = max_hp
        self.start_hp = start_hp
        self.inertia = inertia
        self.sprite_location = sprite_location
        self.sprite_scale = sprite_scale
        self.size = size

        # Situational variables
        self.currently_alive = True
        self.current_hp = self.start_hp
        self.position = [0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.velocit0 = [0.0, 0.0]
        self.can_collide = True
        self.can_expire = False
        self.expiration_timer = 0

        # Physics attributes
        self.accumulated_impulses = [0.0, 0.0]

        # Behavior flags
        self.can_move = False
        self.can_die = False

    def add_impulse(self, dx, dy):
        """Add a physics impulse to the object."""
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

    def update(self):
        """Main update function for game loop."""
        self.update_physics()
        if self.can_expire:
            return self.expiration_timer > 0
        return True
