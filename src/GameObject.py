import json
import math

from tensorflow.python.ops.metrics_impl import false_positives

import src.GameConstants as GameConstants

class GameObject:
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
            if speed > GameConstants.SPEED_LIMIT:
                scale = GameConstants.SPEED_LIMIT / speed
                self.velocity[0] *= scale
                self.velocity[1] *= scale

            self.accumulated_impulses = [0.0, 0.0]
            self.position[0] = (self.position[0] + 0.5 * (
                    self.velocit0[0] + self.velocity[0])) % GameConstants.ARENA_SIZE
            self.position[1] = (self.position[1] + 0.5 * (
                    self.velocit0[1] + self.velocity[1])) % GameConstants.ARENA_SIZE

    def apply_gravity(self, source_position, gravity_strength, min_distance=0):

        if not self.can_move or not self.inertia:
            return

        dx = source_position[0] - self.position[0]
        dy = source_position[1] - self.position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > min_distance:
            gravity_force = GameConstants.GRAVITY_MULTIPLIER * gravity_strength / (distance * distance)
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


class ThrustMarker(GameObject):
    def __init__(self, x, y):
        super().__init__(
            player_num=0,
            max_hp=1,
            start_hp=1,
            inertia=False,
            sprite_location=None,
            sprite_scale=1.0,
            size=[6, 6]
        )
        self.position = [x, y]
        self.life = 30
        self.can_collide = False
        self.can_expire = True
        self.expiration_timer = self.life

    def update(self):
        super().update()
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


class SpaceShip(GameObject):
    def __init__(self, ship_name, player_num):
        self.name = ship_name

        with open('Ships/Ships.json', 'r') as f:
            ships_data = json.load(f)
            ship_data = ships_data[ship_name]

        super().__init__(
            player_num=player_num,
            max_hp=ship_data['MaxHP'],
            start_hp=ship_data['StartHP'],
            inertia=ship_data['Inertia'],
            sprite_location=ship_data['SpriteLocation'],
            sprite_scale=ship_data['SpriteScale'],
            size=[ship_data['Size']['width'], ship_data['Size']['height']]
        )

        # Ship-specific attributes
        self.ship_type = ship_data['ShipType']
        self.cost = ship_data['Cost']
        self.max_energy = ship_data['MaxEnergy']
        self.start_energy = ship_data['StartEnergy']
        self.energy_regen = ship_data['EnergyRegen']
        self.energy_wait = ship_data['EnergyWait']
        self.max_thrust = ship_data['MaxThrust']
        self.thrust_increment = ship_data['ThrustIncrement']
        self.thrust_wait = ship_data['ThrustWait']
        self.turn_wait = ship_data['TurnWait']
        self.ship_mass = ship_data['ShipMass']

        # Ship state variables
        self.current_energy = self.start_energy
        self.heading = 0
        self.rotation = 0.0
        self.energy_timer = 0
        self.thrust_timer = 0
        self.turn_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0
        self.status1 = False
        self.status2 = False
        self.status3 = False
        self.status4 = 0
        self.status5 = 0
        self.status6 = 0
        self.in_battle = False

        self.can_move = True
        self.can_die = True
        try:
            self.ship_module = __import__(f"Ships.{ship_name}.{ship_name}", fromlist=[''])
        except ImportError:
            self.ship_module = None

    def initialize_in_battle(self, position, heading):
        self.position = list(position)
        self.heading = heading % 16
        self.rotation = self.heading * 22.5
        self.velocity = [0.0, 0.0]
        self.thrust_timer = 0
        self.turn_timer = 0
        self.in_battle = True

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
            target_x = math.sin(angle_rad) * self.max_thrust * GameConstants.SPEED_SCALE
            target_y = -math.cos(angle_rad) * self.max_thrust * GameConstants.SPEED_SCALE

            if self.inertia:
                dx = target_x - self.velocity[0]
                dy = target_y - self.velocity[1]

                scale = self.thrust_increment / self.max_thrust
                thrust_x = dx * scale
                thrust_y = dy * scale

            else:
                thrust_x = target_x
                thrust_y = target_y

            self.add_impulse(thrust_x, thrust_y)

            marker_x, marker_y = self.get_thrust_marker_position()
            marker = ThrustMarker(marker_x, marker_y)
            self.thrust_timer = int(self.thrust_wait * GameConstants.THRUST_WAIT_SCALE)
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
            self.turn_timer = int(self.turn_wait * GameConstants.TURN_WAIT_SCALE)

    def turn_right(self):
        if self.can_turn():
            self.heading = (self.heading + 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.turn_wait * GameConstants.TURN_WAIT_SCALE)

    def perform_action1(self):
        if self.ship_module and hasattr(self.ship_module, 'action1'):
            return self.ship_module.action1(self)
        return False

    def perform_action2(self):
        if self.ship_module and hasattr(self.ship_module, 'action2'):
            return self.ship_module.action2(self)
        return False

    def perform_action3(self):
        if self.ship_module and hasattr(self.ship_module, 'action3'):
            return self.ship_module.action3(self)
        return False

    def update(self):
        super().update()
        return True