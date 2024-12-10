import json
import math

# Constants (moved here from Battle.py for convenience)
SPEED_SCALE = 0.75
TURN_WAIT_SCALE = 2.0
THRUST_WAIT_SCALE = 2.0

class PlayerObject:
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
        self.can_collide = True
        self.can_expire = False
        self.expiration_timer = 0


class SpaceShip(PlayerObject):
    def __init__(self, ship_name, player_num):
        self.name = ship_name

        # Load ship data
        with open('Ships/Ships.json', 'r') as f:
            ships_data = json.load(f)
            ship_data = ships_data[ship_name]

        # Initialize parent PlayerObject with values from ship_data
        super().__init__(
            player_num=player_num,
            max_hp=ship_data['MaxHP'],
            start_hp=ship_data['StartHP'],
            inertia=ship_data['Inertia'],
            sprite_location=ship_data['SpriteLocation'],
            sprite_scale=ship_data['SpriteScale'],
            size=[ship_data['Size']['width'], ship_data['Size']['height']]
        )

        # Spaceship-specific attributes
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

        # Spaceship-specific situational variables
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

        # Game state
        self.in_battle = False

        # Load ship-specific module if it exists
        self.module_name = f"Ships.{ship_name}.{ship_name}"
        try:
            self.ship_module = __import__(self.module_name, fromlist=[''])
        except ImportError:
            self.ship_module = None

    def initialize_in_battle(self, position, heading):
        """Initialize battle-specific state, such as position and heading."""
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
        """Update thrust and turn timers. If inertia is off and no forward key is pressed,
        velocity drops to zero when thrust is not applied."""
        if self.thrust_timer > 0:
            self.thrust_timer -= 1
        if self.turn_timer > 0:
            self.turn_timer -= 1

        if not self.inertia and self.thrust_timer == 0 and not forward_pressed:
            self.velocity = [0.0, 0.0]

    def apply_thrust(self):
        """Apply thrust if allowed, factoring in inertia and thrust increment."""
        if self.can_thrust():
            angle_rad = math.radians(self.rotation)
            if self.inertia:
                self.velocity[0] += math.sin(angle_rad) * self.thrust_increment * SPEED_SCALE
                self.velocity[1] -= math.cos(angle_rad) * self.thrust_increment * SPEED_SCALE
            else:
                speed = self.max_thrust * SPEED_SCALE
                self.velocity[0] = math.sin(angle_rad) * speed
                self.velocity[1] = -math.cos(angle_rad) * speed

            self.thrust_timer = int(self.thrust_wait * THRUST_WAIT_SCALE)

    def turn_left(self):
        """Turn the ship to the left if allowed."""
        if self.can_turn():
            self.heading = (self.heading - 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.turn_wait * TURN_WAIT_SCALE)

    def turn_right(self):
        """Turn the ship to the right if allowed."""
        if self.can_turn():
            self.heading = (self.heading + 1) % 16
            self.rotation = self.heading * 22.5
            self.turn_timer = int(self.turn_wait * TURN_WAIT_SCALE)

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