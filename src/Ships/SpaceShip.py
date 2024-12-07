import json
import os


class SpaceShip:
    def __init__(self, ship_name, player_num):
        self.name = ship_name
        self.player = player_num

        # Load ship data
        with open('Ships/Ships.json', 'r') as f:
            ships_data = json.load(f)
            ship_data = ships_data[ship_name]

        # Intrinsic variables from Ships.json
        self.ship_type = ship_data['ShipType']
        self.cost = ship_data['Cost']
        self.max_hp = ship_data['MaxHP']
        self.start_hp = ship_data['StartHP']
        self.max_energy = ship_data['MaxEnergy']
        self.start_energy = ship_data['StartEnergy']
        self.energy_regen = ship_data['EnergyRegen']
        self.energy_wait = ship_data['EnergyWait']
        self.max_thrust = ship_data['MaxThrust']
        self.thrust_increment = ship_data['ThrustIncrement']
        self.thrust_wait = ship_data['ThrustWait']
        self.turn_wait = ship_data['TurnWait']
        self.ship_mass = ship_data['ShipMass']
        self.inertia = ship_data['Inertia']

        # Situational variables
        self.currently_alive = True
        self.current_hp = self.start_hp
        self.current_energy = self.start_energy
        self.direction = 0
        self.position = [0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.energy_timer = 0
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

        # Load ship-specific module
        self.module_name = f"Ships.{ship_name}.{ship_name}"
        try:
            self.ship_module = __import__(self.module_name, fromlist=[''])
        except ImportError as e:
            print(f"Warning: Could not load ship-specific module for {ship_name}: {e}")
            self.ship_module = None

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
