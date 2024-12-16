from src.Objects.Object import PlayerObject, ThrustMarker
import src.Const as Const
import math
import pygame
import json
from pathlib import Path

class SpaceShip(PlayerObject):
    def __init__(self, ship_name, player_num):
        # Load ship-specific data from Ships.json
        with open(Const.SHIPS_JSON_PATH, 'r') as f:
            ships_data = json.load(f)
            ship_data = ships_data[ship_name]

        sprite_location = Path(ship_data['SpriteLocation'])

        # Initialize the PlayerObject base class
        super().__init__(
            name=ship_name,
            sprite_location=sprite_location,
            size=[ship_data['Size']['width'], ship_data['Size']['height']],
            player=player_num,
            sprite_scale=ship_data['SpriteScale']
        )

        # Ship-specific attributes
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
        self.a1_cost = ship_data['A1Cost']
        self.a2_cost = ship_data['A3Cost']
        self.a3_cost = ship_data['A3Cost']
        self.a1_wait = ship_data['A3Wait']
        self.a2_wait = ship_data['A3Wait']
        self.a3_wait = ship_data['A3Wait']
        self.mass = ship_data['Mass']
        self.inertia = ship_data['Inertia']

        self.current_hp = ship_data['StartHP']
        self.current_energy = ship_data['StartEnergy']
        self.energy_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0

        self.status1 = False
        self.status2 = False
        self.status3 = False
        self.status4 = 0
        self.status5 = 0
        self.status6 = 0

        self.can_die = True

        # Load optional ship module
        try:
            self.ship_module = __import__(f"Objects.Ships.{ship_name}.{ship_name}", fromlist=[''])
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
    def add_impulse(self, dx, dy):
        if self.can_move:
            self.accumulated_impulses[0] += dx
            self.accumulated_impulses[1] += dy

    def update(self):
        self.update_physics()
        return True

    def update_physics(self):
        if self.can_move:
            if self.inertia:
                gravity_impulse = self.get_gravity()
                acc0 = [gravity_impulse[0] + self.accumulated_impulses[0],
                        gravity_impulse[1] + self.accumulated_impulses[1]]

                self.position[0] = (self.position[0] + (self.velocity[0]
                                  + 0.5 * acc0[0]) * Const.SPEED_SCALE) % Const.ARENA_SIZE
                self.position[1] = (self.position[1] + (self.velocity[1]
                                  + 0.5 * acc0[1]) * Const.SPEED_SCALE) % Const.ARENA_SIZE

                gravity_impulse = self.get_gravity()
                acc1 = [gravity_impulse[0] + self.accumulated_impulses[0],
                        gravity_impulse[1] + self.accumulated_impulses[1]]
                self.velocity[0] += (acc0[0] +acc1[0]) * 0.5
                self.velocity[1] += (acc0[1] +acc1[1]) * 0.5
                speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
                if speed > Const.SPEED_LIMIT:
                    scale = Const.SPEED_LIMIT / speed
                    self.velocity[0] *= scale
                    self.velocity[1] *= scale
            else:
                self.velocity = self.accumulated_impulses.copy()

                speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
                if speed > Const.SPEED_LIMIT:
                    scale = Const.SPEED_LIMIT / speed
                    self.velocity[0] *= scale
                    self.velocity[1] *= scale

                self.position[0] = (self.position[0] + self.velocity[0] * Const.SPEED_SCALE) % Const.ARENA_SIZE
                self.position[1] = (self.position[1] + self.velocity[1] * Const.SPEED_SCALE) % Const.ARENA_SIZE

            self.accumulated_impulses = [0.0, 0.0]


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

                _, distance = self.planet_distance()
                if speed > self.max_thrust and distance > Const.GRAVITY_RANGE:
                    scale = self.max_thrust / speed

                target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

                diff_vector = [target_velocity[0] - self.velocity[0], target_velocity[1] - self.velocity[1]]

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

    def draw(self, screen, scale_factor, translation):
        sprite = self.sprites[self.heading]
        sprite_rect = sprite.get_rect()

        total_scale = scale_factor * self.sprite_scale
        scaled_sprite = pygame.transform.smoothscale_by(sprite, total_scale)
        scaled_rect = scaled_sprite.get_rect()

        # Calculate screen position with translation
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        # Draw the ship at all potential wrap-around positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                # Only draw if the position would be visible
                if (0 <= pos_x <= Const.SCREEN_HEIGHT and
                        0 <= pos_y <= Const.SCREEN_HEIGHT):
                    screen.blit(scaled_sprite, (
                        Const.SCREEN_LEFT +  pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))