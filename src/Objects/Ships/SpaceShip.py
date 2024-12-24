from src.Objects.Object import PlayerObject, ThrustMarker
import src.Const as Const
import math
import pygame
import json
from pathlib import Path

# Load ship data once at module level
with open(Const.SHIPS_JSON_PATH, 'r') as f:
    SHIPS_DATA = json.load(f)

class SpaceShip(PlayerObject):
    # Class-level storage for shared sprites
    _shared_sprites = {}

    def __init__(self, ship_name, player_num):
        # Get ship-specific data from cached data
        ship_data = SHIPS_DATA[ship_name]
        sprite_location = Path(ship_data['SpriteLocation'])

        # Initialize the PlayerObject base class
        super().__init__(
            name=ship_name,
            sprite_location=sprite_location,
            size=[0,0],
            player=player_num,
            sprite_scale=ship_data['SpriteScale']
        )

        # Load shared sprites if not already loaded for this ship type
        if ship_name not in self._shared_sprites:
            self._shared_sprites[ship_name] = []
            try:
                for i in range(Const.SHIP_DIRECTIONS):
                    sprite_path = self.sprite_location.joinpath(f'{ship_name}{i:02d}.png')
                    base_sprite = pygame.image.load(str(sprite_path)).convert_alpha()
                    scaled_sprite = pygame.transform.smoothscale_by(base_sprite, self.sprite_scale)
                    self._shared_sprites[ship_name].append(scaled_sprite)
                    if i == 0:  # Set size based on first sprite
                        self.size = [scaled_sprite.get_width(), scaled_sprite.get_height()]
            except pygame.error as e:
                print(f"Error loading sprites for {ship_name}: {e}")
                self._shared_sprites[ship_name] = None
                return
        else:
            self.size = [
                self._shared_sprites[ship_name][0].get_width(),
                self._shared_sprites[ship_name][0].get_height()
            ]

        # Use the shared sprites
        self.sprites = self._shared_sprites[ship_name]

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
        self.a2_cost = ship_data['A2Cost']
        self.a3_cost = ship_data['A3Cost']
        self.a1_wait = ship_data['A1Wait']
        self.a2_wait = ship_data['A2Wait']
        self.a3_wait = ship_data['A3Wait']
        self.mass = ship_data['Mass']
        self.inertia = ship_data['Inertia']

        self.current_hp = ship_data['StartHP']
        self.current_energy = ship_data['StartEnergy']
        self.energy_timer = 0

        # Timers
        self.thrust_timer = 0
        self.turn_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0

        # action states
        self.thrust_active = False
        self.turn_left_active = False
        self.turn_right_active = False
        self.action1_active = False
        self.action2_active = False

        self.opponent = None
        self.can_die = True

        # Load optional ship module
        try:
            self.ship_module = __import__(f"Objects.Ships.{ship_name}.{ship_name}", fromlist=[''])
        except ImportError:
            self.ship_module = None

    def initialize_in_battle(self, position, heading):
        self.position = list(position)
        self.heading = heading % Const.SHIP_DIRECTIONS
        self.rotation = self.heading * Const.TURN_ANGLE
        self.velocity = [0.0, 0.0]
        self.thrust_timer = 0
        self.turn_timer = 0
        self.action1_timer = 0
        self.action2_timer = 0
        self.action3_timer = 0
        self.in_battle = True

    def handle_actions(self, key, pressed, forward_key, left_key, right_key, action1_key, action2_key):
        new_objects = []

        # Update internal key state based on event
        if key == forward_key:
            self.thrust_active = pressed
        elif key == left_key:
            self.turn_left_active = pressed
        elif key == right_key:
            self.turn_right_active = pressed
        elif key == action1_key:
            self.action1_active = pressed
        elif key == action2_key:
            self.action2_active = pressed

        # Update timers and check if actions can be performed
        self.update_timers()

        # Handle movement based on active states
        if self.turn_left_active:
            self.turn_left()
        if self.turn_right_active:
            self.turn_right()
        if self.thrust_active:
            marker = self.apply_thrust(self.max_thrust, self.thrust_increment, 0, self.can_thrust())
            if marker:
                new_objects.append(marker)

        # Handle key release events
        if key and not pressed:
            if key == action1_key:
                result = self.perform_action1()
                if result:
                    if isinstance(result, list):
                        new_objects.extend(result)
                    else:
                        new_objects.append(result)
            elif key == action2_key:
                result = self.perform_action2()
                if result:
                    if isinstance(result, list):
                        new_objects.extend(result)
                    else:
                        new_objects.append(result)
            return new_objects

        # Handle actions based on active states
        if self.action1_active and self.action2_active:
            result, is_valid = self.perform_action3()
            if result:
                if isinstance(result, list):
                    new_objects.extend(result)
                else:
                    new_objects.append(result)
            elif not is_valid:
                if self.action1_active:
                    result = self.perform_action1()
                    if result:
                        if isinstance(result, list):
                            new_objects.extend(result)
                        else:
                            new_objects.append(result)
                if self.action2_active:
                    result = self.perform_action2()
                    if result:
                        if isinstance(result, list):
                            new_objects.extend(result)
                        else:
                            new_objects.append(result)
        else:
            if self.action1_active:
                result = self.perform_action1()
                if result:
                    if isinstance(result, list):
                        new_objects.extend(result)
                    else:
                        new_objects.append(result)
            if self.action2_active:
                result = self.perform_action2()
                if result:
                    if isinstance(result, list):
                        new_objects.extend(result)
                    else:
                        new_objects.append(result)

        return new_objects

    def add_impulse(self, dx, dy):
        if self.can_move:
            self.accumulated_impulses[0] += dx
            self.accumulated_impulses[1] += dy

    def update(self):
        self.update_physics()
        return True

    def update_physics(self):
        if self.inertia:
            self.velocity[0] += self.accumulated_impulses[0]
            self.velocity[1] += self.accumulated_impulses[1]
            self.accumulated_impulses = [0.0, 0.0]
            self.apply_verlet()
            self.apply_speed_limit()
        else:
            self.velocity = self.accumulated_impulses.copy()
            self.accumulated_impulses = [0.0, 0.0]
            self.apply_speed_limit()
            self.position[0] = (self.position[0] + self.velocity[0] * Const.SPEED_SCALE) % Const.ARENA_SIZE
            self.position[1] = (self.position[1] + self.velocity[1] * Const.SPEED_SCALE) % Const.ARENA_SIZE


    def can_thrust(self):
        return self.thrust_timer == 0

    def can_turn(self):
        return self.turn_timer == 0

    def can_action1(self):
        return self.action1_timer == 0 and self.current_energy >= self.a1_cost

    def can_action2(self):
        return self.action2_timer == 0 and self.current_energy >= self.a2_cost

    def can_action3(self):
        return self.action3_timer == 0 and self.current_energy >= self.a3_cost

    def update_timers(self):
        if self.thrust_timer > 0:
            self.thrust_timer -= 1
        if self.turn_timer > 0:
            self.turn_timer -= 1
        if self.action1_timer > 0:
            self.action1_timer -= 1
        if self.action2_timer > 0:
            self.action2_timer -= 1
        if self.action3_timer > 0:
            self.action3_timer -= 1
        if not self.inertia and self.thrust_timer == 0 and not self.thrust_active:
            self.velocity = [0.0, 0.0]

        self.energy_timer += 1
        if self.energy_timer >= self.energy_wait*Const.RECHARGE_DELAY_SCALE:
            self.energy_timer = 0
            if self.current_energy < self.max_energy:
                self.current_energy = min(self.max_energy,
                                          self.current_energy + self.energy_regen)

    def apply_thrust(self, max_thrust, thrust_increment, angle, can_thrust):
        if can_thrust:
            angle_rad = math.radians(self.rotation + angle)
            thrust_direction = [math.sin(angle_rad), -math.cos(angle_rad)]

            if self.inertia:
                new_velocity = [
                    self.velocity[0] + thrust_direction[0] * thrust_increment,
                    self.velocity[1] + thrust_direction[1] * thrust_increment
                ]

                speed = math.sqrt(new_velocity[0] ** 2 + new_velocity[1] ** 2)
                scale = 1.0

                _, planet_distance = self.planet_distance()
                if speed > max_thrust and planet_distance > Const.GRAVITY_RANGE:
                    scale = max_thrust / speed
                if speed > Const.MAX_GRAV_WHIP:
                    scale = Const.MAX_GRAV_WHIP / speed

                target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

                diff_vector = [target_velocity[0] - self.velocity[0], target_velocity[1] - self.velocity[1]]

                diff_magnitude = math.sqrt(diff_vector[0] ** 2 + diff_vector[1] ** 2)
                if diff_magnitude > thrust_increment:
                    scale = thrust_increment / diff_magnitude
                    self.add_impulse(diff_vector[0] * scale, diff_vector[1] * scale)
                else:
                    self.add_impulse(diff_vector[0] , diff_vector[1] )
            else:
                self.add_impulse(
                    thrust_direction[0] * max_thrust,
                    thrust_direction[1] * max_thrust
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
            self.heading = (self.heading - 1) % Const.SHIP_DIRECTIONS
            self.rotation = self.heading * Const.TURN_ANGLE
            self.turn_timer = int(self.turn_wait * Const.TURN_WAIT_SCALE)

    def turn_right(self):
        if self.can_turn():
            self.heading = (self.heading + 1) % Const.SHIP_DIRECTIONS
            self.rotation = self.heading * Const.TURN_ANGLE
            self.turn_timer = int(self.turn_wait * Const.TURN_WAIT_SCALE)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * Const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * Const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False

    def draw(self, screen, scale_factor, translation):
        sprite = self.sprites[self.heading]
        sprite_rect = sprite.get_rect()

        scaled_sprite = pygame.transform.smoothscale_by(sprite, scale_factor)
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
                        Const.SCREEN_LEFT + pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))