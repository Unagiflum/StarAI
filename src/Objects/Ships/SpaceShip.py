import json
from pathlib import Path
import math
import pygame

import src.Const as Const
from src.Objects.Object import Object

class ThrustMarker(Object):
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

    def draw(self, screen, scale_factor, translation):
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        positions = [(screen_x, screen_y)]

        if screen_x < 6:
            positions.append((screen_x + screen.get_height(), screen_y))
        elif screen_x > screen.get_height() - 6:
            positions.append((screen_x - screen.get_height(), screen_y))

        if screen_y < 6:
            positions.append((screen_x, screen_y + screen.get_height()))
        elif screen_y > screen.get_height() - 6:
            positions.append((screen_x, screen_y - screen.get_height()))

        if len(positions) > 2:
            positions.append((
                screen_x + (screen.get_height() if screen_x < screen.get_height() // 2 else -screen.get_height()),
                screen_y + (screen.get_height() if screen_y < screen.get_height() // 2 else -screen.get_height())
            ))

        for pos_x, pos_y in positions:
            pygame.draw.circle(
                screen, self.get_color(), (pos_x, pos_y),
                max(1, int(3 * scale_factor))
            )

class SpaceShip(Object):

    def __init__(self, ship_name, player_num):
        self.name = ship_name
        with open(Const.SHIPS_JSON_PATH, 'r') as f:
            ships_data = json.load(f)
            ship_data = ships_data[ship_name]

        sprite_location = Path(ship_data['SpriteLocation'])

        super().__init__(
            player_num=player_num,
            max_hp=ship_data['MaxHP'],
            start_hp=ship_data['StartHP'],
            inertia=ship_data['Inertia'],
            sprite_location=sprite_location,
            sprite_scale=ship_data['SpriteScale'],
            size=[ship_data['Size']['width'], ship_data['Size']['height']]
        )

        # Load sprites
        self.sprites = []
        for i in range(16):
            sprite_path = self.sprite_location.joinpath(f'{self.name}{i:02d}.png')
            self.sprites.append(pygame.image.load(str(sprite_path)).convert_alpha())

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
                # Add thrust_increment in facing direction to current velocity
                new_velocity = [
                    self.velocity[0] + thrust_direction[0] * self.thrust_increment,
                    self.velocity[1] + thrust_direction[1] * self.thrust_increment
                ]

                # Normalize to max_thrust
                speed = math.sqrt(new_velocity[0] ** 2 + new_velocity[1] ** 2)
                scale = 1.0
                if speed > self.max_thrust:
                    scale = self.max_thrust / speed
                target_velocity = [new_velocity[0] * scale, new_velocity[1] * scale]

                # Calculate difference vector
                diff_vector = [
                    target_velocity[0] - self.velocity[0],
                    target_velocity[1] - self.velocity[1]
                ]

                # Normalize difference to thrust_increment
                diff_magnitude = math.sqrt(diff_vector[0] ** 2 + diff_vector[1] ** 2)
                if diff_magnitude > 0:
                    scale = self.thrust_increment / diff_magnitude
                    self.add_impulse(diff_vector[0] * scale, diff_vector[1] * scale)
            else:
                # Non-inertial ships behavior unchanged
                self.add_impulse(
                    thrust_direction[0] * self.max_thrust,
                    thrust_direction[1] * self.max_thrust
                )

            marker_x, marker_y = self.get_thrust_marker_position()
            marker = ThrustMarker(marker_x, marker_y)
            self.thrust_timer = int(self.thrust_wait * Const.THRUST_WAIT_SCALE)
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

    def update(self):
        super().update()
        return True

    def draw(self, screen, scale_factor, translation):
        sprite = self.sprites[self.heading]
        sprite_rect = sprite.get_rect()

        total_scale = scale_factor * self.sprite_scale
        scaled_sprite = pygame.transform.scale(
            sprite,
            (int(sprite_rect.width * total_scale),
             int(sprite_rect.height * total_scale))
        )
        scaled_rect = scaled_sprite.get_rect()

        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        positions = [(screen_x, screen_y)]
        screen_height = screen.get_height()

        if screen_x < scaled_rect.width // 2:
            positions.append((screen_x + screen_height, screen_y))
        elif screen_x > screen_height - scaled_rect.width // 2:
            positions.append((screen_x - screen_height, screen_y))

        if screen_y < scaled_rect.height // 2:
            positions.append((screen_x, screen_y + screen_height))
        elif screen_y > screen_height - scaled_rect.height // 2:
            positions.append((screen_x, screen_y - screen_height))

        if len(positions) > 2:
            positions.append((
                screen_x + (screen_height if screen_x < screen_height // 2 else -screen_height),
                screen_y + (screen_height if screen_y < screen_height // 2 else -screen_height)
            ))

        for pos_x, pos_y in positions:
            screen.blit(scaled_sprite, (
                pos_x - scaled_rect.width // 2,
                pos_y - scaled_rect.height // 2
            ))