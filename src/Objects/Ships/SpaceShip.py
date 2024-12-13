from src.Objects.Object import MovableObject
import src.Const as Const
import pygame
import json
from pathlib import Path
from src.UI import UI

class SpaceShip(MovableObject):
    def __init__(self, ship_name, player_num):
        self.name = ship_name
        with open(Const.SHIPS_JSON_PATH, 'r') as f:
            ships_data = json.load(f)
            ship_data = ships_data[ship_name]

        sprite_location = Path(ship_data['SpriteLocation'])

        super().__init__(
            name=ship_name,
            sprite_location=sprite_location,
            size=[ship_data['Size']['width'], ship_data['Size']['height']],
            player=player_num,
            max_hp=ship_data['MaxHP'],
            start_hp=ship_data['StartHP'],
            inertia=ship_data['Inertia'],
            sprite_scale=ship_data['SpriteScale'],
            max_thrust=ship_data['MaxThrust'],
            thrust_increment=ship_data['ThrustIncrement'],
            thrust_wait=ship_data['ThrustWait'],
            turn_wait=ship_data['TurnWait'],
            mass=ship_data['Mass']
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

        # Ship state variables
        self.current_energy = self.start_energy
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
        scaled_sprite = pygame.transform.scale(
            sprite,
            (int(sprite_rect.width * total_scale),
             int(sprite_rect.height * total_scale))
        )
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
                if (0 <= pos_x <= UI.SCREEN_HEIGHT and
                        0 <= pos_y <= UI.SCREEN_HEIGHT):
                    screen.blit(scaled_sprite, (
                        pos_x - scaled_rect.width // 2,
                        pos_y - scaled_rect.height // 2
                    ))