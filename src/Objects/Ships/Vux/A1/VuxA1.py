from src.Objects.Ships.ability import Ability, wrapped_endpoint
from src.Objects.Ships.catalog import ABILITIES_DATA
from src.toroidal import wrapped_delta
import pygame
import math
import src.const as const

class VuxA1(Ability):
    def __init__(self, parent):
        super().__init__("VuxA1", parent)
        ability_data = ABILITIES_DATA["VuxA1"]
        self.LASER_RANGE = ability_data.get("LASER_RANGE", 644)
        self.LASER_COLOR = (0, 255, 0)
        self.LASER_WIDTH = 4
        
        self.end_position = [0, 0]
        self.place_self()
        self.calculate_end_position()

    def place_self(self):
        # A1 is fired from the front, directly forward
        angle_rad = math.radians(self.parent.rotation)
        spawn_distance = (self.parent.size[1]) / 2
        self.position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance
        ]
        self.heading = self.parent.heading
        self.rotation = self.parent.rotation

    def calculate_end_position(self):
        angle_rad = math.radians(self.rotation)
        self.end_position[0] = self.position[0] + math.sin(angle_rad) * self.LASER_RANGE
        self.end_position[1] = self.position[1] - math.cos(angle_rad) * self.LASER_RANGE

    def update(self):
        if not self.currently_alive:
            return False
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def draw(self, screen, scale_factor, translation):
        self.place_self()
        if not getattr(self, "intercepted", False):
            self.calculate_end_position()

        draw_end_position = wrapped_endpoint(self.position, self.end_position)
        screen_start_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_start_y = int((self.position[1] + translation[1]) * scale_factor)
        screen_end_x = int((draw_end_position[0] + translation[0]) * scale_factor)
        screen_end_y = int((draw_end_position[1] + translation[1]) * scale_factor)
        view_rect = pygame.Rect(0, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)

        # Draw laser at all potential wrap-around positions
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                start_x = screen_start_x + dx * const.ARENA_SIZE * scale_factor
                start_y = screen_start_y + dy * const.ARENA_SIZE * scale_factor
                end_x = screen_end_x + dx * const.ARENA_SIZE * scale_factor
                end_y = screen_end_y + dy * const.ARENA_SIZE * scale_factor

                if view_rect.clipline((start_x, start_y), (end_x, end_y)):
                    pygame.draw.line(
                        screen,
                        self.LASER_COLOR,
                        (const.SCREEN_LEFT + start_x, start_y),
                        (const.SCREEN_LEFT + end_x, end_y),
                        max(1, int(self.LASER_WIDTH * scale_factor))
                    )
