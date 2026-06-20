import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability, ABILITIES_DATA, wrapped_endpoint
from src.toroidal import wrapped_delta


class KzerZaA2Laser(Ability):
    def __init__(self, parent, target=None):
        super().__init__("KzerZaA2Laser", parent)
        data = ABILITIES_DATA["KzerZaA2"]
        self.target = None
        self.aim_target = target or parent
        self.LASER_COLOR = (255, 255, 0)
        self.LASER_WIDTH = 2
        self.offset = data["offset"]
        self.LASER_RANGE = data["range"]
        self.track_directions = data["track_directions"]
        self.start_position = parent.position.copy()
        self.end_position = parent.position.copy()
        self.calculate_end_position()

    def calculate_end_position(self):
        dx, dy = wrapped_delta(self.parent.position, self.aim_target.position)
        target_angle = math.degrees(math.atan2(dx, -dy)) % 360
        direction_step = 360 / self.track_directions
        shot_angle = round(target_angle / direction_step) * direction_step
        angle = math.radians(shot_angle)
        direction = [math.sin(angle), -math.cos(angle)]
        self.start_position = [
            (self.parent.position[0] + direction[0] * self.offset) % const.ARENA_SIZE,
            (self.parent.position[1] + direction[1] * self.offset) % const.ARENA_SIZE,
        ]
        self.position = self.start_position.copy()
        self.end_position = [
            (self.parent.position[0] + direction[0] * self.LASER_RANGE) % const.ARENA_SIZE,
            (self.parent.position[1] + direction[1] * self.LASER_RANGE) % const.ARENA_SIZE,
        ]

    def update(self):
        if not self.currently_alive:
            return False
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def draw(self, screen, scale_factor, translation):
        start = self.start_position
        end = wrapped_endpoint(start, self.end_position)
        start_x = int((start[0] + translation[0]) * scale_factor)
        start_y = int((start[1] + translation[1]) * scale_factor)
        end_x = int((end[0] + translation[0]) * scale_factor)
        end_y = int((end[1] + translation[1]) * scale_factor)
        view_rect = pygame.Rect(0, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)

        for wrap_x in [-1, 0, 1]:
            for wrap_y in [-1, 0, 1]:
                arena_x = wrap_x * const.ARENA_SIZE * scale_factor
                arena_y = wrap_y * const.ARENA_SIZE * scale_factor
                line = (start_x + arena_x, start_y + arena_y, end_x + arena_x, end_y + arena_y)
                if view_rect.clipline(line):
                    pygame.draw.line(
                        screen,
                        self.LASER_COLOR,
                        (const.SCREEN_LEFT + line[0], line[1]),
                        (const.SCREEN_LEFT + line[2], line[3]),
                        max(1, int(self.LASER_WIDTH * scale_factor)),
                    )
