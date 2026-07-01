import pygame

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.toroidal import wrapped_delta


class ChmmrSatelliteLaser(Ability):
    def __init__(self, parent, target=None):
        super().__init__("ChmmrSatelliteLaser", parent)
        owner = getattr(parent, "parent", parent)
        definition = SHIP_DEFINITIONS[owner.name]
        self.target = target
        self.current_damage = definition.satellite_laser_damage
        self.damages = (self.current_damage,)
        self.LASER_RANGE = definition.satellite_laser_range
        self.LASER_COLOR = definition.satellite_laser_color
        self.LASER_WIDTH = definition.satellite_laser_width
        self.position = parent.position.copy()
        self.previous_position = self.position.copy()
        self.end_position = self.position.copy()
        self.calculate_end_position()

    def should_consider_laser_target(self, target):
        return target is self.target

    def calculate_end_position(self):
        self.position = self.parent.position.copy()
        if self.target is None:
            self.end_position = self.position.copy()
            return
        delta = wrapped_delta(self.position, self.target.position)
        self.end_position = [
            self.position[0] + delta[0],
            self.position[1] + delta[1],
        ]

    def update(self):
        if not self.currently_alive:
            return False
        self.previous_position = self.position.copy()
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import interpolated_position

        start = interpolated_position(self.parent, interp_t)
        if getattr(self, "intercepted", False):
            delta = wrapped_delta(self.position, self.end_position)
        elif self.target is not None:
            delta = wrapped_delta(start, interpolated_position(self.target, interp_t))
        else:
            delta = (0, 0)
        end = [start[0] + delta[0], start[1] + delta[1]]
        start_x = int((start[0] + translation[0]) * scale_factor)
        start_y = int((start[1] + translation[1]) * scale_factor)
        end_x = int((end[0] + translation[0]) * scale_factor)
        end_y = int((end[1] + translation[1]) * scale_factor)
        view = pygame.Rect(0, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)
        for wrap_x in (-1, 0, 1):
            for wrap_y in (-1, 0, 1):
                offset_x = wrap_x * const.ARENA_SIZE * scale_factor
                offset_y = wrap_y * const.ARENA_SIZE * scale_factor
                line_start = (start_x + offset_x, start_y + offset_y)
                line_end = (end_x + offset_x, end_y + offset_y)
                if view.clipline(line_start, line_end):
                    self.draw_aa_laser(
                        screen,
                        self.LASER_COLOR,
                        (const.SCREEN_LEFT + line_start[0], line_start[1]),
                        (const.SCREEN_LEFT + line_end[0], line_end[1]),
                        max(1, int(self.LASER_WIDTH * scale_factor)),
                    )
