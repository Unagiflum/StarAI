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
        self.start_position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance,
        ]
        self.position = self.start_position.copy()
        self.heading = self.parent.heading
        self.rotation = self.parent.rotation

    def calculate_end_position(self):
        angle_rad = math.radians(self.rotation)
        spawn_distance = (self.parent.size[1]) / 2
        self.start_position = [
            self.parent.position[0] + math.sin(angle_rad) * spawn_distance,
            self.parent.position[1] - math.cos(angle_rad) * spawn_distance,
        ]
        self.position = self.start_position.copy()
        self.end_position[0] = self.start_position[0] + math.sin(angle_rad) * self.LASER_RANGE
        self.end_position[1] = self.start_position[1] - math.cos(angle_rad) * self.LASER_RANGE

    def update(self):
        if not self.currently_alive:
            return False
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import (
            interpolated_position,
            interpolated_sprite_index,
        )
        from src.toroidal import wrapped_delta

        parent_pos = interpolated_position(self.parent, interp_t)
        visual_idx = interpolated_sprite_index(self.parent, interp_t)
        visual_heading = visual_idx / const.VIDEO_FPS_MULTIPLIER
        angle_rad = math.radians(visual_heading * const.TURN_ANGLE)

        spawn_distance = (self.parent.size[1]) / 2
        visual_pos = [
            parent_pos[0] + math.sin(angle_rad) * spawn_distance,
            parent_pos[1] - math.cos(angle_rad) * spawn_distance,
        ]

        if getattr(self, "intercepted", False):
            if getattr(self, "attached_target", None) and getattr(self, "target_contact_offset", None):
                interp_target_pos = interpolated_position(self.attached_target, interp_t)
                raw_end = [
                    interp_target_pos[0] + self.target_contact_offset[0],
                    interp_target_pos[1] + self.target_contact_offset[1],
                ]
                end_offset = wrapped_delta(visual_pos, raw_end)
            else:
                end_offset = wrapped_delta(self.start_position, self.end_position)
            draw_end_position = [
                visual_pos[0] + end_offset[0],
                visual_pos[1] + end_offset[1],
            ]
        else:
            draw_end_position = [
                visual_pos[0] + math.sin(angle_rad) * self.LASER_RANGE,
                visual_pos[1] - math.cos(angle_rad) * self.LASER_RANGE,
            ]

        screen_start_x = int((visual_pos[0] + translation[0]) * scale_factor)
        screen_start_y = int((visual_pos[1] + translation[1]) * scale_factor)
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
                        max(1, int(self.LASER_WIDTH * scale_factor)),
                    )
