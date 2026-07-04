from src.Objects.Ships.ability import (
    Ability,
    outward_visual_laser_start,
    wrapped_endpoint,
)
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
import pygame
import math
import src.const as const
from src.toroidal import wrapped_delta


class ArilouA1(Ability):
    def __init__(self, parent):
        super().__init__("ArilouA1", parent)
        definition = ABILITY_DEFINITIONS["ArilouA1"]
        self.LASER_RANGE = definition.range
        self.configure_laser_colors(definition.colors)
        self.LASER_WIDTH = definition.stroke_width
        self.end_position = [0, 0]
        self.calculate_end_position()

    def calculate_end_position(self):
        self.position = self.configured_gun_position()
        opponent = self._live_trackable_opponent()
        if opponent is not None:
            # Calculate direction to opponent
            dx, dy = wrapped_delta(self.position, opponent.position)

            # Calculate angle and quantize to nearest SHIP_DIRECTION
            angle = math.atan2(dx, -dy)
            direction = (
                round(angle / (2 * math.pi) * const.SHIP_DIRECTIONS)
                % const.SHIP_DIRECTIONS
            )
        else:
            relative_direction = self.configured_gun()[1] or 0
            direction = (
                self.parent.heading
                + round(relative_direction / const.TURN_ANGLE)
            ) % const.SHIP_DIRECTIONS

        angle = direction * (2 * math.pi / const.SHIP_DIRECTIONS)

        # Calculate end position using sin/cos flipped to match coordinate system
        self.end_position[0] = self.position[0] + math.sin(angle) * self.LASER_RANGE
        self.end_position[1] = self.position[1] - math.cos(angle) * self.LASER_RANGE

    def update_physics(self):
        self.position = self.configured_gun_position()

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.update_physics()
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import (
            interpolated_position,
            interpolated_sprite_index,
        )

        parent_pos = interpolated_position(self.parent, interp_t)
        visual_idx = interpolated_sprite_index(self.parent, interp_t)
        visual_rotation = (
            visual_idx / const.VIDEO_FPS_MULTIPLIER * const.TURN_ANGLE
        )
        pos = self.configured_gun_position(
            rotation=visual_rotation, position=parent_pos
        )

        if not getattr(self, "intercepted", False):
            opponent = self._live_trackable_opponent()
            if opponent is not None:
                target_pos = interpolated_position(opponent, interp_t)
                dx, dy = wrapped_delta(pos, target_pos)
                angle = math.atan2(dx, -dy)
                direction = (
                    round(angle / (2 * math.pi) * const.SHIP_DIRECTIONS)
                    % const.SHIP_DIRECTIONS
                )
            else:
                relative_direction = self.configured_gun()[1] or 0
                direction = (
                    visual_idx / const.VIDEO_FPS_MULTIPLIER
                    + relative_direction / const.TURN_ANGLE
                )

            angle = direction * (2 * math.pi / const.SHIP_DIRECTIONS)
            draw_end_position = [
                pos[0] + math.sin(angle) * self.LASER_RANGE,
                pos[1] - math.cos(angle) * self.LASER_RANGE,
            ]
        else:
            end_offset = wrapped_delta(self.position, self.end_position)
            draw_end_position = [pos[0] + end_offset[0], pos[1] + end_offset[1]]

        draw_start_position = outward_visual_laser_start(
            pos,
            draw_end_position,
            self.parent.sprites[visual_idx].get_height() / 2.0,
        )
        screen_start_x = int(
            (draw_start_position[0] + translation[0]) * scale_factor
        )
        screen_start_y = int(
            (draw_start_position[1] + translation[1]) * scale_factor
        )
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
                    self.draw_aa_laser(
                        screen,
                        self.LASER_COLOR,
                        (const.SCREEN_LEFT + start_x, start_y),
                        (const.SCREEN_LEFT + end_x, end_y),
                        max(1, int(self.LASER_WIDTH * scale_factor)),
                    )
