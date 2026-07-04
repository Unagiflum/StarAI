from dataclasses import dataclass
import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.toroidal import wrapped_delta


@dataclass
class LightningSession:
    suppressed: bool = False


class SlylandroA1(Ability):
    """One independently colored bolt in a shared lightning discharge."""

    DIRECTION_COUNT = const.ASSET_SPRITE_DIRECTIONS
    DIRECTION_ANGLE = 360 / DIRECTION_COUNT

    def __init__(self, parent, session=None, bolt_index=0):
        super().__init__("SlylandroA1", parent)
        definition = ABILITY_DEFINITIONS["SlylandroA1"]
        self.session = session or LightningSession()
        self.bolt_index = bolt_index
        self.cooldown = round(parent.a1_wait)
        self.segment_length_min = definition.segment_length_min
        self.segment_length_max = definition.segment_length_max
        self.bolt_colors = definition.colors
        self.LASER_WIDTH = definition.stroke_width or 1
        self.color = self.bolt_colors[0]
        self.points = []
        self.frame_number = 0
        self.position = self.configured_gun_position()
        self.previous_position = self.position.copy()
        self.start_position = self.position.copy()
        self.end_position = self.position.copy()

    @staticmethod
    def segment_count(frame_number, cooldown):
        if frame_number <= 0 or frame_number >= cooldown:
            return 0
        return min(frame_number + 1, cooldown - frame_number + 1)

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.position = self.configured_gun_position()
        self.frame_number += 1
        if self.frame_number >= self.cooldown:
            self.points = []
            self.currently_alive = False
            return False

        if self.session.suppressed:
            self.points = []
        else:
            self._generate_bolt(
                self.segment_count(self.frame_number, self.cooldown)
            )
        return True

    def _generate_bolt(self, segment_count):
        self.color = self.rng.choice(self.bolt_colors)
        self.intercepted = False
        self.attached_target = None
        self.target_contact_offset = None
        point = self.configured_gun_position()
        self.points = [point]
        target_available = self._target_is_available()
        relative_direction = self.configured_gun()[1] or 0
        direction = round(
            (self.parent.rotation + relative_direction) / self.DIRECTION_ANGLE
        ) % self.DIRECTION_COUNT

        for segment_index in range(segment_count):
            if not target_available:
                direction = self.rng.randrange(self.DIRECTION_COUNT)
            elif segment_index > 0:
                direction = self._next_direction(direction, point)

            length = self.rng.uniform(
                self.segment_length_min, self.segment_length_max
            )
            angle = math.radians(direction * self.DIRECTION_ANGLE)
            point = [
                (point[0] + math.sin(angle) * length) % const.ARENA_SIZE,
                (point[1] - math.cos(angle) * length) % const.ARENA_SIZE,
            ]
            self.points.append(point)

        self.calculate_end_position()

    def _target_is_available(self):
        target = self.opponent
        return bool(
            target is not None
            and target.currently_alive
            and target.current_hp > 0
            and target.trackable
        )

    def _next_direction(self, previous_direction, point):
        dx, dy = wrapped_delta(point, self.opponent.position)
        target_angle = math.degrees(math.atan2(dx, -dy)) % 360
        target_direction = (
            round(target_angle / self.DIRECTION_ANGLE) % self.DIRECTION_COUNT
        )
        half_turn = self.DIRECTION_COUNT // 2
        difference = (
            (target_direction - previous_direction + half_turn)
            % self.DIRECTION_COUNT
        ) - half_turn

        if abs(difference) > 1:
            turn = min(3, abs(difference))
            turn *= 1 if difference > 0 else -1
        else:
            turn = self.rng.choice((-1, 1))
        return (previous_direction + turn) % self.DIRECTION_COUNT

    def calculate_end_position(self):
        if len(self.points) < 2:
            self.start_position = self.configured_gun_position()
            self.end_position = self.start_position.copy()
            return
        self.start_position = self.points[0]
        self.end_position = self.points[-1]

    def collision_segments(self):
        return tuple(zip(self.points, self.points[1:]))

    def should_consider_laser_target(self, target):
        return not (
            target is self.parent
            or getattr(target, "player", None) == self.parent.player
        )

    def on_laser_hit(self, target, contact, segment_index):
        self.session.suppressed = True
        if segment_index is None or not self.points:
            return
        self.points = self.points[: segment_index + 1] + [list(contact)]
        self.calculate_end_position()

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        if len(self.points) < 2:
            return

        from src.Battle.interpolation import (
            interpolated_position,
            interpolated_sprite_index,
        )

        parent_position = interpolated_position(self.parent, interp_t)
        visual_idx = interpolated_sprite_index(self.parent, interp_t)
        visual_rotation = (
            visual_idx / const.VIDEO_FPS_MULTIPLIER * const.TURN_ANGLE
        )
        gun_position = self.configured_gun_position(
            rotation=visual_rotation, position=parent_position
        )
        segments = list(self.collision_segments())
        segments[0] = (gun_position, segments[0][1])
        view_rect = pygame.Rect(0, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)

        for start, end in segments:
            delta = wrapped_delta(start, end)
            unwrapped_end = [start[0] + delta[0], start[1] + delta[1]]
            start_x = int((start[0] + translation[0]) * scale_factor)
            start_y = int((start[1] + translation[1]) * scale_factor)
            end_x = int((unwrapped_end[0] + translation[0]) * scale_factor)
            end_y = int((unwrapped_end[1] + translation[1]) * scale_factor)

            for wrap_x in (-1, 0, 1):
                for wrap_y in (-1, 0, 1):
                    arena_x = wrap_x * const.ARENA_SIZE * scale_factor
                    arena_y = wrap_y * const.ARENA_SIZE * scale_factor
                    line = (
                        start_x + arena_x,
                        start_y + arena_y,
                        end_x + arena_x,
                        end_y + arena_y,
                    )
                    if view_rect.clipline(line):
                        self.draw_aa_laser(
                            screen,
                            self.color,
                            (const.SCREEN_LEFT + line[0], line[1]),
                            (const.SCREEN_LEFT + line[2], line[3]),
                            max(1, int(self.LASER_WIDTH * scale_factor)),
                        )
