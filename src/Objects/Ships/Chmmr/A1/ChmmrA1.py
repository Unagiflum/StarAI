import math

import pygame

import src.const as const
from src.Battle.effects import BattleEffect
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.toroidal import wrapped_delta


class ChmmrA1Spark(BattleEffect):
    """A world-fixed spark animation created once per firing frame."""

    render_layer = "after_lasers"

    def __init__(self, parent, position):
        directory = const.source_path("Objects/Ships/Chmmr/A1")
        frames = parent.resources.animation(
            "ChmmrA1-sparks",
            tuple(directory / f"ChmmrA1Sparks{index:02d}.png" for index in range(16)),
            interpolated=True,
        )
        super().__init__(
            position,
            frames,
            frame_delay=1,
            video_multiplier=const.VIDEO_FPS_MULTIPLIER,
        )


class ChmmrA1(Ability):
    def __init__(self, parent):
        super().__init__("ChmmrA1", parent)
        definition = ABILITY_DEFINITIONS["ChmmrA1"]
        self.LASER_RANGE = definition.laser_range
        self.LASER_COLOR = definition.laser_color
        self.LASER_WIDTH = definition.laser_width
        self.end_position = [0.0, 0.0]
        self._spawned_objects = []

        self.place_self()
        self.calculate_end_position()
        self._create_spark()
        self._load_muzzle_anchor()

    def _load_muzzle_anchor(self):
        anchor_sprite = self.sprites[0]
        mask = pygame.mask.from_surface(anchor_sprite)
        bounds = mask.get_bounding_rects()
        if not bounds:
            anchor = anchor_sprite.get_rect().center
        else:
            bottom = max(rect.bottom for rect in bounds) - 1
            opaque_x = [
                x for x in range(anchor_sprite.get_width()) if mask.get_at((x, bottom))
            ]
            center_x = anchor_sprite.get_rect().centerx
            anchor = min(opaque_x, key=lambda x: (abs(x - center_x), x)), bottom
        center = anchor_sprite.get_rect().center
        self._muzzle_anchor_offset = (anchor[0] - center[0], anchor[1] - center[1])

    def place_self(self):
        self.start_position = self.configured_gun_position()
        self.position = self.start_position.copy()
        relative_direction = self.configured_gun()[1] or 0
        self.rotation = (self.parent.rotation + relative_direction) % 360
        self.heading = round(self.rotation / const.TURN_ANGLE) % const.SHIP_DIRECTIONS

    def calculate_end_position(self):
        self.start_position = self.configured_gun_position()
        self.position = self.start_position.copy()
        angle = math.radians(self.rotation)
        self.end_position = [
            self.start_position[0] + math.sin(angle) * self.LASER_RANGE,
            self.start_position[1] - math.cos(angle) * self.LASER_RANGE,
        ]

    def _create_spark(self):
        distance = self.rng.uniform(0, self.LASER_RANGE)
        angle = math.radians(self.rotation)
        position = [
            (self.start_position[0] + math.sin(angle) * distance) % const.ARENA_SIZE,
            (self.start_position[1] - math.cos(angle) * distance) % const.ARENA_SIZE,
        ]
        self._spawned_objects.append(ChmmrA1Spark(self.parent, position))

    def drain_spawned_objects(self):
        spawned, self._spawned_objects = self._spawned_objects, []
        return spawned

    def should_consider_laser_target(self, target):
        return not (
            getattr(target, "name", None) == "ChmmrSatellite"
            and getattr(target, "parent", None) is self.parent
        )

    def update(self):
        if not self.currently_alive:
            return False
        self.expiration_timer -= 1
        return self.expiration_timer >= 0

    def _visual_geometry(self, interp_t):
        from src.Battle.interpolation import interpolated_position, interpolated_sprite_index

        parent_position = interpolated_position(self.parent, interp_t)
        sprite_index = interpolated_sprite_index(self.parent, interp_t)
        visual_rotation = sprite_index / const.VIDEO_FPS_MULTIPLIER * const.TURN_ANGLE
        relative_direction = self.configured_gun()[1] or 0
        rotation = visual_rotation + relative_direction
        start = self.configured_gun_position(
            rotation=visual_rotation,
            position=parent_position,
        )
        if getattr(self, "intercepted", False):
            end_delta = wrapped_delta(self.start_position, self.end_position)
            end = [start[0] + end_delta[0], start[1] + end_delta[1]]
        else:
            angle = math.radians(rotation)
            end = [
                start[0] + math.sin(angle) * self.LASER_RANGE,
                start[1] - math.cos(angle) * self.LASER_RANGE,
            ]
        return start, end, rotation, sprite_index

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        start, end, _, _ = self._visual_geometry(interp_t)
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

    def draw_foreground(self, screen, scale_factor, translation, interp_t=0.0):
        start, _, rotation, sprite_index = self._visual_geometry(interp_t)
        sprite = self.sprites[sprite_index]
        angle = math.radians(rotation)
        anchor_x, anchor_y = self._muzzle_anchor_offset
        rotated_anchor = (
            math.cos(angle) * anchor_x - math.sin(angle) * anchor_y,
            math.sin(angle) * anchor_x + math.cos(angle) * anchor_y,
        )
        center = [start[0] - rotated_anchor[0], start[1] - rotated_anchor[1]]
        scaled = pygame.transform.smoothscale_by(sprite, scale_factor)
        rect = scaled.get_rect()
        screen_x = int((center[0] + translation[0]) * scale_factor)
        screen_y = int((center[1] + translation[1]) * scale_factor)
        for wrap_x in (-1, 0, 1):
            for wrap_y in (-1, 0, 1):
                x = screen_x + wrap_x * const.ARENA_SIZE * scale_factor
                y = screen_y + wrap_y * const.ARENA_SIZE * scale_factor
                if (
                    -rect.width <= x <= const.SCREEN_HEIGHT + rect.width
                    and -rect.height <= y <= const.SCREEN_HEIGHT + rect.height
                ):
                    screen.blit(
                        scaled,
                        (const.SCREEN_LEFT + x - rect.width // 2, y - rect.height // 2),
                    )
