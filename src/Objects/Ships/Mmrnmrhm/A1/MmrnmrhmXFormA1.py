import math

import pygame

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.launch_geometry import gun_world_position
from src.toroidal import wrapped_delta


class _LaserVolley:
    def __init__(self):
        self._damaged_target_ids = set()

    def claim_damage(self, target):
        target_id = id(target)
        if target_id in self._damaged_target_ids:
            return False
        self._damaged_target_ids.add(target_id)
        return True


class MmrnmrhmXFormA1(Ability):
    def __init__(self, parent, gun_location=None, volley=None):
        super().__init__("MmrnmrhmXFormA1", parent)
        definition = ABILITY_DEFINITIONS["MmrnmrhmXFormA1"]
        self.LASER_RANGE = definition.laser_range
        self.configure_laser_colors(definition.laser_color)
        self.LASER_WIDTH = definition.laser_width
        self.gun_location = gun_location or definition.gun_locations[0]
        self.volley = volley or _LaserVolley()
        self.start_position = parent.position.copy()
        self.end_position = parent.position.copy()
        self.calculate_end_position()

    @classmethod
    def create_beams(cls, ship):
        locations = ABILITY_DEFINITIONS["MmrnmrhmXFormA1"].gun_locations or ()
        volley = _LaserVolley()
        return [cls(ship, location, volley) for location in locations]

    def calculate_end_position(self):
        self.start_position = gun_world_position(
            self.parent, self.gun_location
        )
        self.position = self.start_position.copy()
        angle = math.radians(self.parent.rotation)
        self.end_position = [
            self.parent.position[0] + math.sin(angle) * self.LASER_RANGE,
            self.parent.position[1] - math.cos(angle) * self.LASER_RANGE,
        ]
        self.heading = self.parent.heading
        self.rotation = self.parent.rotation

    def should_damage_target(self, target):
        return self.volley.claim_damage(target)

    def visual_end_position(self, start, parent_position, rotation):
        angle = math.radians(rotation)
        raw_end = [
            parent_position[0] + math.sin(angle) * self.LASER_RANGE,
            parent_position[1] - math.cos(angle) * self.LASER_RANGE,
        ]
        offset = wrapped_delta(start, raw_end)
        return [start[0] + offset[0], start[1] + offset[1]]

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        from src.Battle.interpolation import (
            interpolated_position,
            interpolated_sprite_index,
        )

        parent_position = interpolated_position(self.parent, interp_t)
        visual_index = interpolated_sprite_index(self.parent, interp_t)
        rotation = visual_index / const.VIDEO_FPS_MULTIPLIER * const.TURN_ANGLE
        start = gun_world_position(
            self.parent,
            self.gun_location,
            rotation=rotation,
            position=parent_position,
        )

        if getattr(self, "intercepted", False):
            target = getattr(self, "attached_target", None)
            contact_offset = getattr(self, "target_contact_offset", None)
            if target is not None and contact_offset is not None:
                target_position = interpolated_position(target, interp_t)
                raw_end = [
                    target_position[0] + contact_offset[0],
                    target_position[1] + contact_offset[1],
                ]
                offset = wrapped_delta(start, raw_end)
            else:
                offset = wrapped_delta(self.start_position, self.end_position)
            end = [start[0] + offset[0], start[1] + offset[1]]
        else:
            end = self.visual_end_position(start, parent_position, rotation)

        screen_start = (
            int((start[0] + translation[0]) * scale_factor),
            int((start[1] + translation[1]) * scale_factor),
        )
        screen_end = (
            int((end[0] + translation[0]) * scale_factor),
            int((end[1] + translation[1]) * scale_factor),
        )
        view_rect = pygame.Rect(0, 0, const.SCREEN_HEIGHT, const.SCREEN_HEIGHT)
        for wrap_x in (-1, 0, 1):
            for wrap_y in (-1, 0, 1):
                dx = wrap_x * const.ARENA_SIZE * scale_factor
                dy = wrap_y * const.ARENA_SIZE * scale_factor
                draw_start = (screen_start[0] + dx, screen_start[1] + dy)
                draw_end = (screen_end[0] + dx, screen_end[1] + dy)
                if view_rect.clipline((*draw_start, *draw_end)):
                    self.draw_aa_laser(
                        screen,
                        self.LASER_COLOR,
                        (const.SCREEN_LEFT + draw_start[0], draw_start[1]),
                        (const.SCREEN_LEFT + draw_end[0], draw_end[1]),
                        max(1, int(self.LASER_WIDTH * scale_factor)),
                    )
