import math

import pygame

import src.const as const
from src.Battle.collision_geometry import ship_shape_change_blocked
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Slylandro.A1.SlylandroA1 import SlylandroA1
from src.Objects.Ships.Slylandro.A2.SlylandroA2 import SlylandroA2
from src.Objects.object import PlayerObject
from src.toroidal import wrapped_distance


class Slylandro(SpaceShip):
    crew_bar_color = const.HUD_NONSENTIENT_HP_COLOR
    animation_phases = 32
    animation_rotation_step = 360 / animation_phases

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        # Probe rotation is an animation, not a movement heading. Keep its
        # 16 source phases and their video interpolation frames independent of
        # the number of gameplay headings.
        stride = const.DIRECTIONS_MULTIPLIER
        self.sprites = tuple(self.sprites[::stride])
        self.masks = tuple(self.masks[::stride])
        self.base_sprites = self.sprites
        self.animation_frame = 0
        self.animation_direction = None

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        if self.animation_direction is None:
            self.animation_direction = self.rng.choice((-1, 1))
        self.animation_frame = 0
        self._set_velocity_from_heading()

    def process_controls(self, frame_id=None):
        reverse = self.thrust_active and "thrust" in self.newly_pressed_controls
        saved_thrust = self.thrust_active
        if reverse:
            self.heading = (
                self.heading + const.SHIP_DIRECTIONS // 2
            ) % const.SHIP_DIRECTIONS
            self.rotation = self.heading * const.TURN_ANGLE

        # Slylandro's thrust control is a one-shot reversal, not acceleration.
        self.thrust_active = False
        try:
            objects = super().process_controls(frame_id)
        finally:
            self.thrust_active = saved_thrust
        self._set_velocity_from_heading()
        return objects

    def turn_left(self, max_steps=1):
        turned = super().turn_left(max_steps)
        if turned:
            self._set_velocity_from_heading()
        return turned

    def turn_right(self, max_steps=1):
        turned = super().turn_right(max_steps)
        if turned:
            self._set_velocity_from_heading()
        return turned

    def update(self):
        self.previous_position = self.position.copy()
        self.update_physics()
        next_phase = (
            self.animation_frame + self.animation_direction
        ) % self.animation_phases
        next_sprite = next_phase % len(self.masks)
        if not self._sprite_would_overlap(next_sprite):
            self.animation_frame = next_phase
            self.size = list(self._opaque_mask_size(self.masks[next_sprite]))
        return True

    def update_physics(self):
        candidate = (
            self.collision_velocity
            if self.collision_velocity != [0.0, 0.0]
            else self.velocity
        )
        candidate = [
            candidate[0] + self.accumulated_impulses[0],
            candidate[1] + self.accumulated_impulses[1],
        ]
        expected = self._heading_velocity()
        if (
            math.hypot(candidate[0], candidate[1]) > 0
            and math.hypot(candidate[0] - expected[0], candidate[1] - expected[1])
            > 1e-6
        ):
            angle = math.degrees(math.atan2(candidate[0], -candidate[1])) % 360
            self.heading = round(angle / const.TURN_ANGLE) % const.SHIP_DIRECTIONS
            self.rotation = self.heading * const.TURN_ANGLE

        self.collision_velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self._set_velocity_from_heading()
        PlayerObject.update_physics(self)
        self._set_velocity_from_heading()

    def on_elastic_bounce(self, other):
        self._quantize_heading_from_velocity(self.velocity)
        self._set_velocity_from_heading()

    def predict_unhindered_trajectory(self, frames=60):
        position = list(self.position)
        candidate = (
            self.collision_velocity
            if self.collision_velocity != [0.0, 0.0]
            else self.velocity
        )
        candidate = [
            candidate[0] + self.accumulated_impulses[0],
            candidate[1] + self.accumulated_impulses[1],
        ]
        if math.hypot(candidate[0], candidate[1]) > 0:
            angle = math.degrees(math.atan2(candidate[0], -candidate[1])) % 360
            heading = round(angle / const.TURN_ANGLE) % const.SHIP_DIRECTIONS
        else:
            heading = self.heading
        angle = math.radians(heading * const.TURN_ANGLE)
        velocity = [
            math.sin(angle) * self.max_thrust,
            -math.cos(angle) * self.max_thrust,
        ]

        trajectory = []
        for _ in range(frames):
            position = [
                (position[0] + velocity[0] * const.SPEED_SCALE)
                % const.ARENA_SIZE,
                (position[1] + velocity[1] * const.SPEED_SCALE)
                % const.ARENA_SIZE,
            ]
            trajectory.append(position)
        return trajectory

    def _heading_velocity(self):
        angle = math.radians(self.heading * const.TURN_ANGLE)
        return [
            math.sin(angle) * self.max_thrust,
            -math.cos(angle) * self.max_thrust,
        ]

    def _quantize_heading_from_velocity(self, velocity):
        if math.hypot(velocity[0], velocity[1]) <= 0:
            return
        angle = math.degrees(math.atan2(velocity[0], -velocity[1])) % 360
        self.heading = round(angle / const.TURN_ANGLE) % const.SHIP_DIRECTIONS
        self.rotation = self.heading * const.TURN_ANGLE

    def _set_velocity_from_heading(self):
        self.velocity = self._heading_velocity()

    def _sprite_would_overlap(self, frame):
        candidate_mask = self.masks[self._animation_asset_index(frame)]
        candidate_masks = tuple(candidate_mask for _ in self.masks)
        return ship_shape_change_blocked(
            self,
            candidate_masks,
            self._opaque_mask_size(candidate_mask),
        )

    @staticmethod
    def _opaque_mask_size(mask):
        bounds = mask.get_bounding_rects()
        if not bounds:
            return (0, 0)
        left = min(rect.left for rect in bounds)
        top = min(rect.top for rect in bounds)
        right = max(rect.right for rect in bounds)
        bottom = max(rect.bottom for rect in bounds)
        return (right - left, bottom - top)

    @classmethod
    def _limpet_phase_offset(cls, offset_x, offset_y, phase):
        angle = math.radians(phase * cls.animation_rotation_step)
        return (
            offset_x * math.cos(angle) - offset_y * math.sin(angle),
            offset_x * math.sin(angle) + offset_y * math.cos(angle),
        )

    def _attach_limpet_visual(self):
        visual = self._new_limpet_visual()
        if visual is None:
            return
        limpet_sprite, offset_x, offset_y = visual
        has_full_cycle = len(self.sprites) == self.animation_phases
        new_sprites = []

        for phase in range(self.animation_phases):
            source_index = (
                phase if has_full_cycle else self._animation_asset_index(phase)
            )
            current_sprite = self.sprites[source_index].copy()
            rotated_x, rotated_y = self._limpet_phase_offset(
                offset_x, offset_y, phase
            )
            angle = phase * self.animation_rotation_step
            rotated_limpet = pygame.transform.rotate(limpet_sprite, -angle)
            destination = rotated_limpet.get_rect(
                center=(
                    round(current_sprite.get_width() / 2 + rotated_x),
                    round(current_sprite.get_height() / 2 + rotated_y),
                )
            )
            current_sprite.blit(rotated_limpet, destination)
            new_sprites.append(current_sprite)

        self.sprites = tuple(new_sprites)

    def set_sprite(self, interp_t=0.0):
        if len(self.sprites) == self.animation_phases:
            return self.sprites[self.animation_frame]
        return self.sprites[self._animation_asset_index(self.animation_frame)]

    def get_collision_mask(self):
        return self.masks[self._animation_asset_index(self.animation_frame)]

    def _animation_asset_index(self, phase):
        return (
            (phase % const.ASSET_SPRITE_DIRECTIONS)
            * const.VIDEO_FPS_MULTIPLIER
        ) % len(self.masks)

    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)
        return self.prepare_action_plan(1, SlylandroA1(self))

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        ability_definition = ABILITY_DEFINITIONS["SlylandroA2"]
        if not any(
            asteroid.currently_alive
            and wrapped_distance(self.position, asteroid.position)
            <= ability_definition.range
            for asteroid in self.asteroids
        ):
            return ActionPlan.invalid(2)
        pulse = SlylandroA2(self)
        gain = ability_definition.battery_gain
        energy_change = min(gain, self.max_energy - self.current_energy)
        return self.prepare_action_plan(2, pulse, energy_change=energy_change)
