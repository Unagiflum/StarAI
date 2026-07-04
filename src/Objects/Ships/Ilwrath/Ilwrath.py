from dataclasses import replace
import math

import pygame

import src.const as const
from src.audio import compatibility_audio_service
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.Ilwrath.A1.IlwrathA1 import IlwrathA1
from src.Objects.Ships.Ilwrath.A2.IlwrathA2 import IlwrathA2
from src.Objects.Ships.space_ship import SpaceShip
from src.toroidal import wrapped_delta


class Ilwrath(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.ship_name = ship_name

        a2_def = ABILITY_DEFINITIONS.get(f"{ship_name}A2")
        self.FADE_DURATION = (
            a2_def.fade_duration
            if a2_def and a2_def.fade_duration is not None
            else 5
        )
        # fade_timer is the current blackness level: zero is visible and
        # FADE_DURATION is fully cloaked. fade_direction is -1, 0, or 1.
        self.fade_timer = 0
        self.previous_fade_timer = 0
        self.fade_direction = 0

        audio = self.audio_service or compatibility_audio_service(
            Ability.sound_enabled, self.resources
        )
        self._uncloak_sound = audio.load_effect(
            self.sprite_location / "A2" / "IlwrathA2end.wav",
            const.SOUND_EFFECT_VOLUME,
        )
        self.black_sprites = self.resources.black_ship_sprites(ship_name)
        self.uncloak_look_ahead = (
            a2_def.look_ahead if a2_def and a2_def.look_ahead is not None else 0
        )

    def control_ready(self, control_name, frame_id):
        # Cloaking has no cooldown, so make the toggle edge-triggered instead of
        # repeatedly reversing it while A2 is held.
        if control_name == "action2" and frame_id is not None:
            return control_name in self.newly_pressed_controls
        return super().control_ready(control_name, frame_id)

    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)

        facing = self._opponent_facing() if self.cloaked else None
        flame = self._construct_primary_at_facing(facing)

        side_effects = ()
        if self.fade_timer > 0 or self.fade_direction > 0:

            def uncloak_for_attack():
                if facing is not None:
                    self.heading, self.rotation = facing
                    self.previous_heading = self.heading
                self.uncloak()

            side_effects = (uncloak_for_attack,)

        return self.prepare_action_plan(1, flame, side_effects=side_effects)

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)

        if self.fade_timer > 0 or self.fade_direction > 0:
            plan = self.prepare_action_plan(
                2,
                energy_change=0,
                resets_energy_wait=False,
                side_effects=(self.uncloak,),
                launch_sound=self._uncloak_sound,
                use_first_object_sound=False,
            )
        else:
            cloak_effect = IlwrathA2(self)
            plan = self.prepare_action_plan(
                2,
                side_effects=(self.cloak,),
                launch_sound=cloak_effect.launch_sound,
                use_first_object_sound=False,
            )

        return replace(plan, cooldown_frames=0, cooldown_committed=False)

    def _construct_primary_at_facing(self, facing):
        if facing is None:
            return IlwrathA1(self)
        old_facing = self.heading, self.rotation
        self.heading, self.rotation = facing
        try:
            return IlwrathA1(self)
        finally:
            self.heading, self.rotation = old_facing

    def face_opponent(self):
        facing = self._opponent_facing()
        if facing is not None:
            self.heading, self.rotation = facing
            self.previous_heading = self.heading

    def _opponent_facing(self):
        if not self.opponent or not self.opponent.trackable:
            return None

        source_pos = self.position
        target_pos = self.opponent.position
        if self.uncloak_look_ahead > 0:
            target_trajectory = self.opponent.predict_unhindered_trajectory(
                frames=self.uncloak_look_ahead
            )
            source_trajectory = self.predict_unhindered_trajectory(
                frames=self.uncloak_look_ahead
            )
            if target_trajectory:
                target_pos = target_trajectory[-1]
            if source_trajectory:
                source_pos = source_trajectory[-1]

        dx, dy = wrapped_delta(source_pos, target_pos)
        target_angle = math.degrees(math.atan2(dx, -dy)) % 360
        direction_step = 360 / const.SHIP_DIRECTIONS
        heading = int(target_angle / direction_step + 0.5) % const.SHIP_DIRECTIONS
        return heading, heading * const.TURN_ANGLE

    def can_action2(self):
        if self.is_confused:
            return False
        if self.fade_timer > 0 or self.fade_direction > 0:
            return True
        return self.current_energy >= self.a2_cost

    def cloak(self):
        if self.fade_timer >= self.FADE_DURATION:
            self.fade_direction = 0
            self.cloaked = True
            self.trackable = False
            return
        self.fade_direction = 1
        self.cloaked = False
        self.trackable = True

    def uncloak(self):
        self.fade_direction = -1 if self.fade_timer > 0 else 0
        self.cloaked = False
        self.trackable = True

    def update(self):
        self.previous_fade_timer = self.fade_timer
        if self.fade_direction:
            self.fade_timer = max(
                0,
                min(self.FADE_DURATION, self.fade_timer + self.fade_direction),
            )
            if self.fade_timer >= self.FADE_DURATION:
                self.fade_direction = 0
                self.cloaked = True
                self.trackable = False
            elif self.fade_timer <= 0:
                self.fade_direction = 0
                self.cloaked = False
                self.trackable = True
        return super().update()

    def set_sprite(self, interp_t=0.0):
        from src.Battle.interpolation import interpolated_sprite_index

        sprite_idx = interpolated_sprite_index(self, interp_t)
        black_sprite = self.black_sprites[sprite_idx]
        normal_sprite = self.sprites[sprite_idx]
        fade_timer = (
            self.previous_fade_timer
            + (self.fade_timer - self.previous_fade_timer) * interp_t
        )
        progress = fade_timer / self.FADE_DURATION
        if progress <= 0:
            return normal_sprite
        if progress >= 1:
            return black_sprite

        final_sprite = pygame.Surface(normal_sprite.get_size(), pygame.SRCALPHA)
        normal_copy = normal_sprite.copy()
        black_copy = black_sprite.copy()
        normal_copy.set_alpha(int(255 * (1 - progress)))
        black_copy.set_alpha(int(255 * progress))
        final_sprite.blit(normal_copy, (0, 0))
        final_sprite.blit(black_copy, (0, 0))
        return final_sprite
