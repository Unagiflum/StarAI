from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.Ilwrath.A1.IlwrathA1 import IlwrathA1
from src.Objects.Ships.Ilwrath.A2.IlwrathA2 import IlwrathA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
import src.const as const
import math
import pygame
from src.toroidal import wrapped_delta
from src.audio import compatibility_audio_service


class Ilwrath(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        ship_data = SHIPS_DATA[ship_name]
        self.FADE_DURATION = ship_data.get("FADE_DURATION", 8)
        self.fade_timer = self.FADE_DURATION
        self.previous_fade_timer = self.FADE_DURATION
        self.ship_name = ship_name

        audio = self.audio_service or compatibility_audio_service(
            Ability.sound_enabled, self.resources
        )
        self._uncloak_sound = audio.load_effect(
            self.sprite_location / "A2" / "IlwrathA2end.wav",
            const.SOUND_EFFECT_VOLUME,
        )
        self.black_sprites = self.resources.black_ship_sprites(ship_name)
        
        a2_def = ABILITY_DEFINITIONS.get(f"{ship_name}A2")
        self.uncloak_look_ahead = a2_def.look_ahead if a2_def and a2_def.look_ahead is not None else 0

    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)

        facing = None
        if self.cloaked and self.fade_timer >= self.FADE_DURATION:
            facing = self._opponent_facing()
        flame = self._construct_primary_at_facing(facing)

        side_effects = ()
        if self.cloaked:
            def uncloak_for_attack():
                if facing is not None:
                    self.heading, self.rotation = facing
                self.uncloak()
            side_effects = (uncloak_for_attack,)

        return self.prepare_action_plan(1, flame, side_effects=side_effects)

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        if self.cloaked:
            return self.prepare_action_plan(
                2,
                energy_change=0,
                side_effects=(self.uncloak,),
                launch_sound=self._uncloak_sound,
                use_first_object_sound=False,
            )

        cloak_effect = IlwrathA2(self)
        return self.prepare_action_plan(
            2,
            side_effects=(self.cloak,),
            launch_sound=cloak_effect.launch_sound,
            use_first_object_sound=False,
        )

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

    def _opponent_facing(self):
        if not self.opponent or not self.opponent.trackable:
            return None
            
        target_pos = self.opponent.position
        if self.uncloak_look_ahead > 0:
            t_traj = self.opponent.predict_unhindered_trajectory(frames=self.uncloak_look_ahead)
            if t_traj:
                target_pos = t_traj[-1]
                
        dx, dy = wrapped_delta(self.position, target_pos)
        target_angle = math.degrees(math.atan2(dx, -dy))
        if target_angle < 0:
            target_angle += 360
        direction_step = 360 / const.SHIP_DIRECTIONS
        heading = int(target_angle / direction_step) % const.SHIP_DIRECTIONS
        return heading, heading * const.TURN_ANGLE


    def can_action2(self):
        if self.cloaked:
            return self.action2_timer == 0
        else:
            return self.action2_timer == 0 and self.current_energy >= self.a2_cost

    def cloak(self):
        self.cloaked = True
        self.trackable = False
        self.fade_timer = 0

    def uncloak(self):
        self.cloaked = False
        self.trackable = True

    def update(self):
        self.previous_fade_timer = getattr(self, "fade_timer", self.FADE_DURATION)
        if self.fade_timer < self.FADE_DURATION:
            self.fade_timer += 1
        return super().update()

    def set_sprite(self, interp_t=0.0):
        from src.Battle.interpolation import interpolated_sprite_index
        sprite_idx = interpolated_sprite_index(self, interp_t)
        black_sprite = self.black_sprites[sprite_idx]
        normal_sprite = self.sprites[sprite_idx]

        # If we're still within the fade timer, do a fade transition; otherwise pick final
        fade_timer = self.previous_fade_timer + (self.fade_timer - self.previous_fade_timer) * interp_t
        if fade_timer < self.FADE_DURATION:
            progress = fade_timer / self.FADE_DURATION
            final_sprite = pygame.Surface(normal_sprite.get_size(), pygame.SRCALPHA)

            if self.cloaked:
                # Fade from normal → black
                normal_copy = normal_sprite.copy()
                black_copy = black_sprite.copy()
                alpha_normal = int(255 * (1 - progress))
                alpha_black = int(255 * progress)
                normal_copy.set_alpha(alpha_normal)
                black_copy.set_alpha(alpha_black)
                final_sprite.blit(normal_copy, (0, 0))
                final_sprite.blit(black_copy, (0, 0))
            else:
                final_sprite = normal_sprite
        else:
            final_sprite = black_sprite if self.cloaked else normal_sprite
        return final_sprite
