import math

import src.const as const
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Umgah.A1.UmgahA1 import UmgahA1
from src.Objects.Ships.Umgah.A2.UmgahA2 import UmgahA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS


class Umgah(SpaceShip):
    action_factories = {1: UmgahA1}
    ENERGY_WAIT_INDICATOR_COLOR = (255, 0, 0)
    ENERGY_WAIT_INDICATOR_NEGATIVE_COLOR = (0, 255, 0)

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self._a1_animation_frame = 0
        self._reverse_burst_active = False

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self._a1_animation_frame = 0
        self._reverse_burst_active = False

    @property
    def hud_indicator_color(self):
        return self.ENERGY_WAIT_INDICATOR_COLOR

    @property
    def hud_indicator_fraction(self):
        if self.current_energy >= self.max_energy:
            return 0.0
        if self.energy_wait <= 0:
            return 0.0
        remaining = self.energy_wait - self.energy_timer
        return max(0.0, min(1.0, remaining / self.energy_wait))

    @property
    def hud_indicator_negative_color(self):
        return self.ENERGY_WAIT_INDICATOR_NEGATIVE_COLOR

    @property
    def hud_indicator_size(self):
        return SHIP_DEFINITIONS[self.name].circle_size

    @property
    def hud_indicator_gap(self):
        return SHIP_DEFINITIONS[self.name].circle_gap

    def take_a1_animation_frame(self):
        frame = self._a1_animation_frame
        self._a1_animation_frame = (frame + 1) % 3
        return frame

    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)
        area = UmgahA1(self)
        return self.prepare_action_plan(
            1,
            area,
        )

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        burst = UmgahA2(self)
        return self.prepare_action_plan(
            2,
            burst,
            side_effects=(self._activate_reverse_burst,),
        )

    def can_action2(self):
        # UQM only starts retropropulsion when the ship's thrust wait has
        # expired. ``thrust_timer`` is StarAI's per-instance equivalent of
        # ELEMENT.thrust_wait.
        return self.thrust_timer == 0 and super().can_action2()

    def _activate_reverse_burst(self):
        angle = math.radians(self.rotation)
        speed = ABILITY_DEFINITIONS["UmgahA2"].backup_speed
        self.velocity = [
            -math.sin(angle) * speed,
            math.cos(angle) * speed,
        ]
        self.collision_velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self._reverse_burst_active = True

    def update_physics(self):
        if not self._reverse_burst_active:
            return super().update_physics()

        self.position[0] = (
            self.position[0] + self.velocity[0] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + self.velocity[1] * const.SPEED_SCALE
        ) % const.ARENA_SIZE
        self.accumulated_impulses = [0.0, 0.0]

    def on_elastic_bounce(self, other):
        if self._reverse_burst_active:
            self.velocity = [0.0, 0.0]

    def on_projectile_motion_contact(self, projectile, contact_position):
        if not self._reverse_burst_active:
            return
        if contact_position is not None:
            self.position = list(contact_position)
        self.velocity = [0.0, 0.0]

    def finalize_collision_frame(self):
        if not self._reverse_burst_active:
            return
        self.velocity = [0.0, 0.0]
        self.collision_velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        self._reverse_burst_active = False
