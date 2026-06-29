import math

import src.const as const
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Umgah.A1.UmgahA1 import UmgahA1
from src.Objects.Ships.Umgah.A2.UmgahA2 import UmgahA2


class Umgah(SpaceShip):
    action_factories = {1: UmgahA1}

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self._a1_animation_frame = 0
        self._reverse_burst_active = False
        self._pending_energy_regen = 0

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self._a1_animation_frame = 0
        self._reverse_burst_active = False
        self._pending_energy_regen = 0

    def take_a1_animation_frame(self):
        frame = self._a1_animation_frame
        self._a1_animation_frame = (frame + 1) % 3
        return frame

    def update_timers(self):
        energy_before = self.current_energy
        super().update_timers()
        self._pending_energy_regen = max(0, self.current_energy - energy_before)

    def process_controls(self, frame_id=None):
        try:
            return super().process_controls(frame_id)
        finally:
            self._pending_energy_regen = 0

    def _reset_energy_wait(self):
        if self._pending_energy_regen:
            self.current_energy -= self._pending_energy_regen
            self._pending_energy_regen = 0
        self.energy_timer = 0

    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)
        area = UmgahA1(self)
        return self.prepare_action_plan(
            1,
            area,
            side_effects=(self._reset_energy_wait,),
        )

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        burst = UmgahA2(self)
        return self.prepare_action_plan(
            2,
            burst,
            side_effects=(self._reset_energy_wait, self._activate_reverse_burst),
        )

    def _activate_reverse_burst(self):
        angle = math.radians(self.rotation)
        self.velocity = [
            -math.sin(angle) * const.SPEED_LIMIT,
            math.cos(angle) * const.SPEED_LIMIT,
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
