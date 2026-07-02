from dataclasses import replace

import src.const as const
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Arilou.A1.ArilouA1 import ArilouA1
from src.Objects.Ships.Arilou.A2.ArilouA2 import ArilouA2
from src.Objects.Ships.space_ship import SpaceShip


class Arilou(SpaceShip):
    action_factories = {1: ArilouA1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teleport_frame = 0
        self.teleport_destination = None
        self._teleport_just_started = False
        self._pre_teleport_physics = None

    def initialize_in_battle(self, position, heading):
        if self._pre_teleport_physics is not None:
            self.physical_collision_capabilities = self._pre_teleport_physics
        self.teleport_frame = 0
        self.teleport_destination = None
        self._teleport_just_started = False
        self._pre_teleport_physics = None
        super().initialize_in_battle(position, heading)

    @property
    def ability_actions_paused(self):
        return self.teleport_frame > 0

    @property
    def camera_position(self):
        if self.teleport_frame in (1, 2):
            return self.frozen_camera_position
        return self.position

    def _choose_teleport_destination(self):
        old_position = self.position.copy()
        destination = old_position
        for _ in range(100):
            candidate = [
                self.rng.randint(0, const.ARENA_SIZE),
                self.rng.randint(0, const.ARENA_SIZE),
            ]
            destination = candidate
            self.position = candidate
            if not self.rotation_would_overlap():
                break
        self.position = old_position
        return destination

    def _begin_teleport(self, effect):
        self.frozen_camera_position = self.position.copy()
        self.teleport_destination = self._choose_teleport_destination()
        self.teleport_frame = 1
        self._teleport_just_started = True
        self._pre_teleport_physics = self.physical_collision_capabilities
        self.physical_collision_capabilities = replace(
            self.physical_collision_capabilities,
            is_solid=False,
            is_intangible=True,
        )
        self.velocity = [0.0, 0.0]
        self.collision_velocity = [0.0, 0.0]
        self.accumulated_impulses = [0.0, 0.0]
        effect.begin(self.teleport_destination)

    def _finish_teleport(self):
        self.teleport_frame = 0
        self.teleport_destination = None
        self.physical_collision_capabilities = self._pre_teleport_physics
        self._pre_teleport_physics = None

    def process_controls(self, frame_id=None):
        if self.ability_actions_paused:
            # Suppress commands without losing held keys or their repeat timing.
            return []

        return super().process_controls(frame_id)

    def action2_cancels_other_commands(self):
        return True

    def update_timers(self):
        if not self.ability_actions_paused:
            super().update_timers()

    def update(self):
        if not self.ability_actions_paused:
            return super().update()

        self.previous_position = self.position.copy()
        if self._teleport_just_started:
            self._teleport_just_started = False
        elif self.teleport_frame == 2:
            self.position = self.teleport_destination.copy()
            self.previous_position = self.position.copy()
            self.teleport_frame = 3
        elif self.teleport_frame == 4:
            self._finish_teleport()
        else:
            self.teleport_frame += 1
        return True

    def take_damage(self, damage, *, shieldable=True, non_lethal=False):
        if self.ability_actions_paused:
            return 0
        return super().take_damage(
            damage,
            shieldable=shieldable,
            non_lethal=non_lethal,
        )

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        if not self.ability_actions_paused:
            super().draw(screen, scale_factor, translation, interp_t=interp_t)

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        ability_obj = ArilouA2(self)
        return self.prepare_action_plan(
            2,
            ability_obj,
            side_effects=(lambda: self._begin_teleport(ability_obj),),
        )
