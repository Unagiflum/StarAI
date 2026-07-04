from dataclasses import replace

from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Yehat.A1.YehatA1 import YehatA1
from src.Objects.Ships.Yehat.A2.YehatA2 import YehatA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class Yehat(SpaceShip):
    def control_ready(self, control_name, frame_id):
        if control_name == "action2":
            # Yehat's special counter is evaluated every frame in UQM.
            return True
        return super().control_ready(control_name, frame_id)

    def plan_action1(self):
        definition = ABILITY_DEFINITIONS["YehatA1"]
        return self.validate_action(
            1,
            lambda ship: [
                YehatA1(ship, location, direction)
                for location, direction in zip(
                    definition.gun_locations, definition.gun_directions
                )
            ],
        )

    def plan_action2(self):
        if not self.can_action2():
            return self.validate_action(2)

        shield = YehatA2(self)
        plan = self.prepare_action_plan(
            2,
            shield,
            side_effects=(shield.activate,),
        )
        # UQM sets SPECIAL_WAIT during preprocess, then decrements it in the
        # same frame. A value of 2 therefore permits refresh two frames later.
        return replace(plan, cooldown_frames=round(self.a2_wait))

    def set_sprite(self, interp_t=0.0):
        if self.damage_shield_is_active():
            from src.Battle.interpolation import interpolated_sprite_index

            return self._active_damage_shield.sprites[
                interpolated_sprite_index(self, interp_t)
            ]
        return super().set_sprite(interp_t)
