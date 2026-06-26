from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Orz.A1.OrzA1 import OrzA1
from src.Objects.Ships.Orz.A2.OrzA2 import OrzA2
from src.Objects.Ships.Orz.A3.OrzA3 import OrzA3
from src.Objects.Ships.space_ship import SpaceShip
from src.resources import centered_overlay


class Orz(SpaceShip):
    action_factories = {1: OrzA1}

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.turret = OrzA2(self)
        self._turret_composites = {}
        self.active_marines = []

    @property
    def turret_heading(self):
        return self.turret.absolute_heading

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.turret.reset()

    def plan_action1(self):
        if self.action2_active:
            return ActionPlan.invalid(1)
        return self.validate_action(1, self.action_factories[1])

    def plan_action2(self):
        turn_left = self.turn_left_active
        turn_right = self.turn_right_active
        if not self.can_action2() or turn_left == turn_right:
            return ActionPlan.invalid(2)
        direction = 1 if turn_right else -1
        return self.prepare_action_plan(
            2,
            side_effects=(lambda: self.turret.turn(direction),),
            use_first_object_sound=False,
        )

    def turn_input_enabled(self):
        # A2 changes the direction keys from hull steering to turret steering.
        return not self.action2_active

    def plan_action3(self):
        self.active_marines = [
            marine for marine in self.active_marines
            if marine.currently_alive
        ]
        opponent = self.opponent
        if opponent is None:
            # Preserve the existing combined-input cooldown outside a bound
            # battle (menus and characterization tests have no opponent).
            return self.validate_action(3)
        if (
            not self.can_action3()
            or self.current_hp <= 1
            or len(self.active_marines) >= OrzA3.MAX_MARINES
            or not opponent.currently_alive
            or opponent.current_hp <= 0
            or not opponent.trackable
        ):
            return ActionPlan.invalid(3)

        marine = OrzA3(self)
        return self.prepare_action_plan(
            3,
            marine,
            crew_change=-1,
            side_effects=(lambda: self.active_marines.append(marine),),
        )

    def handles_combined_action(self):
        return True

    def set_sprite(self, interp_t=0.0):
        from src.Battle.interpolation import interpolated_sprite_index
        sprite_idx = interpolated_sprite_index(self, interp_t)
        turret_sprite = self.turret.get_sprite(interp_t)
        key = (sprite_idx, id(turret_sprite))
        if key not in self._turret_composites:
            self._turret_composites[key] = centered_overlay(
                self.sprites[sprite_idx],
                turret_sprite,
            )
        return self._turret_composites[key]
