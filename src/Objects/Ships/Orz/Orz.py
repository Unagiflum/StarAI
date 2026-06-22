from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Orz.A1.OrzA1 import OrzA1
from src.Objects.Ships.Orz.A2.OrzA2 import OrzA2

class Orz(SpaceShip):
    action_factories = {1: OrzA1}

    def __init__(self, ship_name, player_num, resources=None, audio_service=None):
        super().__init__(ship_name, player_num, resources, audio_service)
        self.turret = OrzA2(self)
        self._turret_composites = {}

    @property
    def turret_heading(self):
        return self.turret.absolute_heading

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
        return self.validate_action(3)

    def handles_combined_action(self):
        return True

    def set_sprite(self):
        key = (self.heading, self.turret_heading)
        if key not in self._turret_composites:
            sprite = self.sprites[self.heading].copy()
            turret_sprite = self.turret.get_sprite()
            turret_rect = turret_sprite.get_rect(center=sprite.get_rect().center)
            sprite.blit(turret_sprite, turret_rect)
            self._turret_composites[key] = sprite
        return self._turret_composites[key]
