from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Shofixti.A1.ShofixtiA1 import ShofixtiA1
from src.Objects.Ships.Shofixti.A2.ShofixtiA2 import ShofixtiA2


class Shofixti(SpaceShip):
    action_factories = {1: ShofixtiA1}

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.shofixti_self_destruct = False

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        explosion = ShofixtiA2(self)
        side_effects = ()
        crew_change = 0
        if self.in_battle:
            crew_change = -self.current_hp
            side_effects = (self._mark_self_destruct,)
        return self.prepare_action_plan(
            2,
            explosion,
            crew_change=crew_change,
            side_effects=side_effects,
        )

    def _mark_self_destruct(self):
        self.shofixti_self_destruct = True
        self.destroy_boarded_marines()
