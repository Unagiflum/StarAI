from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.ZoqFot.A1.ZoqFotA1 import ZoqFotA1
from src.Objects.Ships.ZoqFot.A2.ZoqFotA2 import ZoqFotA2


class ZoqFot(SpaceShip):
    action_factories = {1: ZoqFotA1, 2: ZoqFotA2}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_a2 = None

    def plan_action2(self):
        if self.active_a2:
            if self.active_a2.currently_alive:
                return ActionPlan.invalid(2)
            else:
                self.active_a2 = None

        plan = super().plan_action2()
        if plan.valid and plan.spawned_objects:
            self.active_a2 = plan.spawned_objects[0]
        return plan
