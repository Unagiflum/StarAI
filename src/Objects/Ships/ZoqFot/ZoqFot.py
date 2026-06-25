from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.ZoqFot.A1.ZoqFotA1 import ZoqFotA1
from src.Objects.Ships.ZoqFot.A2.ZoqFotA2 import ZoqFotA2


class ZoqFot(SpaceShip):
    action_factories = {1: ZoqFotA1, 2: ZoqFotA2}

    def plan_action2(self):
        active_a2 = getattr(self, 'active_a2', None)
        if active_a2 and active_a2.currently_alive:
            return ActionPlan.invalid(2)
            
        plan = super().plan_action2()
        if plan.valid and plan.spawned_objects:
            self.active_a2 = plan.spawned_objects[0]
        return plan
