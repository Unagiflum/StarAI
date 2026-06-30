from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.ZoqFotPik.A1.ZoqFotPikA1 import ZoqFotPikA1
from src.Objects.Ships.ZoqFotPik.A2.ZoqFotPikA2 import ZoqFotPikA2


class ZoqFotPik(SpaceShip):
    action_factories = {1: ZoqFotPikA1, 2: ZoqFotPikA2}

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
