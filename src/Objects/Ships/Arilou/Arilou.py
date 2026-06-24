from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Arilou.A1.ArilouA1 import ArilouA1
from src.Objects.Ships.Arilou.A2.ArilouA2 import ArilouA2
import src.const as const
import random

class Arilou(SpaceShip):
    action_factories = {1: ArilouA1}

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        ability_obj = ArilouA2(self)

        def teleport():
            for _ in range(100):
                self.position[0] = self.rng.randint(0, const.ARENA_SIZE)
                self.position[1] = self.rng.randint(0, const.ARENA_SIZE)
                if not self.rotation_would_overlap():
                    break

        return self.prepare_action_plan(2, ability_obj, side_effects=(teleport,))
