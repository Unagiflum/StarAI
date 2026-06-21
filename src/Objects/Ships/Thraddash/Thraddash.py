from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Thraddash.A1.ThraddashA1 import ThraddashA1
from src.Objects.Ships.Thraddash.A2.ThraddashA2 import ThraddashA2


class Thraddash(SpaceShip):
    action_factories = {1: ThraddashA1}

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)
        afterburner = ThraddashA2(self)
        thrust = lambda: self.apply_thrust(
            afterburner.REUNK_THRUST,
            afterburner.REUNK_INCREMENT,
            0,
            True,
            False,
        )
        return self.prepare_action_plan(2, afterburner, side_effects=(thrust,))
