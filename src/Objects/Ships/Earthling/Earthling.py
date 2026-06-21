from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Earthling.A1.EarthlingA1 import EarthlingA1
from src.Objects.Ships.Earthling.A2.EarthlingA2 import EarthlingA2


class Earthling(SpaceShip):
    action_factories = {1: EarthlingA1}

    def plan_action2(self):
        if not self.can_action2():
            return ActionPlan.invalid(2)

        max_shots = self.current_energy // self.a2_cost
        if max_shots == 0:
            return ActionPlan.invalid(2)

        point_defense = EarthlingA2(self)
        projectiles = point_defense.get_shots(max_shots)
        if not projectiles:
            return ActionPlan.invalid(2)

        return self.prepare_action_plan(
            2,
            projectiles,
            energy_change=-len(projectiles) * self.a2_cost,
            launch_sound=point_defense.launch_sound,
            use_first_object_sound=False,
        )
