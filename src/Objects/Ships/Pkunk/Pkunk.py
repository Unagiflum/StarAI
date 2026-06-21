from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Pkunk.A1.PkunkA1 import PkunkA1
from src.Objects.Ships.Pkunk.A2.PkunkA2 import PkunkA2


class Pkunk(SpaceShip):
    def plan_action1(self):
        return self.validate_action(
            1,
            lambda ship: [PkunkA1(ship, angle) for angle in (-90, 0, 90)],
        )

    def plan_action2(self):
        if not self.can_action2() or self.current_energy >= self.max_energy:
            return ActionPlan.invalid(2)
        insult = PkunkA2(self)
        energy_change = min(insult.ENERGY_GAIN, self.max_energy - self.current_energy)
        return self.prepare_action_plan(
            2,
            energy_change=energy_change,
            side_effects=(insult.play_insult,),
            use_first_object_sound=False,
        )
