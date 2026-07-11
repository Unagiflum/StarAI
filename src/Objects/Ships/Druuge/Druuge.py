from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Druuge.A1.DruugeA1 import DruugeA1
from src.Objects.Ships.Druuge.A2.DruugeA2 import DruugeA2


class Druuge(SpaceShip):
    def plan_action1(self):
        if not self.can_action1():
            return ActionPlan.invalid(1)
        cannon = DruugeA1(self)
        recoil = lambda: self.apply_thrust(
            cannon.MAX_RECOIL,
            cannon.RECOIL_INCREMENT,
            180,
            False,
        )
        return self.prepare_action_plan(1, cannon, side_effects=(recoil,))

    def plan_action2(self):
        if not (
            self.can_action2()
            and self.current_energy < self.max_energy
            and self.current_hp > 1
        ):
            return ActionPlan.invalid(2)

        furnace = DruugeA2(self)
        final_energy = min(
            self.max_energy,
            self.current_energy - self.a2_cost + furnace.ENERGY_GAIN,
        )
        return self.prepare_action_plan(
            2,
            energy_change=final_energy - self.current_energy,
            crew_change=-1,
            crew_change_source=furnace,
            launch_sound=furnace.launch_sound,
            use_first_object_sound=False,
        )
