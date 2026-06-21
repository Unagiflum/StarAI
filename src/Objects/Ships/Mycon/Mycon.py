from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Mycon.A1.MyconA1 import MyconA1
from src.Objects.Ships.Mycon.A2.MyconA2 import MyconA2


class Mycon(SpaceShip):
    action_factories = {1: MyconA1}

    def plan_action2(self):
        if not self.can_action2() or self.current_hp >= self.max_hp:
            return ActionPlan.invalid(2)
        heal = MyconA2(self)
        crew_change = min(heal.HP_GAIN, self.max_hp - self.current_hp)
        return self.prepare_action_plan(
            2,
            crew_change=crew_change,
            launch_sound=heal.launch_sound,
            use_first_object_sound=False,
        )
