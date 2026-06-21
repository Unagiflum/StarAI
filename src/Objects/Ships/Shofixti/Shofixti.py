from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Shofixti.A1.ShofixtiA1 import ShofixtiA1
from src.Objects.Ships.Shofixti.A2.ShofixtiA2 import ShofixtiA2
import src.const as const


class Shofixti(SpaceShip):
    action_factories = {1: ShofixtiA1}

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.shofixti_self_destruct = False

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = ShofixtiA2(self)
            if ability_obj.launch_sound:
                ability_obj.launch_sound.play()
            return ability_obj
        return None
