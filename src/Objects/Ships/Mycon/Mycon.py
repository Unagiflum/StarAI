from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Mycon.A1.MyconA1 import MyconA1
from src.Objects.Ships.Mycon.A2.MyconA2 import MyconA2
import src.const as const


class Mycon(SpaceShip):
    action_factories = {1: MyconA1}

    def perform_action2(self):
        if self.can_action2() and self.current_hp < self.max_hp:
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = MyconA2(self)
            self.current_hp = min(self.current_hp + ability_obj.HP_GAIN, self.max_hp)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
        return None
