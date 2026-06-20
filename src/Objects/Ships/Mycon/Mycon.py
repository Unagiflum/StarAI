from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Mycon.A1.MyconA1 import MyconA1
from src.Objects.Ships.Mycon.A2.MyconA2 import MyconA2
import src.const as const


class Mycon(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None):
        super().__init__(ship_name, player_num, resources)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            ability_obj = MyconA1(self)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        if self.can_action2() and self.current_hp < self.max_hp:
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = MyconA2(self)
            self.current_hp = min(self.current_hp + ability_obj.HP_GAIN, self.max_hp)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
        return None

    def perform_action3(self):
        return None, False
