from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Supox.A1.SupoxA1 import SupoxA1
import src.const as const


class Supox(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            ability_obj = SupoxA1(self)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False