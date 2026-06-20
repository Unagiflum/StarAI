from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Shofixti.A1.ShofixtiA1 import ShofixtiA1
from src.Objects.Ships.Shofixti.A2.ShofixtiA2 import ShofixtiA2
import src.const as const
import math


class Shofixti(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None):
        super().__init__(ship_name, player_num, resources)
        ship_data = SHIPS_DATA[ship_name]

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self.shofixti_self_destruct = False

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            ability_obj = ShofixtiA1(self)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = ShofixtiA2(self)
            if ability_obj.launch_sound:
                ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action3(self):
        return None, False
