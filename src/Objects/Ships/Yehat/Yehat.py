from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Yehat.A1.YehatA1 import YehatA1
import src.const as const
import math


class Yehat(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            side_offset = self.size[0] / 2
            projectiles = []

            for offset in [-side_offset, side_offset]:
                ability_obj = YehatA1(self, offset)
                projectiles.append(ability_obj)

            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return projectiles
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False