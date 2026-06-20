from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
import src.const as const

class Orz(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None):
        super().__init__(ship_name, player_num, resources)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            print("Action 1", self.current_energy, self.a1_cost)
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action2(self):
        if self.can_action2():
            print("Action 2", self.current_energy, self.a2_cost)
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        if self.can_action3():
            print("Action 3", self.current_energy, self.a3_cost)
            self.current_energy -= self.a3_cost
            self.action3_timer = int(self.a3_wait * const.ACTION_WAIT_SCALE)
            return None, True
        return None, True
