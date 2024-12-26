from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Druuge.A1.DruugeA1 import DruugeA1
from src.Objects.Ships.Druuge.A2.DruugeA2 import DruugeA2
import src.const as const


class Druuge(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]
        self.MAX_RECOIL = ship_data.get("MAX_RECOIL", 96)
        self.RECOIL_INCREMENT = ship_data.get("RECOIL_INCREMENT", 24)
        self.A2_ENERGY = ship_data.get("A2_ENERGY", 16)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            ability_obj = DruugeA1(self)
            self.apply_thrust(
                ability_obj.MAX_RECOIL,
                ability_obj.RECOIL_INCREMENT,
                180,
                True,
                False
            )
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        if self.can_action2() and self.current_energy < self.max_energy and self.current_hp > 1:
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            projectile = DruugeA2(self)
            self.current_energy = min(self.max_energy, self.current_energy + self.A2_ENERGY)
            self.current_hp -= 1

            if projectile.launch_sound: projectile.launch_sound.play()
            return None
        return None

    def perform_action3(self):
        return None, False