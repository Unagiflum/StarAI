from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Pkunk.A1.PkunkA1 import PkunkA1
from src.Objects.Ships.Pkunk.A2.PkunkA2 import PkunkA2
import src.const as const


class Pkunk(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            projectiles = []

            for angle_offset in [-90, 0, 90]:
                ability_obj = PkunkA1(self, angle_offset)
                projectiles.append(ability_obj)

            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return projectiles
        return None

    def perform_action2(self):
        if self.can_action2() and self.current_energy < self.max_energy:
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = PkunkA2(self)
            self.current_energy = min(self.max_energy, self.current_energy + ability_obj.ENERGY_GAIN)
            ability_obj.play_insult()
        return None

    def perform_action3(self):
        return None, False
