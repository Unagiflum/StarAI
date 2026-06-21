from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Pkunk.A1.PkunkA1 import PkunkA1
from src.Objects.Ships.Pkunk.A2.PkunkA2 import PkunkA2
import src.const as const


class Pkunk(SpaceShip):
    def perform_action1(self):
        return self.execute_action(
            1,
            lambda ship: [PkunkA1(ship, angle) for angle in (-90, 0, 90)],
        )

    def perform_action2(self):
        if self.can_action2() and self.current_energy < self.max_energy:
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = PkunkA2(self)
            self.current_energy = min(self.max_energy, self.current_energy + ability_obj.ENERGY_GAIN)
            ability_obj.play_insult()
        return None
