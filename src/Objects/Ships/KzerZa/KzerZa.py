from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.KzerZa.A1.KzerZaA1 import KzerZaA1
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
import src.const as const


class KzerZa(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None):
        super().__init__(ship_name, player_num, resources)
        self.fighter_launch_count = 0

    action_factories = {1: KzerZaA1}

    def perform_action2(self):
        if self.can_action2() and self.current_hp > 1:
            fighter_count = min(2, self.current_hp - 1)
            fighters = []
            for launch_angle in (135, 225)[:fighter_count]:
                fighters.append(KzerZaA2(self, launch_angle, self.fighter_launch_count))
                self.fighter_launch_count += 1

            self.current_hp -= fighter_count
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            if fighters and fighters[0].launch_sound:
                fighters[0].launch_sound.play()
            return fighters
        return None
