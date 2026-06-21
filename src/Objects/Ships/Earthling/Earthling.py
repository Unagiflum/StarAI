from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Earthling.A1.EarthlingA1 import EarthlingA1
from src.Objects.Ships.Earthling.A2.EarthlingA2 import EarthlingA2
import src.const as const


class Earthling(SpaceShip):
    action_factories = {1: EarthlingA1}

    def perform_action2(self):
        if self.can_action2():
            # Calculate maximum possible shots
            act2_count = self.current_energy // self.a2_cost
            if act2_count == 0:
                return None

            ability_obj = EarthlingA2(self)
            projectiles = ability_obj.get_shots(act2_count)
            if not projectiles:
                return None

            self.current_energy -= len(projectiles) * self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            if ability_obj.launch_sound:
                ability_obj.launch_sound.play()

            return projectiles
        return None
