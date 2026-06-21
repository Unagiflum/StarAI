from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Thraddash.A1.ThraddashA1 import ThraddashA1
from src.Objects.Ships.Thraddash.A2.ThraddashA2 import ThraddashA2
import src.const as const


class Thraddash(SpaceShip):
    action_factories = {1: ThraddashA1}

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = ThraddashA2(self)
            self.apply_thrust(
                ability_obj.REUNK_THRUST,
                ability_obj.REUNK_INCREMENT,
                0,
                True,
                False
            )
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None
