from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Arilou.A1.ArilouA1 import ArilouA1
from src.Objects.Ships.Arilou.A2.ArilouA2 import ArilouA2
import src.const as const
import random

class Arilou(SpaceShip):
    action_factories = {1: ArilouA1}

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            ability_obj = ArilouA2(self)
            self.position[0] = random.randint(0, const.ARENA_SIZE)
            self.position[1] = random.randint(0, const.ARENA_SIZE)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None
