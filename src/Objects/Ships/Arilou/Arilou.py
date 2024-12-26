from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Arilou.A1.ArilouA1 import ArilouA1
from src.Objects.Ships.Arilou.A2.ArilouA2 import ArilouA2
import src.const as const
import random

class Arilou(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            projectile = ArilouA1(self)

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            projectile = ArilouA2(self)
            projectile.position = self.position.copy()
            projectile.velocity = [0, 0]
            projectile.can_collide = False
            self.position[0] = random.randint(0, const.ARENA_SIZE)
            self.position[1] = random.randint(0, const.ARENA_SIZE)

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action3(self):
        return None, False