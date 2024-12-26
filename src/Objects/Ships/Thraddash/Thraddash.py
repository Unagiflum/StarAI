from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Thraddash.A1.ThraddashA1 import ThraddashA1
from src.Objects.Ships.Thraddash.A2.ThraddashA2 import ThraddashA2
import src.const as const
import math

class Thraddash(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]
        self.REUNK_THRUST = ship_data.get("REUNK_THRUST", 72)
        self.REUNK_INCREMENT = ship_data.get("REUNK_INCREMENT", 12)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = ThraddashA1(self)

            spawn_distance = const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
            projectile.position = [
                self.position[0] + math.sin(angle_rad) * spawn_distance,
                self.position[1] - math.cos(angle_rad) * spawn_distance
            ]
            projectile.heading = self.heading
            projectile.rotation = self.rotation
            angle_rad = math.radians(self.rotation)
            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]

            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = ThraddashA2(self)

            spawn_distance = const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2
            projectile.position = [
                self.position[0] - math.sin(angle_rad) * spawn_distance,
                self.position[1] + math.cos(angle_rad) * spawn_distance
            ]

            projectile.heading = 0  # omnidirectional, so heading = 0
            projectile.rotation = 0  # non-tracking, doesn't matter

            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]
            self.apply_thrust(self.REUNK_THRUST,self.REUNK_INCREMENT, 0, True)
            if projectile.launch_sound: projectile.launch_sound.play()
            return projectile
        return None

    def perform_action3(self):
        return None, False