from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Earthling.A1.EarthlingA1 import EarthlingA1
from src.Objects.Ships.Earthling.A2.EarthlingA2 import EarthlingA2
import src.const as const
import math


class Earthling(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            angle_rad = math.radians(self.rotation)

            projectile = EarthlingA1(self)

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
            # Calculate maximum possible shots
            act2_count = self.current_energy // self.a2_cost
            if act2_count == 0:
                return None

            projectile = EarthlingA2(self)
            projectiles = projectile.get_shots(act2_count)
            if not projectiles:
                return None

            self.current_energy -= len(projectiles) * self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            if projectile.launch_sound:
                projectile.launch_sound.play()

            return projectiles
        return None

    def perform_action3(self):
        return None, False
