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

            # Get valid targets within range
            valid_targets = []
            if self.opponent and self._is_in_range(self.opponent):
                valid_targets.append(self.opponent)

            for obj in sorted(self.enemy_objects, key=lambda x: self._distance_to(x)):
                if self._is_in_range(obj):
                    valid_targets.append(obj)

            for obj in sorted(self.friendly_objects, key=lambda x: self._distance_to(x)):
                if self._is_in_range(obj):
                    valid_targets.append(obj)

            for obj in sorted(self.asteroids, key=lambda x: self._distance_to(x)):
                if self._is_in_range(obj):
                    valid_targets.append(obj)

            # Create lasers up to act2_count or number of valid targets
            shots = min(act2_count, len(valid_targets))
            if shots == 0:
                return None

            self.current_energy -= shots * self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)

            projectiles = []
            for i in range(shots):
                projectile = EarthlingA2(self, valid_targets[i])
                if projectile.launch_sound:
                    projectile.launch_sound.play()
                projectiles.append(projectile)

            return projectiles

    def _is_in_range(self, target):
        return self._distance_to(target) <= EarthlingA2.LASER_RANGE

    def _distance_to(self, target):
        dx = target.position[0] - self.position[0]
        dy = target.position[1] - self.position[1]
        if abs(dx) > const.ARENA_SIZE / 2:
            dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
        if abs(dy) > const.ARENA_SIZE / 2:
            dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE
        return (dx * dx + dy * dy) ** 0.5

    def _is_in_range(self, target):
        return self._distance_to(target) <= EarthlingA2.LASER_RANGE

    def _distance_to(self, target):
        dx = target.position[0] - self.position[0]
        dy = target.position[1] - self.position[1]
        if abs(dx) > const.ARENA_SIZE / 2:
            dx = dx - const.ARENA_SIZE if dx > 0 else dx + const.ARENA_SIZE
        if abs(dy) > const.ARENA_SIZE / 2:
            dy = dy - const.ARENA_SIZE if dy > 0 else dy + const.ARENA_SIZE
        return (dx * dx + dy * dy) ** 0.5

    def perform_action3(self):
        return None, False