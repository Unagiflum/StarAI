from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.KohrAh.A1.KohrAhA1 import KohrAhA1
import src.const as const
import math


class KohrAh(SpaceShip):

    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA["KohrAh"]
        self.MAX_PROJECTILES = ship_data.get("MAX_PROJECTILES", 8)
        self.active_projectiles = []
        self.last_action1_state = False

    def perform_action1(self):
        # Update active_projectiles from friendly_objects
        self.active_projectiles = [obj for obj in self.friendly_objects if isinstance(obj, KohrAhA1)]

        button_pressed = self.action1_active and not self.last_action1_state
        button_released = not self.action1_active and self.last_action1_state
        self.last_action1_state = self.action1_active

        if button_pressed and self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)

            # Remove oldest projectile if at max
            if len(self.active_projectiles) >= self.MAX_PROJECTILES:
                oldest = self.active_projectiles.pop(0)
                oldest.currently_alive = False

            # Create new projectile
            projectile = KohrAhA1(self)
            angle_rad = math.radians(self.rotation)
            spawn_distance = const.PROJ_GAP + (self.size[1] + projectile.size[1]) / 2

            projectile.position = [
                self.position[0] + math.sin(angle_rad) * spawn_distance,
                self.position[1] - math.cos(angle_rad) * spawn_distance
            ]

            projectile.heading = 0  # omnidirectional
            projectile.rotation = 0
            projectile.velocity = [
                math.sin(angle_rad) * projectile.speed + self.velocity[0] * projectile.parent_vel,
                -math.cos(angle_rad) * projectile.speed + self.velocity[1] * projectile.parent_vel
            ]

            if projectile.launch_sound:
                projectile.launch_sound.play()

            self.active_projectiles.append(projectile)
            return projectile

        elif button_released:
            # Stop movement of active projectiles
            for proj in self.active_projectiles:
                if proj.currently_alive:
                    proj.is_moving = False

        return None

    def perform_action2(self):
        if self.can_action2():
            self.current_energy -= self.a2_cost
            self.action2_timer = int(self.a2_wait * const.ACTION_WAIT_SCALE)
            return None
        return None

    def perform_action3(self):
        return None, False
