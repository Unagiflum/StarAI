from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Supox.A1.SupoxA1 import SupoxA1
import src.const as const
import math


class Supox(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)
        ship_data = SHIPS_DATA[ship_name]

    def perform_action1(self):
        if self.can_action1():
            self.current_energy -= self.a1_cost
            self.action1_timer = int(self.a1_wait * const.ACTION_WAIT_SCALE)
            ability_obj = SupoxA1(self)
            if ability_obj.launch_sound: ability_obj.launch_sound.play()
            return ability_obj
        return None

    def perform_action2(self):
        return None

    def turn_input_enabled(self):
        return not self.action2_active

    def get_active_thrust_angles(self, thrust_ready, turn_left_ready, turn_right_ready):
        if not self.action2_active:
            return super().get_active_thrust_angles(
                thrust_ready, turn_left_ready, turn_right_ready
            )

        thrust_angles = []
        if self.thrust_active and thrust_ready:
            thrust_angles.append(180)
        if self.turn_left_active and turn_left_ready:
            thrust_angles.append(-90)
        if self.turn_right_active and turn_right_ready:
            thrust_angles.append(90)
        return thrust_angles

    def get_thrust_marker_position(self, thrust_angle=0):
        if abs(thrust_angle) != 90:
            return super().get_thrust_marker_position(thrust_angle)

        angle_rad = math.radians(self.rotation + thrust_angle)
        offset = ((self.size[1] / 2) + 6) / 2
        marker_x = self.position[0] - math.sin(angle_rad) * offset
        marker_y = self.position[1] + math.cos(angle_rad) * offset
        return marker_x, marker_y

    def perform_action3(self):
        return None, False
