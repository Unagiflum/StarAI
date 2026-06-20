from src.Objects.Ships.space_ship import SpaceShip, SHIPS_DATA
from src.Objects.Ships.Supox.A1.SupoxA1 import SupoxA1
import src.const as const
import math


class Supox(SpaceShip):
    def __init__(self, ship_name, player_num, resources=None):
        super().__init__(ship_name, player_num, resources)
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

        directions = {
            "thrust": (self.thrust_active, thrust_ready, 180),
            "turn_left": (self.turn_left_active, turn_left_ready, -90),
            "turn_right": (self.turn_right_active, turn_right_ready, 90),
        }

        # Pressed controls retain insertion order. Walking that order backwards
        # selects the most recently pressed direction, and releasing it removes
        # it so the next-most-recent held direction takes over.
        for control_name in reversed(self.input_pressed_frames):
            active, ready, angle = directions.get(control_name, (False, False, 0))
            if active:
                return [angle] if ready else []

        # Keep this method safe for callers that set control state directly.
        for active, ready, angle in reversed(tuple(directions.values())):
            if active:
                return [angle] if ready else []
        return []

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
