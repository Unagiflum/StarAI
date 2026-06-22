from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Supox.A1.SupoxA1 import SupoxA1
import math


class Supox(SpaceShip):
    action_factories = {1: SupoxA1}

    def plan_action2(self):
        # Secondary is a held movement modifier, not a resource action.
        return ActionPlan.invalid(2)

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
        offset = ((self.size[0] / 2) + 6) / 2
        marker_x = self.position[0] - math.sin(angle_rad) * offset
        marker_y = self.position[1] + math.cos(angle_rad) * offset
        return marker_x, marker_y
