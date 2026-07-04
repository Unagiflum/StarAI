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

        controls = {
            "thrust": (self.thrust_active, thrust_ready),
            "turn_left": (self.turn_left_active, turn_left_ready),
            "turn_right": (self.turn_right_active, turn_right_ready),
        }

        # Pressed controls retain insertion order. Use the most recent active
        # control for repeat readiness, and the most recent lateral control to
        # resolve left+right. Forward+lateral produces UQM's diagonal reverse
        # thrust rather than discarding either component.
        active_controls = []
        for control_name in reversed(self.input_pressed_frames):
            active, ready = controls.get(control_name, (False, False))
            if active:
                active_controls.append((control_name, ready))

        # Keep this method safe for callers that set control state directly.
        if not active_controls:
            active_controls = [
                (control_name, ready)
                for control_name, (active, ready) in reversed(tuple(controls.items()))
                if active
            ]
        if not active_controls:
            return []

        if not active_controls[0][1]:
            return []

        lateral = next(
            (
                control_name
                for control_name, _ in active_controls
                if control_name in ("turn_left", "turn_right")
            ),
            None,
        )
        if lateral == "turn_left":
            return [-135 if self.thrust_active else -90]
        if lateral == "turn_right":
            return [135 if self.thrust_active else 90]
        return [180] if self.thrust_active else []

    def get_thrust_marker_position(self, thrust_angle=0):
        if abs(thrust_angle) != 90:
            return super().get_thrust_marker_position(thrust_angle)

        angle_rad = math.radians(self.rotation + thrust_angle)
        offset = ((self.size[0] / 2) + 6) / 2
        marker_x = self.position[0] - math.sin(angle_rad) * offset
        marker_y = self.position[1] + math.cos(angle_rad) * offset
        return marker_x, marker_y
