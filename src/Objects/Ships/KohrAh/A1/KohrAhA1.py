from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math
from src.toroidal import wrapped_delta


class KohrAhA1(Ability):
    def __init__(self, parent):
        super().__init__("KohrAhA1", parent)
        ability_data = ABILITIES_DATA["KohrAhA1"]
        self.TRACK_SPEED = ability_data.get("TRACK_SPEED", 5)
        self.TRACK_RANGE = ability_data.get("TRACK_RANGE", 900)
        self.is_moving = True
        self.original_speed = self.speed
        self.expiration_timer = float("inf")  # Never expires unless removed manually
        self.place_self()

    def stop_and_track(self):
        self.is_moving = False
        self.velocity = [0, 0]

    def place_self(self):
        self.launch_from_gun()

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.update_physics()

        # Continuous frame animation
        if self.frames > 1:
            if self.frame_timer <= 0:
                self.current_frame = (self.current_frame + 1) % self.frames
                self.frame_timer = self.frame_delay
            else:
                self.frame_timer -= 1

        return self.currently_alive and self.current_hp > 0

    def update_heading(self):
        if not self.is_moving and self._live_trackable_opponent():
            # Calculate distance to opponent
            dx, dy = wrapped_delta(self.position, self.opponent.position)

            distance = math.sqrt(dx * dx + dy * dy)

            # Only track if within range
            if distance <= self.TRACK_RANGE:
                target_angle = math.degrees(math.atan2(dx, -dy))
                if target_angle < 0:
                    target_angle += 360

                # Calculate velocity towards opponent
                angle_rad = math.radians(target_angle)
                self.velocity = [
                    math.sin(angle_rad) * self.TRACK_SPEED,
                    -math.cos(angle_rad) * self.TRACK_SPEED,
                ]
            else:
                self.velocity = [0, 0]
        elif not self.is_moving:
            self.velocity = [0, 0]

        self.heading = 0  # Always 0 for omnidirectional projectile
