from src.Objects.Ships.ability import Ability, ABILITIES_DATA
import src.const as const
import math
from src.toroidal import wrapped_delta


class KohrAhA1(Ability):
    def __init__(self, parent):
        super().__init__("KohrAhA1", parent)
        ability_data = ABILITIES_DATA["KohrAhA1"]
        self.TRACK_SPEED = ability_data.get("track_speed", 8)
        self.TRACK_RANGE = ability_data.get("track_range", 900)
        self.TRACK_WAIT = ability_data.get("turn_wait", 4)
        self.DECELERATION_TIME = ability_data.get("deceleration_time", 12)
        self.is_moving = True
        self.original_speed = self.speed
        self.deceleration_timer = 0
        self.track_timer = 0
        self.expiration_timer = float("inf")  # Never expires unless removed manually
        self.place_self()

    def stop_and_track(self):
        self.is_moving = False
        self.deceleration_timer = self.DECELERATION_TIME
        self.track_timer = 0

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
        if self.is_moving:
            self.heading = 0
            return

        if self.deceleration_timer > 0:
            # UQM halves the fixed-point velocity components until both reach
            # zero. Quantizing at 1/32 world unit reproduces that decay.
            self.velocity = [
                math.trunc(component * 16) / 32 for component in self.velocity
            ]
            self.deceleration_timer -= 1
            if self.velocity == [0, 0] or self.deceleration_timer <= 0:
                self.velocity = [0, 0]
                self.deceleration_timer = 0
            self.heading = 0
            return

        if self.track_timer > 0:
            self.track_timer -= 1
        else:
            opponent = self._live_trackable_opponent()
            if opponent is None:
                self.velocity = [0, 0]
                self.heading = 0
                return

            dx, dy = wrapped_delta(self.position, self.opponent.position)
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= self.TRACK_RANGE:
                target_angle = math.degrees(math.atan2(dx, -dy))
                if target_angle < 0:
                    target_angle += 360
                angle_rad = math.radians(target_angle)
                self.velocity = [
                    math.sin(angle_rad) * self.TRACK_SPEED,
                    -math.cos(angle_rad) * self.TRACK_SPEED,
                ]
                self.track_timer = self.TRACK_WAIT
            else:
                self.velocity = [0, 0]

        self.heading = 0  # Always 0 for omnidirectional projectile
