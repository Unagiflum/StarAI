from src.Objects.Ships.Projectile import Projectile
import math

class KohrAhA1(Projectile):
    TRACK_SPEED = 15
    TRACK_RANGE = 500

    def __init__(self, parent):
        super().__init__("KohrAhA1", parent)
        self.is_moving = True
        self.original_speed = self.speed
        self.expiration_timer = float('inf')  # Never expires unless removed manually

    def update(self):
        if not self.currently_alive:
            return False

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
        if not self.is_moving and self.opponent:
            # Calculate distance to opponent
            dx = self.opponent.position[0] - self.position[0]
            dy = self.opponent.position[1] - self.position[1]

            # Handle arena wrapping
            if abs(dx) > self.parent.planet.diameter / 2:
                dx = dx - self.parent.planet.diameter if dx > 0 else dx + self.parent.planet.diameter
            if abs(dy) > self.parent.planet.diameter / 2:
                dy = dy - self.parent.planet.diameter if dy > 0 else dy + self.parent.planet.diameter

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
                    -math.cos(angle_rad) * self.TRACK_SPEED
                ]
            else:
                self.velocity = [0, 0]
        elif not self.is_moving:
            self.velocity = [0, 0]

        self.heading = 0  # Always 0 for omnidirectional projectile
