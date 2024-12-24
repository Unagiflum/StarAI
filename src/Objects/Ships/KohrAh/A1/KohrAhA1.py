from src.Objects.Ships.Projectile import Projectile
import math
import src.Const as Const

class KohrAhA1(Projectile):
    TRACKING_SPEED = 20
    TRACKING_RANGE = 224
    ANIMATION_DELAY = 4

    def __init__(self, parent):
        super().__init__("KohrAhA1", parent)
        self.anim_timer = self.ANIMATION_DELAY
        self.current_frame = 0
        self.moving = True

    def update(self):
        if not self.currently_alive:
            return False

        # Handle animation
        self.anim_timer -= 1
        if self.anim_timer <= 0:
            self.anim_timer = self.ANIMATION_DELAY
            self.current_frame = 1 - self.current_frame

        # Update movement based on parent's button state
        if self.moving and not self.parent.action1_active:
            self.moving = False
            self.velocity = [0, 0]

        if not self.moving and self.opponent:
            dx = self.opponent.position[0] - self.position[0]
            dy = self.opponent.position[1] - self.position[1]

            # Handle arena wrapping
            if abs(dx) > Const.ARENA_SIZE / 2:
                dx = dx - Const.ARENA_SIZE if dx > 0 else dx + Const.ARENA_SIZE
            if abs(dy) > Const.ARENA_SIZE / 2:
                dy = dy - Const.ARENA_SIZE if dy > 0 else dy + Const.ARENA_SIZE

            # Calculate distance and track if within range
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > 0 and distance <= self.TRACKING_RANGE:
                self.velocity = [
                    dx / distance * self.TRACKING_SPEED,
                    dy / distance * self.TRACKING_SPEED
                ]

        self.update_physics()
        self.expiration_timer -= 1
        return self.expiration_timer > 0 and self.current_hp > 0