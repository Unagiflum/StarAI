import math

import src.const as const
from src.Objects.Ships.ability import Ability
from src.toroidal import wrapped_delta


class AndrosynthA1(Ability):
    """A slow bubble projectile that periodically chooses a biased direction."""

    def __init__(self, parent):
        super().__init__("AndrosynthA1", parent)
        self.place_self()

    def place_self(self):
        self.launch_from_gun(inherit_parent_velocity=False)

    def update_heading(self):
        self.heading = 0
        if self.turn_timer > 0:
            self.turn_timer -= 1
            return

        opponent = self._live_trackable_opponent()
        if opponent is None or getattr(opponent, "cloaked", False):
            angle = self.rng.uniform(0.0, 360.0)
        else:
            dx, dy = wrapped_delta(self.position, opponent.position)
            target_angle = math.degrees(math.atan2(dx, -dy)) % 360
            angle = target_angle + self.rng.uniform(-90.0, 90.0)

        self.rotation = angle % 360
        self._set_velocity(self.rotation)
        self.turn_timer = const.cooldown_frames(self.turn_wait)

    def _set_velocity(self, angle):
        angle = math.radians(angle)
        self.velocity = [
            math.sin(angle) * self.speed,
            -math.cos(angle) * self.speed,
        ]

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.previous_heading = self.heading
        self.update_physics()
        self.expiration_timer -= 1

        self.frame_timer -= 1
        if self.frames > 1 and self.frame_timer <= 0:
            self.current_frame = (self.current_frame + 1) % self.frames
            self.size = list(self.sizes[self.current_frame])
            self.frame_timer = self.frame_delay

        return self.expiration_timer > 0 and self.current_hp > 0
