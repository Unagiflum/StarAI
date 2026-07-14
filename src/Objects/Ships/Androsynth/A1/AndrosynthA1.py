import math

import src.const as const
from src.Objects.Ships.ability import Ability
from src.toroidal import wrapped_delta


class AndrosynthA1(Ability):
    """A slow bubble projectile that periodically chooses a biased direction."""

    def __init__(self, parent):
        super().__init__("AndrosynthA1", parent)
        # UQM bubbles choose a direction and advance their animation on their
        # first update. The common Ability defaults delay both operations.
        self.turn_timer = 0
        self.frame_timer = 0
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
            facing = self.rng.randrange(const.SHIP_DIRECTIONS)
        else:
            direction_step = const.TURN_ANGLE
            current_facing = int(
                (self.rotation + direction_step / 2) // direction_step
            ) % const.SHIP_DIRECTIONS
            dx, dy = wrapped_delta(self.position, opponent.position)
            target_angle = math.degrees(math.atan2(dx, -dy)) % 360
            target_facing = int(
                (target_angle + direction_step / 2) // direction_step
            ) % const.SHIP_DIRECTIONS
            delta_facing = (
                target_facing - current_facing
            ) % const.SHIP_DIRECTIONS

            # TrackShip turns one facing toward the target before the bubble
            # chooses a random facing on that side of the target vector.
            tracked_facing = current_facing
            half_circle = const.SHIP_DIRECTIONS // 2
            if delta_facing == half_circle:
                tracked_facing += -1 if self.rng.randrange(2) == 0 else 1
            elif 0 < delta_facing < half_circle:
                tracked_facing += 1
            elif delta_facing > half_circle:
                tracked_facing -= 1

            random_offset = self.rng.randrange(half_circle)
            if delta_facing <= half_circle:
                facing = tracked_facing + random_offset
            else:
                facing = tracked_facing - random_offset

        self.rotation = (facing % const.SHIP_DIRECTIONS) * const.TURN_ANGLE
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

        if self.frame_timer > 0:
            self.frame_timer -= 1
        elif self.frames > 1:
            self.current_frame = (self.current_frame + 1) % self.frames
            if self.sizes:
                self.size = list(self.sizes[self.current_frame % len(self.sizes)])
            self.frame_timer = self.rng.randrange(4)

        return self.expiration_timer > 0 and self.current_hp > 0
