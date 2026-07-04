import math

from src.Objects.Ships.ability import Ability, ABILITIES_DATA


class ZoqFotPikA1(Ability):
    SPEED_STAGES = (96, 88, 80, 72)

    def __init__(self, parent):
        super().__init__("ZoqFotPikA1", parent)
        ability_data = ABILITIES_DATA["ZoqFotPikA1"]
        self.SPREAD_ANGLE = ability_data.get("spread_angle", 3)
        self.variant_index = self.rng.randrange(len(self.masks))
        self.size = list(self.sizes[self.variant_index])
        # These configured frames are independent visual variants, not an
        # evolution animation. One variant remains selected for the shot.
        self.frames = 1
        self._age = 0
        self._speed_stage = 0
        self._stage_frames = max(1, int(self.life_time) // len(self.SPEED_STAGES))
        self.speed = self.SPEED_STAGES[0]
        self.place_self()

    def place_self(self):
        configured_direction = self.configured_gun()[1]
        self.launch_from_gun(relative_direction=configured_direction)

    def _set_forward_velocity(self):
        angle = math.radians(self.rotation)
        self.velocity = [
            math.sin(angle) * self.speed,
            -math.cos(angle) * self.speed,
        ]

    def update(self):
        stage = min(
            self._age // self._stage_frames,
            len(self.SPEED_STAGES) - 1,
        )
        if stage != self._speed_stage:
            self._speed_stage = stage
            self.speed = self.SPEED_STAGES[stage]
            if stage == 1:
                self.rotation = (
                    self.rotation
                    + self.rng.choice((-1, 0, 1)) * self.SPREAD_ANGLE
                ) % 360
            self._set_forward_velocity()

        alive = super().update()
        self._age += 1
        return alive

    def get_sprite(self, interp_t=0.0):
        return self.sprites[0][self.variant_index]

    def get_collision_mask(self):
        return self.masks[self.variant_index]
