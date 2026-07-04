import math

from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class MyconA1(Ability):
    def __init__(self, parent):
        super().__init__("MyconA1", parent)
        self._plasma_hp_before_damage = self.current_hp
        self.place_self()

    def place_self(self):
        self.launch_from_gun()

    def _evolve_plasma(self):
        """Apply UQM's coupled plasma lifetime, strength, and frame rules."""
        definition = ABILITY_DEFINITIONS[self.name]
        stage_duration = max(1, round(definition.frame_delay))
        total_lifetime = max(1, round(definition.life_time))
        maximum_damage = definition.damage[0]

        if self._plasma_hp_before_damage > self.current_hp:
            # Surviving weapon damage ages the plasma immediately. UQM does
            # this on the first preprocess following the collision.
            self.expiration_timer = min(
                self.expiration_timer,
                self.current_hp * stage_duration,
            )
        else:
            strength = math.ceil(
                self.expiration_timer * maximum_damage / total_lifetime
            )
            self.current_hp = max(1, strength)

        self.current_damage = self.current_hp
        self._plasma_hp_before_damage = self.current_hp

        remaining_stages = math.ceil(self.expiration_timer / stage_duration)
        self.current_frame = max(
            0,
            min(self.frames - 1, self.frames - remaining_stages),
        )
        self.size = list(self.sizes[self.current_frame])
        self.frame_timer = ((self.expiration_timer - 1) % stage_duration) + 1

    def update(self):
        if not self.currently_alive:
            return False

        self._evolve_plasma()
        self.previous_position = self.position.copy()
        self.previous_heading = getattr(self, "heading", 0)
        self.update_physics()
        self.expiration_timer -= 1
        if self.expiration_timer <= 0 or self.current_hp <= 0:
            self.currently_alive = False
            return False
        return True
