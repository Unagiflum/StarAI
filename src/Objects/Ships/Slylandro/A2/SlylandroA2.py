from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class SlylandroA2(Ability):
    def __init__(self, parent):
        super().__init__("SlylandroA2", parent)
        definition = ABILITY_DEFINITIONS["SlylandroA2"]
        self.range = definition.range
        self.position = self.configured_gun_position()
        self.previous_position = self.position.copy()
        self.velocity = [0.0, 0.0]
        self._first_update = True
        self.area_damage_pending = parent.in_battle

    def update(self):
        if not self.currently_alive:
            return False
        self.previous_position = self.position.copy()
        self.position = self.configured_gun_position()
        if self._first_update:
            self._first_update = False
            return True
        self.currently_alive = False
        return False

    def area_damage_for_target(self, target, distance):
        if not isinstance(target, Asteroid) or distance > self.range:
            return 0
        return self.damages[0]

    def damage_at_distance(self, distance):
        return self.damages[0] if distance <= self.range else 0

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        pass
