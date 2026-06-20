import math

from src.Objects.Ships.ability import ABILITIES_DATA, Ability
from src.collision_capabilities import AreaDamageCapabilities


class ShofixtiA2(Ability):
    def __init__(self, parent):
        super().__init__("ShofixtiA2", parent)
        ability_data = ABILITIES_DATA["ShofixtiA2"]
        self.range = ability_data["range"]
        self.position = parent.position.copy()
        self.previous_position = self.position.copy()
        self.velocity = [0.0, 0.0]
        self._first_update = True
        self.area_damage_pending = parent.in_battle
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=True,
            vulnerable=False,
        )

        # Fleet setup instantiates abilities once to warm their shared assets.
        # Gameplay effects must only run for a ship active in battle.
        if parent.in_battle:
            parent.shofixti_self_destruct = True
            parent.current_hp = 0

    def damage_at_distance(self, distance):
        if distance > self.range:
            return 0

        maximum_damage = self.damages[0]
        return max(1, math.ceil(maximum_damage * (1.0 - distance / self.range)))

    def update(self):
        if not self.currently_alive:
            return False

        self.previous_position = self.position.copy()
        self.position = self.parent.position.copy()

        # The ability is added before the frame update. Hold frame 00 for that
        # first update so every supplied animation frame is displayed once.
        if self._first_update:
            self._first_update = False
            return True

        if self.current_frame < self.frames - 1:
            self.current_frame += 1
            self.size = self.sizes[self.current_frame]
            return True

        self.currently_alive = False
        return False
