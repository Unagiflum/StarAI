import math

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.launch_geometry import direction_vector, mask_projection_bounds
from src.Objects.Ships.Syreen.A2.SyreenCrew import SyreenCrew


class SyreenA2(Ability):
    def __init__(self, parent):
        super().__init__("SyreenA2", parent)
        definition = ABILITY_DEFINITIONS["SyreenA2"]
        self.range = (
            definition.effect_range if definition.effect_range is not None else 832
        )
        self.separation_distance = (
            definition.separation_distance
            if definition.separation_distance is not None
            else 3
        )
        self.spawn_angle_increment = (
            definition.spawn_angle_increment
            if definition.spawn_angle_increment is not None
            else 25
        )
        self._spawned_crew = []
        self.place_self()

    def place_self(self):
        self.position = self.configured_gun_position()
        self.velocity = [0.0, 0.0]
        self.rotation = 0
        self.heading = 0
        self.area_damage_pending = self.parent.in_battle

    def update_physics(self):
        # The pulse is parent-mounted and collision processing runs after updates.
        self.position = self.configured_gun_position()

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        pass

    def area_damage_for_target(self, target, distance):
        if distance > self.range:
            return 0
        if not hasattr(target, "current_hp"):
            return 0
        durability = getattr(target, "durability_capabilities", None)
        if durability and durability.immune_to_psychic:
            return 0
        if getattr(target, "player", None) == self.parent.player:
            return 0

        raw_damage = max(
            1, round(self.damages[0] * (1.0 - distance / self.range))
        )
        max_allowed = max(0, target.current_hp - 1)
        return min(raw_damage, max_allowed)

    def on_area_damage_hit(self, target, damage):
        from src.Battle.collision_geometry import get_collision_mask, radius
        from src.toroidal import wrapped_delta

        syreen_dx, syreen_dy = wrapped_delta(target.position, self.parent.position)
        base_direction = math.degrees(math.atan2(syreen_dx, -syreen_dy)) % 360
        target_mask = get_collision_mask(target)

        for i in range(damage):
            multiplier = (i + 1) // 2
            sign = 1 if i % 2 == 1 else -1
            if i == 0:
                sign = 0

            direction = (
                base_direction + sign * multiplier * self.spawn_angle_increment
            ) % 360
            dx, dy = direction_vector(direction)

            crew = SyreenCrew(self.parent, list(target.position))
            crew_mask = get_collision_mask(crew)
            if target_mask is not None and crew_mask is not None:
                target_front = mask_projection_bounds(target_mask, direction)[1]
                crew_rear = mask_projection_bounds(crew_mask, direction)[0]
                dist = target_front - crew_rear + self.separation_distance
            else:
                dist = radius(target) + radius(crew) + self.separation_distance
            crew.position = [
                (target.position[0] + dx * dist) % const.ARENA_SIZE,
                (target.position[1] + dy * dist) % const.ARENA_SIZE,
            ]
            self._spawned_crew.append(crew)

    def drain_spawned_objects(self):
        objects = self._spawned_crew
        self._spawned_crew = []
        return objects
