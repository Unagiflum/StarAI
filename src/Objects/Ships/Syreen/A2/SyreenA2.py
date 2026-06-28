import math
from src.Objects.Ships.ability import Ability, ABILITIES_DATA
from src.Objects.Ships.Syreen.A2.SyreenCrew import SyreenCrew

class SyreenA2(Ability):
    def __init__(self, parent):
        super().__init__("SyreenA2", parent)
        ability_data = ABILITIES_DATA["SyreenA2"]
        self.range = ability_data.range if hasattr(ability_data, "range") and ability_data.range is not None else 832
        self.separation_distance = ability_data.separation_distance if ability_data.separation_distance is not None else 3
        self.spawn_angle_increment = ability_data.spawn_angle_increment if ability_data.spawn_angle_increment is not None else 25
        self._spawned_crew = []
        self.place_self()

    def place_self(self):
        self.position = list(self.parent.position)
        self.velocity = [0.0, 0.0]
        self.rotation = 0
        self.heading = 0
        self.area_damage_pending = True

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        pass

    def area_damage_for_target(self, target, distance):
        if distance > self.range:
            return 0
        if not hasattr(target, "current_hp"):
            return 0
        if getattr(target, "immune_to_psychic", False):
            return 0
        if getattr(target, "player", None) == self.parent.player:
            return 0
            
        raw_damage = max(1, round(8 * (1.0 - distance / self.range)))
        max_allowed = max(0, target.current_hp - 1)
        return min(raw_damage, max_allowed)

    def on_area_damage_hit(self, target, damage):
        from src.Battle.collision_geometry import objects_overlap_at_positions
        from src.toroidal import wrapped_delta

        syreen_dx, syreen_dy = wrapped_delta(target.position, self.parent.position)
        base_angle = math.degrees(math.atan2(-syreen_dy, syreen_dx))
        print(f"RAYCAST DEBUG: Target={target.position}, Syreen={self.parent.position}, dx={syreen_dx}, dy={syreen_dy}, base_angle={base_angle}")
        
        for i in range(damage):
            multiplier = (i + 1) // 2
            sign = 1 if i % 2 == 1 else -1
            if i == 0:
                sign = 0
            
            angle = math.radians((base_angle + sign * multiplier * self.spawn_angle_increment) % 360)
            dx = math.cos(angle)
            dy = -math.sin(angle)
            
            crew = SyreenCrew(self.parent, list(target.position))
            dist = 0
            
            # Move radially outward until no longer overlapping
            while objects_overlap_at_positions(crew, target, crew.position, target.position):
                dist += 2
                crew.position = [
                    target.position[0] + dx * dist,
                    target.position[1] + dy * dist,
                ]
                if dist > 400: # safety bailout
                    break
                    
            # Add separation buffer amount
            dist += self.separation_distance
            crew.position = [
                target.position[0] + dx * dist,
                target.position[1] + dy * dist,
            ]
            self._spawned_crew.append(crew)

    def drain_spawned_objects(self):
        objects = self._spawned_crew
        self._spawned_crew = []
        return objects
