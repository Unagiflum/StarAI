import math

import src.const as const
from src.Battle.collision_geometry import collision_info, objects_overlap
from src.collision_capabilities import AreaDamageCapabilities, CollisionRole
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class ZoqFotA2(Ability):
    """A parent-mounted, sprite-shaped area attack that retracts toward the ship."""

    def __init__(self, parent):
        super().__init__("ZoqFotA2", parent)
        definition = ABILITY_DEFINITIONS["ZoqFotA2"]
        self.sprite_scale_x = definition.sprite_scale_x
        self.sprite_scale_y = definition.sprite_scale_y
        self._damaged_targets = set()
        self._age = 0
        self._duration = max(1, round(self.life_time * const.PROJ_LIFE_SCALE))
        self._retraction_assets = self.resources.ability_retraction(
            "ZoqFotA2", self._duration
        )
        self.base_offset = definition.offset
        self.velocity = [0.0, 0.0]
        self.can_move = False
        self.can_die = False
        self.can_expire = True
        self.area_damage_pending = parent.in_battle
        self.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=True,
            vulnerable=False,
            persistent=True,
            plays_impact_sound=True,
        )
        self._current_sprite = None
        self._current_mask = None
        self._sync_to_parent(0)

    @staticmethod
    def _projection_bounds(mask, heading):
        width, height = mask.get_size()
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2
        angle = math.radians(heading * const.TURN_ANGLE)
        forward_x = math.sin(angle)
        forward_y = -math.cos(angle)
        projections = (
            (x - center_x) * forward_x + (y - center_y) * forward_y
            for y in range(height)
            for x in range(width)
            if mask.get_at((x, y))
        )
        projections = tuple(projections)
        return (min(projections), max(projections)) if projections else (0.0, 0.0)

    def _sync_to_parent(self, retraction_frame):
        self.heading = self.parent.heading % const.SHIP_DIRECTIONS
        self.rotation = self.parent.rotation
        self._current_sprite = self._retraction_assets.sprites[
            self.heading
        ][retraction_frame]
        self._current_mask = self._retraction_assets.masks[
            self.heading
        ][retraction_frame]
        self.size = list(self._current_sprite.get_size())

        parent_mask = self.parent.get_collision_mask()
        parent_forward = self._projection_bounds(parent_mask, self.heading)[1]
        effect_rear = self._retraction_assets.projection_bounds[
            self.heading
        ][0]
        base_distance = (
            parent_forward + const.PROJ_GAP
        ) * self.base_offset
        distance = base_distance - effect_rear
        angle = math.radians(self.rotation)
        self.position = [
            (self.parent.position[0] + math.sin(angle) * distance)
            % const.ARENA_SIZE,
            (self.parent.position[1] - math.cos(angle) * distance)
            % const.ARENA_SIZE,
        ]

    def update(self):
        if not self.currently_alive:
            return False
        if self._age >= self._duration:
            self.currently_alive = False
            return False

        self.previous_position = self.position.copy()
        self._sync_to_parent(self._age)
        self._age += 1
        self.area_damage_pending = self.parent.in_battle
        return True

    def area_damage_for_target(self, target, distance):
        if target in self._damaged_targets:
            return 0

        role = target.collision_capabilities.role
        if role == CollisionRole.ASTEROID:
            pass
        elif role not in (
            CollisionRole.SHIP,
            CollisionRole.PROJECTILE,
            CollisionRole.FIGHTER,
        ) or target.player == self.player:
            return 0

        _, _, overlap = collision_info(self, target)
        if not objects_overlap(self, target, overlap):
            return 0
        return self.current_damage

    def on_area_damage_hit(self, target, damage):
        self._damaged_targets.add(target)

    def get_collision_mask(self):
        return self._current_mask

    def get_sprite(self):
        return self._current_sprite
