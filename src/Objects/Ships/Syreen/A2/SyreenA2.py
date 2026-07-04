import math
from collections import OrderedDict
from weakref import WeakKeyDictionary

import pygame
import pygame.gfxdraw

import src.const as const
from src.Battle.effects import BattleEffect
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.launch_geometry import direction_vector, mask_projection_bounds
from src.Objects.Ships.Syreen.A2.SyreenCrew import SyreenCrew
from src.resources import default_assets


def _farthest_alpha_radius(surface, origin):
    """Return the farthest alpha>0 pixel-center distance from origin."""
    mask = pygame.mask.from_surface(surface, threshold=0)
    origin_x, origin_y = origin
    maximum_squared = 0.0
    for bounds in mask.get_bounding_rects():
        for y in range(bounds.top, bounds.bottom):
            for x in range(bounds.left, bounds.right):
                if mask.get_at((x, y)):
                    maximum_squared = max(
                        maximum_squared,
                        (x - origin_x) ** 2 + (y - origin_y) ** 2,
                    )
    return math.sqrt(maximum_squared)


class SyreenSongEffect(BattleEffect):
    """Fixed-origin, expanding visual wave for the Syreen song."""

    render_layer = "after_lasers"
    _overlay_cache = OrderedDict()
    _overlay_cache_limit = 2

    def __init__(
        self,
        position,
        starting_radius,
        target_radius,
        thickness,
        colors,
        total_frames,
    ):
        super().__init__(position=position, frames=(), frame_delay=1)
        self.starting_radius = starting_radius
        self.target_radius = target_radius
        self.thickness = thickness
        self.colors = colors
        self.total_frames = total_frames

    def update(self):
        self.current_frame += 1
        return self.current_frame < self.total_frames

    def radius_and_color(self, interp_t=0.0):
        denominator = max(1, self.total_frames - 1)
        progress = min(1.0, max(0.0, (self.current_frame + interp_t) / denominator))
        radius = self.starting_radius + (
            self.target_radius - self.starting_radius
        ) * progress
        start, end = self.colors
        color = tuple(
            round(start[channel] + (end[channel] - start[channel]) * progress)
            for channel in range(4)
        )
        return radius, color

    @classmethod
    def _overlay(cls, size):
        overlay = cls._overlay_cache.get(size)
        if overlay is None:
            overlay = pygame.Surface(size, pygame.SRCALPHA)
            cls._overlay_cache[size] = overlay
            while len(cls._overlay_cache) > cls._overlay_cache_limit:
                cls._overlay_cache.popitem(last=False)
        else:
            cls._overlay_cache.move_to_end(size)
        return overlay

    @staticmethod
    def _draw_feathered_circle(surface, center, radius, thickness, color):
        center_radius = max(1, round(radius))
        pixel_thickness = max(1.0, thickness)
        half_thickness = max(0.5, pixel_thickness / 2.0)
        maximum_offset = max(0, math.ceil(half_thickness) - 1)

        # Three strokes retain a soft inner/outer edge without drawing every
        # one-pixel radius across the full thickness.
        offsets = sorted(
            {-maximum_offset, 0, maximum_offset}, key=abs, reverse=True
        )
        for offset in offsets:
            # pygame.draw.circle's one-pixel outline lies at radius - 1.
            stroke_radius = center_radius + offset + 1
            if stroke_radius <= 0:
                continue
            alpha = color[3] if offset == 0 else round(color[3] * 0.5)
            if alpha:
                stroke_color = (*color[:3], alpha)
                pygame.gfxdraw.aacircle(
                    surface,
                    center[0],
                    center[1],
                    stroke_radius,
                    stroke_color,
                )
                pygame.draw.circle(
                    surface,
                    stroke_color,
                    center,
                    stroke_radius,
                    width=1,
                )

    @staticmethod
    def _outline_intersects_rect(center, radius, thickness, rect):
        nearest_x = min(max(center[0], rect.left), rect.right)
        nearest_y = min(max(center[1], rect.top), rect.bottom)
        nearest_distance = math.hypot(
            center[0] - nearest_x,
            center[1] - nearest_y,
        )
        farthest_distance = max(
            math.hypot(center[0] - x, center[1] - y)
            for x in (rect.left, rect.right)
            for y in (rect.top, rect.bottom)
        )
        half_thickness = max(0.5, thickness / 2.0)
        return (
            nearest_distance <= radius + half_thickness
            and farthest_distance >= max(0.0, radius - half_thickness)
        )

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        radius, color = self.radius_and_color(interp_t)
        raw_pixel_radius = radius * scale_factor
        pixel_radius = max(1, round(raw_pixel_radius / 2.0) * 2)
        pixel_thickness = self.thickness * scale_factor
        extent = math.ceil(pixel_radius + pixel_thickness / 2.0 + 1)
        if extent <= 0 or color[3] <= 0:
            return

        clip = screen.get_clip()
        screen_x = round((self.position[0] + translation[0]) * scale_factor)
        screen_y = round((self.position[1] + translation[1]) * scale_factor)
        arena_span = const.ARENA_SIZE * scale_factor
        visible_circles = []
        dirty_rect = None

        for wrap_x in (-1, 0, 1):
            for wrap_y in (-1, 0, 1):
                center = (
                    round(const.SCREEN_LEFT + screen_x + wrap_x * arena_span),
                    round(screen_y + wrap_y * arena_span),
                )
                bounds = pygame.Rect(
                    center[0] - extent,
                    center[1] - extent,
                    extent * 2 + 1,
                    extent * 2 + 1,
                )
                if not self._outline_intersects_rect(
                    center,
                    pixel_radius,
                    pixel_thickness,
                    clip,
                ):
                    continue
                visible_circles.append(center)
                clipped_bounds = bounds.clip(clip)
                dirty_rect = (
                    clipped_bounds
                    if dirty_rect is None
                    else dirty_rect.union(clipped_bounds)
                )

        if not visible_circles or not dirty_rect:
            return

        # The same drawing path is used for the battle view and both HUD
        # viewports. The scratch surface only needs to cover the active clip.
        overlay = self._overlay(clip.size)
        local_dirty = dirty_rect.move(-clip.left, -clip.top)
        overlay.set_clip(local_dirty)
        overlay.fill((0, 0, 0, 0), local_dirty)
        for center in visible_circles:
            self._draw_feathered_circle(
                overlay,
                (center[0] - clip.left, center[1] - clip.top),
                pixel_radius,
                pixel_thickness,
                color,
            )

        overlay.set_clip(None)
        screen.blit(overlay, dirty_rect.topleft, area=local_dirty)


class SyreenA2(Ability):
    _starting_radius_cache = WeakKeyDictionary()

    @classmethod
    def preload_resources(cls, ability_name, resources=None):
        resources = resources or default_assets()
        ability_assets = super().preload_resources(ability_name, resources)
        if resources in cls._starting_radius_cache:
            return ability_assets

        definition = ABILITY_DEFINITIONS[ability_name]
        ship_definition = SHIP_DEFINITIONS[definition.ship_name]
        ship_sprite = resources.ship(definition.ship_name).sprites[0]
        gun_x, gun_y = definition.gun_locations[0]
        gun_origin = (
            gun_x * ship_definition.sprite_scale,
            gun_y * ship_definition.sprite_scale,
        )
        farthest_radius = _farthest_alpha_radius(ship_sprite, gun_origin)
        starting_radius = farthest_radius + definition.stroke_width / 2.0
        if starting_radius >= definition.range:
            raise ValueError(
                f"Ability '{ability_name}' circle starts outside its configured range"
            )
        cls._starting_radius_cache[resources] = starting_radius
        return ability_assets

    def __init__(self, parent):
        super().__init__("SyreenA2", parent)
        definition = ABILITY_DEFINITIONS["SyreenA2"]
        self.range = definition.range
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
        resources = getattr(parent, "resources", None) or default_assets()
        if resources not in self._starting_radius_cache:
            self.preload_resources("SyreenA2", resources)

        self._spawned_objects = []
        self.place_self()
        self._spawned_objects.append(
            SyreenSongEffect(
                position=self.position,
                starting_radius=self._starting_radius_cache[resources],
                target_radius=self.range,
                thickness=definition.stroke_width,
                colors=definition.colors,
                total_frames=definition.anim_length,
            )
        )

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

        raw_damage = 1 + math.floor(
            self.damages[0] * (1.0 - distance / self.range)
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

            crew = SyreenCrew(
                self.parent, list(target.position), origin_ship=target
            )
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
            self._spawned_objects.append(crew)

    def drain_spawned_objects(self):
        objects = self._spawned_objects
        self._spawned_objects = []
        return objects
