import math

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.launch_geometry import direction_vector, mask_projection_bounds
from src.Objects.Ships.Syreen.A2.SyreenCrew import SyreenCrew


import pygame
from src.Battle.effects import BattleEffect

class SyreenSongEffect(BattleEffect):
    render_layer = "after_lasers"

    def __init__(self, position, starting_radius, target_radius, thickness, colors, frames):
        super().__init__(
            position=position,
            frames=[],
            frame_delay=1,
            scale=1.0,
            attached_target=None,
        )
        self.starting_radius = starting_radius
        self.target_radius = target_radius
        self.thickness = thickness
        self.colors = colors
        self.total_frames = max(1, frames)
        self.current_frame = 0
        self.can_collide = False
        self.can_expire = True

    def update(self):
        self.current_frame += 1
        return self.current_frame < self.total_frames

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        import pygame
        import src.const as const
        from src.Battle.interpolation import interpolated_position

        if getattr(SyreenA2, "_cached_frames", None) is None:
            return

        video_multiplier = const.VIDEO_FPS_MULTIPLIER
        draw_idx = int((self.current_frame + interp_t) * video_multiplier)
        
        if draw_idx >= len(SyreenA2._cached_frames):
            return
            
        frame_data = SyreenA2._cached_frames[draw_idx]
        if frame_data is None:
            return
            
        super_surf = frame_data["surface"]
        final_size = int((frame_data["radius"] + frame_data["thickness"]) * 2 * scale_factor)
        if final_size <= 0:
            return
            
        surf = pygame.transform.smoothscale(super_surf, (final_size, final_size))

        pos = interpolated_position(self, interp_t)

        screen_x = int((pos[0] + translation[0]) * scale_factor)
        screen_y = int((pos[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                if (
                    -final_size <= pos_x <= const.SCREEN_HEIGHT + final_size
                    and -final_size <= pos_y <= const.SCREEN_HEIGHT + final_size
                ):
                    screen.blit(
                        surf,
                        (
                            const.SCREEN_LEFT + pos_x - final_size // 2,
                            pos_y - final_size // 2,
                        ),
                    )

class SyreenA2(Ability):
    @classmethod
    def preload_resources(cls, ability_name, resources=None):
        super().preload_resources(ability_name, resources)
        
        if getattr(cls, "_cached_frames", None) is not None:
            return
            
        import pygame
        from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
        from src.resources import default_assets
        
        resources = resources or default_assets()
        
        definition = ABILITY_DEFINITIONS[ability_name]
        anim_length = definition.anim_length or 30
        thickness = definition.circle_thickness or 4
        colors = definition.circle_color or ((255, 100, 255, 255), (255, 100, 255, 0))
        target_radius = definition.effect_range if definition.effect_range is not None else 832
        
        ship_def = ability_name[:-2]
                
        max_r = 0
        if ship_def in SHIP_DEFINITIONS:
            ship_assets = resources.ship(ship_def)
            if ship_assets and ship_assets.masks:
                mask = ship_assets.masks[0]
                gx, gy = definition.gun_locations[0]
                ship_scale = getattr(SHIP_DEFINITIONS[ship_def], "sprite_scale", 1.0)
                gx *= ship_scale
                gy *= ship_scale
                for pt in mask.outline():
                    r = ((pt[0] - gx)**2 + (pt[1] - gy)**2)**0.5
                    if r > max_r:
                        max_r = r
                        
        cls._cached_start_radius = max_r
        
        cls._cached_frames = []
        total_physics_frames = max(1, anim_length)
        total_video_frames = total_physics_frames * const.VIDEO_FPS_MULTIPLIER
        
        for v_frame in range(total_video_frames):
            if total_video_frames <= 1:
                progress = 1.0
            else:
                progress = v_frame / (total_video_frames - 1)
                
            current_radius = (max_r + thickness) + (target_radius - max_r) * progress
            if current_radius <= 0:
                cls._cached_frames.append(None)
                continue
                
            c_start, c_end = colors[0], colors[1]
            current_color = (
                int(c_start[0] + (c_end[0] - c_start[0]) * progress),
                int(c_start[1] + (c_end[1] - c_start[1]) * progress),
                int(c_start[2] + (c_end[2] - c_start[2]) * progress),
                int(c_start[3] + (c_end[3] - c_start[3]) * progress),
            )
            
            super_scale = 2
            super_radius = current_radius * super_scale
            super_thickness = thickness * super_scale
            super_size = int((super_radius + super_thickness) * 2)
            
            if super_size <= 0:
                cls._cached_frames.append(None)
                continue
                
            super_surf = pygame.Surface((super_size, super_size), pygame.SRCALPHA)
            pygame.draw.circle(super_surf, current_color, (super_size // 2, super_size // 2), int(super_radius), int(super_thickness))
            
            cls._cached_frames.append({
                "surface": super_surf,
                "radius": current_radius,
                "thickness": thickness
            })

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
        
        if getattr(SyreenA2, "_cached_frames", None) is None:
            SyreenA2.preload_resources("SyreenA2")

        self.anim_length = definition.anim_length or 30
        self.circle_thickness = definition.circle_thickness or 4
        self.circle_color = definition.circle_color or ((255, 100, 255, 255), (255, 100, 255, 0))
        
        self.place_self()
        
        effect = SyreenSongEffect(
            position=self.position,
            starting_radius=SyreenA2._cached_start_radius,
            target_radius=self.range,
            thickness=self.circle_thickness,
            colors=self.circle_color,
            frames=self.anim_length,
        )
        self._spawned_crew.append(effect)

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
            self._spawned_crew.append(crew)

    def drain_spawned_objects(self):
        objects = self._spawned_crew
        self._spawned_crew = []
        return objects
