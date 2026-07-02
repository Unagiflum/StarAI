import re

with open("src/Objects/Ships/Syreen/A2/SyreenA2.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace draw method of SyreenSongEffect
draw_insert = """    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        import pygame
        import src.const as const
        from src.Battle.interpolation import interpolated_position

        if getattr(SyreenA2, "_cached_frames", None) is None:
            return

        if self.current_frame >= len(SyreenA2._cached_frames):
            return
            
        frame_data = SyreenA2._cached_frames[self.current_frame]
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

class SyreenA2(Ability):"""

# Regex to match draw method until class SyreenA2
content = re.sub(r"    def draw\(self.*?class SyreenA2\(Ability\):", draw_insert, content, flags=re.DOTALL)

# Add preload_resources to SyreenA2
preload_insert = """class SyreenA2(Ability):
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
        
        ship_def = None
        for s_name, s_def in SHIP_DEFINITIONS.items():
            if ability_name in (s_def.a1, s_def.a2, s_def.a3):
                ship_def = s_name
                break
                
        max_r = 0
        if ship_def:
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
        total_frames = max(1, anim_length)
        for frame in range(total_frames):
            if total_frames <= 1:
                progress = 1.0
            else:
                progress = frame / (total_frames - 1)
                
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

    def __init__"""

content = content.replace("class SyreenA2(Ability):\n    def __init__", preload_insert)

# Replace __init__ logic
init_search = """        if getattr(SyreenA2, "_cached_start_radius", None) is None:
            max_r = 0
            if self.parent.masks and self.parent.masks[0]:
                mask = self.parent.masks[0]
                gx, gy = definition.gun_locations[0]
                for pt in mask.outline():
                    r = ((pt[0] - gx) ** 2 + (pt[1] - gy) ** 2) ** 0.5
                    if r > max_r:
                        max_r = r
            SyreenA2._cached_start_radius = max_r

        self.anim_length = definition.anim_length or 30
        self.circle_thickness = definition.circle_thickness or 4
        self.circle_color = definition.circle_color or ((255, 100, 255, 255), (255, 100, 255, 0))"""

init_replace = """        if getattr(SyreenA2, "_cached_frames", None) is None:
            SyreenA2.preload_resources("SyreenA2")

        self.anim_length = definition.anim_length or 30
        self.circle_thickness = definition.circle_thickness or 4
        self.circle_color = definition.circle_color or ((255, 100, 255, 255), (255, 100, 255, 0))"""

content = content.replace(init_search, init_replace)

with open("src/Objects/Ships/Syreen/A2/SyreenA2.py", "w", encoding="utf-8") as f:
    f.write(content)
