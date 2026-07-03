import pygame
import math
from dataclasses import dataclass

from src.UI import ui
from src.Battle.status_bar import (
    draw_player_status,
    draw_boarded_marine_icons,
    draw_limpet_count,
    draw_special_indicator,
    StatusBar,
)
from src.Battle.battle_entry import draw_entry_silhouettes
import src.const as const
from src.toroidal import view_center_and_size, wrapped_delta, wrapped_midpoint
from src.Battle.world import World
from src.Battle.interpolation import interpolated_position
from src.Battle.effects import BattleEffect
from src.Objects.object import ThrustMarker
from src.Objects.Space.space_obj import Asteroid, Planet, Star
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip

# HUD layout constants
VIEWPORT_SIZE = 200
VIEWPORT_MARGIN = 7
BAR_WIDTH = 30
VIEWPORT_COLUMN_WIDTH = (
    VIEWPORT_SIZE + 2 * VIEWPORT_MARGIN
)  # gap that holds the viewport

# Shared HUD colour (fill only; HUD_BORDER is imported from status_bar)
HUD_FILL = (0, 0, 0)
HUD_BOTTOM_PADDING = 20
MARINE_REGION_HEIGHT = HUD_BOTTOM_PADDING
HUD_INSTRUCTION_FONT_SIZE = 30
HUD_INSTRUCTION_MARGIN = 20

# Derived layout — constant once the screen geometry is set.
_TOTAL_WIDTH = (BAR_WIDTH * 2) + VIEWPORT_COLUMN_WIDTH
_LEFT_PANEL_W = const.SCREEN_LEFT
_RIGHT_PANEL_W = const.SCREEN_WIDTH - (const.SCREEN_LEFT + const.SCREEN_HEIGHT)
P1_X = const.SCREEN_LEFT - _TOTAL_WIDTH - ((_LEFT_PANEL_W - _TOTAL_WIDTH) // 2)
P2_X = (const.SCREEN_LEFT + const.SCREEN_HEIGHT) + (
    (_RIGHT_PANEL_W - _TOTAL_WIDTH) // 2
)

# Viewport surface geometry.  Object draw methods cull against
# SCREEN_HEIGHT and offset x by SCREEN_LEFT, so the viewport
# translation must keep objects in the range that passes those checks.
# The surface is sized to just contain the clip region.
VP_CENTER_X = const.SCREEN_LEFT + const.SCREEN_HEIGHT // 2
VP_CENTER_Y = const.SCREEN_HEIGHT // 2
VP_SURF_W = VP_CENTER_X + VIEWPORT_SIZE // 2
VP_SURF_H = VP_CENTER_Y + VIEWPORT_SIZE // 2
VP_CLIP_RECT = pygame.Rect(
    VP_CENTER_X - VIEWPORT_SIZE // 2,
    VP_CENTER_Y - VIEWPORT_SIZE // 2,
    VIEWPORT_SIZE,
    VIEWPORT_SIZE,
)

# Cached viewport surface (module-level instead of function attribute).
_viewport_surface = None


@dataclass(frozen=True)
class RenderSnapshot:
    stars: tuple
    planets: tuple
    thrust_markers: tuple
    asteroids: tuple
    abilities: tuple
    ships: tuple
    effects: tuple
    live_ships: tuple

    @classmethod
    def capture(cls, game_objects):
        if isinstance(game_objects, cls):
            return game_objects

        stars = []
        planets = []
        thrust_markers = []
        asteroids = []
        abilities = []
        ships = []
        effects = []

        for obj in World.coerce(game_objects):
            if isinstance(obj, Star):
                stars.append(obj)
            elif isinstance(obj, Planet):
                planets.append(obj)
            elif isinstance(obj, ThrustMarker):
                thrust_markers.append(obj)
            elif isinstance(obj, Asteroid):
                asteroids.append(obj)
            elif isinstance(obj, Ability):
                abilities.append(obj)
            elif isinstance(obj, SpaceShip):
                ships.append(obj)
            elif isinstance(obj, BattleEffect):
                effects.append(obj)

        return cls(
            stars=tuple(stars),
            planets=tuple(planets),
            thrust_markers=tuple(thrust_markers),
            asteroids=tuple(asteroids),
            abilities=tuple(abilities),
            ships=tuple(ships),
            effects=tuple(effects),
            live_ships=tuple(ship for ship in ships if World.is_alive(ship)),
        )


def _draw_empty_rect(surface, rect, border_color, fill_color):
    """Draw a filled rect with a colored border — used for dead-ship HUD slots."""
    pygame.draw.rect(surface, fill_color, rect)
    pygame.draw.rect(surface, border_color, rect, 2)


def _draw_dashed_circle(surface, ship, scale_factor, translation, interp_t=0.0):
    radius = int(100 * scale_factor)
    if radius <= 0:
        return

    color = const.P1_COLOR if ship.player == 1 else const.P2_COLOR
    color_with_alpha = (*color, 255)

    surf_size = radius * 2 + 12
    circle_surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
    rect = pygame.Rect(6, 6, radius * 2, radius * 2)

    angles_deg = [45, 135, 225, 315]
    arc_length_rad = math.radians(4)

    for angle_deg in angles_deg:
        center_rad = math.radians(angle_deg)
        start_angle = center_rad - arc_length_rad / 2
        end_angle = center_rad + arc_length_rad / 2
        pygame.draw.arc(circle_surf, color_with_alpha, rect, start_angle, end_angle, 6)

    pos = interpolated_position(ship, interp_t)
    screen_x = int((pos[0] + translation[0]) * scale_factor)
    screen_y = int((pos[1] + translation[1]) * scale_factor)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
            pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

            if (
                -surf_size <= pos_x <= const.SCREEN_HEIGHT + surf_size
                and -surf_size <= pos_y <= const.SCREEN_HEIGHT + surf_size
            ):
                surface.blit(
                    circle_surf,
                    (
                        const.SCREEN_LEFT + pos_x - surf_size // 2,
                        pos_y - surf_size // 2,
                    ),
                )


class StarFieldRenderer:
    def __init__(self):
        pass

    def draw(self, screen, stars, scale_factor, translation, midpoint):
        stars_by_depth = [[] for _ in range(const.STAR_DEPTHS)]
        for star in stars:
            stars_by_depth[star.depth].append(star)

        scaled_star_cache = {}

        for depth, depth_stars in enumerate(stars_by_depth):
            parallax_factor = 0.5 + 0.5 * (depth / (const.STAR_DEPTHS - 1))
            self.draw_depth_stars(
                screen,
                depth_stars,
                scale_factor,
                translation,
                midpoint,
                parallax_factor,
                scaled_star_cache,
            )

    def draw_depth_stars(
        self,
        screen,
        stars,
        scale_factor,
        translation,
        midpoint,
        parallax_factor,
        scaled_star_cache,
    ):
        for star in stars:
            dx, dy = wrapped_delta(midpoint, star.position)

            relative_x = midpoint[0] + dx * parallax_factor
            relative_y = midpoint[1] + dy * parallax_factor

            screen_x = int((relative_x + translation[0]) * scale_factor)
            screen_y = int((relative_y + translation[1]) * scale_factor)

            img_id = id(star.image)
            if img_id not in scaled_star_cache:
                scaled_image = pygame.transform.smoothscale_by(star.image, scale_factor)
                scaled_image.set_alpha(const.STAR_ALPHA)
                scaled_star_cache[img_id] = scaled_image
            
            scaled_image = scaled_star_cache[img_id]
            star_size = scaled_image.get_width()

            if (
                -star_size <= screen_x <= const.SCREEN_HEIGHT + star_size
                and -star_size <= screen_y <= const.SCREEN_HEIGHT + star_size
            ):
                screen.blit(
                    scaled_image,
                    (
                        const.SCREEN_LEFT + screen_x - star_size // 2,
                        screen_y - star_size // 2,
                    ),
                )


def calculate_view_parameters(game_objects, camera_targets=None, interp_t=0.0):
    snapshot = RenderSnapshot.capture(game_objects)
    targets = camera_targets
    if targets is None:
        targets = snapshot.live_ships

    if len(targets) == 1:
        view_size = (const.SCREEN_HEIGHT / const.MAX_ZOOM) * 1.5
        scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
        base_pos0 = getattr(targets[0], "camera_position", targets[0].position)
        if base_pos0 == targets[0].position:
            pos0 = interpolated_position(targets[0], interp_t)
        else:
            pos0 = base_pos0
        translation = [
            const.SCREEN_HEIGHT / (2 * scale_factor) - pos0[0],
            const.SCREEN_HEIGHT / (2 * scale_factor) - pos0[1],
        ]
        return scale_factor, translation
    if len(targets) < 2:
        return 1.0, [0, 0]

    base_pos0 = getattr(targets[0], "camera_position", targets[0].position)
    if base_pos0 == targets[0].position:
        pos0 = interpolated_position(targets[0], interp_t)
    else:
        pos0 = base_pos0

    base_pos1 = getattr(targets[1], "camera_position", targets[1].position)
    if base_pos1 == targets[1].position:
        pos1 = interpolated_position(targets[1], interp_t)
    else:
        pos1 = base_pos1
    center, view_size = view_center_and_size([pos0, pos1])

    scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
    translation = [
        const.SCREEN_HEIGHT / (2 * scale_factor) - center[0],
        const.SCREEN_HEIGHT / (2 * scale_factor) - center[1],
    ]

    return scale_factor, translation


def _render_world_to_surface(
    surface,
    snapshot,
    scale_factor,
    translation,
    midpoint,
    entry_state,
    frame_id,
    star_field_renderer,
    skip_stars=False,
    show_gravity_range=True,
    show_entry_trails=True,
    show_crosshairs=False,
    interp_t=0.0,
):
    if not skip_stars:
        star_field_renderer.draw(
            surface, snapshot.stars, scale_factor, translation, midpoint
        )

    if show_gravity_range:
        for planet in snapshot.planets:
            planet.draw_gravity_range(
                surface,
                scale_factor,
                translation,
                interp_t=interp_t,
            )

    for planet in snapshot.planets:
        planet.draw(surface, scale_factor, translation, interp_t=interp_t)

    for marker in snapshot.thrust_markers:
        marker.draw(surface, scale_factor, translation, interp_t=interp_t)

    for asteroid in snapshot.asteroids:
        asteroid.draw(surface, scale_factor, translation, interp_t=interp_t)

    foreground_types = {"special_object", "projectile", "laser", "area"}
    for ability in snapshot.abilities:
        if getattr(ability, "type", None) in foreground_types:
            continue
        ability.draw(surface, scale_factor, translation, interp_t=interp_t)

    entering_ships = set(entry_state.entering_ships if entry_state else ())
    if entry_state and show_entry_trails:
        draw_entry_silhouettes(
            surface,
            entry_state,
            frame_id,
            scale_factor,
            translation,
        )

    for ship in snapshot.ships:
        if ship not in entering_ships:
            physics = getattr(ship, "physical_collision_capabilities", None)
            if (
                show_crosshairs
                and not getattr(ship, "cloaked", False)
                and not (physics and physics.is_intangible)
            ):
                _draw_dashed_circle(surface, ship, scale_factor, translation, interp_t)
            ship.draw(surface, scale_factor, translation, interp_t=interp_t)

    for ability_type in ("special_object", "projectile", "laser"):
        for ability in snapshot.abilities:
            if getattr(ability, "type", None) == ability_type:
                ability.draw(surface, scale_factor, translation, interp_t=interp_t)

    after_laser_effects = tuple(
        effect
        for effect in snapshot.effects
        if getattr(effect, "render_layer", None) == "after_lasers"
    )
    for effect in after_laser_effects:
        effect.draw(surface, scale_factor, translation, interp_t=interp_t)

    for ability in snapshot.abilities:
        draw_foreground = getattr(ability, "draw_foreground", None)
        if draw_foreground is not None:
            draw_foreground(
                surface,
                scale_factor,
                translation,
                interp_t=interp_t,
            )

    area_abilities = sorted(
        (
            ability
            for ability in snapshot.abilities
            if getattr(ability, "type", None) == "area"
        ),
        key=lambda ability: getattr(ability, "render_priority", 0),
    )
    for ability in area_abilities:
        ability.draw(surface, scale_factor, translation, interp_t=interp_t)

    for effect in snapshot.effects:
        if effect not in after_laser_effects:
            effect.draw(surface, scale_factor, translation, interp_t=interp_t)


def _should_show_crosshairs(mode, is_mirror):
    return mode == "always" or (mode == "mirror_match_only" and is_mirror)


def draw_battle(
    screen,
    game_objects,
    border_rect,
    border_color,
    star_field_renderer,
    camera_targets=None,
    entry_state=None,
    frame_id=0,
    original_ships=None,
    is_paused=False,
    interp_t=0.0,
):
    snapshot = RenderSnapshot.capture(game_objects)
    scale_factor, translation = calculate_view_parameters(
        snapshot, camera_targets, interp_t
    )

    players = snapshot.live_ships
    is_mirror = False
    if original_ships and len(original_ships) == 2:
        is_mirror = original_ships[0].name == original_ships[1].name
    elif len(players) == 2:
        is_mirror = players[0].name == players[1].name

    show_crosshairs = _should_show_crosshairs(const.SHIP_CROSSHAIRS, is_mirror)

    midpoint = [
        const.SCREEN_HEIGHT / (2 * scale_factor) - translation[0],
        const.SCREEN_HEIGHT / (2 * scale_factor) - translation[1],
    ]

    screen.fill(ui.BLACK)
    screen.set_clip(border_rect)

    _render_world_to_surface(
        screen,
        snapshot,
        scale_factor,
        translation,
        midpoint,
        entry_state,
        frame_id,
        star_field_renderer,
        show_gravity_range=const.SHOW_PLANET_GRAVITY_MARKER,
        show_crosshairs=show_crosshairs,
        interp_t=interp_t,
    )

    pygame.draw.rect(screen, border_color, border_rect, 2)
    screen.set_clip(None)

    # Draw status bars and viewports for both players.
    global _viewport_surface
    if _viewport_surface is None or _viewport_surface.get_size() != (
        VP_SURF_W,
        VP_SURF_H,
    ):
        _viewport_surface = pygame.Surface((VP_SURF_W, VP_SURF_H))
    viewport_surface = _viewport_surface

    for player_id in (1, 2):
        if player_id == 1:
            status_x = 0
            panel_w = _LEFT_PANEL_W
        else:
            status_x = const.SCREEN_LEFT + const.SCREEN_HEIGHT
            panel_w = _RIGHT_PANEL_W

        live_ship = next((s for s in players if s.player == player_id), None)
        ship_to_track = live_ship or next(
            (s for s in snapshot.ships if s.player == player_id), None
        )

        # Calculate max height of elements to size the panel
        if live_ship:
            hp_height = StatusBar.calculate_height(live_ship.max_hp)
            energy_height = StatusBar.calculate_height(live_ship.max_energy)
        else:
            dead_ship = None
            if original_ships:
                dead_ship = next(
                    (s for s in original_ships if s.player == player_id), None
                )
            elif ship_to_track:
                dead_ship = ship_to_track

            if dead_ship:
                hp_height = StatusBar.calculate_height(dead_ship.max_hp)
                energy_height = StatusBar.calculate_height(dead_ship.max_energy)
            else:
                hp_height = VIEWPORT_SIZE
                energy_height = VIEWPORT_SIZE

        highest_bar = max(hp_height, energy_height, VIEWPORT_SIZE)

        border_color = const.P1_COLOR if player_id == 1 else const.P2_COLOR

        # Create the panel surface
        panel_h = MARINE_REGION_HEIGHT + highest_bar + HUD_BOTTOM_PADDING
        panel_surface = pygame.Surface((panel_w, panel_h))
        panel_surface.fill((0, 0, 0))

        # Add translucent player color
        color_surface = pygame.Surface((panel_w, panel_h))
        color_surface.fill(border_color)
        color_surface.set_alpha(const.HUD_PANEL_ALPHA)
        panel_surface.blit(color_surface, (0, 0))

        local_base_y = panel_h - HUD_BOTTOM_PADDING
        draw_x_offset = (panel_w - _TOTAL_WIDTH) // 2

        # Draw status bars onto the panel
        if live_ship:
            draw_player_status(
                panel_surface,
                live_ship,
                draw_x_offset,
                local_base_y,
                BAR_WIDTH,
                VIEWPORT_COLUMN_WIDTH,
                viewport_size=VIEWPORT_SIZE,
            )
        else:
            _draw_empty_rect(
                panel_surface,
                pygame.Rect(
                    draw_x_offset, local_base_y - hp_height, BAR_WIDTH, hp_height
                ),
                const.HUD_BAR_BORDER,
                const.HUD_BAR_BG,
            )
            _draw_empty_rect(
                panel_surface,
                pygame.Rect(
                    draw_x_offset + BAR_WIDTH + VIEWPORT_COLUMN_WIDTH,
                    local_base_y - energy_height,
                    BAR_WIDTH,
                    energy_height,
                ),
                const.HUD_BAR_BORDER,
                const.HUD_BAR_BG,
            )

        # Draw viewport onto the panel
        if ship_to_track:
            viewport_surface.set_clip(VP_CLIP_RECT)
            viewport_surface.fill((0, 0, 0))

            from src.Battle.interpolation import interpolated_position

            base_pos = getattr(ship_to_track, "camera_position", ship_to_track.position)
            if base_pos == ship_to_track.position:
                ship_pos = interpolated_position(ship_to_track, interp_t)
            else:
                ship_pos = base_pos

            ship_translation = [
                const.SCREEN_HEIGHT / 2 - ship_pos[0],
                const.SCREEN_HEIGHT / 2 - ship_pos[1],
            ]

            _render_world_to_surface(
                viewport_surface,
                snapshot,
                1.0,
                ship_translation,
                midpoint,
                entry_state,
                frame_id,
                star_field_renderer,
                skip_stars=True,
                show_gravity_range=False,
                show_entry_trails=False,
                show_crosshairs=False,
                interp_t=interp_t,
            )

            viewport_surface.set_clip(None)

            dest_x = draw_x_offset + BAR_WIDTH + VIEWPORT_MARGIN
            dest_y = local_base_y - VIEWPORT_SIZE
            dest_rect = pygame.Rect(dest_x, dest_y, VIEWPORT_SIZE, VIEWPORT_SIZE)

            panel_surface.blit(viewport_surface, dest_rect, area=VP_CLIP_RECT)
            pygame.draw.rect(panel_surface, const.HUD_VIEWPORT_BORDER, dest_rect, 2)
        else:
            dest_x = draw_x_offset + BAR_WIDTH + VIEWPORT_MARGIN
            dest_y = local_base_y - VIEWPORT_SIZE
            _draw_empty_rect(
                panel_surface,
                pygame.Rect(dest_x, dest_y, VIEWPORT_SIZE, VIEWPORT_SIZE),
                const.HUD_VIEWPORT_BORDER,
                HUD_FILL,
            )

        if live_ship:
            draw_boarded_marine_icons(
                panel_surface,
                live_ship,
                draw_x_offset,
                0,
                _TOTAL_WIDTH,
                MARINE_REGION_HEIGHT,
            )
            draw_limpet_count(
                panel_surface,
                live_ship,
                draw_x_offset,
                local_base_y,
                _TOTAL_WIDTH,
                HUD_BOTTOM_PADDING,
            )
            draw_special_indicator(panel_surface, live_ship)

        # Blit the unified panel to the screen
        screen.blit(panel_surface, (status_x, 0))

    instruction_font = pygame.font.SysFont(None, HUD_INSTRUCTION_FONT_SIZE)
    pause_text = instruction_font.render("Press F1 to Pause", True, ui.WHITE)
    exit_text = instruction_font.render("Press Esc to Exit", True, ui.WHITE)
    screen.blit(
        pause_text,
        pause_text.get_rect(
            bottomleft=(
                HUD_INSTRUCTION_MARGIN,
                const.SCREEN_HEIGHT - HUD_INSTRUCTION_MARGIN,
            )
        ),
    )
    screen.blit(
        exit_text,
        exit_text.get_rect(
            bottomright=(
                const.SCREEN_WIDTH - HUD_INSTRUCTION_MARGIN,
                const.SCREEN_HEIGHT - HUD_INSTRUCTION_MARGIN,
            )
        ),
    )

    if is_paused:
        from src.UI.ui import WHITE

        font = pygame.font.SysFont(None, 72)
        text = font.render("PAUSED", True, WHITE)
        text_rect = text.get_rect(
            center=(const.SCREEN_WIDTH // 2, const.SCREEN_HEIGHT // 2)
        )
        screen.blit(text, text_rect)

    pygame.display.flip()
