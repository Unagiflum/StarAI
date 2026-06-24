import pygame
from src.UI import ui
from src.Battle.status_bar import draw_player_status, StatusBar
from src.Battle.battle_entry import draw_entry_silhouettes
import src.const as const
from src.toroidal import view_center_and_size, wrapped_delta, wrapped_midpoint
from src.Battle.world import World


class StarFieldRenderer:
    def __init__(self):
        self.depth_surfaces = [
            pygame.Surface(
                (const.SCREEN_WIDTH, const.SCREEN_HEIGHT), pygame.SRCALPHA
            )
            for _ in range(const.STAR_DEPTHS)
        ]

    def draw(self, screen, stars, scale_factor, translation, midpoint):
        stars_by_depth = [[] for _ in range(const.STAR_DEPTHS)]
        for star in stars:
            stars_by_depth[star.depth].append(star)

        for depth, depth_stars in enumerate(stars_by_depth):
            parallax_factor = 0.5 + 0.5 * (
                depth / (const.STAR_DEPTHS - 1)
            )
            self.update_depth_surface(
                depth,
                depth_stars,
                scale_factor,
                translation,
                midpoint,
                parallax_factor,
            )
            screen.blit(self.depth_surfaces[depth], (0, 0))

    def update_depth_surface(
        self,
        depth,
        stars,
        scale_factor,
        translation,
        midpoint,
        parallax_factor,
    ):
        surface = self.depth_surfaces[depth]
        surface.fill((0, 0, 0, 0))

        for star in stars:
            dx, dy = wrapped_delta(midpoint, star.position)

            relative_x = midpoint[0] + dx * parallax_factor
            relative_y = midpoint[1] + dy * parallax_factor

            screen_x = int((relative_x + translation[0]) * scale_factor)
            screen_y = int((relative_y + translation[1]) * scale_factor)

            scaled_image = pygame.transform.smoothscale_by(
                star.image, scale_factor
            )
            scaled_image.set_alpha(const.STAR_ALPHA)
            star_size = scaled_image.get_width()

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    pos_x = screen_x + dx * const.ARENA_SIZE * scale_factor
                    pos_y = screen_y + dy * const.ARENA_SIZE * scale_factor

                    if (
                        -star_size
                        <= pos_x
                        <= const.SCREEN_HEIGHT + star_size
                        and -star_size
                        <= pos_y
                        <= const.SCREEN_HEIGHT + star_size
                    ):
                        surface.blit(
                            scaled_image,
                            (
                                const.SCREEN_LEFT + pos_x - star_size // 2,
                                pos_y - star_size // 2,
                            ),
                        )


def calculate_view_parameters(game_objects, camera_targets=None):
    world = World.coerce(game_objects)
    targets = camera_targets
    if targets is None:
        targets = world.live_ships

    if len(targets) == 1:
        view_size = (const.SCREEN_HEIGHT / const.MAX_ZOOM) * 1.5
        scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
        pos0 = getattr(targets[0], "camera_position", targets[0].position)
        translation = [
            const.SCREEN_HEIGHT / (2 * scale_factor) - pos0[0],
            const.SCREEN_HEIGHT / (2 * scale_factor) - pos0[1],
        ]
        return scale_factor, translation
    if len(targets) < 2:
        return 1.0, [0, 0]

    pos0 = getattr(targets[0], "camera_position", targets[0].position)
    pos1 = getattr(targets[1], "camera_position", targets[1].position)
    center, view_size = view_center_and_size(
        [pos0, pos1]
    )

    scale_factor = min(const.MAX_ZOOM, const.SCREEN_HEIGHT / view_size)
    translation = [
        const.SCREEN_HEIGHT / (2 * scale_factor) - center[0],
        const.SCREEN_HEIGHT / (2 * scale_factor) - center[1]
    ]

    return scale_factor, translation


def _render_world_to_surface(
    surface,
    world,
    scale_factor,
    translation,
    midpoint,
    entry_state,
    frame_id,
    star_field_renderer,
):
    star_field_renderer.draw(
        surface, world.stars, scale_factor, translation, midpoint
    )

    for planet in world.planets:
        planet.draw(surface, scale_factor, translation)

    for marker in world.thrust_markers:
        marker.draw(surface, scale_factor, translation)

    for asteroid in world.asteroids:
        asteroid.draw(surface, scale_factor, translation)

    for ability in world.abilities:
        ability.draw(surface, scale_factor, translation)

    entering_ships = set(
        entry_state.entering_ships if entry_state else ()
    )
    if entry_state:
        draw_entry_silhouettes(
            surface,
            entry_state,
            frame_id,
            scale_factor,
            translation,
        )

    for ship in world.ships:
        if ship not in entering_ships:
            ship.draw(surface, scale_factor, translation)

    for effect in world.effects:
        effect.draw(surface, scale_factor, translation)


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
):
    world = World.coerce(game_objects)
    scale_factor, translation = calculate_view_parameters(world, camera_targets)

    players = world.live_ships
    if len(players) == 2:
        p1_pos, p2_pos = players[0].position, players[1].position
        midpoint = wrapped_midpoint(p1_pos, p2_pos)
    else:
        midpoint = [const.ARENA_SIZE / 2, const.ARENA_SIZE / 2]

    screen.fill(ui.BLACK)
    screen.set_clip(border_rect)

    _render_world_to_surface(
        screen,
        world,
        scale_factor,
        translation,
        midpoint,
        entry_state,
        frame_id,
        star_field_renderer,
    )

    pygame.draw.rect(screen, border_color, border_rect, 2)
    screen.set_clip(None)

    # Draw status bars and viewports for both players.
    VIEWPORT_SIZE = 200
    BAR_WIDTH = 30
    BAR_SPACING = VIEWPORT_SIZE + 14

    # Calculate total width of both bars + spacing
    TOTAL_WIDTH = (BAR_WIDTH * 2) + BAR_SPACING

    # Calculate panel widths (space between arena edge and screen edge)
    LEFT_PANEL_WIDTH = const.SCREEN_LEFT
    RIGHT_PANEL_WIDTH = const.SCREEN_WIDTH - (const.SCREEN_LEFT + const.SCREEN_HEIGHT)

    # Center layout horizontally in panels
    P1_X = const.SCREEN_LEFT - TOTAL_WIDTH - ((LEFT_PANEL_WIDTH - TOTAL_WIDTH) // 2)
    P2_X = (const.SCREEN_LEFT + const.SCREEN_HEIGHT) + ((RIGHT_PANEL_WIDTH - TOTAL_WIDTH) // 2)

    # Position at the bottom of the margin, with a bit of padding
    BASE_Y = const.SCREEN_HEIGHT - 20

    viewport_surface = getattr(draw_battle, "_viewport_surface", None)
    if viewport_surface is None or viewport_surface.get_size() != (const.SCREEN_WIDTH, const.SCREEN_HEIGHT):
        viewport_surface = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        draw_battle._viewport_surface = viewport_surface

    for player_id in (1, 2):
        status_x = P1_X if player_id == 1 else P2_X
        live_ship = next((s for s in players if s.player == player_id), None)
        ship_to_track = live_ship or next((s for s in world.ships if s.player == player_id), None)
        
        # Draw status bars
        if live_ship:
            draw_player_status(screen, live_ship, status_x, BASE_Y, BAR_WIDTH, BAR_SPACING)
        else:
            dark_gray = (0, 0, 0)
            border_color_hud = (100, 100, 100)
            
            if original_ships:
                dead_ship = next((s for s in original_ships if s.player == player_id), None)
            else:
                dead_ship = ship_to_track
                
            hp_height = 200
            energy_height = 200
            
            if dead_ship:
                hp_height = StatusBar.calculate_height(dead_ship.max_hp)
                energy_height = StatusBar.calculate_height(dead_ship.max_energy)
            
            hp_rect = pygame.Rect(status_x, BASE_Y - hp_height, BAR_WIDTH, hp_height)
            energy_rect = pygame.Rect(status_x + BAR_WIDTH + BAR_SPACING, BASE_Y - energy_height, BAR_WIDTH, energy_height)
            
            pygame.draw.rect(screen, dark_gray, hp_rect)
            pygame.draw.rect(screen, border_color_hud, hp_rect, 1)
            
            pygame.draw.rect(screen, dark_gray, energy_rect)
            pygame.draw.rect(screen, border_color_hud, energy_rect, 1)

        # Draw viewport
        if ship_to_track:
            center_x = const.SCREEN_LEFT + const.SCREEN_HEIGHT // 2
            center_y = const.SCREEN_HEIGHT // 2
            
            clip_rect = pygame.Rect(
                center_x - VIEWPORT_SIZE // 2,
                center_y - VIEWPORT_SIZE // 2,
                VIEWPORT_SIZE,
                VIEWPORT_SIZE
            )
            viewport_surface.set_clip(clip_rect)
            viewport_surface.fill((0, 0, 0))

            ship_pos = getattr(ship_to_track, "camera_position", ship_to_track.position)
            ship_translation = [
                const.SCREEN_HEIGHT / 2 - ship_pos[0],
                const.SCREEN_HEIGHT / 2 - ship_pos[1]
            ]

            _render_world_to_surface(
                viewport_surface,
                world,
                1.0,
                ship_translation,
                midpoint,
                entry_state,
                frame_id,
                star_field_renderer,
            )

            viewport_surface.set_clip(None)

            dest_x = status_x + BAR_WIDTH + 7
            dest_y = BASE_Y - VIEWPORT_SIZE
            dest_rect = pygame.Rect(dest_x, dest_y, VIEWPORT_SIZE, VIEWPORT_SIZE)

            screen.blit(viewport_surface, dest_rect, area=clip_rect)
            pygame.draw.rect(screen, (100, 100, 100), dest_rect, 1)
        else:
            dark_gray = (0, 0, 0)
            border_color_hud = (100, 100, 100)
            
            dest_x = status_x + BAR_WIDTH + 7
            dest_y = BASE_Y - VIEWPORT_SIZE
            dest_rect = pygame.Rect(dest_x, dest_y, VIEWPORT_SIZE, VIEWPORT_SIZE)
            
            pygame.draw.rect(screen, dark_gray, dest_rect)
            pygame.draw.rect(screen, border_color_hud, dest_rect, 1)

    pygame.display.flip()
