import pygame
from src.UI import ui
from src.Battle.status_bar import draw_player_status
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


def draw_battle(
    screen,
    game_objects,
    border_rect,
    border_color,
    star_field_renderer,
    camera_targets=None,
    entry_state=None,
    frame_id=0,
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

    star_field_renderer.draw(
        screen, world.stars, scale_factor, translation, midpoint
    )

    # Draw other objects normally
    for planet in world.planets:
        planet.draw(screen, scale_factor, translation)

    for marker in world.thrust_markers:
        marker.draw(screen, scale_factor, translation)

    for asteroid in world.asteroids:
        asteroid.draw(screen, scale_factor, translation)

    for ability in world.abilities:
        ability.draw(screen, scale_factor, translation)

    entering_ships = set(
        entry_state.entering_ships if entry_state else ()
    )
    if entry_state:
        draw_entry_silhouettes(
            screen,
            entry_state,
            frame_id,
            scale_factor,
            translation,
        )

    for ship in world.ships:
        if ship not in entering_ships:
            ship.draw(screen, scale_factor, translation)

    for effect in world.effects:
        effect.draw(screen, scale_factor, translation)

    pygame.draw.rect(screen, border_color, border_rect, 2)
    screen.set_clip(None)

    # Draw status bars for each surviving player. During the post-fight winner
    # view only one ship remains alive, but its crew and battery are still
    # useful and should remain visible.
    if players:
        BAR_WIDTH = 30
        BAR_SPACING = 5

        # Calculate total width of both bars + spacing
        TOTAL_BAR_WIDTH = (BAR_WIDTH * 2) + BAR_SPACING

        # Calculate panel widths (space between arena edge and screen edge)
        LEFT_PANEL_WIDTH = const.SCREEN_LEFT
        RIGHT_PANEL_WIDTH = const.SCREEN_WIDTH - (const.SCREEN_LEFT + const.SCREEN_HEIGHT)

        # Center bars in panels
        P1_X = const.SCREEN_LEFT - TOTAL_BAR_WIDTH - ((LEFT_PANEL_WIDTH - TOTAL_BAR_WIDTH) // 2)
        P2_X = (const.SCREEN_LEFT + const.SCREEN_HEIGHT) + ((RIGHT_PANEL_WIDTH - TOTAL_BAR_WIDTH) // 2)

        BASE_Y = const.SCREEN_HEIGHT // 2

        for ship in players:
            status_x = P1_X if ship.player == 1 else P2_X
            draw_player_status(screen, ship, status_x, BASE_Y, BAR_WIDTH, BAR_SPACING)

    pygame.display.flip()
