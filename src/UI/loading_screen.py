"""Full-screen asset loading progress UI."""

import pygame

from src.resources import default_assets


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def loading_bar_rects(screen):
    """Return the outer and inner loading-bar rectangles for ``screen``."""
    width, height = screen.get_size()
    outer_width = max(1, int(width * 0.6))
    outer_height = max(1, int(height * 0.05))
    outer = pygame.Rect(0, 0, outer_width, outer_height)
    outer.center = screen.get_rect().center

    border = max(1, min(4, int(min(width, height) * 0.004)))
    inner = outer.inflate(-2 * border, -2 * border)
    if inner.width < 0 or inner.height < 0:
        inner = outer.copy()
    return outer, inner


def draw_loading_screen(screen, loaded_ships, total_ships):
    """Draw a text-free ship-loading progress screen."""
    screen.fill(BLACK)
    outer, inner = loading_bar_rects(screen)
    pygame.draw.rect(screen, WHITE, outer)
    pygame.draw.rect(screen, BLACK, inner)

    progress = 1.0 if total_ships == 0 else loaded_ships / total_ships
    progress = max(0.0, min(1.0, progress))
    fill_width = round(inner.width * progress)
    if fill_width:
        fill = pygame.Rect(inner.left, inner.top, fill_width, inner.height)
        pygame.draw.rect(screen, GREEN, fill)
    return outer, inner


def preload_assets(screen, resources=None):
    """Preload all assets while presenting ship-loading progress."""
    resources = resources or default_assets()

    def update(loaded_ships, total_ships):
        pygame.event.pump()
        draw_loading_screen(screen, loaded_ships, total_ships)
        pygame.display.flip()

    return resources.preload_all(progress_callback=update)
