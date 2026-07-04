"""Full-screen asset loading progress UI."""

import pygame

from src.resources import default_assets


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)


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


def draw_loading_screen(screen, completed_work, total_work):
    """Draw a text-free asset-loading progress screen."""
    screen.fill(BLACK)
    outer, inner = loading_bar_rects(screen)
    border_radius = max(1, outer.height // 4)
    inner_radius = max(1, border_radius - (outer.width - inner.width) // 2)
    pygame.draw.rect(screen, WHITE, outer, border_radius=border_radius)
    pygame.draw.rect(screen, BLACK, inner, border_radius=inner_radius)

    progress = 1.0 if total_work == 0 else completed_work / total_work
    progress = max(0.0, min(1.0, progress))
    fill_width = round(inner.width * progress)
    if fill_width:
        fill = pygame.Rect(inner.left, inner.top, fill_width, inner.height)
        pygame.draw.rect(screen, GREEN, fill, border_radius=inner_radius)
    return outer, inner


def preload_assets(screen, resources=None):
    """Preload all assets while presenting weighted loading progress."""
    resources = resources or default_assets()

    def update(completed_work, total_work):
        pygame.event.pump()
        draw_loading_screen(screen, completed_work, total_work)
        pygame.display.flip()

    return resources.preload_all(progress_callback=update)
