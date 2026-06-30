"""Shared Pygame sprite loading and menu-scale normalization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import pygame

from src.Objects.Ships.catalog import ShipDefinition
from src.resources import default_assets


def load_menu_ship_sprites(
    ship_names: Iterable[str],
    *,
    resources=None,
    fallback: Callable[[str], pygame.Surface] | None = None,
) -> dict[str, pygame.Surface]:
    resources = resources or default_assets()
    sprites = {}
    for ship_name in ship_names:
        try:
            sprites[ship_name] = resources.menu_ship_sprite(ship_name)
        except (OSError, pygame.error) as error:
            print(f"Error loading sprite for {ship_name}: {error}")
            if fallback is not None:
                sprites[ship_name] = fallback(ship_name)
    return sprites


def scale_ship_sprites(
    original_sprites: Mapping[str, pygame.Surface],
    target_size: int,
    catalog: Mapping[str, ShipDefinition],
) -> dict[str, pygame.Surface]:
    """Scale every sprite equally, preserving relative source-art dimensions."""
    _ = catalog  # Retained for API compatibility with existing menu callers.
    max_dimension = max(
        1,
        max(
            (
                dimension
                for sprite in original_sprites.values()
                for dimension in sprite.get_size()
            ),
            default=1,
        ),
    )
    scale_factor = target_size / max_dimension
    return {
        name: pygame.transform.scale(
            sprite,
            tuple(
                max(1, int(dimension * scale_factor))
                for dimension in sprite.get_size()
            ),
        )
        for name, sprite in original_sprites.items()
    }


def fit_ship_sprites(
    original_sprites: Mapping[str, pygame.Surface], max_size: int
) -> dict[str, pygame.Surface]:
    """Fit each sprite independently without enlarging its source art."""
    fitted = {}
    for name, sprite in original_sprites.items():
        scale_factor = min(1.0, max_size / max(1, *sprite.get_size()))
        if scale_factor == 1.0:
            fitted[name] = sprite
            continue
        fitted[name] = pygame.transform.scale(
            sprite,
            tuple(
                max(1, int(dimension * scale_factor))
                for dimension in sprite.get_size()
            ),
        )
    return fitted


def populate_fleet_panel(panel, ship_names, sprites, catalog) -> None:
    """Populate a fleet view while preserving persisted empty positions."""
    for index, name in enumerate(ship_names):
        if name is None:
            continue
        if hasattr(panel, "set_ship_at_slot"):
            panel.set_ship_at_slot(index, sprites[name], name, catalog[name].cost)
        else:
            panel.add_ship(sprites[name], name, catalog[name].cost)
