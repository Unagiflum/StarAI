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
    """Normalize a catalog's sprites by its largest scaled dimension."""
    max_dimension = max(
        1,
        max((
            dimension * catalog[name].sprite_scale
            for name, sprite in original_sprites.items()
            for dimension in sprite.get_size()
        ), default=1),
    )
    scale_factor = target_size / max_dimension
    return {
        name: pygame.transform.scale(
            sprite,
            tuple(
                int(dimension * catalog[name].sprite_scale * scale_factor)
                for dimension in sprite.get_size()
            ),
        )
        for name, sprite in original_sprites.items()
    }


def populate_fleet_panel(panel, ship_names, sprites, catalog) -> None:
    """Populate a fleet view in persisted order from typed catalog metadata."""
    for name in ship_names:
        panel.add_ship(sprites[name], name, catalog[name].cost)
