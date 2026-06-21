"""Canonical discovery and construction for ships and abilities."""

from functools import lru_cache
from importlib import import_module

from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS


@lru_cache(maxsize=None)
def get_ship_class(ship_name):
    if ship_name not in SHIP_DEFINITIONS:
        raise KeyError(f"Unknown ship: {ship_name}")
    module = import_module(f"src.Objects.Ships.{ship_name}.{ship_name}")
    return getattr(module, ship_name)


@lru_cache(maxsize=None)
def get_ability_class(ability_name):
    if ability_name not in ABILITY_DEFINITIONS:
        raise KeyError(f"Unknown ability: {ability_name}")
    ability_definition = ABILITY_DEFINITIONS[ability_name]
    module = import_module(
        f"src.Objects.Ships.{ability_definition.ship_name}."
        f"{ability_definition.action}.{ability_name}"
    )
    return getattr(module, ability_name)


def create_ship(ship_name, player_num, resources=None):
    ship_class = get_ship_class(ship_name)
    if resources is None:
        return ship_class(ship_name, player_num)
    return ship_class(ship_name, player_num, resources=resources)


def create_ability(ability_name, parent, *args, **kwargs):
    return get_ability_class(ability_name)(parent, *args, **kwargs)


def ability_names_for_ship(ship_name):
    if ship_name not in SHIP_DEFINITIONS:
        raise KeyError(f"Unknown ship: {ship_name}")
    return tuple(
        ability_name
        for ability_name, ability_definition in ABILITY_DEFINITIONS.items()
        if ability_definition.ship_name == ship_name
    )


def preload_ship_ability_resources(ship_name, resources=None):
    for ability_name in ability_names_for_ship(ship_name):
        Ability.preload_resources(ability_name, resources)
