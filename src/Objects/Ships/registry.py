"""Canonical discovery and construction for ships and abilities."""

from functools import lru_cache
from importlib import import_module

from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITIES_DATA, SHIPS_DATA


@lru_cache(maxsize=None)
def get_ship_class(ship_name):
    if ship_name not in SHIPS_DATA:
        raise KeyError(f"Unknown ship: {ship_name}")
    module = import_module(f"src.Objects.Ships.{ship_name}.{ship_name}")
    return getattr(module, ship_name)


@lru_cache(maxsize=None)
def get_ability_class(ability_name):
    if ability_name not in ABILITIES_DATA:
        raise KeyError(f"Unknown ability: {ability_name}")
    ability_data = ABILITIES_DATA[ability_name]
    module = import_module(
        f"src.Objects.Ships.{ability_data['ship_name']}."
        f"{ability_data['action']}.{ability_name}"
    )
    return getattr(module, ability_name)


def create_ship(ship_name, player_num):
    return get_ship_class(ship_name)(ship_name, player_num)


def create_ability(ability_name, parent, *args, **kwargs):
    return get_ability_class(ability_name)(parent, *args, **kwargs)


def ability_names_for_ship(ship_name):
    if ship_name not in SHIPS_DATA:
        raise KeyError(f"Unknown ship: {ship_name}")
    return tuple(
        ability_name
        for ability_name, ability_data in ABILITIES_DATA.items()
        if ability_data['ship_name'] == ship_name
    )


def preload_ship_ability_resources(ship_name):
    for ability_name in ability_names_for_ship(ship_name):
        Ability.preload_resources(ability_name)
