"""Shared safety rules for computer-controlled ship actions."""

from __future__ import annotations

from collections.abc import Mapping

from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.launch_geometry import gun_world_position
from src.toroidal import wrapped_distance


_A2_MASK = 16


def prepare_computer_controlled_ship(ship) -> None:
    """Apply computer-only round initialization without changing human ships."""

    if getattr(ship, "name", None) == "Shofixti":
        ship.shofixti_arming_stage = getattr(ship, "ARMED", 2)


def computer_action2_allowed(ship, enemy) -> bool:
    """Return whether a computer-controlled ship may currently request A2."""

    ship_name = getattr(ship, "name", None)
    if ship_name == "Druuge":
        return float(getattr(ship, "current_energy", 0.0)) < float(
            getattr(ship, "a1_cost", 0.0)
        )
    if ship_name == "Shofixti":
        return _living_enemy_in_shofixti_range(ship, enemy)
    return True


def guard_computer_controls(controls, ship, enemy):
    """Clear A2 from controls when the computer-only safety rule forbids it."""

    if computer_action2_allowed(ship, enemy):
        return controls

    if isinstance(controls, Mapping):
        guarded = dict(controls)
        if "action2" in guarded:
            guarded["action2"] = False
        if "a2" in guarded:
            guarded["a2"] = False
        return guarded

    mask = getattr(controls, "mask", None)
    from_mask = getattr(type(controls), "from_mask", None)
    if isinstance(mask, int) and callable(from_mask):
        return from_mask(mask & ~_A2_MASK)
    return controls


def _living_enemy_in_shofixti_range(ship, enemy) -> bool:
    if not _is_living(enemy):
        return False

    # Cloak and trackability are intentionally irrelevant: a present ship is
    # still affected by the radial explosion.
    definition = ABILITY_DEFINITIONS["ShofixtiA2"]
    effect_range = float(definition.range)
    origin = getattr(ship, "position", (0.0, 0.0))
    locations = definition.gun_locations or ()
    if locations:
        try:
            origin = gun_world_position(ship, locations[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
    target = getattr(enemy, "position", (0.0, 0.0))
    return wrapped_distance(origin, target) <= effect_range


def _is_living(ship) -> bool:
    return bool(
        ship is not None
        and getattr(ship, "currently_alive", True)
        and getattr(ship, "current_hp", 1) > 0
    )
