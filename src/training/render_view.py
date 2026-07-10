"""Stable display snapshots for training UI rendering."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


def freeze_battle_view(battle_view: Mapping[str, Any]) -> dict[str, Any]:
    """Return a render-only copy of a live battle view.

    The battle worker owns the authoritative objects.  Training display code
    needs a stable graph so arena and HUD rendering cannot observe those
    objects halfway through a simulation step.
    """

    game_objects = tuple(battle_view.get("game_objects", ()))
    original_ships = tuple(battle_view.get("original_ships", ()))
    camera_targets = tuple(battle_view.get("camera_targets", ()))

    clones = _clone_object_graph((*game_objects, *original_ships, *camera_targets))
    frozen = dict(battle_view)
    frozen["game_objects"] = tuple(_freeze_value(obj, clones) for obj in game_objects)
    frozen["original_ships"] = tuple(
        _freeze_value(obj, clones) for obj in original_ships
    )
    frozen["camera_targets"] = tuple(
        _freeze_value(obj, clones) for obj in camera_targets
    )
    if "border_rect" in frozen:
        frozen["border_rect"] = _copy_rect(frozen["border_rect"])
    return frozen


def _clone_object_graph(objects: tuple[Any, ...]) -> dict[int, Any]:
    clones: dict[int, Any] = {}
    ordered = []
    for obj in objects:
        if id(obj) in clones:
            continue
        ordered.append(obj)
        try:
            clones[id(obj)] = copy.copy(obj)
        except Exception:
            clones[id(obj)] = obj

    for obj in ordered:
        clone = clones[id(obj)]
        if clone is obj:
            continue
        attributes = getattr(obj, "__dict__", None)
        if not attributes:
            continue
        for name, value in attributes.items():
            try:
                setattr(clone, name, _freeze_value(value, clones))
            except Exception:
                pass
    return clones


def _freeze_value(value: Any, clones: dict[int, Any]) -> Any:
    cloned = clones.get(id(value))
    if cloned is not None:
        return cloned

    if isinstance(value, list):
        return [_freeze_value(item, clones) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_value(item, clones) for item in value)
    if isinstance(value, set):
        return {_freeze_value(item, clones) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(item, clones) for item in value)
    if isinstance(value, dict):
        return {
            _freeze_value(key, clones): _freeze_value(item, clones)
            for key, item in value.items()
        }
    return _copy_rect(value)


def _copy_rect(value: Any) -> Any:
    copy_method = getattr(value, "copy", None)
    if (
        callable(copy_method)
        and value.__class__.__module__.startswith("pygame")
        and value.__class__.__name__ == "Rect"
    ):
        try:
            return copy_method()
        except Exception:
            return value
    return value
