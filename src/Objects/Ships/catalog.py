"""Validated, immutable ship and ability definitions loaded from JSON."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

import src.const as const
from src.persistence import read_json


class CatalogValidationError(ValueError):
    """Raised when a catalog entry does not match its definition schema."""


def _json_value(value):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


class _DefinitionMapping(Mapping):
    """Read-only JSON-key access retained for unmigrated consumers."""

    _json_key_to_attribute: ClassVar[dict[str, str]]
    _source_keys: tuple[str, ...]

    def __getitem__(self, key):
        if key not in self._source_keys:
            raise KeyError(key)
        try:
            attribute = self._json_key_to_attribute[key]
        except KeyError:
            raise KeyError(key) from None
        return _json_value(getattr(self, attribute))

    def __iter__(self):
        return iter(self._source_keys)

    def __len__(self):
        return len(self._source_keys)

    def to_json_dict(self):
        """Return a mutable JSON-compatible copy in the original key format."""
        return {key: self[key] for key in self}


@dataclass(frozen=True)
class ShipDefinition(_DefinitionMapping):
    ship_type: str
    cost: int
    max_hp: int
    start_hp: int
    max_energy: int
    start_energy: int
    energy_regen: int
    energy_wait: int
    max_thrust: float
    thrust_increment: float
    thrust_wait: float
    turn_wait: float
    a1_cost: int
    a2_cost: int
    a3_cost: int
    a1_wait: float
    a2_wait: float
    a3_wait: float
    mass: float
    inertia: bool
    sprite_path: str
    sprite_scale: float = 1.0
    menu_overlay_path: str | None = None
    fade_duration: int = 8
    saw_count: int = 8
    gas_count: int = 16
    initial_rebirth_chance: float | None = None
    rebirth_chance_decay: float | None = None
    _source_keys: tuple[str, ...] = field(default=(), repr=False, compare=False)

    _json_key_to_attribute = {
        "ship_type": "ship_type",
        "cost": "cost",
        "max_hp": "max_hp",
        "start_hp": "start_hp",
        "max_energy": "max_energy",
        "start_energy": "start_energy",
        "energy_regen": "energy_regen",
        "energy_wait": "energy_wait",
        "max_thrust": "max_thrust",
        "thrust_increment": "thrust_increment",
        "thrust_wait": "thrust_wait",
        "turn_wait": "turn_wait",
        "a1_cost": "a1_cost",
        "a2_cost": "a2_cost",
        "a3_cost": "a3_cost",
        "a1_wait": "a1_wait",
        "a2_wait": "a2_wait",
        "a3_wait": "a3_wait",
        "mass": "mass",
        "inertia": "inertia",
        "sprite_path": "sprite_path",
        "sprite_scale": "sprite_scale",
        "menu_overlay_path": "menu_overlay_path",
        "FADE_DURATION": "fade_duration",
        "SAW_COUNT": "saw_count",
        "GAS_COUNT": "gas_count",
        "initial_rebirth_chance": "initial_rebirth_chance",
        "rebirth_chance_decay": "rebirth_chance_decay",
    }


@dataclass(frozen=True)
class AbilityDefinition(_DefinitionMapping):
    ship_name: str
    ability_type: str
    action: str
    start_hp: tuple[int, ...]
    damage: tuple[int, ...]
    tracking: bool
    parent_vel: float
    speed: float
    life_time: float
    inertia: bool
    hit_parent: bool
    hit_self: bool
    omnidirectional: bool
    file_path: str
    turn_wait: float = 0
    end_anim: int = 0
    sprite_scale: float = 1.0
    sprite_scale_x: float = 1.0
    sprite_scale_y: float = 1.0
    frames: int = 1
    frame_delay: int = 0
    has_sprites: bool = True
    has_sound: bool = True
    laser_vulnerable: bool = True
    collide_planets: bool = True
    collide_asteroids: bool = True
    damage_asteroids: bool = True
    collide_projectiles: bool = True
    damage_projectiles: bool = True
    collide_enemy_ships: bool = True
    collide_friendly_ships: bool = False
    collide_fighters: bool = True
    effect_range: float | None = None
    one_way_flight: float | None = None
    life_margin: float | None = None
    launch_time: float | None = None
    mass: float | None = None
    offset: float | None = None
    track_directions: int | None = None
    weapon_wait: float | None = None
    laser_range: float | None = None
    laser_color: tuple[int, int, int] | None = None
    laser_width: int | None = None
    max_recoil: float | None = None
    recoil_increment: float | None = None
    energy_gain: int | None = None
    hp_gain: int | None = None
    track_speed: float | None = None
    track_range: float | None = None
    damage_to_projectiles: int | None = None
    reunk_thrust: float | None = None
    reunk_increment: float | None = None
    spread_angle: float | None = None
    max_thrust: float | None = None
    thrust_increment: float | None = None
    thrust_wait: float | None = None
    look_ahead: int | None = None
    max_marines: int | None = None
    spiral_distance: float | None = None
    retraction_frames: int | None = None
    advancing_frames: int | None = None
    retracting_frames: int | None = None
    area_width: int | None = None
    area_length: int | None = None
    gun_locations: tuple[tuple[int, int], ...] | None = None
    _source_keys: tuple[str, ...] = field(default=(), repr=False, compare=False)

    _json_key_to_attribute = {
        "ship_name": "ship_name",
        "type": "ability_type",
        "action": "action",
        "start_hp": "start_hp",
        "damage": "damage",
        "tracking": "tracking",
        "parent_vel": "parent_vel",
        "speed": "speed",
        "life_time": "life_time",
        "turn_wait": "turn_wait",
        "inertia": "inertia",
        "hit_parent": "hit_parent",
        "hit_self": "hit_self",
        "omnidirectional": "omnidirectional",
        "frames": "frames",
        "frame_delay": "frame_delay",
        "file_path": "file_path",
        "end_anim": "end_anim",
        "sprite_scale": "sprite_scale",
        "sprite_scale_x": "sprite_scale_x",
        "sprite_scale_y": "sprite_scale_y",
        "has_sprites": "has_sprites",
        "has_sound": "has_sound",
        "laser_vulnerable": "laser_vulnerable",
        "collide_planets": "collide_planets",
        "collide_asteroids": "collide_asteroids",
        "damage_asteroids": "damage_asteroids",
        "collide_projectiles": "collide_projectiles",
        "damage_projectiles": "damage_projectiles",
        "collide_enemy_ships": "collide_enemy_ships",
        "collide_friendly_ships": "collide_friendly_ships",
        "collide_fighters": "collide_fighters",
        "range": "effect_range",
        "one_way_flight": "one_way_flight",
        "life_margin": "life_margin",
        "launch_time": "launch_time",
        "mass": "mass",
        "offset": "offset",
        "track_directions": "track_directions",
        "weapon_wait": "weapon_wait",
        "LASER_RANGE": "laser_range",
        "LASER_COLOR": "laser_color",
        "LASER_WIDTH": "laser_width",
        "MAX_RECOIL": "max_recoil",
        "RECOIL_INCREMENT": "recoil_increment",
        "ENERGY_GAIN": "energy_gain",
        "HP_GAIN": "hp_gain",
        "TRACK_SPEED": "track_speed",
        "TRACK_RANGE": "track_range",
        "DMG_TO_PROJ": "damage_to_projectiles",
        "REUNK_THRUST": "reunk_thrust",
        "REUNK_INCREMENT": "reunk_increment",
        "SPREAD_ANGLE": "spread_angle",
        "max_thrust": "max_thrust",
        "thrust_increment": "thrust_increment",
        "thrust_wait": "thrust_wait",
        "look_ahead": "look_ahead",
        "max_marines": "max_marines",
        "spiral_distance": "spiral_distance",
        "retraction_frames": "retraction_frames",
        "advancing_frames": "advancing_frames",
        "retracting_frames": "retracting_frames",
        "area_width": "area_width",
        "area_length": "area_length",
        "gun_locations": "gun_locations",
    }


def _entry_mapping(kind, name, data):
    if not isinstance(name, str) or not name:
        raise CatalogValidationError(f"{kind} names must be non-empty strings")
    if not isinstance(data, Mapping):
        raise CatalogValidationError(f"{kind} '{name}' must be a JSON object")
    return data


def _check_keys(kind, name, data, allowed, required):
    unknown = set(data) - allowed
    if unknown:
        fields = ", ".join(sorted(unknown))
        raise CatalogValidationError(f"{kind} '{name}' has unknown field(s): {fields}")
    missing = required - set(data)
    if missing:
        fields = ", ".join(sorted(missing))
        raise CatalogValidationError(f"{kind} '{name}' is missing field(s): {fields}")


def _typed(kind, name, field_name, value, expected_type):
    if expected_type is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected_type is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected_type)
    if not valid:
        expected_name = "number" if expected_type is float else expected_type.__name__
        raise CatalogValidationError(
            f"{kind} '{name}' field '{field_name}' must be {expected_name}"
        )
    return value


def _optional_typed(kind, name, data, field_name, expected_type, default):
    if field_name not in data:
        return default
    return _typed(kind, name, field_name, data[field_name], expected_type)


def _int_tuple(kind, name, data, field_name, length=None):
    value = data[field_name]
    if not isinstance(value, list) or not value:
        raise CatalogValidationError(
            f"{kind} '{name}' field '{field_name}' must be a non-empty array"
        )
    result = tuple(
        _typed(kind, name, f"{field_name}[{index}]", item, int)
        for index, item in enumerate(value)
    )
    if length is not None and len(result) != length:
        raise CatalogValidationError(
            f"{kind} '{name}' field '{field_name}' must contain {length} values"
        )
    return result


def _int_pair_tuple(kind, name, data, field_name):
    value = data[field_name]
    if not isinstance(value, list) or not value:
        raise CatalogValidationError(
            f"{kind} '{name}' field '{field_name}' must be a non-empty array"
        )
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise CatalogValidationError(
                f"{kind} '{name}' field '{field_name}[{index}]' must be a 2-element array"
            )
        result.append(
            (
                _typed(kind, name, f"{field_name}[{index}][0]", item[0], int),
                _typed(kind, name, f"{field_name}[{index}][1]", item[1], int),
            )
        )
    return tuple(result)


def parse_ship_definition(name, data):
    """Validate one JSON ship object and return its immutable definition."""
    kind = "Ship"
    data = _entry_mapping(kind, name, data)
    allowed = set(ShipDefinition._json_key_to_attribute)
    required = allowed - {
        "sprite_scale",
        "menu_overlay_path",
        "FADE_DURATION",
        "SAW_COUNT",
        "GAS_COUNT",
        "initial_rebirth_chance",
        "rebirth_chance_decay",
    }
    _check_keys(kind, name, data, allowed, required)

    string_fields = ("ship_type", "sprite_path")
    integer_fields = (
        "cost",
        "max_hp",
        "start_hp",
        "max_energy",
        "start_energy",
        "energy_regen",
        "energy_wait",
        "a1_cost",
        "a2_cost",
        "a3_cost",
    )
    number_fields = (
        "max_thrust",
        "thrust_increment",
        "thrust_wait",
        "turn_wait",
        "a1_wait",
        "a2_wait",
        "a3_wait",
        "mass",
    )
    values = {key: _typed(kind, name, key, data[key], str) for key in string_fields}
    values.update(
        {key: _typed(kind, name, key, data[key], int) for key in integer_fields}
    )
    values.update(
        {key: _typed(kind, name, key, data[key], float) for key in number_fields}
    )
    values["inertia"] = _typed(kind, name, "inertia", data["inertia"], bool)
    values["sprite_scale"] = _optional_typed(
        kind, name, data, "sprite_scale", float, 1.0
    )
    values["menu_overlay_path"] = _optional_typed(
        kind, name, data, "menu_overlay_path", str, None
    )
    values["fade_duration"] = _optional_typed(kind, name, data, "FADE_DURATION", int, 8)
    values["saw_count"] = _optional_typed(kind, name, data, "SAW_COUNT", int, 8)
    values["gas_count"] = _optional_typed(kind, name, data, "GAS_COUNT", int, 16)
    values["initial_rebirth_chance"] = _optional_typed(
        kind, name, data, "initial_rebirth_chance", float, None
    )
    values["rebirth_chance_decay"] = _optional_typed(
        kind, name, data, "rebirth_chance_decay", float, None
    )
    values["_source_keys"] = tuple(data)

    if values["start_hp"] > values["max_hp"]:
        raise CatalogValidationError(f"Ship '{name}' start_hp exceeds max_hp")
    if values["start_energy"] > values["max_energy"]:
        raise CatalogValidationError(f"Ship '{name}' start_energy exceeds max_energy")
    if values["sprite_scale"] <= 0:
        raise CatalogValidationError(f"Ship '{name}' sprite_scale must be positive")
    if (
        values["initial_rebirth_chance"] is not None
        and not 0 <= values["initial_rebirth_chance"] <= 1
    ):
        raise CatalogValidationError(
            f"Ship '{name}' initial_rebirth_chance must be between 0 and 1"
        )
    if (
        values["rebirth_chance_decay"] is not None
        and not 0 <= values["rebirth_chance_decay"] <= 1
    ):
        raise CatalogValidationError(
            f"Ship '{name}' rebirth_chance_decay must be between 0 and 1"
        )
    return ShipDefinition(**values)


def parse_ability_definition(name, data):
    """Validate one JSON ability object and return its immutable definition."""
    kind = "Ability"
    data = _entry_mapping(kind, name, data)
    allowed = set(AbilityDefinition._json_key_to_attribute)
    optional = {
        "turn_wait",
        "end_anim",
        "sprite_scale",
        "sprite_scale_x",
        "sprite_scale_y",
        "frames",
        "frame_delay",
        "has_sprites",
        "has_sound",
        "laser_vulnerable",
        "collide_planets",
        "collide_asteroids",
        "damage_asteroids",
        "collide_projectiles",
        "damage_projectiles",
        "collide_enemy_ships",
        "collide_friendly_ships",
        "collide_fighters",
        "range",
        "one_way_flight",
        "life_margin",
        "launch_time",
        "mass",
        "offset",
        "track_directions",
        "weapon_wait",
        "LASER_RANGE",
        "LASER_COLOR",
        "LASER_WIDTH",
        "MAX_RECOIL",
        "RECOIL_INCREMENT",
        "ENERGY_GAIN",
        "HP_GAIN",
        "TRACK_SPEED",
        "TRACK_RANGE",
        "DMG_TO_PROJ",
        "REUNK_THRUST",
        "REUNK_INCREMENT",
        "SPREAD_ANGLE",
        "max_thrust",
        "thrust_increment",
        "thrust_wait",
        "look_ahead",
        "max_marines",
        "spiral_distance",
        "retraction_frames",
        "advancing_frames",
        "retracting_frames",
        "area_width",
        "area_length",
        "gun_locations",
    }
    _check_keys(kind, name, data, allowed, allowed - optional)

    values: dict[str, Any] = {
        "ship_name": _typed(kind, name, "ship_name", data["ship_name"], str),
        "ability_type": _typed(kind, name, "type", data["type"], str),
        "action": _typed(kind, name, "action", data["action"], str),
        "start_hp": _int_tuple(kind, name, data, "start_hp"),
        "damage": _int_tuple(kind, name, data, "damage"),
        "file_path": _typed(kind, name, "file_path", data["file_path"], str),
    }
    for key in ("tracking", "inertia", "hit_parent", "hit_self", "omnidirectional"):
        values[key] = _typed(kind, name, key, data[key], bool)
    for key in ("parent_vel", "speed", "life_time"):
        values[key] = _typed(kind, name, key, data[key], float)

    defaults = {
        "turn_wait": (float, 0),
        "end_anim": (int, 0),
        "sprite_scale": (float, 1.0),
        "frames": (int, 1),
        "sprite_scale_x": (float, 1.0),
        "sprite_scale_y": (float, 1.0),
        "frame_delay": (int, 0),
        "has_sprites": (bool, True),
        "has_sound": (bool, True),
        "laser_vulnerable": (bool, True),
        "collide_planets": (bool, True),
        "collide_asteroids": (bool, True),
        "damage_asteroids": (bool, True),
        "collide_projectiles": (bool, True),
        "damage_projectiles": (bool, True),
        "collide_enemy_ships": (bool, True),
        "collide_friendly_ships": (bool, False),
        "collide_fighters": (bool, True),
    }
    for key, (expected_type, default) in defaults.items():
        values[key] = _optional_typed(kind, name, data, key, expected_type, default)

    optional_fields = {
        "range": ("effect_range", float),
        "one_way_flight": ("one_way_flight", float),
        "life_margin": ("life_margin", float),
        "launch_time": ("launch_time", float),
        "mass": ("mass", float),
        "offset": ("offset", float),
        "track_directions": ("track_directions", int),
        "weapon_wait": ("weapon_wait", float),
        "LASER_RANGE": ("laser_range", float),
        "LASER_WIDTH": ("laser_width", int),
        "MAX_RECOIL": ("max_recoil", float),
        "RECOIL_INCREMENT": ("recoil_increment", float),
        "ENERGY_GAIN": ("energy_gain", int),
        "HP_GAIN": ("hp_gain", int),
        "TRACK_SPEED": ("track_speed", float),
        "TRACK_RANGE": ("track_range", float),
        "DMG_TO_PROJ": ("damage_to_projectiles", int),
        "REUNK_THRUST": ("reunk_thrust", float),
        "REUNK_INCREMENT": ("reunk_increment", float),
        "SPREAD_ANGLE": ("spread_angle", float),
        "max_thrust": ("max_thrust", float),
        "thrust_increment": ("thrust_increment", float),
        "thrust_wait": ("thrust_wait", float),
        "look_ahead": ("look_ahead", int),
        "max_marines": ("max_marines", int),
        "spiral_distance": ("spiral_distance", float),
        "retraction_frames": ("retraction_frames", int),
        "advancing_frames": ("advancing_frames", int),
        "retracting_frames": ("retracting_frames", int),
        "area_width": ("area_width", int),
        "area_length": ("area_length", int),
    }
    for json_key, (attribute, expected_type) in optional_fields.items():
        values[attribute] = _optional_typed(
            kind, name, data, json_key, expected_type, None
        )
    if "LASER_COLOR" in data:
        values["laser_color"] = _int_tuple(kind, name, data, "LASER_COLOR", length=3)
        if any(channel < 0 or channel > 255 for channel in values["laser_color"]):
            raise CatalogValidationError(
                f"Ability '{name}' LASER_COLOR channels must be between 0 and 255"
            )
    else:
        values["laser_color"] = None
    if "gun_locations" in data:
        values["gun_locations"] = _int_pair_tuple(kind, name, data, "gun_locations")
    else:
        values["gun_locations"] = None
    values["_source_keys"] = tuple(data)

    if values["ability_type"] not in {
        "area",
        "fighter",
        "laser",
        "other",
        "projectile",
        "shield",
    }:
        raise CatalogValidationError(f"Ability '{name}' has unsupported type")
    if values["action"] not in {"A1", "A2", "A3"}:
        raise CatalogValidationError(f"Ability '{name}' has unsupported action")
    if values["frames"] <= 0:
        raise CatalogValidationError(f"Ability '{name}' frames must be positive")
    if values["sprite_scale"] <= 0:
        raise CatalogValidationError(f"Ability '{name}' sprite_scale must be positive")
    if values["sprite_scale_x"] <= 0 or values["sprite_scale_y"] <= 0:
        raise CatalogValidationError(
            f"Ability '{name}' directional sprite scales must be positive"
        )
    return AbilityDefinition(**values)


def build_catalogs(ship_entries, ability_entries):
    """Build validated catalogs and verify every ability references a known ship."""
    if not isinstance(ship_entries, Mapping):
        raise CatalogValidationError("Ship catalog must be a JSON object")
    if not isinstance(ability_entries, Mapping):
        raise CatalogValidationError("Ability catalog must be a JSON object")
    ships = {
        name: parse_ship_definition(name, data) for name, data in ship_entries.items()
    }
    abilities = {
        name: parse_ability_definition(name, data)
        for name, data in ability_entries.items()
    }
    for name, definition in abilities.items():
        if definition.ship_name not in ships:
            raise CatalogValidationError(
                f"Ability '{name}' references unknown ship '{definition.ship_name}'"
            )
    return MappingProxyType(ships), MappingProxyType(abilities)


SHIP_DEFINITIONS, ABILITY_DEFINITIONS = build_catalogs(
    read_json(const.SHIPS_JSON_PATH),
    read_json(const.ABILITIES_JSON_PATH),
)

# Temporary read-compatible aliases for specialized definitions and tests that
# still use JSON-key lookup. Values are immutable typed definitions, not dicts.
SHIPS_DATA = SHIP_DEFINITIONS
ABILITIES_DATA = ABILITY_DEFINITIONS
