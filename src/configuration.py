"""Typed codecs and repositories for StarAI's persisted configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pygame

from src.persistence import (
    EXPECTED_READ_ERRORS,
    PersistenceValidationError,
    atomic_write_json,
    merge_validated_defaults,
    read_json,
    require_object,
)


@dataclass(frozen=True)
class GameSettings:
    bindings: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    def key_codes(self) -> dict[str, int]:
        return dict(self.bindings)

    def key_names(self) -> dict[str, str]:
        return {label: pygame.key.name(code) for label, code in self.bindings.items()}


class GameSettingsCodec:
    def __init__(self, defaults: Mapping[str, int]):
        self.defaults = dict(defaults)

    @staticmethod
    def _key_code(label: str, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise PersistenceValidationError(f"{label} must be a Pygame key code")
        if not pygame.key.name(value):
            raise PersistenceValidationError(f"{label} has an unknown Pygame key code")
        return value

    def decode(self, value: Any) -> GameSettings:
        loaded = require_object(value, "Game settings")
        bindings = merge_validated_defaults(self.defaults, loaded, self._key_code)
        return GameSettings(bindings)

    def from_key_names(self, values: Mapping[str, Any]) -> GameSettings:
        loaded: dict[str, Any] = {}
        for label in self.defaults:
            value = values.get(label, self.defaults[label])
            if isinstance(value, str):
                try:
                    value = pygame.key.key_code(value)
                except ValueError as error:
                    raise PersistenceValidationError(
                        f"{label} has an unknown Pygame key name"
                    ) from error
            loaded[label] = value
        return self.decode(loaded)

    def encode(self, settings: GameSettings) -> dict[str, int]:
        loaded = require_object(settings.bindings, "Game settings")
        encoded = {}
        for label in self.defaults:
            if label not in loaded:
                raise PersistenceValidationError(f"Game settings is missing {label}")
            encoded[label] = self._key_code(label, loaded[label])
        return encoded


class GameSettingsRepository:
    def __init__(self, path: Path, defaults: Mapping[str, int]):
        self.path = Path(path)
        self.codec = GameSettingsCodec(defaults)

    def default(self) -> GameSettings:
        return self.codec.decode({})

    def load(self) -> GameSettings:
        try:
            return self.codec.decode(read_json(self.path))
        except EXPECTED_READ_ERRORS:
            return self.default()

    def save(self, settings: GameSettings) -> None:
        atomic_write_json(self.path, self.codec.encode(settings))


@dataclass(frozen=True)
class DisplaySettings:
    video_frame_rate: int
    ship_crosshairs: str
    show_planet_gravity_marker: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "video_frame_rate": self.video_frame_rate,
            "ship_crosshairs": self.ship_crosshairs,
            "show_planet_gravity_marker": self.show_planet_gravity_marker,
        }


class DisplaySettingsCodec:
    VIDEO_FRAME_RATES = (24, 48, 72, 96, 120)
    CROSSHAIR_MODES = ("never", "mirror_match_only", "always")

    def __init__(self, defaults: Mapping[str, Any]):
        self.defaults = dict(defaults)

    def _value(self, name: str, value: Any) -> int | str | bool:
        if name == "video_frame_rate":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value not in self.VIDEO_FRAME_RATES
            ):
                raise PersistenceValidationError(
                    "video_frame_rate must be one of 24, 48, 72, 96, or 120"
                )
            return value
        if name == "ship_crosshairs":
            if value not in self.CROSSHAIR_MODES:
                raise PersistenceValidationError(
                    "ship_crosshairs must be never, mirror_match_only, or always"
                )
            return value
        if name == "show_planet_gravity_marker":
            if not isinstance(value, bool):
                raise PersistenceValidationError(
                    "show_planet_gravity_marker must be a boolean"
                )
            return value
        raise PersistenceValidationError(f"Unknown display setting {name}")

    def decode(self, value: Any) -> DisplaySettings:
        loaded = require_object(value, "Display settings")
        values = merge_validated_defaults(self.defaults, loaded, self._value)
        return DisplaySettings(**values)

    def encode(self, settings: DisplaySettings) -> dict[str, int | str | bool]:
        return {
            name: self._value(name, value)
            for name, value in settings.to_dict().items()
        }


class DisplaySettingsRepository:
    def __init__(self, path: Path, defaults: Mapping[str, Any]):
        self.path = Path(path)
        self.codec = DisplaySettingsCodec(defaults)

    def default(self) -> DisplaySettings:
        return self.codec.decode({})

    def load(self) -> DisplaySettings:
        try:
            return self.codec.decode(read_json(self.path))
        except EXPECTED_READ_ERRORS:
            return self.default()

    def save(self, settings: DisplaySettings) -> None:
        atomic_write_json(self.path, self.codec.encode(settings))


@dataclass(frozen=True)
class PlayerFleet:
    ships: tuple[str | None, ...] = ()
    ai: bool = False


@dataclass(frozen=True)
class Fleets:
    player1: PlayerFleet = PlayerFleet()
    player2: PlayerFleet = PlayerFleet()

    def to_json_dict(self) -> dict[str, dict[str, Any]]:
        return {
            "Player1": {"ships": list(self.player1.ships), "ai": self.player1.ai},
            "Player2": {"ships": list(self.player2.ships), "ai": self.player2.ai},
        }


class FleetsCodec:
    def __init__(self, ship_catalog: Mapping[str, Any]):
        self.ship_names = frozenset(ship_catalog)

    def _player(self, value: Any) -> PlayerFleet:
        if not isinstance(value, Mapping):
            return PlayerFleet()
        raw_ships = value.get("ships", [])
        if not isinstance(raw_ships, list):
            raw_ships = []
        ships = tuple(
            name
            for name in raw_ships
            if name is None or (isinstance(name, str) and name in self.ship_names)
        )
        ai = value.get("ai", False)
        if not isinstance(ai, bool):
            ai = False
        return PlayerFleet(ships, ai)

    def decode(self, value: Any) -> Fleets:
        loaded = require_object(value, "Fleets")
        return Fleets(
            self._player(loaded.get("Player1", {})),
            self._player(loaded.get("Player2", {})),
        )

    def encode(self, fleets: Fleets) -> dict[str, dict[str, Any]]:
        for player in (fleets.player1, fleets.player2):
            if not isinstance(player.ai, bool):
                raise PersistenceValidationError("Fleet AI flags must be booleans")
            for ship_name in player.ships:
                if ship_name is not None and (
                    not isinstance(ship_name, str) or ship_name not in self.ship_names
                ):
                    raise PersistenceValidationError(
                        f"Fleet contains unknown ship {ship_name!r}"
                    )
        return fleets.to_json_dict()


class FleetsRepository:
    def __init__(self, path: Path, ship_catalog: Mapping[str, Any]):
        self.path = Path(path)
        self.codec = FleetsCodec(ship_catalog)

    def load(self) -> Fleets:
        try:
            return self.codec.decode(read_json(self.path))
        except EXPECTED_READ_ERRORS:
            return Fleets()

    def save(self, fleets: Fleets) -> None:
        atomic_write_json(self.path, self.codec.encode(fleets))
