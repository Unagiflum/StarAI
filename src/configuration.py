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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
class TrainingSettings:
    learning_rate: float
    discount_factor: float
    epsilon: float
    number_of_hidden_layers: int
    layer_size: int
    batch_size: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "number_of_hidden_layers": self.number_of_hidden_layers,
            "layer_size": self.layer_size,
            "batch_size": self.batch_size,
        }


class TrainingSettingsCodec:
    _ranges = {
        "learning_rate": (0.0001, 0.01),
        "discount_factor": (0.8, 1.0),
        "epsilon": (0.0, 1.0),
        "number_of_hidden_layers": (1, 20),
        "layer_size": (16, 512),
        "batch_size": (32, 256),
    }
    _integer_fields = {"number_of_hidden_layers", "layer_size", "batch_size"}

    def __init__(self, defaults: Mapping[str, int | float]):
        self.defaults = dict(defaults)

    def _value(self, name: str, value: Any) -> int | float:
        if name in self._integer_fields:
            if not isinstance(value, int) or isinstance(value, bool):
                raise PersistenceValidationError(f"{name} must be an integer")
        elif not _is_number(value):
            raise PersistenceValidationError(f"{name} must be a number")
        minimum, maximum = self._ranges[name]
        if not minimum <= value <= maximum:
            raise PersistenceValidationError(f"{name} is outside its supported range")
        return value if name in self._integer_fields else float(value)

    def decode(self, value: Any) -> TrainingSettings:
        loaded = require_object(value, "Training settings")
        values = merge_validated_defaults(self.defaults, loaded, self._value)
        return TrainingSettings(**values)

    def encode(self, settings: TrainingSettings) -> dict[str, int | float]:
        return {
            name: self._value(name, value) for name, value in settings.to_dict().items()
        }


class TrainingSettingsRepository:
    def __init__(self, path: Path, defaults: Mapping[str, int | float]):
        self.path = Path(path)
        self.codec = TrainingSettingsCodec(defaults)

    def default(self) -> TrainingSettings:
        return self.codec.decode({})

    def load(self) -> TrainingSettings:
        try:
            return self.codec.decode(read_json(self.path))
        except EXPECTED_READ_ERRORS:
            return self.default()

    def save(self, settings: TrainingSettings) -> None:
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
