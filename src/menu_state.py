"""Headless state used by the fleet-building and ship-selection menus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


@dataclass(frozen=True)
class FleetShip:
    name: str
    cost: int


class FleetModel:
    """Ordered, capacity-limited fleet state with no Pygame dependencies."""

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("Fleet capacity cannot be negative")
        self.capacity = capacity
        self._ships: list[FleetShip] = []

    @property
    def ships(self) -> tuple[FleetShip, ...]:
        return tuple(self._ships)

    @property
    def ship_names(self) -> tuple[str, ...]:
        return tuple(ship.name for ship in self._ships)

    @property
    def total_cost(self) -> int:
        return sum(ship.cost for ship in self._ships)

    @property
    def is_empty(self) -> bool:
        return not self._ships

    def add_ship(self, name: str, cost: int) -> bool:
        if len(self._ships) >= self.capacity:
            return False
        self._ships.append(FleetShip(name, cost))
        return True

    def remove_ship(self, index: int) -> FleetShip:
        return self._ships.pop(index)

    def replace_ship(self, index: int, name: str, cost: int) -> FleetShip:
        """Replace one occupied fleet position without changing its order."""
        replacement = FleetShip(name, cost)
        previous = self._ships[index]
        self._ships[index] = replacement
        return previous

    def clear(self) -> None:
        self._ships.clear()

    def replace(self, ships: Sequence[FleetShip]) -> None:
        self.clear()
        for ship in ships:
            if not self.add_ship(ship.name, ship.cost):
                break

    def __len__(self) -> int:
        return len(self._ships)


class SelectableShip(Protocol):
    currently_alive: bool
    current_hp: int


ShipT = TypeVar("ShipT", bound=SelectableShip)


@dataclass(frozen=True)
class ShipSelection(Generic[ShipT]):
    name: str
    ship: ShipT
    index: int


class ShipSelectionState(Generic[ShipT]):
    """Selection rules for two ordered fleets, independent of rendering/input."""

    PLAYERS = (1, 2)

    def __init__(
        self,
        fleets: Mapping[int, Sequence[ShipT]],
        ship_names: Mapping[int, Sequence[str]],
        *,
        preselected: Mapping[int, ShipT | None] | None = None,
        choose_second_player: int | None = None,
    ):
        self.fleets = {player: tuple(fleets[player]) for player in self.PLAYERS}
        self.ship_names = {player: tuple(ship_names[player]) for player in self.PLAYERS}
        for player in self.PLAYERS:
            if len(self.fleets[player]) != len(self.ship_names[player]):
                raise ValueError(f"Player {player} ships and names must align")

        self.choose_second_player = (
            choose_second_player if choose_second_player in self.PLAYERS else None
        )
        self.first_player = (
            3 - self.choose_second_player
            if self.choose_second_player is not None
            else None
        )
        self.active_player = self.first_player
        self.first_locked = False
        self._selections: dict[int, ShipSelection[ShipT] | None] = {
            player: None for player in self.PLAYERS
        }

        preselected = preselected or {}
        self.survivor_locked_players = frozenset(
            player
            for player in self.PLAYERS
            if self._is_survivor(preselected.get(player))
        )
        for player in self.PLAYERS:
            self._set_preselection(player, preselected.get(player))

    @staticmethod
    def _is_alive(ship: ShipT | None) -> bool:
        return bool(ship is not None and ship.currently_alive)

    @classmethod
    def _is_survivor(cls, ship: ShipT | None) -> bool:
        return bool(cls._is_alive(ship) and ship.current_hp > 0)

    def _set_preselection(self, player: int, ship: ShipT | None) -> None:
        if not self._is_alive(ship):
            return
        for index, candidate in enumerate(self.fleets[player]):
            if candidate == ship:
                self._selections[player] = ShipSelection(
                    self.ship_names[player][index], candidate, index
                )
                return

    def selection(self, player: int) -> ShipSelection[ShipT] | None:
        return self._selections[player]

    def selection_allowed(self, player: int) -> bool:
        if player in self.survivor_locked_players:
            return False
        return self.active_player is None or self.active_player == player

    def alive_indices(self, player: int) -> tuple[int, ...]:
        return tuple(
            index
            for index, ship in enumerate(self.fleets[player])
            if self._is_alive(ship)
        )

    def select_index(self, player: int, index: int) -> bool:
        if not self.selection_allowed(player):
            return False
        if index < 0 or index >= len(self.fleets[player]):
            return False
        ship = self.fleets[player][index]
        if not self._is_alive(ship):
            return False
        self._selections[player] = ShipSelection(
            self.ship_names[player][index], ship, index
        )
        self._advance_order(player)
        return True

    def _advance_order(self, player: int) -> None:
        if self.active_player != player or self.choose_second_player is None:
            return
        if player == self.first_player:
            self.first_locked = True
        self.active_player = self.choose_second_player

    @property
    def both_selected(self) -> bool:
        return all(self._selections[player] is not None for player in self.PLAYERS)

    @property
    def confirmation_ready(self) -> bool:
        return all(
            (
                self._is_alive(self._selections[player].ship)
                if self._selections[player] is not None
                else False
            )
            for player in self.PLAYERS
        )

    def selected_ships(self) -> tuple[ShipT, ShipT] | None:
        if not self.confirmation_ready:
            return None
        return (self._selections[1].ship, self._selections[2].ship)
