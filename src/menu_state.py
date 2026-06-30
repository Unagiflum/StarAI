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
        self._ships: list[FleetShip | None] = [None] * capacity

    @property
    def ships(self) -> tuple[FleetShip, ...]:
        return tuple(ship for ship in self._ships if ship is not None)

    @property
    def ship_names(self) -> tuple[str, ...]:
        return tuple(ship.name for ship in self._ships if ship is not None)

    @property
    def ship_slots(self) -> tuple[str | None, ...]:
        """Return every fleet position, including intentional empty slots."""
        return tuple(ship.name if ship is not None else None for ship in self._ships)

    @property
    def total_cost(self) -> int:
        return sum(ship.cost for ship in self._ships if ship is not None)

    @property
    def is_empty(self) -> bool:
        return all(ship is None for ship in self._ships)

    def add_ship(self, name: str, cost: int) -> bool:
        for i in range(self.capacity):
            if self._ships[i] is None:
                self._ships[i] = FleetShip(name, cost)
                return True
        return False

    def remove_ship(self, index: int) -> FleetShip | None:
        if 0 <= index < self.capacity:
            ship = self._ships[index]
            self._ships[index] = None
            return ship
        return None

    def replace_ship(self, index: int, name: str, cost: int) -> FleetShip | None:
        """Replace one occupied fleet position without changing its order."""
        if 0 <= index < self.capacity:
            previous = self._ships[index]
            self._ships[index] = FleetShip(name, cost)
            return previous
        return None

    def clear(self) -> None:
        self._ships = [None] * self.capacity

    def replace(self, ships: Sequence[FleetShip]) -> None:
        self.clear()
        for ship in ships:
            if not self.add_ship(ship.name, ship.cost):
                break

    def __len__(self) -> int:
        return sum(1 for ship in self._ships if ship is not None)


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
        self._random_locked_players: set[int] = set()

        preselected = preselected or {}
        self.survivor_locked_players = frozenset(
            player
            for player in self.PLAYERS
            if self._is_survivor(preselected.get(player))
        )
        for player in self.PLAYERS:
            self._set_preselection(player, preselected.get(player))

        # A surviving ship is already locked in for the next round.  If that
        # player would otherwise choose first, advance the forced order so the
        # player who still needs a ship can select instead of deadlocking.
        if (
            self.choose_second_player is not None
            and self.first_player in self.survivor_locked_players
        ):
            self.first_locked = True
            self.active_player = self.choose_second_player

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
        if (
            player in self.survivor_locked_players
            or player in self._random_locked_players
        ):
            return False
        return self.active_player is None or self.active_player == player

    @property
    def random_locked_players(self) -> frozenset[int]:
        return frozenset(self._random_locked_players)

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

    def select_random_index(self, player: int, index: int) -> bool:
        """Select and permanently lock a hidden random choice for this round."""
        if not self.select_index(player, index):
            return False
        self._random_locked_players.add(player)
        return True

    def deselect(self, player: int) -> bool:
        """Clear an editable selection, leaving locked selections unchanged."""
        if not self.selection_allowed(player) or self._selections[player] is None:
            return False
        self._selections[player] = None
        return True

    def toggle_index(self, player: int, index: int) -> bool:
        """Select an alive ship, or deselect it when it is already selected."""
        selection = self._selections[player]
        if selection is not None and selection.index == index:
            return self.deselect(player)
        return self.select_index(player, index)

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
