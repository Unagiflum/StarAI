"""Seeded, display-free API for running StarAI battle simulations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
from types import MappingProxyType
from typing import Any

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Objects.Space.space_obj import Asteroid, Planet, Star
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.space_ship import SpaceShip
from src.audio import NullAudioService
from src.resources import HeadlessAssetManager


@dataclass(frozen=True)
class EntityObservation:
    """Immutable, renderer-independent state for one gameplay entity."""

    kind: str
    name: str
    player: int | None
    position: tuple[float, float]
    velocity: tuple[float, float]
    heading: int | None
    hp: int | float | None
    energy: int | float | None
    alive: bool


@dataclass(frozen=True)
class BattleObservation:
    """State exposed to an agent after reset or a simulation step."""

    frame_id: int
    arena_size: int
    entities: tuple[EntityObservation, ...]
    needs_selection: bool

    @property
    def ships(self) -> tuple[EntityObservation, ...]:
        return tuple(entity for entity in self.entities if entity.kind == "ship")


@dataclass(frozen=True)
class BattleTransition:
    """Result of advancing the environment by one simulation frame."""

    observation: BattleObservation
    rewards: Mapping[int, float]
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class HeadlessBattleEnvironment:
    """Own a reproducible single-round battle without display or audio setup.

    The environment exposes every simulation frame through the complete
    aftermath and ends when the interactive application would request fleet
    selection. Fleet selection and multi-round orchestration remain concerns
    of the interactive application.
    """

    def __init__(
        self,
        ship1: str = "Earthling",
        ship2: str = "Earthling",
        *,
        seed: int | None = None,
        max_steps: int | None = None,
        resources=None,
    ):
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.ship_names = (ship1, ship2)
        self.default_seed = seed
        self.max_steps = max_steps
        self.resources = resources or HeadlessAssetManager()
        self.audio = NullAudioService()
        self.simulation: BattleSimulation | None = None
        self._episode_done = False
        self._seed: int | None = None

    def reset(self, *, seed: int | None = None) -> BattleObservation:
        """Start a fresh episode and return its deterministic initial state."""
        self._seed = self.default_seed if seed is None else seed
        rng = random.Random(self._seed)
        first = create_ship(
            self.ship_names[0],
            1,
            resources=self.resources,
            audio_service=self.audio,
        )
        second = create_ship(
            self.ship_names[1],
            2,
            resources=self.resources,
            audio_service=self.audio,
        )
        self.simulation = BattleSimulation(
            None,
            first,
            second,
            sound_enabled=False,
            audio_service=self.audio,
            rng=rng,
            resources=self.resources,
            include_stars=False,
        )
        self._episode_done = False
        return self.observe()

    def observe(self) -> BattleObservation:
        """Return a value snapshot that contains no Pygame surfaces or masks."""
        simulation = self._require_simulation()
        entities = tuple(
            self._observe_entity(entity)
            for entity in simulation.world
            if isinstance(entity, (SpaceShip, Ability, Asteroid, Planet))
            and not isinstance(entity, Star)
        )
        return BattleObservation(
            frame_id=simulation.frame_id,
            arena_size=const.ARENA_SIZE,
            entities=entities,
            needs_selection=simulation.needs_selection,
        )

    def step(
        self, actions: Mapping[int | str, Mapping[str, Any]] | None = None
    ) -> BattleTransition:
        """Advance one frame using action dictionaries accepted by the simulation."""
        simulation = self._require_simulation()
        if self._episode_done:
            raise RuntimeError("Episode is complete; call reset() before step()")

        state = simulation.step(actions=actions or {})
        terminated = simulation.needs_selection
        truncated = (
            not terminated
            and self.max_steps is not None
            and simulation.frame_id >= self.max_steps
        )
        self._episode_done = terminated or truncated

        winner_ship = simulation.winner() if terminated else None
        winner = winner_ship.player if winner_ship is not None else None
        rewards = {1: 0.0, 2: 0.0}
        if winner is not None:
            rewards[winner] = 1.0
            rewards[1 if winner == 2 else 2] = -1.0

        return BattleTransition(
            observation=self.observe(),
            rewards=MappingProxyType(rewards),
            terminated=terminated,
            truncated=truncated,
            info=MappingProxyType(
                {
                    "frame_id": state["frame_id"],
                    "seed": self._seed,
                    "winner": winner,
                }
            ),
        )

    def close(self) -> None:
        if self.simulation is not None:
            self.simulation.audio.stop_music()
        self.simulation = None
        self._episode_done = True

    def _require_simulation(self) -> BattleSimulation:
        if self.simulation is None:
            raise RuntimeError("Environment has not been reset")
        return self.simulation

    @staticmethod
    def _observe_entity(entity) -> EntityObservation:
        if isinstance(entity, SpaceShip):
            kind = "ship"
        elif isinstance(entity, Ability):
            kind = entity.type
        elif isinstance(entity, Asteroid):
            kind = "asteroid"
        else:
            kind = "planet"

        position = getattr(entity, "position", (0.0, 0.0))
        velocity = getattr(entity, "velocity", (0.0, 0.0))
        return EntityObservation(
            kind=kind,
            name=entity.name,
            player=getattr(entity, "player", None),
            position=(float(position[0]), float(position[1])),
            velocity=(float(velocity[0]), float(velocity[1])),
            heading=getattr(entity, "heading", None),
            hp=getattr(entity, "current_hp", None),
            energy=getattr(entity, "current_energy", None),
            alive=bool(getattr(entity, "currently_alive", True)),
        )
