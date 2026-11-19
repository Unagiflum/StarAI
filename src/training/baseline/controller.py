"""Baseline controllers: per-decision strategy runner, trainee phase mixer,
and a simulation-driven opponent adapter."""

from __future__ import annotations

import random
from typing import Any, Sequence

from src.training.baseline.engine import BaselineContext, BaselineMemory
from src.training.baseline.observation_view import ObservationView
from src.training.baseline.strategies import strategy_for
from src.training.contracts import TrainingAction, action_for_index
from src.training.replay_contracts import ActionHold


class BaselineController:
    """Choose one action per decision from a ship's baseline strategy."""

    def __init__(self, ship_name: str, *, action_frame_span: int = 1):
        self.ship_name = str(ship_name)
        self.action_frame_span = max(1, int(action_frame_span))
        self._strategy = strategy_for(self.ship_name)
        self.memory = BaselineMemory()

    def reset(self) -> None:
        """Forget episode state; keep learned turning estimates."""

        self.memory.reset_episode()

    def begin_phase(self) -> None:
        """Start a fresh run of baseline control after other control."""

        self.memory.reset_episode()

    def decide(self, observation: Sequence[float]) -> int:
        view = ObservationView(observation)
        self.memory.begin_decision(view)
        context = BaselineContext(
            ship_name=self.ship_name,
            view=view,
            memory=self.memory,
            action_frame_span=self.action_frame_span,
        )
        context.avoid_planet()
        if view.enemy is not None:
            self._strategy(context)
        self.memory.end_decision(view)
        return context.plan.action_index()


class BaselinePhaseMixer:
    """Split a trainee's decisions into baseline and network phases.

    Each phase lasts ``decisions_per_phase`` decisions, that is the action
    frame span times the frame span multiplier in frames. At the start of each
    phase a roll against ``baseline_use`` decides whether the baseline
    strategy or the network controls it. Exploration applies on top of both.
    """

    def __init__(
        self,
        ship_name: str,
        *,
        baseline_use: float,
        decisions_per_phase: int,
        action_frame_span: int,
        rng: Any | None = None,
    ):
        self.baseline_use = max(0.0, min(1.0, float(baseline_use)))
        self.decisions_per_phase = max(1, int(decisions_per_phase))
        self.rng = rng or random
        self.controller = BaselineController(
            ship_name,
            action_frame_span=action_frame_span,
        )
        self.decisions = 0
        self.baseline_decisions = 0
        self._phase_remaining = 0
        self._baseline_phase = False

    @property
    def enabled(self) -> bool:
        return self.baseline_use > 0.0

    def reset(self) -> None:
        """Start the next episode with a fresh phase roll."""

        self._phase_remaining = 0
        self._baseline_phase = False
        self.controller.reset()

    def begin_decision(self) -> bool:
        """Advance one decision; return whether the baseline controls it."""

        self.decisions += 1
        if not self.enabled:
            return False
        if self._phase_remaining <= 0:
            self._baseline_phase = (
                self.baseline_use >= 1.0 or self.rng.random() < self.baseline_use
            )
            self._phase_remaining = self.decisions_per_phase
            if self._baseline_phase:
                self.controller.begin_phase()
        self._phase_remaining -= 1
        return self._baseline_phase

    def baseline_action(self, observation: Sequence[float]) -> int:
        self.baseline_decisions += 1
        return self.controller.decide(observation)


def baseline_decisions_per_phase(frame_span_multiplier: int) -> int:
    return max(1, int(frame_span_multiplier))


class BaselineOpponentController:
    """Drive an opponent ship from its baseline strategy.

    Decisions are made every ``action_frame_span`` frames from the opponent's
    own observation and held in between, like a trained opponent.
    """

    def __init__(self, *, action_frame_span: int = 1):
        self.action_frame_span = max(1, int(action_frame_span))
        self._controller: BaselineController | None = None
        self._hold = ActionHold()

    def reset(self) -> None:
        self._hold.reset()
        if self._controller is not None:
            self._controller.reset()

    def direct_controls_for_frame(self, simulation) -> TrainingAction:
        ship = simulation.player2
        ship_name = str(getattr(ship, "name", ""))
        if self._controller is None or self._controller.ship_name != ship_name:
            self._controller = BaselineController(
                ship_name,
                action_frame_span=self.action_frame_span,
            )
            self._hold.reset()
        action_index = self._hold.held_action()
        if action_index is None:
            from src.training.observation import encode_observation

            observation = encode_observation(
                ship,
                simulation.player1,
                frame_id=getattr(simulation, "frame_id", None),
                game_objects=getattr(simulation, "world", None),
            )
            action_index = self._controller.decide(observation)
            self._hold.begin(action_index, self.action_frame_span)
        from src.Battle.computer_control import guard_computer_controls

        return guard_computer_controls(
            action_for_index(action_index),
            ship,
            simulation.player1,
        )

    def controls_for_frame(self, simulation) -> dict[str, bool]:
        action = self.direct_controls_for_frame(simulation)
        return {
            "forward": action.thrust,
            "left": action.turn_left,
            "right": action.turn_right,
            "action1": action.a1,
            "action2": action.a2,
        }
