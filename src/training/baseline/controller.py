"""Baseline controllers: per-frame strategy runner, trainee phase mixer,
and a simulation-driven opponent adapter."""

from __future__ import annotations

import random
from typing import Any, Sequence

from src.training.baseline.behaviours import ShipBehaviour
from src.training.baseline.engine import BaselineContext, BaselineMemory
from src.training.baseline.observation_view import ObservationView
from src.training.baseline.strategies import may_evade, shared, strategy_for
from src.training.contracts import TrainingAction, action_for_index


class BaselineController:
    """Choose one action per frame from a ship's baseline strategy.

    With ``entry``, the ship flies ``strategies.shared`` from that table
    entry instead of its own (candidate settings in the arena).
    """

    def __init__(
        self,
        ship_name: str,
        variant: str | None = None,
        *,
        entry: ShipBehaviour | None = None,
    ):
        self.ship_name = str(ship_name)
        self.variant = variant
        self.entry = entry
        self._strategy = shared if entry is not None else strategy_for(self.ship_name, variant)
        self.memory = BaselineMemory()

    def reset(self) -> None:
        """Forget episode state; keep learned turning estimates."""

        self.memory.reset_episode()

    def begin_phase(self) -> None:
        """Start a fresh run of baseline control after other control."""

        self.memory.reset_episode()

    def interrupt(self) -> None:
        """Note that other control flew the ship since the last decision."""

        self.memory.forget_previous_frame()

    def decide(self, observation: Sequence[float]) -> int:
        view = ObservationView(observation)
        self.memory.begin_decision(view)
        context = BaselineContext(
            ship_name=self.ship_name,
            view=view,
            memory=self.memory,
            entry=self.entry,
        )
        context.avoid_planet()
        if may_evade(context):
            context.evade_threats()
        if view.enemy is not None:
            self._strategy(context)
            context.keep_moving()
            context.hold_thrust_while_confused()
        self.memory.end_decision(view)
        return context.plan.action_index()


class BaselinePhaseMixer:
    """Split a trainee's frames into baseline and network phases.

    Each phase lasts ``frames_per_phase`` frames, that is the action frame
    span times the frame span multiplier. At the start of each phase a roll
    against ``baseline_use`` decides whether the baseline strategy or the
    network controls it. Exploration applies on top of both.
    """

    def __init__(
        self,
        ship_name: str,
        *,
        baseline_use: float,
        frames_per_phase: int,
        rng: Any | None = None,
    ):
        self.baseline_use = max(0.0, min(1.0, float(baseline_use)))
        self.frames_per_phase = max(1, int(frames_per_phase))
        self.rng = rng or random
        self.controller = BaselineController(ship_name)
        # Frames seen and frames the baseline chose the action for.
        self.decisions = 0
        self.baseline_decisions = 0
        self._phase_remaining = 0
        self._baseline_phase = False
        self._last_baseline_frame: int | None = None

    @property
    def enabled(self) -> bool:
        return self.baseline_use > 0.0

    def reset(self) -> None:
        """Start the next episode with a fresh phase roll."""

        self._phase_remaining = 0
        self._baseline_phase = False
        self._last_baseline_frame = None
        self.controller.reset()

    def begin_decision(self) -> bool:
        """Advance one frame; return whether the baseline phase is active."""

        self.decisions += 1
        if not self.enabled:
            return False
        if self._phase_remaining <= 0:
            self._baseline_phase = (
                self.baseline_use >= 1.0 or self.rng.random() < self.baseline_use
            )
            self._phase_remaining = self.frames_per_phase
            if self._baseline_phase:
                self.controller.begin_phase()
                self._last_baseline_frame = None
        self._phase_remaining -= 1
        return self._baseline_phase

    def baseline_action(self, observation: Sequence[float]) -> int:
        if (
            self._last_baseline_frame is not None
            and self._last_baseline_frame != self.decisions - 1
        ):
            # An exploratory hold flew the ship in between.
            self.controller.interrupt()
        self._last_baseline_frame = self.decisions
        self.baseline_decisions += 1
        return self.controller.decide(observation)


def baseline_frames_per_phase(action_frame_span: int, frame_span_multiplier: int) -> int:
    return max(1, int(action_frame_span)) * max(1, int(frame_span_multiplier))


class BaselineOpponentController:
    """Drive an opponent ship from its baseline strategy, deciding every
    frame from the opponent's own observation."""

    def __init__(self) -> None:
        self._controller: BaselineController | None = None

    def reset(self) -> None:
        if self._controller is not None:
            self._controller.reset()

    def direct_controls_for_frame(self, simulation) -> TrainingAction:
        from src.Battle.computer_control import guard_computer_controls
        from src.training.observation import encode_observation

        ship = simulation.player2
        ship_name = str(getattr(ship, "name", ""))
        if self._controller is None or self._controller.ship_name != ship_name:
            self._controller = BaselineController(ship_name)
        observation = encode_observation(
            ship,
            simulation.player1,
            frame_id=getattr(simulation, "frame_id", None),
            game_objects=getattr(simulation, "world", None),
        )
        action_index = self._controller.decide(observation)
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
