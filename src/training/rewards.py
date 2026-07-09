"""Reward components and rolling finite-horizon returns for training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from src.training import combat_adapters
from src.training.event_ledger import (
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_REMOVED,
    EVENT_OBJECT_SPAWNED,
    EVENT_SHIP_DIED,
    TrainingBattleEvent,
)


REWARD_POINT_A1 = "Point A1 at enemy"
REWARD_A1_RANGE = "Get in A1 weapon range"
REWARD_SPAWN_A1 = "Spawn A1 object"
REWARD_POINT_A2 = "Point A2 at enemy"
REWARD_A2_RANGE = "Get in A2 weapon range"
REWARD_SPAWN_A2 = "Spawn A2 object"
REWARD_HIGH_SPEED = "Get to high speed"
REWARD_ENEMY_LOSES_CREW = "Enemy loses crew"
REWARD_DEBUFF_ENEMY = "Debuff enemy"
REWARD_KILL_ENEMY_OBJECT = "Kill enemy object"
REWARD_KILL_ENEMY = "Kill enemy"
REWARD_GAIN_CREW = "Gain crew"
REWARD_GAIN_BATTERY = "Gain battery"
REWARD_LOSE_CREW = "Lose crew"
REWARD_LOSE_BATTERY = "Lose battery"
REWARD_BATTERY_AT_ZERO = "Battery at zero"
REWARD_GET_DEBUFFED = "Get debuffed"
REWARD_DIE = "Die"

REWARD_COMPONENTS = (
    REWARD_POINT_A1,
    REWARD_A1_RANGE,
    REWARD_SPAWN_A1,
    REWARD_POINT_A2,
    REWARD_A2_RANGE,
    REWARD_SPAWN_A2,
    REWARD_HIGH_SPEED,
    REWARD_ENEMY_LOSES_CREW,
    REWARD_DEBUFF_ENEMY,
    REWARD_KILL_ENEMY_OBJECT,
    REWARD_KILL_ENEMY,
    REWARD_GAIN_CREW,
    REWARD_GAIN_BATTERY,
    REWARD_LOSE_CREW,
    REWARD_LOSE_BATTERY,
    REWARD_BATTERY_AT_ZERO,
    REWARD_GET_DEBUFFED,
    REWARD_DIE,
)

NORMALIZED_SHAPING_COMPONENTS = (
    REWARD_POINT_A1,
    REWARD_A1_RANGE,
    REWARD_POINT_A2,
    REWARD_A2_RANGE,
)


@dataclass(frozen=True)
class RewardDecisionFrame:
    """The pre-action state for one training decision frame."""

    frame_id: int
    observation: Sequence[float]
    action_index: int
    self_ship: object | None = None
    enemy_ship: object | None = None
    self_battery: float = 0.0
    self_speed: float = 0.0
    self_max_thrust: float = 0.0
    a1_pointing: bool = False
    a1_in_range: bool = False
    a2_pointing: bool = False
    a2_in_range: bool = False


@dataclass(frozen=True)
class RewardFrameOutcome:
    """The post-simulation outcome for the matching decision frame."""

    frame_id: int
    self_battery: float = 0.0
    self_speed: float = 0.0
    self_max_thrust: float = 0.0
    events: tuple[TrainingBattleEvent, ...] = ()
    terminal: bool = False


@dataclass(frozen=True)
class MatureTrainingSample:
    observation: tuple[float, ...]
    action_index: int
    return_value: float
    component_values: dict[str, float]
    weighted_components: dict[str, float]
    start_frame_id: int
    end_frame_id: int
    actual_frame_count: int
    terminal_truncated: bool = False


@dataclass
class _PendingSample:
    decision: RewardDecisionFrame
    decisions: list[RewardDecisionFrame] = field(default_factory=list)
    outcomes: list[RewardFrameOutcome] = field(default_factory=list)


def decision_frame_from_battle_state(
    *,
    frame_id: int,
    observation: Sequence[float],
    action_index: int,
    self_ship,
    enemy_ship,
    world=None,
) -> RewardDecisionFrame:
    """Create a read-only reward snapshot from live battle objects."""

    velocity = _vector(self_ship, "velocity")
    return RewardDecisionFrame(
        frame_id=int(frame_id),
        observation=tuple(float(value) for value in observation),
        action_index=int(action_index),
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        self_battery=_number(self_ship, "current_energy"),
        self_speed=math.hypot(velocity[0], velocity[1]),
        self_max_thrust=_number(self_ship, "max_thrust"),
        a1_pointing=combat_adapters.is_a1_pointing_at_enemy(
            self_ship, enemy_ship, world
        ),
        a1_in_range=combat_adapters.is_enemy_in_a1_effective_range(
            self_ship, enemy_ship, world
        ),
        a2_pointing=combat_adapters.is_a2_pointing_at_enemy(
            self_ship, enemy_ship, world
        ),
        a2_in_range=combat_adapters.is_enemy_in_a2_effective_range(
            self_ship, enemy_ship, world
        ),
    )


def frame_outcome_from_battle_state(
    *,
    frame_id: int,
    self_ship,
    events: Iterable[TrainingBattleEvent] = (),
    terminal: bool = False,
) -> RewardFrameOutcome:
    velocity = _vector(self_ship, "velocity")
    return RewardFrameOutcome(
        frame_id=int(frame_id),
        self_battery=_number(self_ship, "current_energy"),
        self_speed=math.hypot(velocity[0], velocity[1]),
        self_max_thrust=_number(self_ship, "max_thrust"),
        events=tuple(events),
        terminal=bool(terminal),
    )


class RollingReturnPipeline:
    """Collect pending decisions until their windows mature or terminate."""

    def __init__(
        self,
        *,
        prediction_window: int,
        reward_weights: Mapping[str, float] | None = None,
    ):
        if int(prediction_window) <= 0:
            raise ValueError("prediction_window must be positive")
        self.prediction_window = int(prediction_window)
        self.reward_weights = normalize_reward_weights(reward_weights or {})
        self._pending: list[_PendingSample] = []
        self._terminal_seen = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()
        self._terminal_seen = False

    def add_frame(
        self,
        decision: RewardDecisionFrame,
        outcome: RewardFrameOutcome,
    ) -> list[MatureTrainingSample]:
        if self._terminal_seen:
            raise RuntimeError("cannot append frames after a terminal outcome")
        if decision.frame_id != outcome.frame_id:
            raise ValueError("decision and outcome frame IDs must match")

        self._pending.append(_PendingSample(decision=decision))
        matured: list[MatureTrainingSample] = []
        for pending in list(self._pending):
            pending.decisions.append(decision)
            pending.outcomes.append(outcome)
            if len(pending.outcomes) >= self.prediction_window or outcome.terminal:
                matured.append(
                    build_training_sample(
                        pending.decision,
                        pending.decisions,
                        pending.outcomes,
                        self.reward_weights,
                        terminal_truncated=outcome.terminal
                        and len(pending.outcomes) < self.prediction_window,
                    )
                )
                self._pending.remove(pending)

        if outcome.terminal:
            self._terminal_seen = True
        return matured


def normalize_reward_weights(weights: Mapping[str, float]) -> dict[str, float]:
    return {component: float(weights.get(component, 0.0)) for component in REWARD_COMPONENTS}


def calculate_reward_components(
    start_decision: RewardDecisionFrame,
    decisions: Sequence[RewardDecisionFrame],
    outcomes: Sequence[RewardFrameOutcome],
) -> dict[str, float]:
    if not decisions or not outcomes or len(decisions) != len(outcomes):
        raise ValueError("reward windows require matching decisions and outcomes")
    components = {component: 0.0 for component in REWARD_COMPONENTS}
    actual_frames = len(outcomes)
    end_outcome = outcomes[-1]

    components[REWARD_POINT_A1] = _fraction(
        decision.a1_pointing for decision in decisions
    )
    components[REWARD_A1_RANGE] = _fraction(
        decision.a1_in_range for decision in decisions
    )
    components[REWARD_POINT_A2] = _fraction(
        decision.a2_pointing for decision in decisions
    )
    components[REWARD_A2_RANGE] = _fraction(
        decision.a2_in_range for decision in decisions
    )

    if (
        start_decision.self_speed <= start_decision.self_max_thrust
        and end_outcome.self_speed > end_outcome.self_max_thrust
    ):
        components[REWARD_HIGH_SPEED] = 1.0

    battery_delta = end_outcome.self_battery - start_decision.self_battery
    if battery_delta > 0:
        components[REWARD_GAIN_BATTERY] = battery_delta
    elif battery_delta < 0:
        components[REWARD_LOSE_BATTERY] = -battery_delta
    if end_outcome.self_battery <= 0:
        components[REWARD_BATTERY_AT_ZERO] = 1.0

    self_ship = start_decision.self_ship
    enemy_ship = start_decision.enemy_ship
    for event in _events_for_window(outcomes):
        if event.event_type == EVENT_OBJECT_SPAWNED and _same_entity(event.owner, self_ship):
            if event.action == "A1":
                components[REWARD_SPAWN_A1] = 1.0
            elif _is_non_a1_ability_event(event):
                components[REWARD_SPAWN_A2] = 1.0
        elif event.event_type == EVENT_CREW_CHANGED:
            if _same_entity(event.target, enemy_ship) and event.magnitude < 0:
                components[REWARD_ENEMY_LOSES_CREW] += -event.magnitude
            elif _same_entity(event.target, self_ship) and event.magnitude < 0:
                components[REWARD_LOSE_CREW] += -event.magnitude
            elif _same_entity(event.target, self_ship) and event.magnitude > 0:
                components[REWARD_GAIN_CREW] += event.magnitude
        elif event.event_type == EVENT_DEBUFF_APPLIED:
            if _same_entity(event.target, enemy_ship):
                components[REWARD_DEBUFF_ENEMY] += event.magnitude
            elif _same_entity(event.target, self_ship):
                components[REWARD_GET_DEBUFFED] += event.magnitude
        elif event.event_type == EVENT_OBJECT_REMOVED:
            if event.destroyed and _same_entity(event.owner, enemy_ship):
                components[REWARD_KILL_ENEMY_OBJECT] += 1.0
        elif event.event_type == EVENT_SHIP_DIED:
            if _same_entity(event.target, enemy_ship):
                components[REWARD_KILL_ENEMY] += 1.0
            elif _same_entity(event.target, self_ship):
                components[REWARD_DIE] += 1.0

    if actual_frames <= 0:
        raise ValueError("reward windows must contain at least one frame")
    return components


def build_training_sample(
    start_decision: RewardDecisionFrame,
    decisions: Sequence[RewardDecisionFrame],
    outcomes: Sequence[RewardFrameOutcome],
    reward_weights: Mapping[str, float],
    *,
    terminal_truncated: bool = False,
) -> MatureTrainingSample:
    components = calculate_reward_components(start_decision, decisions, outcomes)
    weights = normalize_reward_weights(reward_weights)
    weighted = {
        component: components[component] * weights[component]
        for component in REWARD_COMPONENTS
    }
    return MatureTrainingSample(
        observation=tuple(float(value) for value in start_decision.observation),
        action_index=int(start_decision.action_index),
        return_value=sum(weighted.values()),
        component_values=components,
        weighted_components=weighted,
        start_frame_id=start_decision.frame_id,
        end_frame_id=outcomes[-1].frame_id,
        actual_frame_count=len(outcomes),
        terminal_truncated=bool(terminal_truncated),
    )


def _events_for_window(
    outcomes: Sequence[RewardFrameOutcome],
) -> Iterable[TrainingBattleEvent]:
    for outcome in outcomes:
        yield from outcome.events


def _fraction(values: Iterable[bool]) -> float:
    total = 0
    qualifying = 0
    for value in values:
        total += 1
        if value:
            qualifying += 1
    return qualifying / total if total else 0.0


def _is_non_a1_ability_event(event: TrainingBattleEvent) -> bool:
    if event.action in {"A2", "A3"}:
        return True
    if event.action == "A1":
        return False
    ability_name = event.ability_name or getattr(event.obj, "name", "")
    return bool(ability_name) and not str(ability_name).endswith("A1")


def _same_entity(left, right) -> bool:
    if left is None or right is None:
        return False
    return left is right


def _number(obj, attribute: str, default: float = 0.0) -> float:
    value = getattr(obj, attribute, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return float(default)


def _vector(obj, attribute: str) -> tuple[float, float]:
    value = getattr(obj, attribute, (0.0, 0.0))
    if not isinstance(value, Sequence) or len(value) < 2:
        return 0.0, 0.0
    return _finite_float(value[0]), _finite_float(value[1])


def _finite_float(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return 0.0
