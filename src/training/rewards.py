"""Reward components and discounted normalized returns for training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import src.const as const
from src.training import combat_adapters
from src.training.event_ledger import (
    EVENT_ACTION_USED,
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_REMOVED,
    EVENT_OBJECT_SPAWNED,
    EVENT_SHIP_DIED,
    TrainingBattleEvent,
)


REWARD_POINT_A1 = "Point A1 at enemy"
REWARD_A1_RANGE = "In A1 range"
REWARD_SPAWN_A1 = "Use A1"
REWARD_POINT_A2 = "Point A2 at enemy"
REWARD_A2_RANGE = "In A2 range"
REWARD_SPAWN_A2 = "Use A2"
REWARD_HIGH_SPEED = "Be at high speed"
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

LEGACY_REWARD_ALIASES = {
    REWARD_SPAWN_A1: "Spawn A1 object",
    REWARD_SPAWN_A2: "Spawn A2 object",
    REWARD_A1_RANGE: "Get in A1 weapon range",
    REWARD_A2_RANGE: "Get in A2 weapon range",
}

NORMALIZED_SHAPING_COMPONENTS = (
    REWARD_POINT_A1,
    REWARD_A1_RANGE,
    REWARD_POINT_A2,
    REWARD_A2_RANGE,
)

DISCOUNTED_SUM_COMPONENTS = (
    REWARD_KILL_ENEMY,
    REWARD_DIE,
)

DISCOUNT_CUTOFF_WEIGHT = 0.01


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
    components: list[dict[str, float]] = field(default_factory=list)


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
    """Collect pending decisions until discounted future rewards are negligible."""

    def __init__(
        self,
        *,
        gamma: float,
        reward_weights: Mapping[str, float] | None = None,
    ):
        self.gamma = _validate_gamma(gamma)
        self.discount_cutoff_frames = discount_cutoff_frames(self.gamma)
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
        components = calculate_immediate_reward_components(decision, outcome)
        matured: list[MatureTrainingSample] = []
        for pending in list(self._pending):
            pending.decisions.append(decision)
            pending.outcomes.append(outcome)
            pending.components.append(components)
            if len(pending.outcomes) >= self.discount_cutoff_frames or outcome.terminal:
                matured.append(
                    build_discounted_training_sample(
                        pending.decision,
                        pending.components,
                        self.reward_weights,
                        self.gamma,
                        end_frame_id=pending.outcomes[-1].frame_id,
                        terminal_truncated=outcome.terminal
                        and len(pending.outcomes) < self.discount_cutoff_frames,
                    )
                )
                self._pending.remove(pending)

        if outcome.terminal:
            self._terminal_seen = True
        return matured


def normalize_reward_weights(weights: Mapping[str, float]) -> dict[str, float]:
    normalized = {}
    for component in REWARD_COMPONENTS:
        value = weights.get(component, None)
        if value is None:
            legacy_component = LEGACY_REWARD_ALIASES.get(component)
            if legacy_component is not None:
                value = weights.get(legacy_component, None)
        normalized[component] = float(value) if value is not None else 0.0
    return normalized


def discount_cutoff_frames(
    gamma: float,
    cutoff_weight: float = DISCOUNT_CUTOFF_WEIGHT,
) -> int:
    gamma = _validate_gamma(gamma)
    cutoff = float(cutoff_weight)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff_weight must be between 0 and 1")
    if gamma == 0.0:
        return 1
    return max(1, int(math.ceil(math.log(cutoff) / math.log(gamma))))


def calculate_immediate_reward_components(
    decision: RewardDecisionFrame,
    outcome: RewardFrameOutcome,
) -> dict[str, float]:
    if decision.frame_id != outcome.frame_id:
        raise ValueError("decision and outcome frame IDs must match")
    components = {component: 0.0 for component in REWARD_COMPONENTS}

    components[REWARD_POINT_A1] = 1.0 if decision.a1_pointing else 0.0
    components[REWARD_A1_RANGE] = 1.0 if decision.a1_in_range else 0.0
    components[REWARD_POINT_A2] = 1.0 if decision.a2_pointing else 0.0
    components[REWARD_A2_RANGE] = 1.0 if decision.a2_in_range else 0.0

    if outcome.self_max_thrust > 0.0 and outcome.self_speed > outcome.self_max_thrust:
        components[REWARD_HIGH_SPEED] = (
            outcome.self_speed - outcome.self_max_thrust
        ) / outcome.self_max_thrust

    battery_delta = outcome.self_battery - decision.self_battery
    if battery_delta > 0:
        components[REWARD_GAIN_BATTERY] = battery_delta
    elif battery_delta < 0:
        components[REWARD_LOSE_BATTERY] = -battery_delta
    if outcome.self_battery <= 0:
        components[REWARD_BATTERY_AT_ZERO] = 1.0

    self_ship = decision.self_ship
    enemy_ship = decision.enemy_ship
    for event in outcome.events:
        if _is_self_action_use_event(event, self_ship):
            if _is_a1_use_event(event):
                components[REWARD_SPAWN_A1] = 1.0
            elif _is_a2_use_event(event):
                components[REWARD_SPAWN_A2] = 1.0
        elif event.event_type == EVENT_CREW_CHANGED:
            if _same_entity(event.target, enemy_ship) and event.magnitude < 0:
                components[REWARD_ENEMY_LOSES_CREW] += (
                    -event.magnitude * _source_reward_credit(event)
                )
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
                components[REWARD_KILL_ENEMY] += _kill_reward_credit(event)
            elif _same_entity(event.target, self_ship):
                components[REWARD_DIE] += 1.0

    return components


def calculate_reward_components(
    start_decision: RewardDecisionFrame,
    decisions: Sequence[RewardDecisionFrame],
    outcomes: Sequence[RewardFrameOutcome],
) -> dict[str, float]:
    if not decisions or not outcomes or len(decisions) != len(outcomes):
        raise ValueError("reward windows require matching decisions and outcomes")
    components = {component: 0.0 for component in REWARD_COMPONENTS}
    actual_frames = len(outcomes)

    if actual_frames <= 0:
        raise ValueError("reward windows must contain at least one frame")
    for decision, outcome in zip(decisions, outcomes):
        for component, value in calculate_immediate_reward_components(
            decision, outcome
        ).items():
            components[component] += value
    for component in REWARD_COMPONENTS:
        if component not in DISCOUNTED_SUM_COMPONENTS:
            components[component] /= actual_frames
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
    return _sample_from_components(
        start_decision,
        components,
        reward_weights,
        end_frame_id=outcomes[-1].frame_id,
        actual_frame_count=len(outcomes),
        terminal_truncated=terminal_truncated,
    )


def build_discounted_training_sample(
    start_decision: RewardDecisionFrame,
    immediate_components: Sequence[Mapping[str, float]],
    reward_weights: Mapping[str, float],
    gamma: float,
    *,
    end_frame_id: int,
    terminal_truncated: bool = False,
) -> MatureTrainingSample:
    if not immediate_components:
        raise ValueError("reward windows must contain at least one frame")
    components = discounted_average_components(immediate_components, gamma)
    return _sample_from_components(
        start_decision,
        components,
        reward_weights,
        end_frame_id=end_frame_id,
        actual_frame_count=len(immediate_components),
        terminal_truncated=terminal_truncated,
    )


def discounted_average_components(
    immediate_components: Sequence[Mapping[str, float]],
    gamma: float,
) -> dict[str, float]:
    gamma = _validate_gamma(gamma)
    if not immediate_components:
        raise ValueError("reward windows must contain at least one frame")
    totals = {component: 0.0 for component in REWARD_COMPONENTS}
    discount = 1.0
    weight_sum = 0.0
    for frame_components in immediate_components:
        weight_sum += discount
        for component in REWARD_COMPONENTS:
            totals[component] += float(frame_components.get(component, 0.0)) * discount
        discount *= gamma
    return {
        component: (
            totals[component]
            if component in DISCOUNTED_SUM_COMPONENTS
            else totals[component] / weight_sum
        )
        for component in REWARD_COMPONENTS
    }


def _sample_from_components(
    start_decision: RewardDecisionFrame,
    components: Mapping[str, float],
    reward_weights: Mapping[str, float],
    *,
    end_frame_id: int,
    actual_frame_count: int,
    terminal_truncated: bool = False,
) -> MatureTrainingSample:
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
        end_frame_id=int(end_frame_id),
        actual_frame_count=int(actual_frame_count),
        terminal_truncated=bool(terminal_truncated),
    )


def _is_self_action_use_event(
    event: TrainingBattleEvent,
    self_ship: object | None,
) -> bool:
    return event.event_type in {
        EVENT_ACTION_USED,
        EVENT_OBJECT_SPAWNED,
    } and _same_entity(event.owner, self_ship)


def _is_a1_use_event(event: TrainingBattleEvent) -> bool:
    return event.action == "A1" or _ability_name(event).endswith("A1")


def _is_a2_use_event(event: TrainingBattleEvent) -> bool:
    if event.action == "A2":
        return not _is_orz_turret_turn_event(event)
    if event.action == "A3":
        return _ability_name(event) == "OrzA3" or _owner_name(event) == "Orz"
    if event.action == "A1":
        return False
    ability_name = _ability_name(event)
    return bool(ability_name) and not ability_name.endswith("A1")


def _is_orz_turret_turn_event(event: TrainingBattleEvent) -> bool:
    return (
        event.event_type == EVENT_ACTION_USED
        and event.action == "A2"
        and _owner_name(event) == "Orz"
    )


def _ability_name(event: TrainingBattleEvent) -> str:
    ability_name = event.ability_name or getattr(event.obj, "name", "")
    return str(ability_name or "")


def _owner_name(event: TrainingBattleEvent) -> str:
    return str(getattr(event.owner, "name", "") or "")


def _source_reward_credit(event: TrainingBattleEvent) -> float:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    if "source_credit" in metadata:
        return _clamp01(_finite_float(metadata["source_credit"], default=1.0))
    source_name = _ability_name(event)
    if source_name == "DruugeA2":
        return const.DRUUGE_A2_KILL_CREDIT
    role = getattr(getattr(event.obj, "collision_capabilities", None), "role", None)
    if getattr(role, "name", None) == "PLANET":
        return const.PLANET_KILL_CREDIT
    return 1.0


def _kill_reward_credit(event: TrainingBattleEvent) -> float:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    if "kill_credit" in metadata:
        return _clamp01(_finite_float(metadata["kill_credit"], default=1.0))
    return _source_reward_credit(event)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _same_entity(left, right) -> bool:
    if left is None or right is None:
        return False
    return left is right


def _validate_gamma(value: float) -> float:
    gamma = float(value)
    if not math.isfinite(gamma) or gamma < 0.0 or gamma >= 1.0:
        raise ValueError("gamma must be in [0, 1)")
    return gamma


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


def _finite_float(value, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return float(default)
