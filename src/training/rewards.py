"""Reward components and discounted normalized returns for training."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import src.const as const
from src.training import combat_adapters
from src.training.event_ledger import (
    EVENT_ACTION_USED,
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_REMOVED,
    EVENT_OBJECT_HP_CHANGED,
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
REWARD_DESTROY_OWN_OBJECT = "Destroy own object"
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
    REWARD_KILL_ENEMY,
    REWARD_ENEMY_LOSES_CREW,
    REWARD_DEBUFF_ENEMY,
    REWARD_KILL_ENEMY_OBJECT,
    REWARD_GAIN_CREW,
    REWARD_GAIN_BATTERY,
    REWARD_HIGH_SPEED,
    REWARD_POINT_A1,
    REWARD_A1_RANGE,
    REWARD_SPAWN_A1,
    REWARD_POINT_A2,
    REWARD_A2_RANGE,
    REWARD_SPAWN_A2,
    REWARD_DESTROY_OWN_OBJECT,
    REWARD_LOSE_BATTERY,
    REWARD_BATTERY_AT_ZERO,
    REWARD_GET_DEBUFFED,
    REWARD_LOSE_CREW,
    REWARD_DIE,
)
_REWARD_COMPONENT_INDEX = {
    component: index for index, component in enumerate(REWARD_COMPONENTS)
}
_REWARD_COMPONENT_COUNT = len(REWARD_COMPONENTS)

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
_DISCOUNTED_SUM_COMPONENT_INDICES = frozenset(
    _REWARD_COMPONENT_INDEX[component]
    for component in DISCOUNTED_SUM_COMPONENTS
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
    self_sustained_a2_active: bool = False
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
        self_sustained_a2_active=_sustained_a2_active(self_ship),
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
        self._discount_powers = _discount_powers(
            self.gamma,
            self.discount_cutoff_frames,
        )
        self._discount_weight_sums = _discount_weight_sums(
            self._discount_powers,
        )
        self._pending: deque[_PendingSample] = deque()
        self._component_window: deque[tuple[float, ...]] = deque()
        self._window_totals = _zero_component_vector()
        self._terminal_seen = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()
        self._component_window.clear()
        self._window_totals = _zero_component_vector()
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
        components = _calculate_immediate_reward_component_vector(decision, outcome)
        self._append_component_vector(components)

        if outcome.terminal:
            matured = self._mature_all_pending(
                end_frame_id=outcome.frame_id,
                terminal=True,
            )
            self._terminal_seen = True
            return matured

        if len(self._component_window) < self.discount_cutoff_frames:
            return []

        pending = self._pending.popleft()
        matured = [
            _sample_from_component_vector(
                pending.decision,
                _discounted_average_component_vector(
                    self._window_totals,
                    self._discount_weight_sums[self.discount_cutoff_frames],
                ),
                self.reward_weights,
                end_frame_id=outcome.frame_id,
                actual_frame_count=self.discount_cutoff_frames,
                terminal_truncated=False,
            )
        ]
        self._drop_oldest_component_vector()
        return matured

    def flush_pending(self, *, end_frame_id: int) -> list[MatureTrainingSample]:
        if self._terminal_seen:
            raise RuntimeError("cannot flush after a terminal outcome")
        matured = self._mature_all_pending(
            end_frame_id=end_frame_id,
            terminal=True,
        )
        self._terminal_seen = True
        return matured

    def _append_component_vector(self, components: tuple[float, ...]) -> None:
        offset = len(self._component_window)
        self._component_window.append(components)
        discount = self._discount_powers[offset]
        self._window_totals = tuple(
            total + float(value) * discount
            for total, value in zip(self._window_totals, components)
        )

    def _drop_oldest_component_vector(self) -> None:
        oldest = self._component_window.popleft()
        if not self._component_window:
            self._window_totals = _zero_component_vector()
            return
        if self.gamma == 0.0:
            self._window_totals = tuple(
                float(value) for value in self._component_window[0]
            )
            return
        self._window_totals = tuple(
            (total - float(value)) / self.gamma
            for total, value in zip(self._window_totals, oldest)
        )

    def _mature_all_pending(
        self,
        *,
        end_frame_id: int,
        terminal: bool,
    ) -> list[MatureTrainingSample]:
        components = list(self._component_window)
        suffix_totals = _zero_component_vector()
        samples_by_offset: list[tuple[float, ...]] = [
            _zero_component_vector()
            for _ in components
        ]
        for offset in range(len(components) - 1, -1, -1):
            current = components[offset]
            suffix_totals = tuple(
                float(value) + self.gamma * total
                for value, total in zip(current, suffix_totals)
            )
            samples_by_offset[offset] = suffix_totals

        matured = []
        for offset, pending in enumerate(self._pending):
            actual_frame_count = len(components) - offset
            if actual_frame_count <= 0:
                continue
            averaged = _discounted_average_component_vector(
                samples_by_offset[offset],
                self._discount_weight_sums[actual_frame_count],
            )
            matured.append(
                _sample_from_component_vector(
                    pending.decision,
                    averaged,
                    self.reward_weights,
                    end_frame_id=end_frame_id,
                    actual_frame_count=actual_frame_count,
                    terminal_truncated=terminal
                    and actual_frame_count < self.discount_cutoff_frames,
                )
            )
        self._pending.clear()
        self._component_window.clear()
        self._window_totals = _zero_component_vector()
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


def _zero_component_vector() -> tuple[float, ...]:
    return (0.0,) * _REWARD_COMPONENT_COUNT


def _discount_powers(gamma: float, frame_count: int) -> tuple[float, ...]:
    powers = []
    discount = 1.0
    for _ in range(max(1, int(frame_count))):
        powers.append(discount)
        discount *= gamma
    return tuple(powers)


def _discount_weight_sums(discount_powers: Sequence[float]) -> tuple[float, ...]:
    sums = [0.0]
    total = 0.0
    for discount in discount_powers:
        total += float(discount)
        sums.append(total)
    return tuple(sums)


def _component_dict_from_vector(components: Sequence[float]) -> dict[str, float]:
    return {
        component: float(components[index])
        for index, component in enumerate(REWARD_COMPONENTS)
    }


def _discounted_average_component_vector(
    totals: Sequence[float],
    weight_sum: float,
) -> tuple[float, ...]:
    return tuple(
        float(total)
        if index in _DISCOUNTED_SUM_COMPONENT_INDICES
        else float(total) / weight_sum
        for index, total in enumerate(totals)
    )


def calculate_immediate_reward_components(
    decision: RewardDecisionFrame,
    outcome: RewardFrameOutcome,
) -> dict[str, float]:
    return _component_dict_from_vector(
        _calculate_immediate_reward_component_vector(decision, outcome)
    )


def _calculate_immediate_reward_component_vector(
    decision: RewardDecisionFrame,
    outcome: RewardFrameOutcome,
) -> tuple[float, ...]:
    if decision.frame_id != outcome.frame_id:
        raise ValueError("decision and outcome frame IDs must match")
    components = [0.0] * _REWARD_COMPONENT_COUNT

    components[_REWARD_COMPONENT_INDEX[REWARD_POINT_A1]] = (
        1.0 if decision.a1_pointing else 0.0
    )
    components[_REWARD_COMPONENT_INDEX[REWARD_A1_RANGE]] = (
        1.0 if decision.a1_in_range else 0.0
    )
    components[_REWARD_COMPONENT_INDEX[REWARD_POINT_A2]] = (
        1.0 if decision.a2_pointing else 0.0
    )
    components[_REWARD_COMPONENT_INDEX[REWARD_A2_RANGE]] = (
        1.0 if decision.a2_in_range else 0.0
    )

    if outcome.self_max_thrust > 0.0 and outcome.self_speed > outcome.self_max_thrust:
        components[_REWARD_COMPONENT_INDEX[REWARD_HIGH_SPEED]] = (
            outcome.self_speed - outcome.self_max_thrust
        ) / outcome.self_max_thrust

    battery_delta = outcome.self_battery - decision.self_battery
    if battery_delta > 0:
        components[_REWARD_COMPONENT_INDEX[REWARD_GAIN_BATTERY]] = battery_delta
    elif battery_delta < 0:
        components[_REWARD_COMPONENT_INDEX[REWARD_LOSE_BATTERY]] = -battery_delta
    if outcome.self_battery <= 0:
        components[_REWARD_COMPONENT_INDEX[REWARD_BATTERY_AT_ZERO]] = 1.0

    self_ship = decision.self_ship
    enemy_ship = decision.enemy_ship
    if _uses_sustained_a2_reward(self_ship) and outcome.self_sustained_a2_active:
        components[_REWARD_COMPONENT_INDEX[REWARD_SPAWN_A2]] = 1.0
    for event in outcome.events:
        if _is_self_action_use_event(event, self_ship):
            if _is_a1_use_event(event):
                components[_REWARD_COMPONENT_INDEX[REWARD_SPAWN_A1]] = 1.0
            elif (
                not _uses_sustained_a2_reward(self_ship)
                and _is_a2_use_event(event)
            ):
                components[_REWARD_COMPONENT_INDEX[REWARD_SPAWN_A2]] = 1.0
        elif event.event_type == EVENT_CREW_CHANGED:
            if _same_entity(event.target, enemy_ship) and event.magnitude < 0:
                components[_REWARD_COMPONENT_INDEX[REWARD_ENEMY_LOSES_CREW]] += (
                    -event.magnitude * _source_reward_credit(event)
                )
            elif _same_entity(event.target, self_ship) and event.magnitude < 0:
                components[_REWARD_COMPONENT_INDEX[REWARD_LOSE_CREW]] += -event.magnitude
            elif _same_entity(event.target, self_ship) and event.magnitude > 0:
                components[_REWARD_COMPONENT_INDEX[REWARD_GAIN_CREW]] += event.magnitude
        elif event.event_type == EVENT_DEBUFF_APPLIED:
            if _same_entity(event.target, enemy_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_DEBUFF_ENEMY]] += (
                    event.magnitude
                )
            elif _same_entity(event.target, self_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_GET_DEBUFFED]] += (
                    event.magnitude
                )
        elif event.event_type == EVENT_OBJECT_REMOVED:
            if _is_chmmr_satellite_event(event):
                if event.destroyed and _same_entity(event.owner, enemy_ship):
                    if _same_entity(_removal_source_owner(event), self_ship):
                        components[_REWARD_COMPONENT_INDEX[REWARD_KILL_ENEMY]] += 0.5
                elif event.destroyed and _same_entity(event.owner, self_ship):
                    components[_REWARD_COMPONENT_INDEX[REWARD_DIE]] += (
                        1.0 / _configured_satellite_count(event.owner)
                    )
            elif _object_removed_by_owner_weapon(event, enemy_ship, self_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_KILL_ENEMY_OBJECT]] += 1.0
            elif _object_removed_by_owner_weapon(event, self_ship, self_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_DESTROY_OWN_OBJECT]] += 1.0
        elif event.event_type == EVENT_OBJECT_HP_CHANGED:
            if not _is_chmmr_satellite_event(event):
                continue
            damage = max(0.0, -float(event.magnitude)) * 0.5
            if _same_entity(event.owner, self_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_LOSE_CREW]] += damage
            elif (
                _same_entity(event.owner, enemy_ship)
                and _same_entity(event.actor, self_ship)
            ):
                components[_REWARD_COMPONENT_INDEX[REWARD_ENEMY_LOSES_CREW]] += damage
        elif event.event_type == EVENT_SHIP_DIED:
            if _same_entity(event.target, enemy_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_KILL_ENEMY]] += (
                    _enemy_death_reward_credit(event)
                )
            elif _same_entity(event.target, self_ship):
                components[_REWARD_COMPONENT_INDEX[REWARD_DIE]] += 1.0

    return tuple(components)


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


def _sample_from_component_vector(
    start_decision: RewardDecisionFrame,
    components: Sequence[float],
    reward_weights: Mapping[str, float],
    *,
    end_frame_id: int,
    actual_frame_count: int,
    terminal_truncated: bool = False,
) -> MatureTrainingSample:
    component_values = _component_dict_from_vector(components)
    weighted = {
        component: component_values[component] * float(reward_weights[component])
        for component in REWARD_COMPONENTS
    }
    return MatureTrainingSample(
        observation=tuple(float(value) for value in start_decision.observation),
        action_index=int(start_decision.action_index),
        return_value=sum(weighted.values()),
        component_values=component_values,
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


def _uses_sustained_a2_reward(ship: object | None) -> bool:
    return getattr(ship, "name", None) in {"Ilwrath", "Androsynth"}


def _sustained_a2_active(ship: object | None) -> bool:
    ship_name = getattr(ship, "name", None)
    if ship_name == "Ilwrath":
        return bool(getattr(ship, "cloaked", False))
    if ship_name == "Androsynth":
        is_blazer = getattr(ship, "is_blazer", None)
        if isinstance(is_blazer, bool):
            return is_blazer
        return getattr(ship, "form", None) == "A2"
    return False


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
        return const.DRUUGE_A2_CREW_LOSS_REWARD_FACTOR
    if source_name == "ShofixtiA2":
        return const.SHOFIXTI_A2_CREW_LOSS_REWARD_FACTOR
    role = getattr(getattr(event.obj, "collision_capabilities", None), "role", None)
    if getattr(role, "name", None) == "PLANET":
        return const.PLANET_CREW_LOSS_REWARD_FACTOR
    return 1.0


def _enemy_death_reward_credit(event: TrainingBattleEvent) -> float:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    if "enemy_death_reward_credit" in metadata:
        return _clamp01(
            _finite_float(
                metadata["enemy_death_reward_credit"],
                default=1.0,
            )
        )
    source_name = _ability_name(event)
    if source_name == "DruugeA2":
        return const.DRUUGE_A2_ENEMY_DEATH_REWARD_FACTOR
    if source_name == "ShofixtiA2":
        return const.SHOFIXTI_A2_ENEMY_DEATH_REWARD_FACTOR
    role = getattr(getattr(event.obj, "collision_capabilities", None), "role", None)
    if getattr(role, "name", None) == "PLANET":
        return const.PLANET_ENEMY_DEATH_REWARD_FACTOR
    return 1.0


def _object_removed_by_owner_weapon(
    event: TrainingBattleEvent,
    target_owner: object | None,
    source_owner: object | None,
) -> bool:
    if event.event_type != EVENT_OBJECT_REMOVED or not event.destroyed:
        return False
    if getattr(event.obj, "type", None) not in {"projectile", "special_object"}:
        return False
    if not _same_entity(event.owner, target_owner):
        return False
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    if not _same_entity(metadata.get("source_owner"), source_owner):
        return False
    return metadata.get("source_type") in {"projectile", "special_object", "laser"}


def _is_chmmr_satellite_event(event: TrainingBattleEvent) -> bool:
    target = event.target if event.event_type == EVENT_OBJECT_HP_CHANGED else event.obj
    return getattr(target, "name", None) == "ChmmrSatellite"


def _removal_source_owner(event: TrainingBattleEvent):
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    return metadata.get("source_owner")


def _configured_satellite_count(owner) -> int:
    from src.Objects.Ships.catalog import SHIP_DEFINITIONS

    definition = SHIP_DEFINITIONS.get(getattr(owner, "name", ""))
    return max(1, int(getattr(definition, "satellite_count", 1)))


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
