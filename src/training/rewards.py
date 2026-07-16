"""Per-event and per-second reward components with discounted returns."""

from __future__ import annotations

import math
import time
from array import array
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import src.const as const
from src.training import combat_adapters
from src.training.causal_credit import (
    AbilityRewardCredit,
    REWARD_MODE_CAUSAL,
    REWARD_MODE_LEGACY,
    REWARD_MODE_SHADOW,
    REWARD_MODES,
)
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
REWARD_ENEMY_LOSES_CREW = "Reduce enemy crew"
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
    REWARD_ENEMY_LOSES_CREW: "Enemy loses crew",
    REWARD_SPAWN_A1: "Spawn A1 object",
    REWARD_SPAWN_A2: "Spawn A2 object",
    REWARD_A1_RANGE: "Get in A1 weapon range",
    REWARD_A2_RANGE: "Get in A2 weapon range",
}

ONGOING_REWARD_COMPONENTS = (
    REWARD_POINT_A1,
    REWARD_A1_RANGE,
    REWARD_POINT_A2,
    REWARD_A2_RANGE,
    REWARD_HIGH_SPEED,
    REWARD_BATTERY_AT_ZERO,
)
SUSTAINED_A2_REWARD_SHIPS = frozenset({"Ilwrath", "Androsynth"})

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


@dataclass(frozen=True)
class ReturnDistributionSummary:
    """Compact exact distribution statistics for one finalized trajectory."""

    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    maximum_absolute: float


@dataclass(frozen=True)
class ReturnComparisonSummary:
    baseline: ReturnDistributionSummary
    proposed: ReturnDistributionSummary
    delta: ReturnDistributionSummary


@dataclass(frozen=True)
class ShadowReturnComparison:
    """Legacy-versus-causal finalized target diagnostics for shadow mode."""

    sample_count: int
    overall: ReturnComparisonSummary
    by_component: dict[str, ReturnComparisonSummary]
    by_action: dict[int, ReturnComparisonSummary]
    by_component_and_action: dict[tuple[str, int], ReturnComparisonSummary]


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
                self._window_totals,
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
            matured.append(
                _sample_from_component_vector(
                    pending.decision,
                    samples_by_offset[offset],
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


class StagedTrajectoryPipeline:
    """Packed mutable trajectory storage finalized before replay insertion.

    In legacy mode a trajectory still closes at the historical combat terminal;
    later rollout phases can keep the same storage open across enemy deaths.
    """

    def __init__(
        self,
        *,
        gamma: float,
        reward_weights: Mapping[str, float] | None = None,
        mode: str = REWARD_MODE_LEGACY,
    ):
        self.gamma = _validate_gamma(gamma)
        self.discount_cutoff_frames = discount_cutoff_frames(self.gamma)
        self.reward_weights = normalize_reward_weights(reward_weights or {})
        if mode not in REWARD_MODES:
            raise ValueError("unsupported reward mode")
        self.mode = str(mode)
        self._cutoff_discount = self.gamma ** self.discount_cutoff_frames
        self.trajectory_id: str | None = None
        self._observation_size: int | None = None
        self._observations = array("f")
        self._actions = array("i")
        self._frame_ids = array("q")
        self._components = array("d")
        self._shadow_components = (
            array("d") if self.mode == REWARD_MODE_SHADOW else None
        )
        self._frame_to_index: dict[int, int] = {}
        self._outcome_count = 0
        self._closed = False
        self.peak_staged_frames = 0
        self.peak_staged_bytes = 0
        self.finalization_seconds = 0.0
        self.last_shadow_samples: tuple[MatureTrainingSample, ...] = ()
        self.last_shadow_comparison: ShadowReturnComparison | None = None
        self._diagnostics = None

    @property
    def pending_count(self) -> int:
        return len(self._frame_ids)

    @property
    def is_open(self) -> bool:
        return bool(self._frame_ids) and not self._closed

    @property
    def staged_storage_bytes(self) -> int:
        return (
            len(self._observations) * self._observations.itemsize
            + len(self._actions) * self._actions.itemsize
            + len(self._frame_ids) * self._frame_ids.itemsize
            + len(self._components) * self._components.itemsize
            + (
                len(self._shadow_components) * self._shadow_components.itemsize
                if self._shadow_components is not None
                else 0
            )
        )

    def reset(self) -> None:
        self.trajectory_id = None
        self._observation_size = None
        self._observations = array("f")
        self._actions = array("i")
        self._frame_ids = array("q")
        self._components = array("d")
        self._shadow_components = (
            array("d") if self.mode == REWARD_MODE_SHADOW else None
        )
        self._frame_to_index.clear()
        self._outcome_count = 0
        self._closed = False
        self._diagnostics = None

    def stage_decision(
        self,
        decision: RewardDecisionFrame,
        *,
        trajectory_id: str | None = None,
    ) -> int:
        if self._closed:
            raise RuntimeError("cannot stage a decision after trajectory closure")
        frame_id = int(decision.frame_id)
        existing = self._frame_to_index.get(frame_id)
        if existing is not None:
            return existing
        if self._frame_ids and frame_id <= self._frame_ids[-1]:
            raise ValueError("staged decision frame IDs must increase")
        if self.trajectory_id is None:
            self.trajectory_id = str(trajectory_id) if trajectory_id is not None else None
        elif trajectory_id is not None and str(trajectory_id) != self.trajectory_id:
            raise ValueError("cannot mix reward trajectories in staged storage")

        observation = tuple(float(value) for value in decision.observation)
        if self._observation_size is None:
            self._observation_size = len(observation)
        elif len(observation) != self._observation_size:
            raise ValueError("staged observations must have a stable width")

        index = len(self._frame_ids)
        self._frame_to_index[frame_id] = index
        self._observations.extend(observation)
        self._actions.append(int(decision.action_index))
        self._frame_ids.append(frame_id)
        self._components.extend((0.0,) * _REWARD_COMPONENT_COUNT)
        if self._shadow_components is not None:
            self._shadow_components.extend((0.0,) * _REWARD_COMPONENT_COUNT)
        self.peak_staged_frames = max(self.peak_staged_frames, self.pending_count)
        self.peak_staged_bytes = max(self.peak_staged_bytes, self.staged_storage_bytes)
        return index

    def add_frame(
        self,
        decision: RewardDecisionFrame,
        outcome: RewardFrameOutcome,
        *,
        ledger=None,
    ) -> list[MatureTrainingSample]:
        if decision.frame_id != outcome.frame_id:
            raise ValueError("decision and outcome frame IDs must match")
        index = self.stage_decision(decision)
        if index != self._outcome_count:
            raise ValueError("outcomes must be added once in staged decision order")
        components = _calculate_immediate_reward_component_vector(decision, outcome)
        offset = index * _REWARD_COMPONENT_COUNT
        self._components[offset : offset + _REWARD_COMPONENT_COUNT] = array(
            "d", components
        )
        if self._shadow_components is not None:
            self._shadow_components[
                offset : offset + _REWARD_COMPONENT_COUNT
            ] = array("d", components)
        if ledger is not None:
            self._diagnostics = ledger.diagnostics
            ledger.diagnostics.peak_staged_frames = max(
                ledger.diagnostics.peak_staged_frames,
                self.peak_staged_frames,
            )
            ledger.diagnostics.peak_staged_bytes = max(
                ledger.diagnostics.peak_staged_bytes,
                self.peak_staged_bytes,
            )
            if self.mode in {REWARD_MODE_CAUSAL, REWARD_MODE_SHADOW}:
                _route_causal_event_components(
                    self,
                    decision,
                    outcome,
                    ledger,
                )
        self._outcome_count += 1
        if outcome.terminal:
            return self._finalize(end_frame_id=outcome.frame_id, terminal=True)
        return []

    def add_component_at_frame(
        self,
        frame_id: int,
        component: str,
        amount: float,
    ) -> None:
        index = self._frame_to_index.get(int(frame_id))
        if index is None:
            raise KeyError(f"origin frame {frame_id} is not staged")
        component_index = _REWARD_COMPONENT_INDEX[component]
        offset = index * _REWARD_COMPONENT_COUNT + component_index
        self._components[offset] += float(amount)

    def _add_shadow_component_at_frame(
        self,
        frame_id: int,
        component: str,
        amount: float,
    ) -> None:
        if self._shadow_components is None:
            raise RuntimeError("shadow components require shadow reward mode")
        index = self._frame_to_index.get(int(frame_id))
        if index is None:
            raise KeyError(f"origin frame {frame_id} is not staged")
        component_index = _REWARD_COMPONENT_INDEX[component]
        offset = index * _REWARD_COMPONENT_COUNT + component_index
        self._shadow_components[offset] += float(amount)

    def immediate_components_for_frame(self, frame_id: int) -> dict[str, float]:
        index = self._frame_to_index[int(frame_id)]
        offset = index * _REWARD_COMPONENT_COUNT
        return _component_dict_from_vector(
            self._components[offset : offset + _REWARD_COMPONENT_COUNT]
        )

    def shadow_immediate_components_for_frame(
        self, frame_id: int
    ) -> dict[str, float]:
        if self._shadow_components is None:
            raise RuntimeError("shadow components require shadow reward mode")
        index = self._frame_to_index[int(frame_id)]
        offset = index * _REWARD_COMPONENT_COUNT
        return _component_dict_from_vector(
            self._shadow_components[offset : offset + _REWARD_COMPONENT_COUNT]
        )

    def flush_pending(self, *, end_frame_id: int) -> list[MatureTrainingSample]:
        if self._closed:
            raise RuntimeError("cannot flush after trajectory closure")
        return self._finalize(end_frame_id=end_frame_id, terminal=True)

    def _finalize(
        self,
        *,
        end_frame_id: int,
        terminal: bool,
    ) -> list[MatureTrainingSample]:
        started_at = time.perf_counter()
        if self._outcome_count != self.pending_count:
            raise RuntimeError("cannot finalize a trajectory with an open decision frame")
        frame_count = self.pending_count
        if frame_count == 0:
            self._closed = True
            return []

        totals = self._discounted_totals(self._components, frame_count)
        samples = self._samples_from_totals(
            totals,
            frame_count=frame_count,
            end_frame_id=end_frame_id,
            terminal=terminal,
        )
        shadow_samples: list[MatureTrainingSample] = []
        shadow_comparison = None
        if self.mode == REWARD_MODE_SHADOW:
            if self._shadow_components is None:
                raise RuntimeError("shadow component storage is not initialized")
            shadow_totals = self._discounted_totals(
                self._shadow_components,
                frame_count,
            )
            shadow_samples = self._samples_from_totals(
                shadow_totals,
                frame_count=frame_count,
                end_frame_id=end_frame_id,
                terminal=terminal,
            )
            shadow_comparison = summarize_shadow_returns(samples, shadow_samples)

        elapsed = time.perf_counter() - started_at
        self.finalization_seconds += elapsed
        diagnostics = self._diagnostics
        if diagnostics is not None:
            diagnostics.finalized_trajectory_lengths.append(frame_count)
            diagnostics.finalization_seconds.append(elapsed)
            if shadow_comparison is not None:
                diagnostics.shadow_comparison_count += 1
                diagnostics.shadow_comparisons.append(shadow_comparison)
                diagnostics.last_shadow_comparison = shadow_comparison
        self.reset()
        self.last_shadow_samples = tuple(shadow_samples)
        self.last_shadow_comparison = shadow_comparison
        self._closed = True
        return samples

    def _discounted_totals(
        self,
        component_buffer: array,
        frame_count: int,
    ) -> list[list[float]]:
        totals: list[list[float]] = [
            [0.0] * _REWARD_COMPONENT_COUNT for _ in range(frame_count)
        ]
        running = [0.0] * _REWARD_COMPONENT_COUNT
        horizon = self.discount_cutoff_frames
        for frame_index in range(frame_count - 1, -1, -1):
            offset = frame_index * _REWARD_COMPONENT_COUNT
            expired_index = frame_index + horizon
            expired_offset = expired_index * _REWARD_COMPONENT_COUNT
            for component_index in range(_REWARD_COMPONENT_COUNT):
                value = component_buffer[offset + component_index]
                total = value + self.gamma * running[component_index]
                if expired_index < frame_count:
                    total -= (
                        self._cutoff_discount
                        * component_buffer[expired_offset + component_index]
                    )
                running[component_index] = total
                totals[frame_index][component_index] = total
        return totals

    def _samples_from_totals(
        self,
        totals: Sequence[Sequence[float]],
        *,
        frame_count: int,
        end_frame_id: int,
        terminal: bool,
    ) -> list[MatureTrainingSample]:
        observation_size = int(self._observation_size or 0)
        samples: list[MatureTrainingSample] = []
        horizon = self.discount_cutoff_frames
        for index in range(frame_count):
            actual_frame_count = min(horizon, frame_count - index)
            sample_end_index = index + actual_frame_count - 1
            observation_offset = index * observation_size
            decision = RewardDecisionFrame(
                frame_id=int(self._frame_ids[index]),
                observation=tuple(
                    self._observations[
                        observation_offset : observation_offset + observation_size
                    ]
                ),
                action_index=int(self._actions[index]),
            )
            samples.append(
                _sample_from_component_vector(
                    decision,
                    totals[index],
                    self.reward_weights,
                    end_frame_id=int(self._frame_ids[sample_end_index]),
                    actual_frame_count=actual_frame_count,
                    terminal_truncated=terminal and actual_frame_count < horizon,
                )
            )
        return samples


def summarize_shadow_returns(
    baseline_samples: Sequence[MatureTrainingSample],
    proposed_samples: Sequence[MatureTrainingSample],
) -> ShadowReturnComparison:
    """Summarize aligned legacy and causal targets without retaining rollouts."""

    baseline = tuple(baseline_samples)
    proposed = tuple(proposed_samples)
    if len(baseline) != len(proposed):
        raise ValueError("shadow sample sets must have equal lengths")
    for old, new in zip(baseline, proposed):
        if (
            old.start_frame_id != new.start_frame_id
            or old.action_index != new.action_index
        ):
            raise ValueError("shadow sample sets must be frame/action aligned")

    def comparison(old_values, new_values):
        old_values = tuple(float(value) for value in old_values)
        new_values = tuple(float(value) for value in new_values)
        return ReturnComparisonSummary(
            baseline=_distribution_summary(old_values),
            proposed=_distribution_summary(new_values),
            delta=_distribution_summary(
                tuple(new - old for old, new in zip(old_values, new_values))
            ),
        )

    overall = comparison(
        (sample.return_value for sample in baseline),
        (sample.return_value for sample in proposed),
    )
    by_component = {
        component: comparison(
            (sample.component_values[component] for sample in baseline),
            (sample.component_values[component] for sample in proposed),
        )
        for component in REWARD_COMPONENTS
    }
    actions = sorted({sample.action_index for sample in baseline})
    by_action = {}
    by_component_and_action = {}
    for action in actions:
        old_action = tuple(
            sample for sample in baseline if sample.action_index == action
        )
        new_action = tuple(
            sample for sample in proposed if sample.action_index == action
        )
        by_action[action] = comparison(
            (sample.return_value for sample in old_action),
            (sample.return_value for sample in new_action),
        )
        for component in REWARD_COMPONENTS:
            by_component_and_action[(component, action)] = comparison(
                (sample.component_values[component] for sample in old_action),
                (sample.component_values[component] for sample in new_action),
            )
    return ShadowReturnComparison(
        sample_count=len(baseline),
        overall=overall,
        by_component=by_component,
        by_action=by_action,
        by_component_and_action=by_component_and_action,
    )


def _distribution_summary(values: Sequence[float]) -> ReturnDistributionSummary:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return ReturnDistributionSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return ReturnDistributionSummary(
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        p50=_linear_percentile(ordered, 0.50),
        p95=_linear_percentile(ordered, 0.95),
        p99=_linear_percentile(ordered, 0.99),
        minimum=ordered[0],
        maximum=ordered[-1],
        maximum_absolute=max(abs(value) for value in ordered),
    )


def _linear_percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower]) + (
        float(ordered[upper]) - float(ordered[lower])
    ) * fraction

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


def _component_dict_from_vector(components: Sequence[float]) -> dict[str, float]:
    return {
        component: float(components[index])
        for index, component in enumerate(REWARD_COMPONENTS)
    }


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

    frame_seconds = 1.0 / float(const.FPS)
    components[_REWARD_COMPONENT_INDEX[REWARD_POINT_A1]] = (
        frame_seconds if decision.a1_pointing else 0.0
    )
    components[_REWARD_COMPONENT_INDEX[REWARD_A1_RANGE]] = (
        frame_seconds if decision.a1_in_range else 0.0
    )
    components[_REWARD_COMPONENT_INDEX[REWARD_POINT_A2]] = (
        frame_seconds if decision.a2_pointing else 0.0
    )
    components[_REWARD_COMPONENT_INDEX[REWARD_A2_RANGE]] = (
        frame_seconds if decision.a2_in_range else 0.0
    )

    if outcome.self_max_thrust > 0.0 and outcome.self_speed > outcome.self_max_thrust:
        components[_REWARD_COMPONENT_INDEX[REWARD_HIGH_SPEED]] = (
            outcome.self_speed - outcome.self_max_thrust
        ) / outcome.self_max_thrust * frame_seconds

    battery_delta = outcome.self_battery - decision.self_battery
    if battery_delta > 0:
        components[_REWARD_COMPONENT_INDEX[REWARD_GAIN_BATTERY]] = battery_delta
    elif battery_delta < 0:
        components[_REWARD_COMPONENT_INDEX[REWARD_LOSE_BATTERY]] = -battery_delta
    if outcome.self_battery <= 0:
        components[_REWARD_COMPONENT_INDEX[REWARD_BATTERY_AT_ZERO]] = frame_seconds

    self_ship = decision.self_ship
    enemy_ship = decision.enemy_ship
    if _uses_sustained_a2_reward(self_ship) and outcome.self_sustained_a2_active:
        components[_REWARD_COMPONENT_INDEX[REWARD_SPAWN_A2]] = frame_seconds
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


def _route_causal_event_components(
    pipeline: StagedTrajectoryPipeline,
    decision: RewardDecisionFrame,
    outcome: RewardFrameOutcome,
    ledger,
) -> None:
    """Relocate covered positive effects after their legacy amount is known."""

    for event in outcome.events:
        if (
            pipeline.mode == REWARD_MODE_CAUSAL
            and _is_own_launched_crew_loss(decision, event)
        ):
            _route_own_launched_crew_loss(
                pipeline,
                decision,
                event,
                ledger,
            )
            continue
        source_ability = str(
            event.metadata.get("source_ability_name")
            or event.ability_name
            or ""
        )
        source_owner = event.metadata.get("source_owner") or event.actor
        source_type = event.metadata.get("source_type") or getattr(
            event.obj, "type", None
        )
        expected_ability = (
            source_type in {"projectile", "special_object", "laser", "area"}
            and source_ability != "ChmmrSatellite"
        )
        has_credit = event.metadata.get("reward_credit") is not None
        if not has_credit and (
            not expected_ability
            or not _same_entity(source_owner, decision.self_ship)
        ):
            continue
        contributions = _routeable_positive_event_components(decision, event)
        for component, amount in contributions.items():
            if amount <= 0.0:
                continue
            credit = ledger.resolve_event_credit(
                event,
                component=component,
                expected=expected_ability,
            )
            if credit is None:
                continue
            if any(
                int(origin.frame_index) not in pipeline._frame_to_index
                for origin in credit.origins
            ):
                ledger.diagnostics.missing_provenance[source_ability] += 1
                continue

            effect_index = pipeline._frame_to_index[int(decision.frame_id)]
            component_index = _REWARD_COMPONENT_INDEX[component]
            effect_offset = effect_index * _REWARD_COMPONENT_COUNT + component_index
            if pipeline.mode == REWARD_MODE_SHADOW:
                if pipeline._shadow_components is None:
                    raise RuntimeError("shadow component storage is not initialized")
                pipeline._shadow_components[effect_offset] -= float(amount)
                for origin in credit.origins:
                    pipeline._add_shadow_component_at_frame(
                        origin.frame_index,
                        component,
                        float(amount) * float(origin.weight),
                    )
            if pipeline.mode == REWARD_MODE_CAUSAL:
                pipeline._components[effect_offset] -= float(amount)
                for origin in credit.origins:
                    pipeline.add_component_at_frame(
                        origin.frame_index,
                        component,
                        float(amount) * float(origin.weight),
                    )


def _is_own_launched_crew_loss(
    decision: RewardDecisionFrame,
    event: TrainingBattleEvent,
) -> bool:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    return bool(
        event.event_type == EVENT_CREW_CHANGED
        and event.magnitude < 0
        and _same_entity(event.target, decision.self_ship)
        and metadata.get("launched_crew_loss")
    )


def _route_own_launched_crew_loss(
    pipeline: StagedTrajectoryPipeline,
    decision: RewardDecisionFrame,
    event: TrainingBattleEvent,
    ledger,
) -> None:
    """Move a permanent launched-unit loss to its selected causal launch."""

    amount = -float(event.magnitude)
    status, destinations = _own_launched_crew_loss_destinations(
        decision,
        event,
        ledger,
    )
    if status == "closed":
        _adjust_effect_component(
            pipeline,
            decision.frame_id,
            REWARD_LOSE_CREW,
            -amount,
        )
        return
    if status != "open":
        return

    if any(
        int(origin.frame_index) not in pipeline._frame_to_index
        for credit, _ in destinations
        for origin in credit.origins
    ):
        ability = str(
            event.metadata.get("launched_unit_ability_name") or "unknown"
        )
        ledger.diagnostics.missing_provenance[ability] += 1
        ledger.diagnostics.launched_crew_loss_routes["missing_origin_frame"] += 1
        return

    _adjust_effect_component(
        pipeline,
        decision.frame_id,
        REWARD_LOSE_CREW,
        -amount,
    )
    for credit, destination_weight in destinations:
        for origin in credit.origins:
            pipeline.add_component_at_frame(
                origin.frame_index,
                REWARD_LOSE_CREW,
                amount * float(destination_weight) * float(origin.weight),
            )


def _own_launched_crew_loss_destinations(
    decision: RewardDecisionFrame,
    event: TrainingBattleEvent,
    ledger,
) -> tuple[str, tuple[tuple[AbilityRewardCredit, float], ...]]:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    unit = metadata.get("launched_unit")
    unit_credit = metadata.get("launched_unit_credit")
    unit_ability = str(metadata.get("launched_unit_ability_name") or "unknown")
    if not isinstance(unit_credit, AbilityRewardCredit):
        ledger.diagnostics.missing_provenance[unit_ability] += 1
        ledger.diagnostics.launched_crew_loss_routes["missing_unit"] += 1
        return "missing", ()

    source = metadata.get("source")
    source_credit = metadata.get("reward_credit")
    source_owner = metadata.get("source_owner")
    source_type = metadata.get("source_type")
    friendly_weapon = bool(
        source is not None
        and source is not unit
        and _same_entity(source_owner, decision.self_ship)
        and source_type in {"projectile", "special_object", "laser", "area"}
    )

    route_key = "natural" if source is unit else "external"
    destinations: tuple[tuple[AbilityRewardCredit, float], ...] = (
        (unit_credit, 1.0),
    )
    if friendly_weapon:
        if not isinstance(source_credit, AbilityRewardCredit):
            source_ability = str(
                metadata.get("source_ability_name") or "unknown"
            )
            ledger.diagnostics.missing_provenance[source_ability] += 1
            ledger.diagnostics.launched_crew_loss_routes[
                "missing_friendly_source"
            ] += 1
            return "missing", ()
        unit_stamp = _spawn_stamp_from_metadata(
            metadata.get("launched_unit_spawn_stamp")
        )
        source_stamp = _spawn_stamp_from_metadata(
            metadata.get("source_spawn_stamp")
        )
        if unit_stamp is None or source_stamp is None:
            ledger.diagnostics.missing_provenance[unit_ability] += 1
            ledger.diagnostics.launched_crew_loss_routes[
                "missing_spawn_stamp"
            ] += 1
            return "missing", ()
        if unit_stamp > source_stamp:
            route_key = "friendly_fire_fighter"
        elif source_stamp > unit_stamp:
            route_key = "friendly_fire_source"
            destinations = ((source_credit, 1.0),)
        else:
            route_key = "friendly_fire_tie"
            destinations = ((unit_credit, 0.5), (source_credit, 0.5))

    if any(not ledger.credit_is_open(credit) for credit, _ in destinations):
        ledger.diagnostics.closed_trajectory_rejections[unit_ability] += 1
        ledger.diagnostics.launched_crew_loss_routes["closed"] += 1
        return "closed", ()

    ledger.diagnostics.routed_events[(REWARD_LOSE_CREW, unit_ability)] += 1
    ledger.diagnostics.launched_crew_loss_routes[route_key] += 1
    return "open", destinations


def _spawn_stamp_from_metadata(value) -> tuple[int, int] | None:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return value
    return None


def _adjust_effect_component(
    pipeline: StagedTrajectoryPipeline,
    frame_id: int,
    component: str,
    amount: float,
) -> None:
    index = pipeline._frame_to_index[int(frame_id)]
    offset = index * _REWARD_COMPONENT_COUNT + _REWARD_COMPONENT_INDEX[component]
    pipeline._components[offset] += float(amount)


def _routeable_positive_event_components(
    decision: RewardDecisionFrame,
    event: TrainingBattleEvent,
) -> dict[str, float]:
    self_ship = decision.self_ship
    enemy_ship = decision.enemy_ship
    contributions: dict[str, float] = {}
    if event.event_type == EVENT_CREW_CHANGED:
        if _same_entity(event.target, enemy_ship) and event.magnitude < 0:
            contributions[REWARD_ENEMY_LOSES_CREW] = (
                -float(event.magnitude) * _source_reward_credit(event)
            )
    elif event.event_type == EVENT_DEBUFF_APPLIED:
        if _same_entity(event.target, enemy_ship) and event.magnitude > 0:
            contributions[REWARD_DEBUFF_ENEMY] = float(event.magnitude)
    elif event.event_type == EVENT_OBJECT_REMOVED:
        if _is_chmmr_satellite_event(event):
            if (
                event.destroyed
                and _same_entity(event.owner, enemy_ship)
                and _same_entity(_removal_source_owner(event), self_ship)
            ):
                contributions[REWARD_KILL_ENEMY] = 0.5
        elif _object_removed_by_owner_weapon(event, enemy_ship, self_ship):
            contributions[REWARD_KILL_ENEMY_OBJECT] = 1.0
    elif event.event_type == EVENT_OBJECT_HP_CHANGED:
        if (
            _is_chmmr_satellite_event(event)
            and _same_entity(event.owner, enemy_ship)
            and _same_entity(event.actor, self_ship)
        ):
            contributions[REWARD_ENEMY_LOSES_CREW] = (
                max(0.0, -float(event.magnitude)) * 0.5
            )
    elif event.event_type == EVENT_SHIP_DIED:
        if _same_entity(event.target, enemy_ship):
            contributions[REWARD_KILL_ENEMY] = _enemy_death_reward_credit(event)
    return contributions


def calculate_reward_components(
    start_decision: RewardDecisionFrame,
    decisions: Sequence[RewardDecisionFrame],
    outcomes: Sequence[RewardFrameOutcome],
) -> dict[str, float]:
    if not decisions or not outcomes or len(decisions) != len(outcomes):
        raise ValueError("reward windows require matching decisions and outcomes")
    components = {component: 0.0 for component in REWARD_COMPONENTS}
    for decision, outcome in zip(decisions, outcomes):
        for component, value in calculate_immediate_reward_components(
            decision, outcome
        ).items():
            components[component] += value
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
    components = discounted_sum_components(immediate_components, gamma)
    return _sample_from_components(
        start_decision,
        components,
        reward_weights,
        end_frame_id=end_frame_id,
        actual_frame_count=len(immediate_components),
        terminal_truncated=terminal_truncated,
    )


def discounted_sum_components(
    immediate_components: Sequence[Mapping[str, float]],
    gamma: float,
) -> dict[str, float]:
    gamma = _validate_gamma(gamma)
    if not immediate_components:
        raise ValueError("reward windows must contain at least one frame")
    totals = {component: 0.0 for component in REWARD_COMPONENTS}
    discount = 1.0
    for frame_components in immediate_components:
        for component in REWARD_COMPONENTS:
            totals[component] += float(frame_components.get(component, 0.0)) * discount
        discount *= gamma
    return totals


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
    return getattr(ship, "name", None) in SUSTAINED_A2_REWARD_SHIPS


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
