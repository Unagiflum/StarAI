"""CPU-only coordinated battle runtime shared by parent and workers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Objects.Ships.registry import create_ship
from src.toroidal import wrapped_delta
from src.training import event_ledger
from src.training.causal_credit import REWARD_MODE_LEGACY
from src.training.contracts import TrainingAction, action_for_index
from src.training.coordinated_contracts import (
    CoordinatedFixedFrameWindowResult,
    TrainingEpisodeResult,
)
from src.training.cpu_contracts import (
    OpponentSpec,
    TrainingBatchAborted,
    TrainingOrchestrationConfig,
)
from src.training.episode_metrics import PendingCombatEpisode, finalize_pending_episodes
from src.training.replay_contracts import ActionSelection
from src.training.rewards import (
    MatureTrainingSample,
    REWARD_COMPONENTS,
    StagedTrajectoryPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)


@dataclass
class CoordinatedWindowRuntime:
    state: Any
    opponent: OpponentSpec
    policy: Any
    simulation: Any
    ledger: Any
    pipeline: StagedTrajectoryPipeline
    simple_controller: "SimpleOpponentController"
    frames_consumed: int = 0
    total_mature_count: int = 0
    return_sum: float = 0.0
    component_sums: dict[str, float] = field(default_factory=dict)
    episode_results: list[TrainingEpisodeResult] = field(default_factory=list)
    pending_episodes: list[PendingCombatEpisode] = field(default_factory=list)
    episode_start_frame: int = 0
    episode_mature_count: int = 0
    episode_return_sum: float = 0.0
    episode_component_sums: dict[str, float] = field(default_factory=dict)
    episode_needs_timeout: bool = True

    @property
    def frame_limit(self) -> int:
        return int(self.state.record.config.match_time_limit)

    @property
    def complete(self) -> bool:
        return self.frames_consumed >= self.frame_limit


def direct_controls_for_action_index(action_index: int) -> TrainingAction:
    return action_for_index(int(action_index))


def _controls_mapping_from_action(action: TrainingAction) -> dict[str, bool]:
    return {
        "forward": action.thrust,
        "left": action.turn_left,
        "right": action.turn_right,
        "action1": action.a1,
        "action2": action.a2,
    }


def controls_for_action_index(action_index: int) -> dict[str, bool]:
    return _controls_mapping_from_action(action_for_index(int(action_index)))


class SimpleOpponentController:
    def __init__(self, config: TrainingOrchestrationConfig, *, rng=None):
        self.config = config
        self.rng = rng or random
        self.forward_held = False
        self.action1_held = False
        self.action2_held = False
        self.face_opponent_active = False
        self.next_face_decision_frame: int | None = None

    def controls_for_frame(self, simulation):
        return _controls_mapping_from_action(self.direct_controls_for_frame(simulation))

    def direct_controls_for_frame(self, simulation):
        self.forward_held = self._next_key_state(
            self.forward_held, self.config.forward_activity
        )
        self.action1_held = self._next_key_state(
            self.action1_held, self.config.a1_activity
        )
        self.action2_held = self._next_key_state(
            self.action2_held, self.config.a2_activity
        )
        left = right = False
        if self._should_face_opponent(getattr(simulation, "frame_id", 0)):
            left, right = _turn_toward_target(simulation.player2, simulation.player1)
        mask = (
            (1 if self.forward_held else 0)
            | (2 if left else 0)
            | (4 if right else 0)
            | (8 if self.action1_held else 0)
            | (16 if self.action2_held else 0)
        )
        return TrainingAction.from_mask(mask)

    def _next_key_state(self, held: bool, activity: float) -> bool:
        probability = _activity_probability(activity)
        if probability <= 0.0:
            return False
        if probability >= 1.0 or self.rng.random() < probability:
            return not held
        return held

    def _should_face_opponent(self, frame_id: int) -> bool:
        probability = _activity_probability(self.config.face_opponent_activity)
        if probability <= 0.0:
            self.face_opponent_active = False
            return False
        if probability >= 1.0:
            self.face_opponent_active = True
            return True
        frame_id = int(frame_id)
        if self.next_face_decision_frame is None or frame_id >= self.next_face_decision_frame:
            self.face_opponent_active = self.rng.random() < probability
            self.next_face_decision_frame = frame_id + max(1, int(const.FPS))
        return self.face_opponent_active


def _turn_toward_target(ship, target) -> tuple[bool, bool]:
    dx, dy = wrapped_delta(ship.position, target.position)
    target_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
    rotation = float(getattr(ship, "rotation", 0.0)) % 360.0
    diff = (target_angle - rotation + 540.0) % 360.0 - 180.0
    if abs(diff) <= max(1.0, const.TURN_ANGLE / 2.0):
        return False, False
    return diff < 0.0, diff > 0.0


def _activity_probability(activity: float) -> float:
    return max(0.0, min(1.0, float(activity) / 100.0))


def initialize_training_simulation_ships(simulation, rng) -> None:
    if getattr(simulation, "training_spawn_initialized", False):
        return
    for ship in (simulation.player1, simulation.player2):
        if getattr(ship, "name", None) == "Shofixti":
            setattr(ship, "shofixti_arming_stage", getattr(ship, "ARMED", 2))
        start_hp = max(1, int(getattr(ship, "start_hp", getattr(ship, "current_hp", 1))))
        ship.current_hp = max(1, math.ceil(float(rng.random()) * start_hp))


def new_coordinated_battle(
    config: TrainingOrchestrationConfig,
    opponent: OpponentSpec,
    *,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    resources: Any | None = None,
    ship_factory: Callable[..., Any] = create_ship,
):
    trainee = ship_factory(
        config.trainee_ship, 1, resources=resources, audio_service=audio_service
    )
    opponent_ship = ship_factory(
        opponent.ship, 2, resources=resources, audio_service=audio_service
    )
    ledger = event_ledger.BattleEventLedger()
    simulation = simulation_factory(
        None,
        trainee,
        opponent_ship,
        audio_service=audio_service,
        rng=rng,
        resources=resources,
        include_stars=False,
        training_event_ledger=ledger,
    )
    set_visual_effects = getattr(simulation, "set_visual_effects_enabled", None)
    if callable(set_visual_effects):
        set_visual_effects(bool(config.display_on))
    initialize_training_simulation_ships(simulation, rng)
    return (
        simulation,
        ledger,
        StagedTrajectoryPipeline(
            gamma=config.gamma,
            reward_weights=config.reward_weights,
            mode=config.reward_mode,
        ),
        SimpleOpponentController(config, rng=rng),
    )


def create_coordinated_window_runtime(
    *,
    state: Any,
    opponent: OpponentSpec,
    policy: Any,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    ship_factory: Callable[..., Any] = create_ship,
) -> CoordinatedWindowRuntime:
    simulation, ledger, pipeline, simple_controller = new_coordinated_battle(
        state.record.config,
        opponent,
        rng=rng,
        simulation_factory=simulation_factory,
        audio_service=audio_service,
        ship_factory=ship_factory,
    )
    return CoordinatedWindowRuntime(
        state=state,
        opponent=opponent,
        policy=policy,
        simulation=simulation,
        ledger=ledger,
        pipeline=pipeline,
        simple_controller=simple_controller,
        component_sums={component: 0.0 for component in REWARD_COMPONENTS},
        episode_component_sums={component: 0.0 for component in REWARD_COMPONENTS},
    )


def advance_coordinated_window_frame(
    runtime: CoordinatedWindowRuntime,
    *,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    observation: Sequence[float],
    selection: ActionSelection,
    opponent_controls: Mapping[str, bool],
    resources: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    timing_seconds: dict[str, float] | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> None:
    if runtime.complete:
        return
    raise_if_stop_requested(stop_requested)
    components = runtime.state.components
    if components is None:
        raise RuntimeError("coordinated components were not loaded")
    config = runtime.state.record.config
    simulation = runtime.simulation
    self_ship = simulation.player1
    enemy_ship = simulation.player2
    reward_decision_started_at = timing_started_at(timing_seconds)
    decision = decision_frame_from_battle_state(
        frame_id=simulation.frame_id + 1,
        observation=observation,
        action_index=selection.action_index,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        world=simulation.world,
    )
    runtime.ledger.begin_decision(
        self_ship,
        decision.frame_id,
        selection.action_index,
        reward_mode=config.reward_mode,
    )
    staged_index = runtime.pipeline.stage_decision(
        decision,
        trajectory_id=runtime.ledger.active_trajectory_id,
    )
    add_timing_seconds(timing_seconds, "reward_decision", reward_decision_started_at)
    event_start = len(runtime.ledger.events)
    simulation_started_at = timing_started_at(timing_seconds)
    step_state = step_simulation_with_optional_timing(
        simulation,
        actions={
            1: direct_controls_for_action_index(selection.action_index),
            2: opponent_controls,
        },
        timing_seconds=timing_seconds,
    )
    add_timing_seconds(timing_seconds, "simulation", simulation_started_at)
    runtime.frames_consumed += 1
    reward_started_at = timing_started_at(timing_seconds)
    terminal_started_at = timing_started_at(timing_seconds)
    terminal, terminal_reason = permanent_terminal_state(simulation)
    training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
    causal_lifecycle = config.reward_mode != REWARD_MODE_LEGACY
    can_continue = bool(training_deaths) and runtime.frames_consumed < runtime.frame_limit
    reward_terminal = terminal and (
        not causal_lifecycle or 1 in training_deaths or not can_continue
    )
    events = tuple(runtime.ledger.events[event_start:])
    add_timing_seconds(timing_seconds, "reward_terminal", terminal_started_at)
    outcome_started_at = timing_started_at(timing_seconds)
    outcome = frame_outcome_from_battle_state(
        frame_id=step_state["frame_id"],
        self_ship=self_ship,
        events=events,
        terminal=reward_terminal,
    )
    add_timing_seconds(timing_seconds, "reward_outcome", outcome_started_at)
    pipeline_started_at = timing_started_at(timing_seconds)
    mature_samples = runtime.pipeline.add_frame(
        decision,
        outcome,
        ledger=runtime.ledger,
        staged_index=staged_index,
    )
    add_timing_seconds(timing_seconds, "reward_pipeline", pipeline_started_at)
    replay_started_at = timing_started_at(timing_seconds)
    components.replay_buffer.extend(mature_samples)
    add_timing_seconds(timing_seconds, "reward_replay_insert", replay_started_at)
    accumulate_started_at = timing_started_at(timing_seconds)
    mature_count = len(mature_samples)
    runtime.total_mature_count += mature_count
    runtime.episode_mature_count += mature_count
    sample_return = sum(sample.return_value for sample in mature_samples)
    runtime.return_sum += sample_return
    runtime.episode_return_sum += sample_return
    accumulate_weighted_components(
        runtime.component_sums,
        mature_samples,
        runtime.episode_component_sums,
    )
    add_timing_seconds(timing_seconds, "reward_accumulate", accumulate_started_at)
    if progress_callback is not None:
        progress_started_at = timing_started_at(timing_seconds)
        emit_window_progress(
            progress_callback,
            frame=runtime.frames_consumed,
            opponent=runtime.opponent,
            action_index=selection.action_index,
            exploratory=selection.exploratory,
            replay_size=len(components.replay_buffer),
            weighted_total_return=average_value(
                runtime.return_sum, runtime.total_mature_count
            ),
            component_totals=average_components(
                runtime.component_sums, runtime.total_mature_count
            ),
        )
        add_timing_seconds(timing_seconds, "reward_progress", progress_started_at)
    add_timing_seconds(timing_seconds, "reward", reward_started_at)
    raise_if_stop_requested(stop_requested)
    if terminal:
        record_coordinated_terminal_episode(
            runtime,
            terminal_reason=terminal_reason,
            mature_samples=mature_samples,
            reward_terminal=reward_terminal,
        )
        if runtime.frames_consumed < runtime.frame_limit:
            reset_coordinated_window_battle(
                runtime,
                reward_terminal=reward_terminal,
                rng=rng,
                simulation_factory=simulation_factory,
                audio_service=audio_service,
                resources=resources,
                ship_factory=ship_factory,
            )


def step_simulation_with_optional_timing(
    simulation: Any,
    *,
    actions: Mapping[int, Mapping[str, bool]],
    timing_seconds: dict[str, float] | None,
):
    if timing_seconds is None:
        return simulation.step(actions=actions)
    try:
        return simulation.step(actions=actions, timing_seconds=timing_seconds)
    except TypeError as exc:
        if "timing_seconds" not in str(exc):
            raise
        return simulation.step(actions=actions)


def record_coordinated_terminal_episode(
    runtime: CoordinatedWindowRuntime,
    *,
    terminal_reason: str,
    mature_samples: Sequence[MatureTrainingSample],
    reward_terminal: bool,
) -> None:
    win, loss, draw = classify_round_outcome(runtime.simulation, terminal_reason)
    kills, deaths = classify_kills_deaths(runtime.simulation)
    config = runtime.state.record.config
    if config.reward_mode != REWARD_MODE_LEGACY:
        runtime.pending_episodes.append(
            PendingCombatEpisode(
                opponent=runtime.opponent,
                start_frame_id=runtime.episode_start_frame,
                end_frame_id=runtime.simulation.frame_id,
                terminal_reason=terminal_reason,
                win=win,
                loss=loss,
                draw=draw,
                kills=kills,
                deaths=deaths,
            )
        )
        if reward_terminal:
            runtime.episode_results.extend(
                finalize_pending_episodes(runtime.pending_episodes, mature_samples)
            )
            runtime.pending_episodes.clear()
            runtime.ledger.close_reward_trajectory()
    else:
        runtime.episode_results.append(TrainingEpisodeResult(
            opponent=runtime.opponent,
            frames=runtime.frames_consumed - runtime.episode_start_frame,
            terminal_reason=terminal_reason,
            mature_samples=runtime.episode_mature_count,
            total_return=average_value(
                runtime.episode_return_sum, runtime.episode_mature_count
            ),
            win=win,
            loss=loss,
            draw=draw,
            kills=kills,
            deaths=deaths,
            component_totals=average_components(
                runtime.episode_component_sums, runtime.episode_mature_count
            ),
        ))
    runtime.episode_needs_timeout = False
    reset_span = getattr(runtime.policy, "reset_exploration_span", None)
    if callable(reset_span):
        reset_span()


def reset_coordinated_window_battle(
    runtime: CoordinatedWindowRuntime,
    *,
    reward_terminal: bool,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    resources: Any | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> None:
    config = runtime.state.record.config
    if getattr(runtime.simulation, "training_episode_deaths", ()):
        if reward_terminal:
            runtime.pipeline = StagedTrajectoryPipeline(
                gamma=config.gamma,
                reward_weights=config.reward_weights,
                mode=config.reward_mode,
            )
        runtime.simple_controller = SimpleOpponentController(config, rng=rng)
    else:
        (
            runtime.simulation,
            runtime.ledger,
            runtime.pipeline,
            runtime.simple_controller,
        ) = new_coordinated_battle(
            config,
            runtime.opponent,
            rng=rng,
            simulation_factory=simulation_factory,
            audio_service=audio_service,
            resources=resources,
            ship_factory=ship_factory,
        )
    runtime.episode_start_frame = runtime.simulation.frame_id
    runtime.episode_mature_count = 0
    runtime.episode_return_sum = 0.0
    runtime.episode_component_sums = {
        component: 0.0 for component in REWARD_COMPONENTS
    }
    runtime.episode_needs_timeout = True


def finish_coordinated_window(
    runtime: CoordinatedWindowRuntime,
    *,
    timing_seconds: dict[str, float] | None = None,
) -> CoordinatedFixedFrameWindowResult:
    components = runtime.state.components
    if components is None:
        raise RuntimeError("coordinated components were not loaded")
    if runtime.episode_needs_timeout:
        reward_started_at = timing_started_at(timing_seconds)
        flush_started_at = timing_started_at(timing_seconds)
        mature_samples = tuple(
            runtime.pipeline.flush_pending(end_frame_id=runtime.simulation.frame_id)
        )
        add_timing_seconds(timing_seconds, "reward_flush", flush_started_at)
        replay_started_at = timing_started_at(timing_seconds)
        components.replay_buffer.extend(mature_samples)
        add_timing_seconds(timing_seconds, "reward_replay_insert", replay_started_at)
        accumulate_started_at = timing_started_at(timing_seconds)
        runtime.total_mature_count += len(mature_samples)
        runtime.episode_mature_count += len(mature_samples)
        sample_return = sum(sample.return_value for sample in mature_samples)
        runtime.return_sum += sample_return
        runtime.episode_return_sum += sample_return
        accumulate_weighted_components(
            runtime.component_sums,
            mature_samples,
            runtime.episode_component_sums,
        )
        add_timing_seconds(timing_seconds, "reward_accumulate", accumulate_started_at)
        terminal_started_at = timing_started_at(timing_seconds)
        win, loss, draw = classify_round_outcome(runtime.simulation, "timeout")
        kills, deaths = classify_kills_deaths(runtime.simulation)
        add_timing_seconds(timing_seconds, "reward_terminal", terminal_started_at)
        if runtime.state.record.config.reward_mode != REWARD_MODE_LEGACY:
            runtime.pending_episodes.append(
                PendingCombatEpisode(
                    opponent=runtime.opponent,
                    start_frame_id=runtime.episode_start_frame,
                    end_frame_id=runtime.simulation.frame_id,
                    terminal_reason="timeout",
                    win=win,
                    loss=loss,
                    draw=draw,
                    kills=kills,
                    deaths=deaths,
                )
            )
            runtime.episode_results.extend(
                finalize_pending_episodes(runtime.pending_episodes, mature_samples)
            )
            runtime.pending_episodes.clear()
            runtime.ledger.close_reward_trajectory()
        else:
            runtime.episode_results.append(TrainingEpisodeResult(
                opponent=runtime.opponent,
                frames=runtime.frames_consumed - runtime.episode_start_frame,
                terminal_reason="timeout",
                mature_samples=runtime.episode_mature_count,
                total_return=average_value(
                    runtime.episode_return_sum, runtime.episode_mature_count
                ),
                win=win,
                loss=loss,
                draw=draw,
                kills=kills,
                deaths=deaths,
                component_totals=average_components(
                    runtime.episode_component_sums, runtime.episode_mature_count
                ),
            ))
        reset_span = getattr(runtime.policy, "reset_exploration_span", None)
        if callable(reset_span):
            reset_span()
        add_timing_seconds(timing_seconds, "reward", reward_started_at)
    return CoordinatedFixedFrameWindowResult(
        opponent=runtime.opponent,
        frames=runtime.frames_consumed,
        mature_samples=runtime.total_mature_count,
        episode_results=tuple(runtime.episode_results),
        total_return=average_value(runtime.return_sum, runtime.total_mature_count),
        win=any(result.win for result in runtime.episode_results),
        loss=any(result.loss for result in runtime.episode_results),
        draw=any(result.draw for result in runtime.episode_results),
        component_totals=average_components(
            runtime.component_sums, runtime.total_mature_count
        ),
        reward_diagnostics=runtime.ledger.diagnostics,
    )


def permanent_terminal_state(simulation) -> tuple[bool, str]:
    if getattr(simulation, "training_episode_deaths", ()):
        return True, "resolved"
    aftermath = getattr(simulation, "aftermath", None)
    if bool(getattr(aftermath, "pending_rebirths", None)):
        return False, "pending_rebirth"
    if aftermath is not None:
        return True, "resolved"
    if not ship_alive(simulation.player1) or not ship_alive(simulation.player2):
        return True, "resolved"
    return False, "running"


def ship_alive(ship) -> bool:
    return bool(getattr(ship, "currently_alive", True)) and getattr(
        ship, "current_hp", 1
    ) > 0


def classify_round_outcome(simulation, terminal_reason: str) -> tuple[bool, bool, bool]:
    training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
    if training_deaths:
        if training_deaths == {2}:
            return True, False, False
        if training_deaths == {1}:
            return False, True, False
        return False, False, True
    trainee_alive = ship_alive(simulation.player1)
    opponent_alive = ship_alive(simulation.player2)
    if terminal_reason == "timeout":
        return False, False, True
    if opponent_alive is False and trainee_alive:
        return True, False, False
    if trainee_alive is False and opponent_alive:
        return False, True, False
    return False, False, True


def classify_kills_deaths(simulation) -> tuple[int, int]:
    trainee_player = 1
    credited_killers = set(getattr(simulation, "training_episode_kills", ()))
    dead_players = set(getattr(simulation, "training_episode_deaths", ()))
    return (
        int(trainee_player in credited_killers),
        int(trainee_player in dead_players),
    )


def accumulate_weighted_components(
    totals: dict[str, float],
    samples: Sequence[MatureTrainingSample],
    *additional_totals: dict[str, float],
) -> None:
    destinations = (totals, *additional_totals)
    for sample in samples:
        for component, value in sample.weighted_components.items():
            numeric_value = float(value)
            for destination in destinations:
                destination[component] = (
                    destination.get(component, 0.0) + numeric_value
                )


def average_value(total: float, count: int) -> float:
    return float(total) / int(count) if int(count) > 0 else 0.0


def average_components(totals: Mapping[str, float], count: int) -> dict[str, float]:
    if int(count) <= 0:
        return {component: 0.0 for component in REWARD_COMPONENTS}
    return {
        component: float(totals.get(component, 0.0)) / int(count)
        for component in REWARD_COMPONENTS
    }


def raise_if_stop_requested(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise TrainingBatchAborted("training stop requested")


def emit_window_progress(
    callback: Callable[[Mapping[str, Any]], None] | None, **payload: Any
) -> None:
    if callback is not None:
        callback({"event": "frame", **payload})


def add_timing_seconds(
    timing_seconds: dict[str, float] | None, bucket: str, started_at: float
) -> None:
    if timing_seconds is not None:
        timing_seconds[bucket] = timing_seconds.get(bucket, 0.0) + max(
            0.0, time.perf_counter() - started_at
        )


def timing_started_at(timing_seconds: Mapping[str, float] | None) -> float:
    return time.perf_counter() if timing_seconds is not None else 0.0
