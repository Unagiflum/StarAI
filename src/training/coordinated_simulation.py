"""CPU-only coordinated battle runtime shared by parent and workers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import time
from typing import Any

from src.Battle.battle import BattleSimulation
from src.Battle.computer_control import guard_computer_controls
from src.Objects.Ships.registry import create_ship
from src.training import event_ledger
from src.training.action_safety import guard_training_selection
from src.training.baseline import BaselineOpponentController
from src.training.causal_credit import REWARD_MODE_LEGACY
from src.training.contracts import (
    ACTION_OUTPUT_SIZE,
    TrainingAction,
    action_for_index,
)
from src.training.coordinated_contracts import (
    CoordinatedFixedFrameWindowResult,
    TrainingEpisodeResult,
)
from src.training.cpu_contracts import (
    OpponentControllerPlan,
    OpponentSpec,
    TrainingBatchAborted,
    TrainingOrchestrationConfig,
    single_controller_opponent_plan,
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
    baseline_controller: BaselineOpponentController
    opponent_plan: OpponentControllerPlan | None = None
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
    post_death_episode: PendingCombatEpisode | None = None
    action_counts: list[int] = field(
        default_factory=lambda: [0] * ACTION_OUTPUT_SIZE
    )
    rng: Any | None = None

    def __post_init__(self) -> None:
        if self.opponent_plan is None:
            self.opponent_plan = single_controller_opponent_plan(
                self.opponent,
                self.frame_limit,
            )
        if int(self.opponent_plan.frame_limit) != self.frame_limit:
            raise ValueError("opponent plan frame_limit must match the window")
        if self.opponent_plan.ship != self.opponent.ship:
            raise ValueError("opponent plan ship must match the window opponent")
        self.activate_opponent()

    @property
    def frame_limit(self) -> int:
        return int(self.state.record.config.match_time_limit)

    @property
    def complete(self) -> bool:
        return self.frames_consumed >= self.frame_limit

    def activate_opponent(self) -> OpponentSpec:
        if self.complete:
            return self.opponent
        if self.opponent_plan is None:
            raise RuntimeError("coordinated window has no opponent plan")
        self.opponent = self.opponent_plan.opponent_for_frame(
            self.frames_consumed
        )
        return self.opponent


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
        BaselineOpponentController(),
    )


def create_coordinated_window_runtime(
    *,
    state: Any,
    opponent: OpponentSpec,
    policy: Any,
    rng: Any,
    simulation_factory: Callable[..., BattleSimulation],
    audio_service: Any,
    opponent_plan: OpponentControllerPlan | None = None,
    ship_factory: Callable[..., Any] = create_ship,
) -> CoordinatedWindowRuntime:
    simulation, ledger, pipeline, baseline_controller = new_coordinated_battle(
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
        baseline_controller=baseline_controller,
        opponent_plan=opponent_plan,
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
    runtime.activate_opponent()
    raise_if_stop_requested(stop_requested)
    components = runtime.state.components
    if components is None:
        raise RuntimeError("coordinated components were not loaded")
    config = runtime.state.record.config
    simulation = runtime.simulation
    self_ship = simulation.player1
    enemy_ship = simulation.player2
    selection = guard_training_selection(selection, self_ship, enemy_ship)
    if not selection.held:
        runtime.action_counts[selection.action_index] += 1
    opponent_controls = guard_computer_controls(
        opponent_controls,
        enemy_ship,
        self_ship,
    )
    reward_decision_started_at = timing_started_at(timing_seconds)
    decision = decision_frame_from_battle_state(
        frame_id=simulation.frame_id + 1,
        observation=observation,
        action_index=selection.action_index,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        world=simulation.world,
        enabled_components=runtime.pipeline.enabled_reward_components,
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
        emit_sample=not selection.held,
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
    if (
        terminal
        and training_deaths == {2}
        and bool(getattr(simulation, "training_opponent_absent", False))
    ):
        win, loss, draw = classify_round_outcome(simulation, terminal_reason)
        kills, deaths = classify_kills_deaths(simulation)
        runtime.post_death_episode = PendingCombatEpisode(
            opponent=runtime.opponent,
            start_frame_id=runtime.episode_start_frame,
            end_frame_id=simulation.frame_id,
            terminal_reason=terminal_reason,
            win=win,
            loss=loss,
            draw=draw,
            kills=kills,
            deaths=deaths,
        )
        terminal = False
        terminal_reason = "post_death"
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
        enabled_components=runtime.pipeline.enabled_reward_components,
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
    if runtime.post_death_episode is not None:
        finalize_coordinated_post_death_episode(
            runtime,
            mature_samples=mature_samples,
        )
        return
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


def finalize_coordinated_post_death_episode(
    runtime: CoordinatedWindowRuntime,
    *,
    mature_samples: Sequence[MatureTrainingSample],
) -> None:
    pending = runtime.post_death_episode
    if pending is None:
        raise RuntimeError("coordinated runtime has no post-death episode")
    current_deaths = set(
        getattr(runtime.simulation, "training_episode_deaths", ())
    )
    boundary = PendingCombatEpisode(
        opponent=pending.opponent,
        start_frame_id=pending.start_frame_id,
        end_frame_id=runtime.simulation.frame_id,
        terminal_reason=pending.terminal_reason,
        win=pending.win,
        loss=pending.loss,
        draw=pending.draw,
        kills=pending.kills,
        deaths=max(pending.deaths, int(1 in current_deaths)),
    )
    config = runtime.state.record.config
    if config.reward_mode != REWARD_MODE_LEGACY:
        runtime.pending_episodes.append(boundary)
        runtime.episode_results.extend(
            finalize_pending_episodes(runtime.pending_episodes, mature_samples)
        )
        runtime.pending_episodes.clear()
        runtime.ledger.close_reward_trajectory()
    else:
        runtime.episode_results.append(TrainingEpisodeResult(
            opponent=boundary.opponent,
            frames=boundary.frames,
            terminal_reason=boundary.terminal_reason,
            mature_samples=runtime.episode_mature_count,
            total_return=average_value(
                runtime.episode_return_sum, runtime.episode_mature_count
            ),
            win=boundary.win,
            loss=boundary.loss,
            draw=boundary.draw,
            kills=boundary.kills,
            deaths=boundary.deaths,
            component_totals=average_components(
                runtime.episode_component_sums,
                runtime.episode_mature_count,
            ),
        ))
    runtime.post_death_episode = None
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
        runtime.baseline_controller = BaselineOpponentController()
    else:
        (
            runtime.simulation,
            runtime.ledger,
            runtime.pipeline,
            runtime.baseline_controller,
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
        handled_post_death = runtime.post_death_episode is not None
        if handled_post_death:
            finalize_coordinated_post_death_episode(
                runtime,
                mature_samples=mature_samples,
            )
        elif runtime.state.record.config.reward_mode != REWARD_MODE_LEGACY:
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
        if not handled_post_death:
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
        action_counts=tuple(runtime.action_counts),
    )


def permanent_terminal_state(simulation) -> tuple[bool, str]:
    if getattr(simulation, "training_episode_deaths", ()):
        return True, "resolved"
    if bool(getattr(simulation, "training_post_death_completed", False)):
        return True, "post_death_complete"
    if bool(getattr(simulation, "training_opponent_absent", False)):
        return False, "post_death"
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
