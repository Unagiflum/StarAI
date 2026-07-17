"""Headless training orchestration for battle rounds and batches."""

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
from src.audio import NullAudioService
from src.toroidal import wrapped_delta
from src.training import event_ledger, torch_backend
from src.training.contracts import (
    ACTION_OUTPUT_SIZE,
    SHIP_TYPE_CATALOG_ORDER,
    TrainingAction,
    action_for_index,
)
from src.training.coordinated_contracts import TrainingEpisodeResult
from src.training.causal_credit import CausalRewardDiagnostics, REWARD_MODE_LEGACY
from src.training.cpu_contracts import (
    DEFAULT_MINIBATCH_SIZE,
    DEFAULT_REPLAY_UPDATES_PER_BATCH,
    OPPONENT_MODE_EXISTING_AI,
    OPPONENT_MODE_SIMPLE,
    OpponentSpec,
    TrainingBatchAborted,
    TrainingOrchestrationConfig,
)
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    TrainingModelRepository,
    TrainingModelSlot,
)
from src.training.episode_metrics import (
    PendingCombatEpisode,
    finalize_pending_episodes,
)
from src.training.opponent_cache import load_opponent_model
from src.training.observation import encode_observation
from src.training.replay import (
    ActionSelection,
    TrainingReplayBuffer,
    optimize_from_replay,
    select_action_epsilon_greedy,
)
from src.training.rewards import (
    MatureTrainingSample,
    REWARD_COMPONENTS,
    StagedTrajectoryPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
    predict_action_values_read_only,
)


@dataclass(frozen=True)
class OpponentDiscoveryResult:
    opponents: tuple[OpponentSpec, ...]
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingRoundResult:
    opponent: OpponentSpec
    frames: int
    terminal_reason: str
    mature_samples: int
    total_return: float
    win: bool
    loss: bool
    draw: bool
    component_totals: Mapping[str, float] = field(default_factory=dict)
    episode_results: tuple[TrainingEpisodeResult, ...] = ()
    reward_diagnostics: CausalRewardDiagnostics | None = None


@dataclass(frozen=True)
class TrainingBatchResult:
    completed_rounds: int
    replay_size: int
    optimization_losses: tuple[float, ...]
    round_results: tuple[TrainingRoundResult, ...]

    @property
    def average_loss(self) -> float | None:
        if not self.optimization_losses:
            return None
        return sum(self.optimization_losses) / len(self.optimization_losses)


class ValueNetworkPolicy:
    """Epsilon-greedy policy over the Phase 1 value network."""

    def __init__(
        self,
        model,
        *,
        epsilon: float,
        epsilon_frame_span: int = 1,
        rng: Any | None = None,
    ):
        self.model = model
        self.epsilon = float(epsilon)
        self.epsilon_frame_span = max(1, int(epsilon_frame_span))
        self.rng = rng or random
        self.last_selection: ActionSelection | None = None
        self._span_frames_remaining = 0
        self._span_action_index: int | None = None

    def select_action(self, observation: Sequence[float]) -> ActionSelection:
        prepared = self.prepare_action_selection(observation)
        if prepared is not None:
            return prepared
        selection = select_action_epsilon_greedy(
            self.model,
            observation,
            epsilon=0.0,
            rng=self.rng,
        )
        return self.complete_greedy_selection(selection)

    def prepare_action_selection(
        self,
        observation: Sequence[float],
    ) -> ActionSelection | None:
        del observation
        if self._span_frames_remaining > 0:
            self._span_frames_remaining -= 1
            if self._span_action_index is not None:
                self.last_selection = ActionSelection(
                    action_index=self._span_action_index,
                    exploratory=True,
                )
                return self.last_selection
            return None

        self._span_action_index = None
        self._span_frames_remaining = self.epsilon_frame_span - 1
        if self.epsilon >= 1.0 or (
            self.epsilon > 0.0 and self.rng.random() < self.epsilon
        ):
            self._span_action_index = int(self.rng.randrange(ACTION_OUTPUT_SIZE))
            self.last_selection = ActionSelection(
                action_index=self._span_action_index,
                exploratory=True,
            )
            return self.last_selection

        return None

    def complete_greedy_selection(
        self,
        selection: ActionSelection | int,
        action_values: Sequence[float] | None = None,
    ) -> ActionSelection:
        if isinstance(selection, ActionSelection):
            self.last_selection = selection
        else:
            self.last_selection = ActionSelection(
                action_index=int(selection),
                exploratory=False,
                action_values=(
                    tuple(float(value) for value in action_values)
                    if action_values is not None
                    else None
                ),
            )
        return self.last_selection

    def reset_exploration_span(self) -> None:
        self._span_frames_remaining = 0
        self._span_action_index = None


def simple_opponent_schedule(rounds_per_batch: int) -> tuple[OpponentSpec, ...]:
    """Return the deterministic simple-opponent batch order."""

    if int(rounds_per_batch) <= 0:
        raise ValueError("rounds_per_batch must be positive")
    opponents: list[OpponentSpec] = []
    for _ in range(int(rounds_per_batch)):
        opponents.extend(
            OpponentSpec(ship=ship_name, mode=OPPONENT_MODE_SIMPLE)
            for ship_name in SHIP_TYPE_CATALOG_ORDER
        )
    return tuple(opponents)


def existing_ai_opponent_schedule(
    rounds_per_batch: int,
    discovered_opponents: Sequence[OpponentSpec],
    *,
    ai_opponent_chance: float = 100.0,
    rng: Any | None = None,
) -> tuple[OpponentSpec, ...]:
    """Return one batch schedule with one selected controller per ship type."""

    if int(rounds_per_batch) <= 0:
        raise ValueError("rounds_per_batch must be positive")
    ai_probability = _activity_probability(ai_opponent_chance)
    rng = rng or random
    by_ship: dict[str, list[OpponentSpec]] = {
        ship_name: []
        for ship_name in SHIP_TYPE_CATALOG_ORDER
    }
    for opponent in discovered_opponents:
        if opponent.ship not in by_ship:
            continue
        by_ship[opponent.ship].append(opponent)

    selected_by_ship = {
        ship_name: _choose_opponent_controller(
            ship_name,
            trained_options,
            ai_probability=ai_probability,
            rng=rng,
        )
        for ship_name, trained_options in by_ship.items()
    }
    opponents: list[OpponentSpec] = []
    for _ in range(int(rounds_per_batch)):
        opponents.extend(
            selected_by_ship[ship_name] for ship_name in SHIP_TYPE_CATALOG_ORDER
        )
    return tuple(opponents)


def discover_existing_ai_opponents(
    repository: TrainingModelRepository,
    *,
    device_choice: str | None = torch_backend.DEVICE_AUTO,
) -> OpponentDiscoveryResult:
    """Load and freeze available stored AI opponents at a batch boundary."""

    opponents: list[OpponentSpec] = []
    skipped: list[str] = []
    torch_available = torch_backend.get_torch() is not None
    for ship_name in SHIP_TYPE_CATALOG_ORDER:
        for slot_number in range(1, MODEL_SLOT_COUNT + 1):
            slot = repository.slot_for(ship_name, slot_number)
            if slot.source == SLOT_EMPTY:
                continue
            if not torch_available:
                skipped.append(f"{ship_name}-{slot_number:02d}: PyTorch unavailable")
                continue
            loaded = _load_opponent_model(slot, device_choice=device_choice)
            if isinstance(loaded, str):
                skipped.append(f"{ship_name}-{slot_number:02d}: {loaded}")
                continue
            opponents.append(
                OpponentSpec(
                    ship=ship_name,
                    mode=OPPONENT_MODE_EXISTING_AI,
                    slot=slot_number,
                    model=loaded,
                    description=slot.description,
                )
            )
    return OpponentDiscoveryResult(opponents=tuple(opponents), skipped=tuple(skipped))


def run_training_batch(
    *,
    model,
    optimizer,
    replay_buffer: TrainingReplayBuffer,
    config: TrainingOrchestrationConfig,
    rng: Any | None = None,
    model_repository: TrainingModelRepository | None = None,
    discovered_opponents: Sequence[OpponentSpec] | None = None,
    simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
    audio_service: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    battle_view_enabled: Callable[[], bool] | None = None,
) -> TrainingBatchResult:
    """Run one complete Phase 7 batch and perform end-of-batch replay updates."""

    rng = rng or random.Random()
    if config.opponent_mode == OPPONENT_MODE_SIMPLE:
        opponents = simple_opponent_schedule(config.rounds_per_batch)
    else:
        if discovered_opponents is None:
            if model_repository is None:
                raise ValueError("model_repository is required for existing-AI mode")
            discovered_opponents = discover_existing_ai_opponents(
                model_repository,
                device_choice=config.training_device,
            ).opponents
        opponents = existing_ai_opponent_schedule(
            config.rounds_per_batch,
            discovered_opponents,
            ai_opponent_chance=config.ai_opponent_chance,
            rng=rng,
        )

    trainee_policy = ValueNetworkPolicy(
        model,
        epsilon=config.epsilon,
        epsilon_frame_span=config.epsilon_frame_span,
        rng=rng,
    )
    round_results: list[TrainingRoundResult] = []
    total_rounds = len(opponents)
    for index, opponent in enumerate(opponents, start=1):
        _raise_if_stop_requested(stop_requested)
        _emit_progress(
            progress_callback,
            event="round_start",
            round_index=index,
            total_rounds=total_rounds,
            opponent=opponent,
        )
        result = run_training_round(
            opponent=opponent,
            trainee_policy=trainee_policy,
            replay_buffer=replay_buffer,
            config=config,
            rng=rng,
            simulation_factory=simulation_factory,
            audio_service=audio_service,
            progress_callback=progress_callback,
            stop_requested=stop_requested,
            battle_view_enabled=battle_view_enabled,
        )
        round_results.append(result)
        _raise_if_stop_requested(stop_requested)
        _emit_progress(
            progress_callback,
            event="round_end",
            round_index=index,
            total_rounds=total_rounds,
            opponent=opponent,
            result=result,
        )

    _emit_progress(
        progress_callback,
        event="batch_optimization_start",
        replay_updates=config.replay_updates_per_batch,
        replay_size=len(replay_buffer),
    )
    losses: list[float] = []
    # Optimization is the commit phase for a completed simulation batch. Once it
    # starts, finish every configured update so a stop cannot leave the batch
    # partially optimized and unrecorded.
    for _ in range(config.replay_updates_per_batch):
        result = optimize_from_replay(
            model,
            optimizer,
            replay_buffer,
            batch_size=config.minibatch_size,
            rng=rng,
        )
        if result is not None:
            losses.append(result.loss)

    return TrainingBatchResult(
        completed_rounds=len(round_results),
        replay_size=len(replay_buffer),
        optimization_losses=tuple(losses),
        round_results=tuple(round_results),
    )


def run_training_round(
    *,
    opponent: OpponentSpec,
    trainee_policy,
    replay_buffer: TrainingReplayBuffer,
    config: TrainingOrchestrationConfig,
    rng: Any | None = None,
    simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
    audio_service: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    battle_view_enabled: Callable[[], bool] | None = None,
) -> TrainingRoundResult:
    rng = rng or random.Random()
    audio = audio_service or NullAudioService()
    trainee = create_ship(config.trainee_ship, 1, audio_service=audio)
    opponent_ship = create_ship(opponent.ship, 2, audio_service=audio)
    ledger = event_ledger.BattleEventLedger()
    simulation = simulation_factory(
        None,
        trainee,
        opponent_ship,
        audio_service=audio,
        rng=rng,
        include_stars=False,
        training_event_ledger=ledger,
    )
    _initialize_training_simulation_ships(simulation, rng)
    if _battle_view_enabled(battle_view_enabled):
        _emit_progress(
            progress_callback,
            event="battle_view",
            opponent=opponent,
            battle_view=_battle_view_from_simulation(simulation),
        )

    pipeline = StagedTrajectoryPipeline(
        gamma=config.gamma,
        reward_weights=config.reward_weights,
        mode=config.reward_mode,
    )

    def abort_if_requested() -> None:
        if stop_requested is None or not stop_requested():
            return
        if pipeline.pending_count:
            replay_buffer.extend(
                pipeline.flush_pending(end_frame_id=simulation.frame_id)
            )
        raise TrainingBatchAborted("training batch stopped")

    return_sum = 0.0
    component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
    mature_count = 0
    terminal_reason = "timeout"
    terminal_seen = False
    episode_results: list[TrainingEpisodeResult] = []
    pending_episodes: list[PendingCombatEpisode] = []
    causal_lifecycle = config.reward_mode != REWARD_MODE_LEGACY
    episode_start_frame = 0
    episode_mature_count = 0
    episode_return_sum = 0.0
    episode_component_sums = {
        component: 0.0 for component in REWARD_COMPONENTS
    }
    next_display_frame_time = time.perf_counter()
    simple_opponent_controller = SimpleOpponentController(config, rng=rng)

    for _ in range(config.match_time_limit):
        abort_if_requested()
        self_ship = simulation.player1
        enemy_ship = simulation.player2
        observation = encode_observation(
            self_ship,
            enemy_ship,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
        selection = _select_policy_action(trainee_policy, observation)
        decision = decision_frame_from_battle_state(
            frame_id=simulation.frame_id + 1,
            observation=observation,
            action_index=selection.action_index,
            self_ship=self_ship,
            enemy_ship=enemy_ship,
            world=simulation.world,
        )
        ledger.begin_decision(
            self_ship,
            decision.frame_id,
            selection.action_index,
            reward_mode=config.reward_mode,
        )
        staged_index = pipeline.stage_decision(
            decision,
            trajectory_id=ledger.active_trajectory_id,
        )
        event_start = len(ledger.events)
        actions = {
            1: direct_controls_for_action_index(selection.action_index),
            2: _opponent_direct_controls(
                opponent,
                simulation,
                config,
                simple_opponent_controller,
            ),
        }
        state = simulation.step(actions=actions)
        terminal, terminal_reason = _round_terminal_state(
            simulation,
            elapsed_frames=state["frame_id"],
            frame_limit=config.match_time_limit,
        )
        training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
        can_continue = bool(training_deaths) and state["frame_id"] < config.match_time_limit
        reward_terminal = terminal and (
            not causal_lifecycle or 1 in training_deaths or not can_continue
        )
        events = tuple(ledger.events[event_start:])
        outcome = frame_outcome_from_battle_state(
            frame_id=state["frame_id"],
            self_ship=self_ship,
            events=events,
            terminal=reward_terminal,
        )
        mature_samples = pipeline.add_frame(
            decision,
            outcome,
            ledger=ledger,
            staged_index=staged_index,
        )
        replay_buffer.extend(mature_samples)
        mature_count += len(mature_samples)
        episode_mature_count += len(mature_samples)
        sample_return = sum(sample.return_value for sample in mature_samples)
        return_sum += sample_return
        episode_return_sum += sample_return
        _accumulate_weighted_components(
            component_sums,
            mature_samples,
            episode_component_sums,
        )
        normalized_return = _average_value(return_sum, mature_count)
        progress_payload = {
            "event": "frame",
            "frame": state["frame_id"],
            "opponent": opponent,
            "action_index": selection.action_index,
            "exploratory": selection.exploratory,
            "replay_size": len(replay_buffer),
            "weighted_total_return": normalized_return,
            "component_totals": _average_components(component_sums, mature_count),
        }
        view_enabled = _battle_view_enabled(battle_view_enabled)
        if view_enabled:
            progress_payload["battle_view"] = _battle_view_from_simulation(simulation)
        _emit_progress(progress_callback, **progress_payload)
        abort_if_requested()
        if config.display_on and view_enabled:
            next_display_frame_time += 1.0 / const.FPS
            sleep_seconds = next_display_frame_time - time.perf_counter()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            else:
                next_display_frame_time = time.perf_counter()
            abort_if_requested()
        if terminal:
            win, loss, draw = _classify_round_outcome(
                simulation, terminal_reason
            )
            kills, deaths = _classify_kills_deaths(simulation)
            if causal_lifecycle:
                pending_episodes.append(
                    PendingCombatEpisode(
                        opponent=opponent,
                        start_frame_id=episode_start_frame,
                        end_frame_id=state["frame_id"],
                        terminal_reason=terminal_reason,
                        win=win,
                        loss=loss,
                        draw=draw,
                        kills=kills,
                        deaths=deaths,
                    )
                )
                if reward_terminal:
                    episode_results.extend(
                        finalize_pending_episodes(pending_episodes, mature_samples)
                    )
                    pending_episodes.clear()
            else:
                episode_results.append(TrainingEpisodeResult(
                    opponent=opponent,
                    frames=state["frame_id"] - episode_start_frame,
                    terminal_reason=terminal_reason,
                    mature_samples=episode_mature_count,
                    total_return=_average_value(
                        episode_return_sum, episode_mature_count
                    ),
                    win=win,
                    loss=loss,
                    draw=draw,
                    kills=kills,
                    deaths=deaths,
                    component_totals=_average_components(
                        episode_component_sums, episode_mature_count
                    ),
                ))
            reset_span = getattr(trainee_policy, "reset_exploration_span", None)
            if callable(reset_span):
                reset_span()
            if not training_deaths:
                terminal_seen = True
                ledger.close_reward_trajectory()
                break
            if state["frame_id"] >= config.match_time_limit:
                terminal_seen = True
                ledger.close_reward_trajectory()
                break
            if reward_terminal:
                ledger.close_reward_trajectory()
                pipeline = StagedTrajectoryPipeline(
                    gamma=config.gamma,
                    reward_weights=config.reward_weights,
                    mode=config.reward_mode,
                )
            simple_opponent_controller = SimpleOpponentController(config, rng=rng)
            episode_start_frame = state["frame_id"]
            episode_mature_count = 0
            episode_return_sum = 0.0
            episode_component_sums = {
                component: 0.0 for component in REWARD_COMPONENTS
            }

    if not terminal_seen:
        terminal_reason = "timeout"
        mature_samples = _flush_timeout_frame(
            pipeline,
            simulation,
            replay_buffer,
        )
        mature_count += len(mature_samples)
        episode_mature_count += len(mature_samples)
        sample_return = sum(sample.return_value for sample in mature_samples)
        return_sum += sample_return
        episode_return_sum += sample_return
        _accumulate_weighted_components(
            component_sums,
            mature_samples,
            episode_component_sums,
        )
        win, loss, draw = _classify_round_outcome(simulation, terminal_reason)
        kills, deaths = _classify_kills_deaths(simulation)
        if causal_lifecycle:
            pending_episodes.append(
                PendingCombatEpisode(
                    opponent=opponent,
                    start_frame_id=episode_start_frame,
                    end_frame_id=simulation.frame_id,
                    terminal_reason=terminal_reason,
                    win=win,
                    loss=loss,
                    draw=draw,
                    kills=kills,
                    deaths=deaths,
                )
            )
            episode_results.extend(
                finalize_pending_episodes(pending_episodes, mature_samples)
            )
            pending_episodes.clear()
            ledger.close_reward_trajectory()
        else:
            episode_results.append(TrainingEpisodeResult(
                opponent=opponent,
                frames=simulation.frame_id - episode_start_frame,
                terminal_reason=terminal_reason,
                mature_samples=episode_mature_count,
                total_return=_average_value(
                    episode_return_sum, episode_mature_count
                ),
                win=win,
                loss=loss,
                draw=draw,
                kills=kills,
                deaths=deaths,
                component_totals=_average_components(
                    episode_component_sums, episode_mature_count
                ),
            ))

    win = any(episode.win for episode in episode_results)
    loss = any(episode.loss for episode in episode_results)
    draw = any(episode.draw for episode in episode_results)
    return TrainingRoundResult(
        opponent=opponent,
        frames=simulation.frame_id,
        terminal_reason=terminal_reason,
        mature_samples=mature_count,
        total_return=_average_value(return_sum, mature_count),
        win=win,
        loss=loss,
        draw=draw,
        component_totals=_average_components(component_sums, mature_count),
        episode_results=tuple(episode_results),
        reward_diagnostics=ledger.diagnostics,
    )


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
    action = action_for_index(int(action_index))
    return _controls_mapping_from_action(action)


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
            self.forward_held,
            self.config.forward_activity,
        )
        self.action1_held = self._next_key_state(
            self.action1_held,
            self.config.a1_activity,
        )
        self.action2_held = self._next_key_state(
            self.action2_held,
            self.config.a2_activity,
        )

        left = right = False
        if self._should_face_opponent(getattr(simulation, "frame_id", 0)):
            left, right = _turn_toward_target(
                simulation.player2,
                simulation.player1,
            )
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
        if (
            self.next_face_decision_frame is None
            or frame_id >= self.next_face_decision_frame
        ):
            self.face_opponent_active = self.rng.random() < probability
            self.next_face_decision_frame = frame_id + max(1, int(const.FPS))
        return self.face_opponent_active


def controls_for_simple_behavior(
    simulation,
    config: TrainingOrchestrationConfig,
    *,
    rng=None,
):
    return SimpleOpponentController(config, rng=rng).controls_for_frame(simulation)


def build_training_components(config: TrainingOrchestrationConfig):
    """Construct the model, optimizer, and replay buffer for a new session."""

    device = torch_backend.training_device(config.training_device)
    model = build_value_network(
        ValueNetworkConfig(
            hidden_layer_width=config.hidden_layer_width,
            hidden_layer_count=config.hidden_layer_count,
        ),
        device=device,
    )
    optimizer = build_optimizer(model, learning_rate=config.learning_rate)
    replay_buffer = TrainingReplayBuffer(config.replay_capacity)
    return model, optimizer, replay_buffer


def _select_policy_action(policy, observation: Sequence[float]) -> ActionSelection:
    selection = policy.select_action(observation)
    if isinstance(selection, ActionSelection):
        return selection
    return ActionSelection(action_index=int(selection), exploratory=False)


def _opponent_controls(
    opponent: OpponentSpec,
    simulation,
    config: TrainingOrchestrationConfig,
    simple_controller: SimpleOpponentController,
) -> dict[str, bool]:
    return _controls_mapping_from_action(
        _opponent_direct_controls(opponent, simulation, config, simple_controller)
    )


def _opponent_direct_controls(
    opponent: OpponentSpec,
    simulation,
    config: TrainingOrchestrationConfig,
    simple_controller: SimpleOpponentController,
) -> TrainingAction:
    if opponent.model is None:
        return simple_controller.direct_controls_for_frame(simulation)
    observation = encode_observation(
        simulation.player2,
        simulation.player1,
        frame_id=simulation.frame_id,
        game_objects=simulation.world,
    )
    selection = select_action_epsilon_greedy(
        opponent.model,
        observation,
        epsilon=0.0,
        value_predictor=predict_action_values_read_only,
    )
    return direct_controls_for_action_index(selection.action_index)


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


def _choose_from_sequence(values: Sequence[Any], *, rng: Any) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    chooser = getattr(rng, "choice", None)
    if callable(chooser):
        return chooser(values)
    index = int(float(rng.random()) * len(values))
    return values[min(index, len(values) - 1)]


def _choose_opponent_controller(
    ship_name: str,
    trained_options: Sequence[OpponentSpec],
    *,
    ai_probability: float,
    rng: Any,
) -> OpponentSpec:
    simple = OpponentSpec(ship=ship_name, mode=OPPONENT_MODE_SIMPLE)
    if not trained_options or ai_probability <= 0.0:
        return simple
    if ai_probability < 1.0 and float(rng.random()) >= ai_probability:
        return simple
    return _choose_from_sequence(trained_options, rng=rng)


def _round_terminal_state(simulation, *, elapsed_frames: int, frame_limit: int):
    if getattr(simulation, "training_episode_deaths", ()):
        return True, "resolved"
    aftermath = getattr(simulation, "aftermath", None)
    pending_rebirths = bool(getattr(aftermath, "pending_rebirths", None))
    if pending_rebirths:
        return False, "pending_rebirth"
    if elapsed_frames >= frame_limit:
        return True, "timeout"
    if aftermath is not None:
        return True, "resolved"
    if not _ship_alive(simulation.player1) or not _ship_alive(simulation.player2):
        return True, "resolved"
    return False, "running"


def _flush_timeout_frame(
    pipeline: StagedTrajectoryPipeline,
    simulation,
    replay_buffer: TrainingReplayBuffer,
) -> tuple[MatureTrainingSample, ...]:
    mature_samples = tuple(pipeline.flush_pending(end_frame_id=simulation.frame_id))
    replay_buffer.extend(mature_samples)
    return mature_samples


def _classify_round_outcome(simulation, terminal_reason: str) -> tuple[bool, bool, bool]:
    training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
    if training_deaths:
        if training_deaths == {2}:
            return True, False, False
        if training_deaths == {1}:
            return False, True, False
        return False, False, True
    trainee_alive = _ship_alive(simulation.player1)
    opponent_alive = _ship_alive(simulation.player2)
    if terminal_reason == "timeout":
        return False, False, True
    if opponent_alive is False and trainee_alive:
        return True, False, False
    if trainee_alive is False and opponent_alive:
        return False, True, False
    return False, False, True


def _classify_kills_deaths(simulation) -> tuple[int, int]:
    trainee_player = 1
    credited_killers = set(getattr(simulation, "training_episode_kills", ()))
    dead_players = set(getattr(simulation, "training_episode_deaths", ()))
    return (
        int(trainee_player in credited_killers),
        int(trainee_player in dead_players),
    )


def _ship_alive(ship) -> bool:
    return bool(getattr(ship, "currently_alive", True)) and getattr(ship, "current_hp", 1) > 0


def _fully_arm_training_shofixti(ship) -> None:
    if getattr(ship, "name", None) != "Shofixti":
        return
    armed = getattr(ship, "ARMED", 2)
    setattr(ship, "shofixti_arming_stage", armed)


def _initialize_training_simulation_ships(simulation, rng) -> None:
    if getattr(simulation, "training_spawn_initialized", False):
        return
    _fully_arm_training_shofixti(simulation.player1)
    _fully_arm_training_shofixti(simulation.player2)
    _randomize_training_start_hp(simulation.player1, rng)
    _randomize_training_start_hp(simulation.player2, rng)


def _randomize_training_start_hp(ship, rng) -> None:
    start_hp = max(1, int(getattr(ship, "start_hp", getattr(ship, "current_hp", 1))))
    ship.current_hp = max(1, math.ceil(float(rng.random()) * start_hp))


def _accumulate_weighted_components(
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


def _average_value(total: float, count: int) -> float:
    return float(total) / int(count) if int(count) > 0 else 0.0


def _average_components(
    totals: Mapping[str, float],
    count: int,
) -> dict[str, float]:
    if int(count) <= 0:
        return {component: 0.0 for component in REWARD_COMPONENTS}
    return {
        component: float(totals.get(component, 0.0)) / int(count)
        for component in REWARD_COMPONENTS
    }


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(payload)


def _raise_if_stop_requested(callback: Callable[[], bool] | None) -> None:
    if callback is not None and callback():
        raise TrainingBatchAborted("training stop requested")


def _battle_view_enabled(callback: Callable[[], bool] | None) -> bool:
    return True if callback is None else bool(callback())


def _battle_view_from_simulation(simulation) -> dict[str, Any]:
    return {
        "game_objects": tuple(simulation.world.snapshot()),
        "border_rect": simulation.border_rect.copy(),
        "border_color": tuple(simulation.border_color),
        "frame_id": int(simulation.frame_id),
        "original_ships": (simulation.player1, simulation.player2),
        "camera_targets": (simulation.player1, simulation.player2),
        "entry_state": None,
    }


def _load_opponent_model(
    slot: TrainingModelSlot,
    *,
    device_choice: str | None = torch_backend.DEVICE_AUTO,
):
    return load_opponent_model(slot, device_choice=device_choice)
