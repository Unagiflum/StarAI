"""Headless training orchestration for battle rounds and batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import math
import random
import time
from typing import Any

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.computer_control import guard_computer_controls
from src.Objects.Ships.registry import create_ship
from src.audio import NullAudioService
from src.training import event_ledger, torch_backend
from src.training.action_safety import guard_training_selection
from src.training.baseline import (
    BaselineOpponentController,
    BaselinePhaseMixer,
    baseline_decisions_per_phase,
)
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
    OpponentControllerPlan,
    OpponentControllerSegment,
    OpponentSpec,
    TrainingBatchAborted,
    TrainingOrchestrationConfig,
    single_controller_opponent_plan,
)
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    TrainingModelRepository,
    TrainingModelSlot,
    model_action_frame_span,
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
from src.training.replay_contracts import ActionHold
from src.training.rewards import (
    MatureTrainingSample,
    REWARD_COMPONENTS,
    StagedTrajectoryPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)
from src.training.contracts import DEFAULT_NEW_MODEL_VALUE_HEAD
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
    action_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainingBatchResult:
    completed_rounds: int
    replay_size: int
    optimization_losses: tuple[float, ...]
    round_results: tuple[TrainingRoundResult, ...]
    baseline_decisions: int = 0
    total_decisions: int = 0

    @property
    def baseline_share(self) -> float:
        if self.total_decisions <= 0:
            return 0.0
        return self.baseline_decisions / self.total_decisions

    @property
    def average_loss(self) -> float | None:
        if not self.optimization_losses:
            return None
        return sum(self.optimization_losses) / len(self.optimization_losses)


class ValueNetworkPolicy:
    """Epsilon-greedy policy that holds each decision for a frame span.

    Every decision, exploratory or greedy, is repeated for
    ``action_frame_span`` frames. Repeated frames are marked ``held`` so only
    the deciding frame becomes a replay sample. With a ``baseline_mixer``,
    phases of decisions are flown by the ship's baseline strategy instead of
    the network; exploration applies to both.
    """

    def __init__(
        self,
        model,
        *,
        epsilon: float,
        action_frame_span: int = 1,
        rng: Any | None = None,
        baseline_mixer: BaselinePhaseMixer | None = None,
    ):
        self.model = model
        self.epsilon = float(epsilon)
        self.action_frame_span = max(1, int(action_frame_span))
        self.rng = rng or random
        self.baseline_mixer = baseline_mixer
        self.last_selection: ActionSelection | None = None
        self._hold = ActionHold()
        self._held_exploratory = False
        self._held_guided = False

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
        """Return a held, exploratory or baseline action, or ``None`` when
        network inference is due."""

        held_action = self._hold.held_action()
        if held_action is not None:
            self.last_selection = ActionSelection(
                action_index=held_action,
                exploratory=self._held_exploratory,
                held=True,
                guided=self._held_guided,
            )
            return self.last_selection

        baseline_phase = (
            self.baseline_mixer.begin_decision()
            if self.baseline_mixer is not None
            else False
        )
        if self.epsilon >= 1.0 or (
            self.epsilon > 0.0 and self.rng.random() < self.epsilon
        ):
            action_index = int(self.rng.randrange(ACTION_OUTPUT_SIZE))
            self._begin_hold(action_index, exploratory=True)
            self.last_selection = ActionSelection(
                action_index=action_index,
                exploratory=True,
            )
            return self.last_selection

        if baseline_phase:
            action_index = int(self.baseline_mixer.baseline_action(observation))
            self._begin_hold(action_index, exploratory=False, guided=True)
            self.last_selection = ActionSelection(
                action_index=action_index,
                exploratory=False,
                guided=True,
            )
            return self.last_selection

        return None

    def complete_greedy_selection(
        self,
        selection: ActionSelection | int,
        action_values: Sequence[float] | None = None,
    ) -> ActionSelection:
        if isinstance(selection, ActionSelection):
            completed = ActionSelection(
                action_index=int(selection.action_index),
                exploratory=False,
                action_values=selection.action_values,
            )
        else:
            completed = ActionSelection(
                action_index=int(selection),
                exploratory=False,
                action_values=(
                    tuple(float(value) for value in action_values)
                    if action_values is not None
                    else None
                ),
            )
        self._begin_hold(completed.action_index, exploratory=False)
        self.last_selection = completed
        return completed

    @property
    def baseline_counts(self) -> tuple[int, int]:
        """Return ``(baseline decisions, all decisions)`` made so far."""

        mixer = self.baseline_mixer
        if mixer is None:
            return 0, 0
        return int(mixer.baseline_decisions), int(mixer.decisions)

    def reset_exploration_span(self) -> None:
        """Drop any held action so the next frame makes a fresh decision.

        Called at episode boundaries, so a new baseline phase roll follows.
        """

        self._hold.reset()
        self._held_exploratory = False
        self._held_guided = False
        if self.baseline_mixer is not None:
            self.baseline_mixer.reset()

    reset_action_span = reset_exploration_span

    def _begin_hold(
        self,
        action_index: int,
        *,
        exploratory: bool,
        guided: bool = False,
    ) -> None:
        self._hold.begin(action_index, self.action_frame_span)
        self._held_exploratory = bool(exploratory)
        self._held_guided = bool(guided)


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


def build_deterministic_opponent_plan(
    ship_name: str,
    frame_limit: int,
    trained_options: Sequence[OpponentSpec],
    *,
    ai_opponent_chance: float = 100.0,
    batch_number: int = 0,
) -> OpponentControllerPlan:
    """Build one predictable, rotating controller plan for a ship match.

    ``batch_number`` is the zero-based count of successfully completed batches
    before this batch. Frame allocations are calculated before rotation so each
    controller retains the same number of frames in every ordering.
    """

    frame_limit = int(frame_limit)
    if frame_limit <= 0:
        raise ValueError("frame_limit must be positive")
    ai_percent = max(
        Decimal("0"),
        min(Decimal("100"), Decimal(str(ai_opponent_chance))),
    )
    trained_by_slot: dict[int, OpponentSpec] = {}
    for opponent in trained_options:
        if opponent.ship != ship_name or opponent.slot is None:
            continue
        trained_by_slot.setdefault(int(opponent.slot), opponent)
    trained = tuple(trained_by_slot[slot] for slot in sorted(trained_by_slot))

    simple = OpponentSpec(ship=ship_name, mode=OPPONENT_MODE_SIMPLE)
    if not trained or ai_percent <= 0:
        return single_controller_opponent_plan(simple, frame_limit)

    simple_frames = _round_half_up(
        Decimal(frame_limit) * (Decimal("100") - ai_percent) / Decimal("100")
    )
    ai_frames = frame_limit - simple_frames
    if ai_frames <= 0:
        return single_controller_opponent_plan(simple, frame_limit)

    controllers_with_lengths: list[tuple[OpponentSpec, int]] = []
    if simple_frames > 0:
        controllers_with_lengths.append((simple, simple_frames))
    previous_boundary = 0
    for index, opponent in enumerate(trained, start=1):
        boundary = _round_half_up(
            Decimal(ai_frames) * Decimal(index) / Decimal(len(trained))
        )
        controller_frames = boundary - previous_boundary
        previous_boundary = boundary
        if controller_frames > 0:
            controllers_with_lengths.append((opponent, controller_frames))

    if not controllers_with_lengths:
        return single_controller_opponent_plan(simple, frame_limit)
    rotation = max(0, int(batch_number)) % len(controllers_with_lengths)
    ordered = (
        controllers_with_lengths[rotation:]
        + controllers_with_lengths[:rotation]
    )
    segments = []
    start_frame = 0
    for opponent, controller_frames in ordered:
        end_frame = start_frame + int(controller_frames)
        segments.append(
            OpponentControllerSegment(
                opponent=opponent,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        start_frame = end_frame
    return OpponentControllerPlan(
        ship=ship_name,
        frame_limit=frame_limit,
        segments=tuple(segments),
    )


def deterministic_opponent_plan_schedule(
    rounds_per_batch: int,
    frame_limit: int,
    discovered_opponents: Sequence[OpponentSpec],
    *,
    ai_opponent_chance: float = 100.0,
    batch_number: int = 0,
) -> tuple[OpponentControllerPlan, ...]:
    """Return the deterministic match plans for one training batch."""

    if int(rounds_per_batch) <= 0:
        raise ValueError("rounds_per_batch must be positive")
    by_ship: dict[str, list[OpponentSpec]] = {
        ship_name: []
        for ship_name in SHIP_TYPE_CATALOG_ORDER
    }
    for opponent in discovered_opponents:
        if opponent.ship in by_ship:
            by_ship[opponent.ship].append(opponent)
    plans_by_ship = {
        ship_name: build_deterministic_opponent_plan(
            ship_name,
            frame_limit,
            by_ship[ship_name],
            ai_opponent_chance=ai_opponent_chance,
            batch_number=batch_number,
        )
        for ship_name in SHIP_TYPE_CATALOG_ORDER
    }
    return tuple(
        plans_by_ship[ship_name]
        for _ in range(int(rounds_per_batch))
        for ship_name in SHIP_TYPE_CATALOG_ORDER
    )


def _round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


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
                    action_frame_span=model_action_frame_span(slot.metadata),
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
    resources: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    battle_view_enabled: Callable[[], bool] | None = None,
    opponent_batch_number: int = 0,
) -> TrainingBatchResult:
    """Run one complete Phase 7 batch and perform end-of-batch replay updates."""

    rng = rng or random.Random()
    if config.opponent_mode == OPPONENT_MODE_SIMPLE:
        discovered_opponents = ()
        ai_opponent_chance = 0.0
    else:
        if discovered_opponents is None:
            if model_repository is None:
                raise ValueError("model_repository is required for existing-AI mode")
            discovered_opponents = discover_existing_ai_opponents(
                model_repository,
                device_choice=config.training_device,
            ).opponents
        ai_opponent_chance = config.ai_opponent_chance
    opponent_plans = deterministic_opponent_plan_schedule(
        config.rounds_per_batch,
        config.match_time_limit,
        discovered_opponents,
        ai_opponent_chance=ai_opponent_chance,
        batch_number=opponent_batch_number,
    )

    trainee_policy = ValueNetworkPolicy(
        model,
        epsilon=config.epsilon,
        action_frame_span=config.action_frame_span,
        rng=rng,
        baseline_mixer=baseline_mixer_for_config(config, rng=rng),
    )
    round_results: list[TrainingRoundResult] = []
    total_rounds = len(opponent_plans)
    for index, opponent_plan in enumerate(opponent_plans, start=1):
        opponent = opponent_plan.initial_opponent
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
            opponent_plan=opponent_plan,
            trainee_policy=trainee_policy,
            replay_buffer=replay_buffer,
            config=config,
            rng=rng,
            simulation_factory=simulation_factory,
            audio_service=audio_service,
            resources=resources,
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

    baseline_decisions, total_decisions = trainee_policy.baseline_counts
    return TrainingBatchResult(
        completed_rounds=len(round_results),
        replay_size=len(replay_buffer),
        optimization_losses=tuple(losses),
        round_results=tuple(round_results),
        baseline_decisions=baseline_decisions,
        total_decisions=total_decisions,
    )


def baseline_mixer_for_config(
    config: TrainingOrchestrationConfig,
    *,
    rng: Any | None = None,
) -> BaselinePhaseMixer | None:
    """Return the trainee's baseline phase mixer, or ``None`` when unused."""

    if float(config.baseline_use) <= 0.0:
        return None
    return BaselinePhaseMixer(
        config.trainee_ship,
        baseline_use=config.baseline_use,
        decisions_per_phase=baseline_decisions_per_phase(config.frame_span_multiplier),
        action_frame_span=config.action_frame_span,
        rng=rng,
    )


def run_training_round(
    *,
    opponent: OpponentSpec,
    opponent_plan: OpponentControllerPlan | None = None,
    trainee_policy,
    replay_buffer: TrainingReplayBuffer,
    config: TrainingOrchestrationConfig,
    rng: Any | None = None,
    simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
    audio_service: Any | None = None,
    resources: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    battle_view_enabled: Callable[[], bool] | None = None,
) -> TrainingRoundResult:
    rng = rng or random.Random()
    audio = audio_service or NullAudioService()
    ship_kwargs = {"audio_service": audio}
    if resources is not None:
        ship_kwargs["resources"] = resources
    opponent_plan = opponent_plan or single_controller_opponent_plan(
        opponent,
        config.match_time_limit,
    )
    if int(opponent_plan.frame_limit) != int(config.match_time_limit):
        raise ValueError("opponent plan frame_limit must match match_time_limit")
    if opponent_plan.ship != opponent.ship:
        raise ValueError("opponent plan ship must match the round opponent")

    trainee = create_ship(config.trainee_ship, 1, **ship_kwargs)
    opponent_ship = create_ship(opponent.ship, 2, **ship_kwargs)
    ledger = event_ledger.BattleEventLedger()
    simulation_kwargs = {
        "audio_service": audio,
        "rng": rng,
        "include_stars": False,
        "training_event_ledger": ledger,
    }
    if resources is not None:
        simulation_kwargs["resources"] = resources
    simulation = simulation_factory(None, trainee, opponent_ship, **simulation_kwargs)
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
    post_death_episode: PendingCombatEpisode | None = None
    next_display_frame_time = time.perf_counter()
    baseline_opponent_controller = BaselineOpponentController(
        action_frame_span=config.action_frame_span
    )
    opponent_hold = ActionHold()
    action_counts = [0] * ACTION_OUTPUT_SIZE

    for match_frame in range(config.match_time_limit):
        abort_if_requested()
        active_opponent = opponent_plan.opponent_for_frame(match_frame)
        self_ship = simulation.player1
        enemy_ship = simulation.player2
        observation = encode_observation(
            self_ship,
            enemy_ship,
            frame_id=simulation.frame_id,
            game_objects=simulation.world,
        )
        selection = guard_training_selection(
            _select_policy_action(trainee_policy, observation),
            self_ship,
            enemy_ship,
        )
        if not selection.held:
            action_counts[selection.action_index] += 1
        decision = decision_frame_from_battle_state(
            frame_id=simulation.frame_id + 1,
            observation=observation,
            action_index=selection.action_index,
            self_ship=self_ship,
            enemy_ship=enemy_ship,
            world=simulation.world,
            enabled_components=pipeline.enabled_reward_components,
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
            emit_sample=not selection.held,
        )
        event_start = len(ledger.events)
        actions = {
            1: direct_controls_for_action_index(selection.action_index),
            2: _opponent_direct_controls(
                active_opponent,
                simulation,
                config,
                baseline_opponent_controller,
                opponent_hold=opponent_hold,
            ),
        }
        state = simulation.step(actions=actions)
        terminal, terminal_reason = _round_terminal_state(
            simulation,
            elapsed_frames=state["frame_id"],
            frame_limit=config.match_time_limit,
        )
        training_deaths = set(getattr(simulation, "training_episode_deaths", ()))
        if (
            terminal
            and training_deaths == {2}
            and bool(getattr(simulation, "training_opponent_absent", False))
        ):
            win, loss, draw = _classify_round_outcome(
                simulation, terminal_reason
            )
            kills, deaths = _classify_kills_deaths(simulation)
            post_death_episode = PendingCombatEpisode(
                opponent=active_opponent,
                start_frame_id=episode_start_frame,
                end_frame_id=state["frame_id"],
                terminal_reason=terminal_reason,
                win=win,
                loss=loss,
                draw=draw,
                kills=kills,
                deaths=deaths,
            )
            terminal = False
            terminal_reason = "post_death"
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
            enabled_components=pipeline.enabled_reward_components,
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
            "opponent": active_opponent,
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
            handled_post_death = post_death_episode is not None
            if handled_post_death:
                current_deaths = set(
                    getattr(simulation, "training_episode_deaths", ())
                )
                pending = post_death_episode
                boundary = PendingCombatEpisode(
                    opponent=pending.opponent,
                    start_frame_id=pending.start_frame_id,
                    end_frame_id=state["frame_id"],
                    terminal_reason=pending.terminal_reason,
                    win=pending.win,
                    loss=pending.loss,
                    draw=pending.draw,
                    kills=pending.kills,
                    deaths=max(pending.deaths, int(1 in current_deaths)),
                )
                win, loss, draw = boundary.win, boundary.loss, boundary.draw
                kills, deaths = boundary.kills, boundary.deaths
            else:
                win, loss, draw = _classify_round_outcome(
                    simulation, terminal_reason
                )
                kills, deaths = _classify_kills_deaths(simulation)
                boundary = PendingCombatEpisode(
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
            if causal_lifecycle:
                pending_episodes.append(
                    boundary
                )
                if reward_terminal:
                    episode_results.extend(
                        finalize_pending_episodes(pending_episodes, mature_samples)
                    )
                    pending_episodes.clear()
            else:
                episode_results.append(TrainingEpisodeResult(
                    opponent=boundary.opponent,
                    frames=state["frame_id"] - episode_start_frame,
                    terminal_reason=boundary.terminal_reason,
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
            post_death_episode = None
            if handled_post_death:
                pipeline = StagedTrajectoryPipeline(
                    gamma=config.gamma,
                    reward_weights=config.reward_weights,
                    mode=config.reward_mode,
                )
                baseline_opponent_controller.reset()
                opponent_hold.reset()
                episode_start_frame = state["frame_id"]
                episode_mature_count = 0
                episode_return_sum = 0.0
                episode_component_sums = {
                    component: 0.0 for component in REWARD_COMPONENTS
                }
                continue
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
            baseline_opponent_controller.reset()
            opponent_hold.reset()
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
        if post_death_episode is not None:
            pending = post_death_episode
            boundary = PendingCombatEpisode(
                opponent=pending.opponent,
                start_frame_id=pending.start_frame_id,
                end_frame_id=simulation.frame_id,
                terminal_reason=pending.terminal_reason,
                win=pending.win,
                loss=pending.loss,
                draw=pending.draw,
                kills=pending.kills,
                deaths=pending.deaths,
            )
            win, loss, draw = boundary.win, boundary.loss, boundary.draw
            kills, deaths = boundary.kills, boundary.deaths
        else:
            win, loss, draw = _classify_round_outcome(
                simulation, terminal_reason
            )
            kills, deaths = _classify_kills_deaths(simulation)
            boundary = PendingCombatEpisode(
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
        if causal_lifecycle:
            pending_episodes.append(boundary)
            episode_results.extend(
                finalize_pending_episodes(pending_episodes, mature_samples)
            )
            pending_episodes.clear()
            ledger.close_reward_trajectory()
        else:
            episode_results.append(TrainingEpisodeResult(
                opponent=boundary.opponent,
                frames=simulation.frame_id - episode_start_frame,
                terminal_reason=boundary.terminal_reason,
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
        action_counts=tuple(action_counts),
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


def build_training_components(config: TrainingOrchestrationConfig):
    """Construct the model, optimizer, and replay buffer for a new session."""

    device = torch_backend.training_device(config.training_device)
    model = build_value_network(
        ValueNetworkConfig(
            hidden_layer_width=config.hidden_layer_width,
            hidden_layer_count=config.hidden_layer_count,
            head=DEFAULT_NEW_MODEL_VALUE_HEAD,
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
    baseline_controller: BaselineOpponentController,
) -> dict[str, bool]:
    return _controls_mapping_from_action(
        _opponent_direct_controls(opponent, simulation, config, baseline_controller)
    )


def _opponent_direct_controls(
    opponent: OpponentSpec,
    simulation,
    config: TrainingOrchestrationConfig,
    baseline_controller: BaselineOpponentController,
    *,
    opponent_hold: ActionHold | None = None,
) -> TrainingAction:
    if opponent.model is None:
        controls = baseline_controller.direct_controls_for_frame(simulation)
    else:
        hold_key = id(opponent.model)
        held_action = (
            opponent_hold.held_action(hold_key)
            if opponent_hold is not None
            else None
        )
        if held_action is None:
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
            held_action = selection.action_index
            if opponent_hold is not None:
                opponent_hold.begin(
                    held_action,
                    opponent.action_frame_span,
                    key=hold_key,
                )
        controls = direct_controls_for_action_index(held_action)
    return guard_computer_controls(
        controls,
        simulation.player2,
        simulation.player1,
    )


def _round_terminal_state(simulation, *, elapsed_frames: int, frame_limit: int):
    if getattr(simulation, "training_episode_deaths", ()):
        return True, "resolved"
    if bool(getattr(simulation, "training_post_death_completed", False)):
        return True, "post_death_complete"
    if bool(getattr(simulation, "training_opponent_absent", False)):
        return False, "post_death"
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
