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
from src.persistence import EXPECTED_READ_ERRORS, read_json
from src.toroidal import wrapped_delta
from src.training import event_ledger, torch_backend
from src.training.contracts import SHIP_TYPE_CATALOG_ORDER, action_for_index
from src.training.model_registry import (
    MODEL_SLOT_COUNT,
    SLOT_EMPTY,
    TrainingModelRepository,
    TrainingModelSlot,
    normalize_architecture_metadata,
)
from src.training.observation import encode_observation
from src.training.replay import (
    ActionSelection,
    TrainingReplayBuffer,
    load_training_checkpoint,
    optimize_from_replay,
    select_action_epsilon_greedy,
)
from src.training.rewards import (
    MatureTrainingSample,
    REWARD_COMPONENTS,
    RollingReturnPipeline,
    decision_frame_from_battle_state,
    frame_outcome_from_battle_state,
)
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
)


MOVEMENT_FORWARD = "Move forward continuously"
MOVEMENT_HOLD_A1 = "Hold A1 continuously"
MOVEMENT_HOLD_A2 = "Hold A2 continuously"

TURN_NONE = "none"
TURN_FACE_TRAINEE = "face_trainee"
TURN_FACE_AWAY = "face_away"
TURN_RIGHT = "turn_right"
TURN_LEFT = "turn_left"

OPPONENT_MODE_SIMPLE = "simple"
OPPONENT_MODE_EXISTING_AI = "all"

DEFAULT_MINIBATCH_SIZE = 32
DEFAULT_REPLAY_UPDATES_PER_BATCH = 1


@dataclass(frozen=True)
class TrainingOrchestrationConfig:
    trainee_ship: str
    reward_weights: Mapping[str, float] = field(default_factory=dict)
    opponent_mode: str = OPPONENT_MODE_SIMPLE
    movement_behaviors: frozenset[str] = field(default_factory=frozenset)
    turning_behavior: str = TURN_NONE
    rounds_per_batch: int = 1
    gamma: float = 0.99
    match_time_limit: int = 2400
    replay_capacity: int = 10000
    learning_rate: float = 0.001
    epsilon: float = 0.1
    hidden_layer_width: int = 128
    hidden_layer_count: int = 2
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE
    replay_updates_per_batch: int = DEFAULT_REPLAY_UPDATES_PER_BATCH
    display_on: bool = False

    def __post_init__(self) -> None:
        if self.opponent_mode not in {OPPONENT_MODE_SIMPLE, OPPONENT_MODE_EXISTING_AI}:
            raise ValueError("unsupported opponent mode")
        if self.rounds_per_batch <= 0:
            raise ValueError("rounds_per_batch must be positive")
        if not 0.0 <= float(self.gamma) < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if self.match_time_limit <= 0:
            raise ValueError("match_time_limit must be positive")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.replay_updates_per_batch < 0:
            raise ValueError("replay_updates_per_batch cannot be negative")


@dataclass(frozen=True)
class OpponentSpec:
    ship: str
    mode: str = OPPONENT_MODE_SIMPLE
    slot: int | None = None
    model: Any | None = None
    description: str = ""


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


class TrainingBatchAborted(RuntimeError):
    """Raised when a requested stop abandons the active training batch."""


class ValueNetworkPolicy:
    """Epsilon-greedy policy over the Phase 1 value network."""

    def __init__(self, model, *, epsilon: float, rng: Any | None = None):
        self.model = model
        self.epsilon = float(epsilon)
        self.rng = rng or random
        self.last_selection: ActionSelection | None = None

    def select_action(self, observation: Sequence[float]) -> ActionSelection:
        self.last_selection = select_action_epsilon_greedy(
            self.model,
            observation,
            epsilon=self.epsilon,
            rng=self.rng,
        )
        return self.last_selection


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


def discover_existing_ai_opponents(
    repository: TrainingModelRepository,
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
            loaded = _load_opponent_model(slot)
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
    simulation_factory: Callable[..., BattleSimulation] = BattleSimulation,
    audio_service: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> TrainingBatchResult:
    """Run one complete Phase 7 batch and perform end-of-batch replay updates."""

    rng = rng or random.Random()
    if config.opponent_mode == OPPONENT_MODE_SIMPLE:
        opponents = simple_opponent_schedule(config.rounds_per_batch)
    else:
        if model_repository is None:
            raise ValueError("model_repository is required for existing-AI mode")
        discovered = discover_existing_ai_opponents(model_repository)
        opponents = tuple(
            opponent
            for _ in range(config.rounds_per_batch)
            for opponent in discovered.opponents
        )

    trainee_policy = ValueNetworkPolicy(model, epsilon=config.epsilon, rng=rng)
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

    losses: list[float] = []
    for _ in range(config.replay_updates_per_batch):
        _raise_if_stop_requested(stop_requested)
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
    _fully_arm_training_shofixti(simulation.player1)
    _fully_arm_training_shofixti(simulation.player2)
    _emit_progress(
        progress_callback,
        event="battle_view",
        opponent=opponent,
        battle_view=_battle_view_from_simulation(simulation),
    )

    pipeline = RollingReturnPipeline(
        gamma=config.gamma,
        reward_weights=config.reward_weights,
    )
    return_sum = 0.0
    component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
    mature_count = 0
    terminal_reason = "timeout"
    terminal_seen = False
    next_display_frame_time = time.perf_counter()

    for _ in range(config.match_time_limit):
        _raise_if_stop_requested(stop_requested)
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
        event_start = len(ledger.events)
        actions = {
            1: controls_for_action_index(selection.action_index),
            2: _opponent_controls(opponent, simulation, config),
        }
        state = simulation.step(actions=actions)
        terminal, terminal_reason = _round_terminal_state(
            simulation,
            elapsed_frames=state["frame_id"],
            frame_limit=config.match_time_limit,
        )
        events = tuple(ledger.events[event_start:])
        outcome = frame_outcome_from_battle_state(
            frame_id=state["frame_id"],
            self_ship=self_ship,
            events=events,
            terminal=terminal,
        )
        mature_samples = pipeline.add_frame(decision, outcome)
        replay_buffer.extend(mature_samples)
        mature_count += len(mature_samples)
        return_sum += sum(sample.return_value for sample in mature_samples)
        _accumulate_weighted_components(component_sums, mature_samples)
        normalized_return = _average_value(return_sum, mature_count)
        _emit_progress(
            progress_callback,
            event="frame",
            frame=state["frame_id"],
            opponent=opponent,
            action_index=selection.action_index,
            exploratory=selection.exploratory,
            replay_size=len(replay_buffer),
            weighted_total_return=normalized_return,
            component_totals=_average_components(component_sums, mature_count),
            battle_view=_battle_view_from_simulation(simulation),
        )
        _raise_if_stop_requested(stop_requested)
        if config.display_on:
            next_display_frame_time += 1.0 / const.FPS
            sleep_seconds = next_display_frame_time - time.perf_counter()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            else:
                next_display_frame_time = time.perf_counter()
            _raise_if_stop_requested(stop_requested)
        if terminal:
            terminal_seen = True
            break

    if not terminal_seen:
        terminal_reason = "timeout"
        mature_samples = _flush_timeout_frame(
            pipeline,
            simulation,
            trainee_policy,
            replay_buffer,
        )
        mature_count += len(mature_samples)
        return_sum += sum(sample.return_value for sample in mature_samples)
        _accumulate_weighted_components(component_sums, mature_samples)

    win, loss, draw = _classify_round_outcome(simulation, terminal_reason)
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
    )


def controls_for_action_index(action_index: int) -> dict[str, bool]:
    action = action_for_index(int(action_index))
    return {
        "forward": action.thrust,
        "left": action.turn_left,
        "right": action.turn_right,
        "action1": action.a1,
        "action2": action.a2,
    }


def controls_for_simple_behavior(simulation, config: TrainingOrchestrationConfig):
    controls = {
        "forward": MOVEMENT_FORWARD in config.movement_behaviors,
        "action1": MOVEMENT_HOLD_A1 in config.movement_behaviors,
        "action2": MOVEMENT_HOLD_A2 in config.movement_behaviors,
        "left": False,
        "right": False,
    }
    turning = config.turning_behavior
    if turning == TURN_LEFT:
        controls["left"] = True
    elif turning == TURN_RIGHT:
        controls["right"] = True
    elif turning in {TURN_FACE_TRAINEE, TURN_FACE_AWAY}:
        left, right = _turn_toward_target(
            simulation.player2,
            simulation.player1,
            face_away=turning == TURN_FACE_AWAY,
        )
        controls["left"] = left
        controls["right"] = right
    return controls


def build_training_components(config: TrainingOrchestrationConfig):
    """Construct the model, optimizer, and replay buffer for a new session."""

    device = torch_backend.preferred_device()
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
) -> dict[str, bool]:
    if opponent.model is None:
        return controls_for_simple_behavior(simulation, config)
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
    )
    return controls_for_action_index(selection.action_index)


def _turn_toward_target(ship, target, *, face_away: bool) -> tuple[bool, bool]:
    dx, dy = wrapped_delta(ship.position, target.position)
    target_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
    if face_away:
        target_angle = (target_angle + 180.0) % 360.0
    rotation = float(getattr(ship, "rotation", 0.0)) % 360.0
    diff = (target_angle - rotation + 540.0) % 360.0 - 180.0
    if abs(diff) <= max(1.0, const.TURN_ANGLE / 2.0):
        return False, False
    return diff < 0.0, diff > 0.0


def _round_terminal_state(simulation, *, elapsed_frames: int, frame_limit: int):
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
    pipeline: RollingReturnPipeline,
    simulation,
    trainee_policy,
    replay_buffer: TrainingReplayBuffer,
) -> tuple[MatureTrainingSample, ...]:
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
        frame_id=simulation.frame_id,
        observation=observation,
        action_index=selection.action_index,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        world=simulation.world,
    )
    outcome = frame_outcome_from_battle_state(
        frame_id=simulation.frame_id,
        self_ship=self_ship,
        events=(),
        terminal=True,
    )
    mature_samples = tuple(pipeline.add_frame(decision, outcome))
    replay_buffer.extend(mature_samples)
    return mature_samples


def _classify_round_outcome(simulation, terminal_reason: str) -> tuple[bool, bool, bool]:
    trainee_alive = _ship_alive(simulation.player1)
    opponent_alive = _ship_alive(simulation.player2)
    if terminal_reason == "timeout":
        return False, False, True
    if opponent_alive is False and trainee_alive:
        return True, False, False
    if trainee_alive is False and opponent_alive:
        return False, True, False
    return False, False, True


def _ship_alive(ship) -> bool:
    return bool(getattr(ship, "currently_alive", True)) and getattr(ship, "current_hp", 1) > 0


def _fully_arm_training_shofixti(ship) -> None:
    if getattr(ship, "name", None) != "Shofixti":
        return
    armed = getattr(ship, "ARMED", 2)
    setattr(ship, "shofixti_arming_stage", armed)


def _accumulate_weighted_components(
    totals: dict[str, float],
    samples: Sequence[MatureTrainingSample],
) -> None:
    for sample in samples:
        for component, value in sample.weighted_components.items():
            totals[component] = totals.get(component, 0.0) + float(value)


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


def _load_opponent_model(slot: TrainingModelSlot):
    if slot.pth_path is None or not slot.pth_path.exists():
        return "missing weights"
    if slot.pth_path.stat().st_size <= 0:
        return "empty weights"
    metadata = _metadata_for_slot(slot)
    architecture = normalize_architecture_metadata(metadata.get("architecture", {}))
    try:
        config = ValueNetworkConfig(
            hidden_layer_width=int(architecture["hidden_layer_width"]),
            hidden_layer_count=int(architecture["hidden_layer_count"]),
        )
        device = torch_backend.preferred_device()
        model = build_value_network(config, device=device)
        load_training_checkpoint(slot.pth_path, model, map_location=device)
        model.eval()
        return model
    except Exception as exc:
        return str(exc)


def _metadata_for_slot(slot: TrainingModelSlot) -> Mapping[str, Any]:
    if isinstance(slot.metadata, Mapping):
        return slot.metadata
    if slot.metadata_path is None or not slot.metadata_path.exists():
        return {}
    try:
        metadata = read_json(slot.metadata_path)
    except EXPECTED_READ_ERRORS:
        return {}
    return metadata if isinstance(metadata, Mapping) else {}
