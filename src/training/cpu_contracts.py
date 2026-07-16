"""CPU-safe training configuration and opponent contracts.

This module is intentionally limited to standard-library and lightweight StarAI
dependencies so spawned simulation workers can deserialize training commands
without importing the model or replay stacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.training.causal_credit import REWARD_MODE_CAUSAL, REWARD_MODES


OPPONENT_MODE_SIMPLE = "simple"
OPPONENT_MODE_EXISTING_AI = "all"

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"
DEVICE_CHOICES = (DEVICE_AUTO, DEVICE_CPU, DEVICE_GPU)

DEFAULT_MINIBATCH_SIZE = 32
DEFAULT_REPLAY_UPDATES_PER_BATCH = 1


@dataclass(frozen=True)
class TrainingOrchestrationConfig:
    trainee_ship: str
    reward_weights: Mapping[str, float] = field(default_factory=dict)
    opponent_mode: str = OPPONENT_MODE_SIMPLE
    ai_opponent_chance: float = 100.0
    forward_activity: float = 0.0
    a1_activity: float = 0.0
    a2_activity: float = 0.0
    face_opponent_activity: float = 0.0
    rounds_per_batch: int = 1
    gamma: float = 0.99
    match_time_limit: int = 2400
    replay_capacity: int = 10000
    learning_rate: float = 0.001
    starting_epsilon: float = 0.1
    epsilon: float = 0.1
    epsilon_floor: float = 0.05
    epsilon_decay: float = 0.998
    epsilon_frame_span: int = 1
    hidden_layer_width: int = 128
    hidden_layer_count: int = 2
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE
    replay_updates_per_batch: int = DEFAULT_REPLAY_UPDATES_PER_BATCH
    training_device: str = DEVICE_AUTO
    display_on: bool = False
    reward_mode: str = REWARD_MODE_CAUSAL

    def __post_init__(self) -> None:
        if self.opponent_mode not in {OPPONENT_MODE_SIMPLE, OPPONENT_MODE_EXISTING_AI}:
            raise ValueError("unsupported opponent mode")
        if not 0.0 <= float(self.ai_opponent_chance) <= 100.0:
            raise ValueError("ai_opponent_chance must be in [0, 100]")
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
        for label, value in (
            ("starting_epsilon", self.starting_epsilon),
            ("epsilon", self.epsilon),
            ("epsilon_floor", self.epsilon_floor),
            ("epsilon_decay", self.epsilon_decay),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if int(self.epsilon_frame_span) <= 0:
            raise ValueError("epsilon_frame_span must be positive")
        if self.training_device not in DEVICE_CHOICES:
            raise ValueError("unsupported training device")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError("unsupported reward mode")
        for label, value in (
            ("forward_activity", self.forward_activity),
            ("a1_activity", self.a1_activity),
            ("a2_activity", self.a2_activity),
            ("face_opponent_activity", self.face_opponent_activity),
        ):
            if not 0.0 <= float(value) <= 100.0:
                raise ValueError(f"{label} must be in [0, 100]")


@dataclass(frozen=True)
class OpponentSpec:
    ship: str
    mode: str = OPPONENT_MODE_SIMPLE
    slot: int | None = None
    model: Any | None = None
    description: str = ""


class TrainingBatchAborted(RuntimeError):
    """Raised when a requested stop abandons the active training batch."""
