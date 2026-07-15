"""Lightweight action and replay-transfer contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.training.contracts import ACTION_OUTPUT_SIZE
from src.training.observation_transfer import (
    PackedObservation,
    pack_observation,
    unpack_observation,
)


@dataclass(frozen=True)
class ActionSelection:
    action_index: int
    exploratory: bool
    action_values: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ReplayTransferSample:
    """Spawn-picklable replay sample with a packed observation."""

    observation: PackedObservation
    action_index: int
    return_value: float

    @classmethod
    def from_mature_sample(cls, sample) -> "ReplayTransferSample":
        return cls(
            observation=pack_observation(sample.observation),
            action_index=int(sample.action_index),
            return_value=float(sample.return_value),
        )

    def __post_init__(self) -> None:
        unpack_observation(self.observation, validate_finite=True)
        if not 0 <= int(self.action_index) < ACTION_OUTPUT_SIZE:
            raise ValueError(f"action_index must be in [0, {ACTION_OUTPUT_SIZE})")
        if not math.isfinite(float(self.return_value)):
            raise ValueError("return_value must be finite")

