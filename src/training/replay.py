"""Replay storage, action selection, optimization, and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from src.training import torch_backend
from src.training.contracts import ACTION_OUTPUT_SIZE, OBSERVATION_INPUT_SIZE
from src.training.model_registry import replay_checkpoint_path
from src.training.rewards import MatureTrainingSample
from src.training.value_network import (
    predict_action_values,
    train_selected_action_regression,
)


REPLAY_CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class ReplaySample:
    observation: tuple[float, ...]
    action_index: int
    return_value: float

    @classmethod
    def from_mature_sample(cls, sample: MatureTrainingSample) -> "ReplaySample":
        return cls(
            observation=tuple(sample.observation),
            action_index=int(sample.action_index),
            return_value=float(sample.return_value),
        )

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> "ReplaySample":
        return cls(
            observation=tuple(float(x) for x in value["observation"]),
            action_index=int(value["action_index"]),
            return_value=float(value["return_value"]),
        )

    def __post_init__(self) -> None:
        if len(self.observation) != OBSERVATION_INPUT_SIZE:
            raise ValueError(f"observation must have length {OBSERVATION_INPUT_SIZE}")
        if not 0 <= self.action_index < ACTION_OUTPUT_SIZE:
            raise ValueError(f"action_index must be in [0, {ACTION_OUTPUT_SIZE})")
        if not math.isfinite(self.return_value):
            raise ValueError("return_value must be finite")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in self.observation):
            raise ValueError("observation values must be finite numbers")

    def to_state(self) -> dict[str, Any]:
        return {
            "observation": list(self.observation),
            "action_index": self.action_index,
            "return_value": self.return_value,
        }


class TrainingReplayBuffer:
    """Capacity-bound FIFO replay store with deterministic eviction."""

    def __init__(self, capacity: int):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._samples: list[ReplaySample] = []

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, index: int) -> ReplaySample:
        return self._samples[index]

    @property
    def samples(self) -> tuple[ReplaySample, ...]:
        return tuple(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def add(self, sample: ReplaySample | MatureTrainingSample) -> None:
        if isinstance(sample, MatureTrainingSample):
            sample = ReplaySample.from_mature_sample(sample)
        if not isinstance(sample, ReplaySample):
            raise TypeError("sample must be a ReplaySample or MatureTrainingSample")
        if len(self._samples) >= self.capacity:
            del self._samples[0]
        self._samples.append(sample)

    def extend(self, samples: Sequence[ReplaySample | MatureTrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample_minibatch(self, batch_size: int, rng: Any | None = None) -> tuple[ReplaySample, ...]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if int(batch_size) > len(self._samples):
            raise ValueError("batch_size cannot exceed replay occupancy")
        rng = rng or random
        indices = rng.sample(range(len(self._samples)), int(batch_size))
        return tuple(self._samples[index] for index in indices)

    def to_state(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "samples": [sample.to_state() for sample in self._samples],
        }

    def load_state(self, value: Mapping[str, Any]) -> None:
        capacity = int(value["capacity"])
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        samples = [ReplaySample.from_state(sample) for sample in value.get("samples", ())]
        self.capacity = capacity
        self._samples = samples[-capacity:]


@dataclass(frozen=True)
class ActionSelection:
    action_index: int
    exploratory: bool
    action_values: tuple[float, ...] | None = None


def select_action_epsilon_greedy(
    model,
    observation: Sequence[float],
    *,
    epsilon: float,
    rng: Any | None = None,
) -> ActionSelection:
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    rng = rng or random
    if epsilon >= 1.0 or (epsilon > 0.0 and rng.random() < epsilon):
        return ActionSelection(
            action_index=int(rng.randrange(ACTION_OUTPUT_SIZE)),
            exploratory=True,
        )

    values = predict_action_values(model, [observation])
    row = values[0].detach().cpu()
    return ActionSelection(
        action_index=int(row.argmax().item()),
        exploratory=False,
        action_values=tuple(float(value) for value in row.tolist()),
    )


@dataclass(frozen=True)
class OptimizationResult:
    loss: float
    batch_size: int


def optimize_from_replay(
    model,
    optimizer,
    replay_buffer: TrainingReplayBuffer,
    *,
    batch_size: int,
    rng: Any | None = None,
) -> OptimizationResult | None:
    if len(replay_buffer) < int(batch_size):
        return None
    batch = replay_buffer.sample_minibatch(batch_size, rng=rng)
    loss = train_selected_action_regression(
        model,
        optimizer,
        [sample.observation for sample in batch],
        [sample.action_index for sample in batch],
        [sample.return_value for sample in batch],
    )
    return OptimizationResult(loss=loss, batch_size=int(batch_size))


class TrainingCheckpointError(RuntimeError):
    """Raised when a training checkpoint cannot be read or applied."""


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    format_version: int
    has_optimizer_state: bool
    replay_sample_count: int | None
    extra_state: Mapping[str, Any]


def save_training_checkpoint(
    path: Path,
    model,
    *,
    optimizer=None,
    replay_buffer: TrainingReplayBuffer | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> None:
    torch = torch_backend.require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": REPLAY_CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra_state": dict(extra_state or {}),
    }
    _atomic_torch_save(torch, payload, path)
    if replay_buffer is not None:
        replay_payload = {
            "format_version": REPLAY_CHECKPOINT_FORMAT_VERSION,
            "replay_buffer": replay_buffer.to_state(),
        }
        _atomic_torch_save(torch, replay_payload, replay_checkpoint_path(path))
    else:
        try:
            replay_checkpoint_path(path).unlink()
        except FileNotFoundError:
            pass


def _atomic_torch_save(torch, payload: Mapping[str, Any], path: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def load_training_checkpoint(
    path: Path,
    model,
    *,
    optimizer=None,
    replay_buffer: TrainingReplayBuffer | None = None,
    map_location: Any | None = None,
) -> LoadedTrainingCheckpoint:
    torch = torch_backend.require_torch()
    try:
        try:
            payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=map_location)
    except Exception as exc:
        raise TrainingCheckpointError(f"Could not load training checkpoint: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise TrainingCheckpointError("Training checkpoint must contain a mapping")
    if payload.get("format_version") != REPLAY_CHECKPOINT_FORMAT_VERSION:
        raise TrainingCheckpointError("Unsupported training checkpoint format")
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise TrainingCheckpointError("Training checkpoint is missing model weights")

    try:
        model.load_state_dict(model_state)
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        replay_state = payload.get("replay_buffer")
        replay_sample_count = None
        if replay_buffer is not None and replay_state is not None:
            replay_buffer.load_state(replay_state)
            replay_sample_count = len(replay_buffer)
        elif replay_buffer is not None:
            replay_path = replay_checkpoint_path(path)
            if replay_path.exists() and replay_path.stat().st_size > 0:
                try:
                    try:
                        replay_payload = torch.load(
                            replay_path,
                            map_location=map_location,
                            weights_only=False,
                        )
                    except TypeError:
                        replay_payload = torch.load(replay_path, map_location=map_location)
                except Exception as exc:
                    raise TrainingCheckpointError(
                        f"Could not load replay checkpoint: {exc}"
                    ) from exc
                if not isinstance(replay_payload, Mapping):
                    raise TrainingCheckpointError("Replay checkpoint must contain a mapping")
                if replay_payload.get("format_version") != REPLAY_CHECKPOINT_FORMAT_VERSION:
                    raise TrainingCheckpointError("Unsupported replay checkpoint format")
                replay_state = replay_payload.get("replay_buffer")
                if replay_state is not None:
                    replay_buffer.load_state(replay_state)
                    replay_sample_count = len(replay_buffer)
    except TrainingCheckpointError:
        raise
    except Exception as exc:
        raise TrainingCheckpointError(f"Training checkpoint is incompatible: {exc}") from exc

    extra_state = payload.get("extra_state", {})
    if not isinstance(extra_state, Mapping):
        extra_state = {}
    return LoadedTrainingCheckpoint(
        format_version=int(payload["format_version"]),
        has_optimizer_state=payload.get("optimizer_state_dict") is not None,
        replay_sample_count=replay_sample_count,
        extra_state=dict(extra_state),
    )
