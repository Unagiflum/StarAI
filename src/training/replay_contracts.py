"""Lightweight action and replay-transfer contracts."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys

from src.training.contracts import ACTION_OUTPUT_SIZE
from src.training.observation_transfer import (
    PACKED_OBSERVATION_BYTES,
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


@dataclass(frozen=True)
class ReplayTransferChunk:
    """A contiguous process-transfer block of replay samples.

    Observations and returns use canonical little-endian float32 storage;
    actions use one unsigned byte per sample.  Workers can therefore serialize
    one compact object and the parent can insert the entire block without
    rebuilding per-sample Python objects.
    """

    count: int
    observations: bytes
    actions: bytes
    returns: bytes

    def __post_init__(self) -> None:
        count = int(self.count)
        observations = bytes(self.observations)
        actions = bytes(self.actions)
        returns = bytes(self.returns)
        if count < 0:
            raise ValueError("replay transfer chunk count cannot be negative")
        if len(observations) != count * PACKED_OBSERVATION_BYTES:
            raise ValueError("replay transfer observations have the wrong size")
        if len(actions) != count:
            raise ValueError("replay transfer actions have the wrong size")
        if len(returns) != count * 4:
            raise ValueError("replay transfer returns have the wrong size")
        if any(action >= ACTION_OUTPUT_SIZE for action in actions):
            raise ValueError(f"action_index must be in [0, {ACTION_OUTPUT_SIZE})")
        numeric_returns = array("f")
        numeric_returns.frombytes(returns)
        if sys.byteorder != "little":
            numeric_returns.byteswap()
        if not all(math.isfinite(value) for value in numeric_returns):
            raise ValueError("return_value must be finite float32")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "returns", returns)

    def __len__(self) -> int:
        return self.count

    @classmethod
    def from_samples(cls, samples) -> "ReplayTransferChunk":
        samples = tuple(samples)
        observations = bytearray()
        actions = bytearray()
        returns = array("f")
        for sample in samples:
            if not isinstance(sample, ReplayTransferSample):
                sample = ReplayTransferSample.from_mature_sample(sample)
            observations.extend(sample.observation.data)
            actions.append(int(sample.action_index))
            returns.append(float(sample.return_value))
        if sys.byteorder != "little":
            returns.byteswap()
        return cls(
            count=len(samples),
            observations=bytes(observations),
            actions=bytes(actions),
            returns=returns.tobytes(),
        )

    @classmethod
    def from_mature_samples(cls, samples) -> "ReplayTransferChunk":
        """Pack trusted finalized trajectory samples as one float32 block."""

        observations = array("f")
        actions = bytearray()
        returns = array("f")
        count = 0
        expected_values = PACKED_OBSERVATION_BYTES // observations.itemsize
        for sample in samples:
            before = len(observations)
            observations.extend(sample.observation)
            if len(observations) - before != expected_values:
                raise ValueError("observation has the wrong size")
            actions.append(int(sample.action_index))
            returns.append(float(sample.return_value))
            count += 1
        if sys.byteorder != "little":
            observations.byteswap()
            returns.byteswap()
        return cls(
            count=count,
            observations=observations.tobytes(),
            actions=bytes(actions),
            returns=returns.tobytes(),
        )
