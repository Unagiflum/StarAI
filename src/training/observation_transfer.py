"""Compact float32 observation payloads for process boundaries."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import math
import sys

from src.training.contracts import OBSERVATION_INPUT_SIZE


FLOAT32_BYTES = 4
PACKED_OBSERVATION_BYTES = OBSERVATION_INPUT_SIZE * FLOAT32_BYTES


@dataclass(frozen=True)
class PackedObservation:
    """One canonical observation encoded as native little-endian float32 bytes."""

    data: bytes
    finite_validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        payload = bytes(self.data)
        if len(payload) != PACKED_OBSERVATION_BYTES:
            raise ValueError(
                f"packed observation must contain {PACKED_OBSERVATION_BYTES} bytes"
            )
        object.__setattr__(self, "data", payload)

    def __len__(self) -> int:
        return OBSERVATION_INPUT_SIZE


def pack_observation(
    values: Sequence[float] | Iterable[float],
    *,
    finite_validated: bool = False,
) -> PackedObservation:
    """Pack exactly one observation, validating values once by default."""

    numeric = array("f", values)
    if len(numeric) != OBSERVATION_INPUT_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_INPUT_SIZE}")
    if not finite_validated and not all(math.isfinite(value) for value in numeric):
        raise ValueError("observation values must be finite float32 numbers")
    if sys.byteorder != "little":
        numeric.byteswap()
    return PackedObservation(numeric.tobytes(), finite_validated=True)


def unpack_observation(
    packed: PackedObservation,
    *,
    validate_finite: bool = True,
) -> memoryview:
    """Return a zero-copy native float32 view over a packed observation."""

    if not isinstance(packed, PackedObservation):
        raise TypeError("packed must be a PackedObservation")
    if sys.byteorder != "little":
        numeric = array("f")
        numeric.frombytes(packed.data)
        numeric.byteswap()
        view = memoryview(numeric)
    else:
        view = memoryview(packed.data).cast("f")
    if len(view) != OBSERVATION_INPUT_SIZE:
        raise ValueError(f"observation must have length {OBSERVATION_INPUT_SIZE}")
    if (
        validate_finite
        and not packed.finite_validated
        and not all(math.isfinite(value) for value in view)
    ):
        raise ValueError("observation values must be finite float32 numbers")
    return view


def unpack_observation_array(
    packed: PackedObservation,
    *,
    validate_finite: bool = False,
):
    """Return a zero-copy NumPy float32 view for parent-side inference."""

    # NumPy is deliberately imported only on this parent-side path. Workers use
    # ``unpack_observation`` and lightweight builds may exclude NumPy entirely.
    import numpy as np

    if not isinstance(packed, PackedObservation):
        raise TypeError("packed must be a PackedObservation")
    values = np.frombuffer(packed.data, dtype="<f4")
    if values.shape != (OBSERVATION_INPUT_SIZE,):
        raise ValueError(f"observation must have length {OBSERVATION_INPUT_SIZE}")
    if (
        validate_finite
        and not packed.finite_validated
        and not bool(np.all(np.isfinite(values)))
    ):
        raise ValueError("observation values must be finite float32 numbers")
    return values
