"""Epsilon and baseline-use decay, set as half-lives in batches.

The Regimen sets each decay as the number of batches over which the value
halves. Training multiplies the value by the per-batch factor the half-life
implies, ``0.5 ** (1 / n)``.
"""

from __future__ import annotations

import math

DECAY_HALF_LIFE_OFF = "off"
DECAY_HALF_LIFE_BATCHES = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)
DECAY_HALF_LIFE_VALUES = DECAY_HALF_LIFE_BATCHES + (DECAY_HALF_LIFE_OFF,)
DEFAULT_EPSILON_DECAY_HALF_LIFE = 300
DEFAULT_BASELINE_DECAY_HALF_LIFE = 100


def is_decay_off(half_life) -> bool:
    return isinstance(half_life, str) and half_life.strip().lower() == DECAY_HALF_LIFE_OFF


def decay_factor_for_half_life(half_life) -> float:
    """Per-batch factor that halves a value over ``half_life`` batches.

    Off gives 1, no decay.
    """

    if is_decay_off(half_life):
        return 1.0
    batches = float(half_life)
    if not math.isfinite(batches) or batches <= 0.0:
        raise ValueError("decay half-life must be a positive number of batches")
    return 0.5 ** (1.0 / batches)


def half_life_for_decay_factor(factor) -> float:
    """Batches over which ``factor`` halves a value; infinite at 1 or more."""

    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("decay factor must be positive")
    if factor >= 1.0:
        return math.inf
    return math.log(0.5) / math.log(factor)


def _nearest_batches(batches: float) -> int:
    return min(
        DECAY_HALF_LIFE_BATCHES,
        key=lambda candidate: abs(math.log(batches / candidate)),
    )


def nearest_decay_half_life(factor):
    """The setting nearest a per-batch factor on the log scale; 1 is Off."""

    batches = half_life_for_decay_factor(factor)
    if math.isinf(batches):
        return DECAY_HALF_LIFE_OFF
    return _nearest_batches(batches)


def normalize_decay_half_life(value, default):
    """The setting nearest a stored half-life, or ``default`` if unreadable."""

    if is_decay_off(value):
        return DECAY_HALF_LIFE_OFF
    if isinstance(value, bool):
        return default
    try:
        batches = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(batches) or batches <= 0.0:
        return default
    return _nearest_batches(batches)


def format_decay_half_life(half_life) -> str:
    if is_decay_off(half_life):
        return "Off"
    return f"{int(half_life):,} batches"
