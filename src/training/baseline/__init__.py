"""Scripted baseline strategies for every ship.

The baseline plays opponents, shares control of trainees during training, and
is available as an AI choice in battle mode. It reads only the training
observation vector.
"""

from src.training.baseline.controller import (
    BaselineController,
    BaselineOpponentController,
    BaselinePhaseMixer,
    baseline_decisions_per_phase,
)
from src.training.baseline.strategies import STRATEGIES, strategy_for

__all__ = (
    "BaselineController",
    "BaselineOpponentController",
    "BaselinePhaseMixer",
    "STRATEGIES",
    "baseline_decisions_per_phase",
    "strategy_for",
)
