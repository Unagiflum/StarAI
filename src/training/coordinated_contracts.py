"""Lightweight coordinated-window result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from src.training.cpu_contracts import OpponentSpec


@dataclass(frozen=True)
class TrainingEpisodeResult:
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
class CoordinatedFixedFrameWindowResult:
    opponent: OpponentSpec
    frames: int
    mature_samples: int
    episode_results: tuple[TrainingEpisodeResult, ...]
    total_return: float
    win: bool
    loss: bool
    draw: bool
    component_totals: Mapping[str, float] = field(default_factory=dict)

