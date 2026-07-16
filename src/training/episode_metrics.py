"""Delayed combat-episode reporting for trainee-life reward trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.training.coordinated_contracts import TrainingEpisodeResult
from src.training.cpu_contracts import OpponentSpec
from src.training.rewards import MatureTrainingSample, REWARD_COMPONENTS


@dataclass(frozen=True)
class PendingCombatEpisode:
    opponent: OpponentSpec
    start_frame_id: int
    end_frame_id: int
    terminal_reason: str
    win: bool
    loss: bool
    draw: bool
    kills: int
    deaths: int

    @property
    def frames(self) -> int:
        return max(0, int(self.end_frame_id) - int(self.start_frame_id))


def finalize_pending_episodes(
    boundaries: Sequence[PendingCombatEpisode],
    samples: Sequence[MatureTrainingSample],
) -> tuple[TrainingEpisodeResult, ...]:
    """Group finalized samples by the combat containing their decision frame."""

    results = []
    assigned = 0
    for boundary in boundaries:
        grouped = tuple(
            sample
            for sample in samples
            if boundary.start_frame_id < sample.start_frame_id <= boundary.end_frame_id
        )
        assigned += len(grouped)
        component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
        for sample in grouped:
            for component in REWARD_COMPONENTS:
                component_sums[component] += float(
                    sample.weighted_components.get(component, 0.0)
                )
        count = len(grouped)
        results.append(
            TrainingEpisodeResult(
                opponent=boundary.opponent,
                frames=boundary.frames,
                terminal_reason=boundary.terminal_reason,
                mature_samples=count,
                total_return=(
                    sum(sample.return_value for sample in grouped) / count
                    if count
                    else 0.0
                ),
                win=boundary.win,
                loss=boundary.loss,
                draw=boundary.draw,
                kills=boundary.kills,
                deaths=boundary.deaths,
                component_totals={
                    component: component_sums[component] / count if count else 0.0
                    for component in REWARD_COMPONENTS
                },
            )
        )
    if assigned != len(samples):
        raise ValueError("every finalized sample must belong to one combat episode")
    return tuple(results)
