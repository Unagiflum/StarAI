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

    if not _ordered_nonoverlapping(boundaries, samples):
        return _finalize_pending_episodes_compatibility(boundaries, samples)

    results = []
    sample_index = 0
    for boundary in boundaries:
        component_sums = {component: 0.0 for component in REWARD_COMPONENTS}
        return_sum = 0.0
        count = 0
        while sample_index < len(samples):
            sample = samples[sample_index]
            if sample.start_frame_id <= boundary.start_frame_id:
                raise ValueError(
                    "every finalized sample must belong to one combat episode"
                )
            if sample.start_frame_id > boundary.end_frame_id:
                break
            sample_index += 1
            count += 1
            return_sum += float(sample.return_value)
            for component in REWARD_COMPONENTS:
                component_sums[component] += float(
                    sample.weighted_components.get(component, 0.0)
                )
        results.append(_episode_result(boundary, count, return_sum, component_sums))
    if sample_index != len(samples):
        raise ValueError("every finalized sample must belong to one combat episode")
    return tuple(results)


def _ordered_nonoverlapping(boundaries, samples) -> bool:
    return all(
        boundaries[index - 1].end_frame_id <= boundaries[index].start_frame_id
        for index in range(1, len(boundaries))
    ) and all(
        samples[index - 1].start_frame_id <= samples[index].start_frame_id
        for index in range(1, len(samples))
    )


def _finalize_pending_episodes_compatibility(boundaries, samples):
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
        return_sum = 0.0
        for sample in grouped:
            return_sum += float(sample.return_value)
            for component in REWARD_COMPONENTS:
                component_sums[component] += float(
                    sample.weighted_components.get(component, 0.0)
                )
        results.append(
            _episode_result(
                boundary,
                len(grouped),
                return_sum,
                component_sums,
            )
        )
    if assigned != len(samples):
        raise ValueError("every finalized sample must belong to one combat episode")
    return tuple(results)


def _episode_result(boundary, count, return_sum, component_sums):
    return TrainingEpisodeResult(
        opponent=boundary.opponent,
        frames=boundary.frames,
        terminal_reason=boundary.terminal_reason,
        mature_samples=count,
        total_return=return_sum / count if count else 0.0,
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
