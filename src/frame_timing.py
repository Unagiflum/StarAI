"""Shared adaptive presentation timing for battle and menu loops."""

from collections import deque

import pygame


class AdaptiveRenderRate:
    """Lower the presentation rate when frame work exceeds its budget."""

    def __init__(self, base_fps, max_multiplier, sample_frames=60):
        self.base_fps = base_fps
        self.multiplier = max(1, max_multiplier)
        self.sample_frames = max(1, sample_frames)
        self._work_samples = deque(maxlen=self.sample_frames)

    @property
    def target_fps(self):
        return self.base_fps * self.multiplier

    def observe(self, work_seconds):
        if work_seconds <= 0:
            return
        self._work_samples.append(work_seconds)
        if len(self._work_samples) < self.sample_frames:
            return

        average_work = sum(self._work_samples) / len(self._work_samples)
        self._work_samples.clear()
        if average_work > 1.0 / self.target_fps and self.multiplier > 1:
            self.multiplier -= 1

    def reset_samples(self):
        self._work_samples.clear()


class PresentationClock:
    """Pace a render loop and report elapsed wall time in seconds."""

    def __init__(self, base_fps, max_multiplier, *, sample_frames=60, clock=None):
        self.rate = AdaptiveRenderRate(base_fps, max_multiplier, sample_frames)
        self._clock = clock or pygame.time.Clock()

    @property
    def target_fps(self):
        return self.rate.target_fps

    def tick(self):
        elapsed_ms = self._clock.tick(self.target_fps)
        self.rate.observe(self._clock.get_rawtime() / 1000.0)
        return elapsed_ms / 1000.0

    def reset(self):
        """Discard stale samples and elapsed time after a nested loop or pause."""
        self.rate.reset_samples()
        self._clock.tick()
