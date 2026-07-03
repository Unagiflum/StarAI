"""Shared presentation timing for battle and menu loops."""

import pygame


class PresentationClock:
    """Pace a render loop and report elapsed wall time in seconds."""

    def __init__(self, base_fps, multiplier, *, clock=None):
        self.base_fps = base_fps
        self.multiplier = max(1, multiplier)
        self._clock = clock or pygame.time.Clock()

    @property
    def target_fps(self):
        return self.base_fps * self.multiplier

    def tick(self):
        elapsed_ms = self._clock.tick(self.target_fps)
        return elapsed_ms / 1000.0

    def reset(self):
        """Discard stale elapsed time after a nested loop or pause."""
        self._clock.tick()

    def set_multiplier(self, multiplier):
        """Apply a new presentation rate and discard stale elapsed time."""
        self.multiplier = max(1, multiplier)
        self.reset()
