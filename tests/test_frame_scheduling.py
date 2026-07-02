import unittest
from unittest.mock import Mock

from src.Battle.battle import FixedStepScheduler
from src.frame_timing import AdaptiveRenderRate, PresentationClock


class FixedStepSchedulerTests(unittest.TestCase):
    def test_physics_rate_is_independent_of_render_rate(self):
        scheduler = FixedStepScheduler(24, start_ready=False)

        steps = sum(scheduler.advance(1 / 60)[0] for _ in range(60))

        self.assertEqual(steps, 24)

    def test_catch_up_work_is_bounded(self):
        scheduler = FixedStepScheduler(24, max_catch_up_steps=5, start_ready=False)

        steps, interpolation = scheduler.advance(1.0)

        self.assertEqual(steps, 5)
        self.assertEqual(interpolation, 0.0)


class AdaptiveRenderRateTests(unittest.TestCase):
    def test_render_multiplier_falls_until_work_fits_the_budget(self):
        rate = AdaptiveRenderRate(24, 5, sample_frames=3)

        for _ in range(3):
            rate.observe(0.010)
        self.assertEqual(rate.multiplier, 4)
        self.assertEqual(rate.target_fps, 96)

        for _ in range(3):
            rate.observe(0.010)
        self.assertEqual(rate.multiplier, 4)


class PresentationClockTests(unittest.TestCase):
    def test_paces_at_adaptive_presentation_rate_and_returns_seconds(self):
        pygame_clock = Mock()
        pygame_clock.tick.return_value = 25
        pygame_clock.get_rawtime.return_value = 10
        clock = PresentationClock(24, 5, sample_frames=1, clock=pygame_clock)

        elapsed_seconds = clock.tick()

        pygame_clock.tick.assert_called_once_with(120)
        self.assertEqual(elapsed_seconds, 0.025)
        self.assertEqual(clock.target_fps, 96)

    def test_reset_discards_samples_and_stale_elapsed_time(self):
        pygame_clock = Mock()
        clock = PresentationClock(24, 5, clock=pygame_clock)
        clock.rate.observe(0.01)

        clock.reset()

        self.assertEqual(len(clock.rate._work_samples), 0)
        pygame_clock.tick.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
