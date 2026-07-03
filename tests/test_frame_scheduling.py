import unittest
from unittest.mock import Mock

from src.Battle.battle import FixedStepScheduler
from src.frame_timing import PresentationClock


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


class PresentationClockTests(unittest.TestCase):
    def test_paces_at_configured_presentation_rate_and_returns_seconds(self):
        pygame_clock = Mock()
        pygame_clock.tick.return_value = 25
        clock = PresentationClock(24, 5, clock=pygame_clock)

        elapsed_seconds = clock.tick()

        pygame_clock.tick.assert_called_once_with(120)
        self.assertEqual(elapsed_seconds, 0.025)
        self.assertEqual(clock.target_fps, 120)

    def test_repeated_ticks_do_not_reduce_configured_rate(self):
        pygame_clock = Mock()
        pygame_clock.tick.return_value = 25
        clock = PresentationClock(24, 5, clock=pygame_clock)

        for _ in range(120):
            clock.tick()

        self.assertEqual(clock.target_fps, 120)
        pygame_clock.tick.assert_called_with(120)

    def test_reset_discards_stale_elapsed_time(self):
        pygame_clock = Mock()
        clock = PresentationClock(24, 5, clock=pygame_clock)

        clock.reset()

        pygame_clock.tick.assert_called_once_with()

    def test_set_multiplier_applies_new_configured_rate(self):
        pygame_clock = Mock()
        clock = PresentationClock(24, 5, clock=pygame_clock)

        clock.set_multiplier(2)

        self.assertEqual(clock.target_fps, 48)
        pygame_clock.tick.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
