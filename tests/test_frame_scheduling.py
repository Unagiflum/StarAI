import unittest

from src.Battle.battle import AdaptiveRenderRate, FixedStepScheduler


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


if __name__ == "__main__":
    unittest.main()
