import math
import unittest

from src.training.decay import (
    DECAY_HALF_LIFE_BATCHES,
    DECAY_HALF_LIFE_OFF,
    DECAY_HALF_LIFE_VALUES,
    decay_factor_for_half_life,
    half_life_for_decay_factor,
    nearest_decay_half_life,
    normalize_decay_half_life,
)


class DecayHalfLifeTests(unittest.TestCase):
    def test_values_run_from_10_batches_to_off(self):
        self.assertEqual(
            DECAY_HALF_LIFE_VALUES,
            (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, "off"),
        )

    def test_half_life_and_factor_round_trip_for_every_value(self):
        for batches in DECAY_HALF_LIFE_BATCHES:
            factor = decay_factor_for_half_life(batches)
            self.assertAlmostEqual(factor ** batches, 0.5)
            self.assertAlmostEqual(
                half_life_for_decay_factor(factor) / batches, 1.0, places=9
            )
            self.assertEqual(nearest_decay_half_life(factor), batches)
        self.assertEqual(decay_factor_for_half_life(DECAY_HALF_LIFE_OFF), 1.0)
        self.assertTrue(math.isinf(half_life_for_decay_factor(1.0)))
        self.assertEqual(nearest_decay_half_life(1.0), DECAY_HALF_LIFE_OFF)

    def test_factors_match_the_specification_table(self):
        expected = {
            10: 0.93303,
            30: 0.97716,
            100: 0.99309,
            300: 0.99769,
            1_000: 0.999307,
            3_000: 0.999769,
            10_000: 0.9999307,
            30_000: 0.9999769,
            100_000: 0.9999931,
        }
        for batches, factor in expected.items():
            places = len(str(factor).split(".")[1])
            self.assertAlmostEqual(
                decay_factor_for_half_life(batches), factor, places=places
            )

    def test_old_factors_snap_to_the_nearest_half_life_on_the_log_scale(self):
        self.assertEqual(nearest_decay_half_life(0.998), 300)
        self.assertEqual(nearest_decay_half_life(0.995), 100)
        self.assertEqual(nearest_decay_half_life(0.95), 10)
        self.assertEqual(nearest_decay_half_life(1.000), "off")

    def test_stored_half_lives_normalise_to_settings(self):
        self.assertEqual(normalize_decay_half_life("Off", 300), "off")
        self.assertEqual(normalize_decay_half_life(1_000, 300), 1_000)
        self.assertEqual(normalize_decay_half_life("3000", 300), 3_000)
        self.assertEqual(normalize_decay_half_life(400, 300), 300)
        self.assertEqual(normalize_decay_half_life("bad", 300), 300)
        self.assertEqual(normalize_decay_half_life(None, 100), 100)
        self.assertEqual(normalize_decay_half_life(True, 100), 100)
        self.assertEqual(normalize_decay_half_life(-5, 100), 100)
        with self.assertRaises(ValueError):
            decay_factor_for_half_life(0)


if __name__ == "__main__":
    unittest.main()
