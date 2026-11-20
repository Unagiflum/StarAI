"""Baseline-versus-baseline arena used to compare strategy variants."""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from src.training.baseline import arena
from src.training.baseline.arena import (
    ArenaMatch,
    ArenaResult,
    OUTCOME_DRAW,
    OUTCOME_LOSS,
    OUTCOME_TIMEOUT,
    OUTCOME_TRADE,
    OUTCOME_WIN,
    build_matches,
    format_report,
    run_match,
    tally,
)


def _result(outcome, *, variant="default", opponent="Spathi", crew_end=5.0):
    return ArenaResult(
        ship="ZoqFotPik",
        variant=variant,
        opponent=opponent,
        seed=0,
        outcome=outcome,
        seconds=10.0,
        crew_start=10.0,
        crew_end=crew_end,
        opponent_crew_start=30.0,
        opponent_crew_end=15.0,
    )


class ArenaTests(unittest.TestCase):
    def test_short_fight_reports_crew_and_outcome(self):
        result = run_match(ArenaMatch("ZoqFotPik", "Spathi", seed=0), seconds=0.5)

        self.assertEqual(result.outcome, OUTCOME_TIMEOUT)
        self.assertEqual(result.seconds, 0.5)
        self.assertGreater(result.crew_start, 0.0)
        self.assertGreater(result.opponent_crew_start, 0.0)

    def test_fights_use_real_ship_assets_like_training(self):
        captured = {}
        original = arena._arena_resources

        def spy():
            captured["resources"] = original()
            return captured["resources"]

        with mock.patch.object(arena, "_arena_resources", side_effect=spy):
            run_match(ArenaMatch("Umgah", "Spathi", seed=0), seconds=1 / 24)

        from src.resources import HeadlessAssetManager

        self.assertIsInstance(captured["resources"], HeadlessAssetManager)
        # Real Umgah sprites differ from the 30x50 placeholder rectangle.
        ship = arena._arena_resources().ship("Umgah")
        self.assertNotEqual(tuple(ship.size), (30, 50))

    def test_variants_meet_the_same_battles(self):
        default = arena._match_rng(ArenaMatch("Pkunk", "Spathi", 3))
        variant = arena._match_rng(ArenaMatch("Pkunk", "Spathi", 3, "full_range"))

        self.assertEqual(default.random(), variant.random())

    def test_matches_cover_every_variant_opponent_and_seed(self):
        matches = build_matches(
            ["ZoqFotPik", "Spathi"],
            ["Spathi", "Earthling"],
            seeds=2,
            variants={"ZoqFotPik": ["a1_first"]},
        )

        self.assertEqual(len(matches), (2 + 1) * 2 * 2)
        self.assertEqual(
            {(match.ship, match.variant) for match in matches},
            {("ZoqFotPik", "default"), ("ZoqFotPik", "a1_first"), ("Spathi", "default")},
        )
        with self.assertRaises(KeyError):
            build_matches(["Spathi"], ["Spathi"], 1, {"Spathi": ["missing"]})

    def test_tally_counts_outcomes_and_damage(self):
        table = tally(
            [
                _result(OUTCOME_WIN),
                _result(OUTCOME_LOSS, opponent="Earthling"),
                _result(OUTCOME_TIMEOUT, variant="a1_first"),
            ]
        )

        overall = table[("ZoqFotPik", "default")]["all"]
        self.assertEqual((overall.wins, overall.losses, overall.fights), (1, 1, 2))
        # A win deals full damage; the loss dealt half of 30 crew.
        self.assertAlmostEqual(overall.mean_dealt, 0.75)
        self.assertAlmostEqual(overall.mean_taken, 0.75)
        timeout = table[("ZoqFotPik", "a1_first")]["all"]
        self.assertAlmostEqual(timeout.mean_taken, 0.5)
        report = format_report(table, ["Spathi", "Earthling"])
        self.assertIn("ZoqFotPik [a1_first]", report)
        self.assertIn("vs Earthling", report)

    def test_trades_count_apart_from_draws_and_cost_both_crews(self):
        table = tally([_result(OUTCOME_TRADE), _result(OUTCOME_DRAW)])

        overall = table[("ZoqFotPik", "default")]["all"]
        self.assertEqual((overall.trades, overall.draws, overall.wins), (1, 1, 0))
        self.assertAlmostEqual(overall.mean_dealt, 1.0)
        self.assertAlmostEqual(overall.mean_taken, 1.0)
        self.assertIn("1D 1X", overall.cell())

    def test_cli_expands_all_variants_for_the_chosen_ships(self):
        captured = []

        def fake_run_arena(matches, **_kwargs):
            captured.extend(matches)
            return [_result(OUTCOME_WIN, variant=match.variant) for match in matches]

        with mock.patch.object(arena, "run_arena", side_effect=fake_run_arena):
            with redirect_stdout(io.StringIO()) as output, mock.patch("sys.stderr"):
                arena.main(
                    ["--ships", "ZoqFotPik", "--opponents", "Spathi", "--seeds", "1",
                     "--variant", "all"]
                )

        self.assertEqual(
            [match.variant for match in captured],
            ["default", "a1_first"],
        )
        self.assertIn("ZoqFotPik [default]", output.getvalue())


if __name__ == "__main__":
    unittest.main()


class ArenaSatelliteTests(unittest.TestCase):
    def test_fights_against_the_chmmr_report_satellite_hp_lost(self):
        result = run_match(ArenaMatch("Vux", "Chmmr", seed=0), seconds=0.5)
        other = run_match(ArenaMatch("Vux", "Spathi", seed=0), seconds=0.5)

        self.assertIsNotNone(result.opponent_satellite_hp_lost)
        self.assertGreaterEqual(result.opponent_satellite_hp_lost, 0.0)
        self.assertIsNone(other.opponent_satellite_hp_lost)

    def test_tally_averages_satellite_hp_over_satellite_fights_only(self):
        with_satellites = ArenaResult(
            "Vux", "default", "Chmmr", 0, OUTCOME_LOSS, 5.0, 20.0, 0.0, 42.0, 40.0,
            opponent_satellite_hp_lost=6.0,
        )
        table = tally([with_satellites, _result(OUTCOME_WIN, opponent="Spathi")])

        chmmr = table[("Vux", "default")]["all"]
        self.assertAlmostEqual(chmmr.mean_satellite_hp_lost, 6.0)
        self.assertIn("sat HP  6.0", chmmr.cell())
        spathi = table[("ZoqFotPik", "default")]["all"]
        self.assertIsNone(spathi.mean_satellite_hp_lost)
        self.assertNotIn("sat HP", spathi.cell())
        self.assertIn("fight  10s", spathi.cell())
