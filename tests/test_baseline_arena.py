"""Baseline-versus-baseline arena, scored by harm, used to compare strategies."""

import io
import math
import random
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from unittest import mock

from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.training.baseline import arena
from src.training.baseline.arena import (
    CELLS,
    STRONG,
    WEAK,
    ArenaMatch,
    ArenaResult,
    ArenaTally,
    build_matches,
    compare,
    format_report,
    new_arena_battle,
    paired_difference,
    rank_order,
    run_match,
    screen,
    spawn_crew,
    tally,
)


def _result(
    *,
    variant="default",
    opponent="Spathi",
    seed=0,
    cell=(STRONG, STRONG),
    dealt=0.5,
    taken=0.2,
    weight=0.5,
    **kwargs,
):
    return ArenaResult(
        ship="ZoqFotPik",
        variant=variant,
        opponent=opponent,
        seed=seed,
        ship_condition=cell[0],
        opponent_condition=cell[1],
        frames=1200,
        harm_taken_weight=weight,
        damage_dealt=dealt,
        damage_taken=taken,
        **kwargs,
    )


def _live_satellites(simulation, owner):
    return [
        obj
        for obj in simulation.world.objects
        if getattr(obj, "name", None) == "ChmmrSatellite"
        and getattr(obj, "parent", None) is owner
    ]


class ArenaFightTests(unittest.TestCase):
    def test_short_fight_reports_harm_and_diagnostics(self):
        result = run_match(ArenaMatch("ZoqFotPik", "Spathi", seed=0), frames=12)

        self.assertEqual(result.frames, 12)
        self.assertEqual((result.kills, result.deaths), (0, 0))
        self.assertGreaterEqual(result.harm_dealt, 0.0)
        self.assertEqual(result.harm_taken_weight, 0.5)
        self.assertTrue(0 <= result.still_frames <= 12)

    def test_fights_repeat_exactly(self):
        match = ArenaMatch("Earthling", "Mycon", seed=1, ship_condition=WEAK)

        self.assertEqual(run_match(match, frames=240), run_match(match, frames=240))

    def test_fights_use_real_ship_assets_like_training(self):
        captured = {}
        original = arena._arena_resources

        def spy():
            captured["resources"] = original()
            return captured["resources"]

        with mock.patch.object(arena, "_arena_resources", side_effect=spy):
            run_match(ArenaMatch("Umgah", "Spathi", seed=0), frames=1)

        from src.resources import HeadlessAssetManager

        self.assertIsInstance(captured["resources"], HeadlessAssetManager)
        # Real Umgah sprites differ from the 30x50 placeholder rectangle.
        ship = arena._arena_resources().ship("Umgah")
        self.assertNotEqual(tuple(ship.size), (30, 50))

    def test_variants_meet_the_same_battles(self):
        default = arena._match_rng(ArenaMatch("Pkunk", "Spathi", 3))
        variant = arena._match_rng(ArenaMatch("Pkunk", "Spathi", 3, "full_range"))
        other_cell = arena._match_rng(ArenaMatch("Pkunk", "Spathi", 3, ship_condition=WEAK))

        first = default.random()
        self.assertEqual(first, variant.random())
        self.assertNotEqual(first, other_cell.random())

    def test_sides_start_strong_or_weak(self):
        strong_crew = SHIP_DEFINITIONS["Chmmr"].start_hp
        simulation, _ = new_arena_battle(ArenaMatch("Chmmr", "Chmmr", 0, ship_condition=STRONG, opponent_condition=WEAK))

        self.assertEqual(simulation.player1.current_hp, strong_crew)
        self.assertEqual(simulation.player2.current_hp, math.ceil(strong_crew / 4))
        self.assertEqual(len(_live_satellites(simulation, simulation.player1)), 3)
        self.assertEqual(len(_live_satellites(simulation, simulation.player2)), 1)
        self.assertTrue(
            all(sat.current_hp == SHIP_DEFINITIONS["Chmmr"].satellite_hp
                for sat in _live_satellites(simulation, simulation.player2))
        )

    def test_weak_crew_is_a_quarter_rounded_up_and_at_least_one(self):
        ship = mock.Mock(start_hp=6)
        self.assertEqual(spawn_crew(ship, WEAK), 2)
        self.assertEqual(spawn_crew(ship, STRONG), 6)
        self.assertEqual(spawn_crew(mock.Mock(start_hp=1), WEAK), 1)

    def test_a_destroyed_ship_respawns_in_its_condition(self):
        simulation, _ = new_arena_battle(ArenaMatch("Earthling", "Spathi", 0, opponent_condition=WEAK))
        dead = simulation.player2
        dead.current_hp = 0
        simulation.step(actions={})

        self.assertIsNot(simulation.player2, dead)
        self.assertEqual(simulation.player2.current_hp, math.ceil(SHIP_DEFINITIONS["Spathi"].start_hp / 4))
        self.assertEqual(set(simulation.training_episode_deaths), {2})

    def test_shared_observation_context_changes_nothing(self):
        from src.training.observation import build_observation_context, encode_observation

        simulation, _ = new_arena_battle(ArenaMatch("Chmmr", "Vux", 0))
        for _ in range(30):
            simulation.step(actions={})
        p1, p2 = simulation.player1, simulation.player2
        context = build_observation_context(p1, p2, simulation.world)
        for ship, enemy in ((p1, p2), (p2, p1)):
            plain = encode_observation(ship, enemy, frame_id=simulation.frame_id, game_objects=simulation.world)
            shared = encode_observation(
                ship, enemy, frame_id=simulation.frame_id, game_objects=simulation.world,
                context=context, validate=False,
            )
            self.assertEqual(plain, shared)

    def test_matches_cover_every_variant_opponent_seed_and_cell(self):
        matches = build_matches(
            ["ZoqFotPik", "Spathi"],
            ["Spathi", "Earthling"],
            seeds=2,
            variants={"ZoqFotPik": ["a1_first"]},
        )

        self.assertEqual(len(matches), (2 + 1) * 2 * 2 * 4)
        self.assertEqual({(m.ship_condition, m.opponent_condition) for m in matches}, set(CELLS))
        self.assertEqual(
            {(match.ship, match.variant) for match in matches},
            {("ZoqFotPik", "default"), ("ZoqFotPik", "a1_first"), ("Spathi", "default")},
        )
        self.assertEqual({m.seed for m in build_matches(["Spathi"], ["Spathi"], [5, 6])}, {5, 6})
        with self.assertRaises(KeyError):
            build_matches(["Spathi"], ["Spathi"], 1, {"Spathi": ["missing"]})


class ArenaScoreTests(unittest.TestCase):
    def test_score_is_nothing_without_harm_dealt(self):
        self.assertEqual(_result(dealt=0.0, taken=1.0).score, 0.0)
        self.assertAlmostEqual(_result(dealt=1.0, taken=0.4, weight=0.25).score, 0.9)
        both = _result(dealt=0.3, kills_dealt=0.7, taken=0.1, deaths_taken=0.5)
        self.assertAlmostEqual((both.harm_dealt, both.harm_taken), (1.0, 0.6))

    def test_tally_totals_harm_by_opponent_and_cell(self):
        table = tally(
            [
                _result(dealt=1.0, taken=0.4, kills=1, still_frames=120),
                _result(dealt=0.0, taken=0.6, opponent="Earthling", cell=(WEAK, STRONG), deaths=1),
                _result(variant="a1_first"),
            ]
        )

        overall = table[("ZoqFotPik", "default")]["all"]
        self.assertEqual(overall.fights, 2)
        self.assertAlmostEqual(overall.harm_dealt, 1.0)
        self.assertAlmostEqual(overall.harm_taken, 1.0)
        self.assertAlmostEqual(overall.score, 0.8)
        self.assertEqual((overall.kills, overall.deaths), (1, 1))
        self.assertAlmostEqual(overall.harmless_share, 0.5)
        self.assertAlmostEqual(overall.still_share, 120 / 2400)
        self.assertEqual(table[("ZoqFotPik", "default")]["cell WS"].fights, 1)
        report = format_report(table, ["Spathi", "Earthling"], by_cell=True)
        self.assertIn("ZoqFotPik [a1_first]", report)
        self.assertIn("vs Earthling", report)
        self.assertIn("cell WS", report)
        self.assertIn("no harm  50%", report)


class ArenaRankingTests(unittest.TestCase):
    def _fights(self, variant, dealt, taken, *, noise=0.0, count=30):
        rng = random.Random(variant)
        return [
            _result(
                variant=variant,
                seed=seed,
                dealt=max(0.0, dealt + rng.uniform(-noise, noise)),
                taken=taken,
            )
            for seed in range(count)
        ]

    def test_paired_difference_has_a_95_percent_interval(self):
        difference = paired_difference([1.0, 2.0, 3.0, 4.0], [2.0, 2.5, 4.5, 5.0])

        self.assertAlmostEqual(difference.mean, 1.0)
        # stdev 0.408, t(3) 3.182: half-width 0.650.
        self.assertAlmostEqual(difference.high - difference.mean, 3.182 * 0.4082 / 2.0, places=3)
        self.assertEqual(difference.lead, 1)
        self.assertEqual(paired_difference([1.0], [1.0]).lead, 0)
        self.assertEqual(arena._t_critical(11), 2.228)
        self.assertEqual(arena._t_critical(5000), 1.960)

    def test_more_harm_dealt_ranks_higher_whatever_it_costs(self):
        base = self._fights("default", 1.0, 0.0, noise=0.05)
        bolder = self._fights("bold", 1.5, 2.5, noise=0.05)

        result = compare(base, bolder)

        self.assertLess(result.score.mean, 0.0)
        self.assertEqual(result.verdict, 1)
        self.assertEqual(result.reason, "harm dealt")

    def test_tied_harm_dealt_ranks_by_the_lambda_score(self):
        base = self._fights("default", 1.0, 1.0)
        safer = self._fights("safe", 1.02, 0.2)

        result = compare(base, safer)

        self.assertEqual(result.verdict, 1)
        self.assertIn("lambda score", result.reason)
        self.assertEqual(compare(base, self._fights("same", 1.0, 1.0)).verdict, 0)

    def test_comparisons_pair_only_shared_fights(self):
        base = self._fights("default", 1.0, 0.0, count=10)
        other = self._fights("other", 2.0, 0.0, count=20)

        self.assertEqual(compare(base, other).dealt.count, 10)

    def test_rank_order_puts_the_lead_group_first_by_lambda_score(self):
        def entry(dealt, score):
            result = ArenaTally()
            result.fights, result.harm_dealt, result.score = 1, dealt, score
            return result

        order = rank_order(
            {
                "far_behind": entry(0.5, 0.5),
                "top": entry(1.00, 0.2),
                "near_top_safer": entry(0.97, 0.9),
                "behind": entry(0.8, 0.8),
            }
        )

        self.assertEqual(order, ["near_top_safer", "top", "behind", "far_behind"])

    @staticmethod
    def _screen_run(dealt_for):
        flown = []

        def run(matches):
            flown.append(sorted({(m.variant, m.seed) for m in matches}))
            return [
                _result(variant=m.variant, seed=m.seed, opponent=m.opponent, dealt=dealt_for(m.variant, m.seed))
                for m in matches
            ]

        return flown, run

    @staticmethod
    def _noise(key, seed):
        return random.Random(f"{key}:{seed}").uniform(-1.0, 1.0)

    def test_screen_halves_the_field_and_doubles_the_seeds(self):
        # Close settings and wide noise: nobody is clearly behind.
        strength = {"default": 1.0, "a": 0.99, "b": 0.98, "c": 0.97, "d": 0.96}
        flown, run = self._screen_run(lambda v, s: 2.0 + strength[v] + self._noise(v, s))

        order, results = screen(
            "ZoqFotPik", list(strength), ["Spathi"], first_seeds=2, max_seeds=16,
            cells=[(STRONG, STRONG)], run=run,
        )

        # 5 settings on seeds 0-1, the best 3 on 2-3, the best 2 on 4-7.
        self.assertEqual(len(flown), 3)
        self.assertEqual({variant for variant, _ in flown[0]}, set(strength))
        self.assertEqual(len({variant for variant, _ in flown[1]}), 3)
        self.assertEqual({seed for _, seed in flown[1]}, {2, 3})
        self.assertEqual(len({variant for variant, _ in flown[2]}), 2)
        self.assertEqual({seed for _, seed in flown[2]}, {4, 5, 6, 7})
        # Halving to one ends the screen: the survivor isn't flown on to 16.
        self.assertEqual(len(results[order[0]]), 8)

    def test_screen_drops_settings_clearly_behind_and_stops_on_a_clear_leader(self):
        strength = {"default": 1.0, "a": 0.2, "b": 0.9, "c": 0.5, "d": 0.8}
        flown, run = self._screen_run(lambda v, s: strength[v])
        lines = []

        order, results = screen(
            "ZoqFotPik", list(strength), ["Spathi"], first_seeds=2, max_seeds=8,
            cells=[(STRONG, STRONG)], run=run, log=lines.append,
        )

        self.assertEqual(order[0], "default")
        self.assertEqual(len(flown), 1)
        self.assertEqual(len(results["default"]), 2)
        self.assertEqual(sum("clearly behind" in line for line in lines), 4)

    def test_screen_keeps_a_setting_that_is_not_clearly_behind(self):
        # "b" trails closely and stays; "a" is clearly behind and goes.
        strength = {"default": 1.0, "a": 0.2, "b": 0.99}
        flown, run = self._screen_run(
            lambda v, s: strength[v] + (0.0 if v == "a" else self._noise(v, s))
        )

        screen(
            "ZoqFotPik", list(strength), ["Spathi"], first_seeds=2, max_seeds=4,
            cells=[(STRONG, STRONG)], run=run,
        )

        self.assertEqual(len(flown), 2)
        self.assertEqual({variant for variant, _ in flown[1]}, {"default", "b"})

    def test_screen_merges_settings_that_fly_identically(self):
        # "same" flies exactly as "default"; the others are close and noisy.
        key = {"default": "default", "same": "default", "b": "b", "c": "c"}
        strength = {"default": 1.0, "same": 1.0, "b": 0.99, "c": 0.98}
        flown, run = self._screen_run(lambda v, s: 2.0 + strength[v] + self._noise(key[v], s))
        lines = []

        screen(
            "ZoqFotPik", list(strength), ["Spathi"], first_seeds=2, max_seeds=8,
            cells=[(STRONG, STRONG)], run=run, log=lines.append,
        )

        self.assertIn("same", {variant for variant, _ in flown[0]})
        self.assertIn("    same flies as default; merged", lines)
        # Three distinct settings left, the better two flown on.
        self.assertEqual(len({variant for variant, _ in flown[1]}), 2)
        self.assertNotIn("same", {variant for variant, _ in flown[1]})


class ArenaProgressTests(unittest.TestCase):
    def test_progress_text_shows_fights_time_left_and_the_fight_in_flight(self):
        match = arena.ArenaMatch("Spathi", "Umgah", 3, "approach", STRONG, WEAK)

        text = arena.progress_text(10, 40, 20.0, match)

        self.assertEqual(
            text,
            "10/40 fights (25%)  20s, ~60s left  now: Spathi [approach] vs Umgah, seed 3 SW",
        )
        self.assertEqual(arena.progress_text(40, 40, 80.0, None), "40/40 fights (100%)  80s")

    def test_progress_off_a_terminal_writes_a_line_each_tenth(self):
        matches = [arena.ArenaMatch("ZoqFotPik", "Spathi", seed, "default", STRONG, STRONG) for seed in range(20)]

        def fake_run_arena(batch, **_kwargs):
            return [_result(seed=m.seed) for m in batch]

        stream = io.StringIO()
        with mock.patch.object(arena, "run_arena", side_effect=fake_run_arena):
            results = arena.run_with_progress(matches, stream=stream)

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(results), 20)
        self.assertEqual(len(lines), 10)
        self.assertTrue(lines[0].startswith("2/20 fights (10%)"))
        self.assertIn("now: ZoqFotPik [default] vs Spathi, seed 2 SS", lines[0])
        self.assertTrue(lines[-1].startswith("20/20 fights (100%)"))


class ArenaCliTests(unittest.TestCase):
    def test_cli_flies_every_variant_and_compares_it_with_the_default(self):
        captured = []

        def fake_run_arena(matches, **_kwargs):
            captured.extend(matches)
            return [
                _result(variant=m.variant, seed=m.seed, cell=(m.ship_condition, m.opponent_condition))
                for m in matches
            ]

        with mock.patch.object(arena, "run_arena", side_effect=fake_run_arena):
            with redirect_stdout(io.StringIO()) as output, mock.patch("sys.stderr"):
                arena.main(
                    ["--ships", "ZoqFotPik", "--opponents", "Spathi", "--seeds", "2",
                     "--first-seed", "10", "--cells", "SS,WW", "--variant", "all"]
                )

        self.assertEqual({m.variant for m in captured}, {"default", "a1_first"})
        self.assertEqual({m.seed for m in captured}, {10, 11})
        self.assertEqual({(m.ship_condition, m.opponent_condition) for m in captured}, {(STRONG, STRONG), (WEAK, WEAK)})
        text = output.getvalue()
        self.assertIn("ZoqFotPik [default]", text)
        self.assertIn("ZoqFotPik [a1_first] vs [default]: equal", text)

    def test_cli_rejects_unknown_cells(self):
        with self.assertRaises(SystemExit):
            arena.main(["--ships", "Spathi", "--cells", "XX"])


if __name__ == "__main__":
    unittest.main()
