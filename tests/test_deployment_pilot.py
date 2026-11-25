import unittest
from types import SimpleNamespace

import src.const as const
from src.training.baseline import arena, deployment_pilot as pilot
from src.training.event_ledger import EVENT_CREW_CHANGED, EVENT_SHIP_DIED, TrainingBattleEvent
from src.training.harm import HARM_DAMAGE_ENEMY, HARM_KILL_ENEMY


class DeploymentPilotTests(unittest.TestCase):
    def test_intercept_uses_wrap_relative_motion_and_remaining_life(self):
        shot = SimpleNamespace(position=(const.ARENA_SIZE - 20, 100), velocity=(10, 0),
                               size=(2, 2), expiration_timer=3)
        target = SimpleNamespace(position=(10, 100), velocity=(0, 0), size=(2, 2))
        self.assertTrue(pilot.on_intercept(shot, target))
        shot.expiration_timer = 2
        self.assertFalse(pilot.on_intercept(shot, target))
        shot.expiration_timer = 30
        target.velocity = (10, 0)
        self.assertFalse(pilot.on_intercept(shot, target))

    def test_quiet_clock_includes_opening_and_tail_and_pauses_when_unavailable(self):
        pace = pilot.Pace()
        for _ in range(8):
            pace.step(True, False)
        pace.step(False, True)
        pace.step(True, False)
        self.assertEqual((pace.longest, pace.quiet, pace.pressure), (9, 9, 0))
        pace.step(True, True)
        for _ in range(13):
            pace.step(True, False)
        self.assertEqual((pace.longest, pace.active, pace.pressure), (13, 23, 1))

    def test_mutual_destruction_counts_final_hit_and_excludes_replacements(self):
        def ship(name):
            return SimpleNamespace(name=name, current_hp=1, currently_alive=True, limpets_attached=0)

        orz, chmmr = ship("Orz"), ship("Chmmr")
        events = []
        sim = SimpleNamespace(player1=orz, player2=chmmr, world=SimpleNamespace(objects=[]),
                              training_event_ledger=SimpleNamespace(events=events), entry=None,
                              training_episode_deaths=(), training_episode_kills=())
        observer = pilot.Deployment(facing_shield=False)
        observer.on_step(sim)
        shot = SimpleNamespace(parent=orz)
        events.extend([
            TrainingBattleEvent(1, EVENT_CREW_CHANGED, target=chmmr, obj=shot, magnitude=-1),
            TrainingBattleEvent(1, EVENT_SHIP_DIED, target=chmmr, obj=shot),
        ])
        sim.player1, sim.player2 = ship("Orz"), ship("Chmmr")
        sim.training_episode_deaths = sim.training_episode_kills = (1, 2)
        observer.on_step(sim)
        self.assertTrue(observer.lost)
        self.assertGreater(observer.harm.totals[HARM_DAMAGE_ENEMY], 0.0)
        self.assertGreater(observer.harm.totals[HARM_KILL_ENEMY], 0.0)
        self.assertEqual((observer.kills, observer.frames, observer.pace.pressure), (1, 1, 1))
        final_total = observer.harm.totals.copy()
        events.append(TrainingBattleEvent(2, EVENT_SHIP_DIED, target=sim.player2, obj=sim.player1))
        observer.on_step(sim)
        self.assertEqual(observer.harm.totals, final_total)
        self.assertEqual(observer.frames, 1)

    def test_pilot_observation_preserves_the_original_arena_result(self):
        match = arena.ArenaMatch("Spathi", "Utwig", seed=0)
        observer = pilot.Deployment(facing_shield=True)
        result = arena.run_match(match, frames=240, on_step=observer.on_step)
        self.assertEqual(result, arena.run_match(match, frames=240))
        self.assertEqual(observer.frames, 240)
        self.assertFalse(observer.lost)
        self.assertAlmostEqual(observer.harm.totals[HARM_DAMAGE_ENEMY]
                               + observer.harm.totals[HARM_KILL_ENEMY], result.harm_dealt)

    def test_challenger_needs_both_pace_and_harm(self):
        def rows(harm, gap):
            return [SimpleNamespace(seed=i, cell="SS", deployment_harm=harm, longest_gap=gap)
                    for i in range(10)]

        incumbent = rows(1.0, pilot.PACE_LIMIT)
        slow = rows(2.0, pilot.PACE_LIMIT + 1)
        slow[0].longest_gap = 0
        self.assertFalse(pilot.eligible_challenger(incumbent, slow))
        self.assertTrue(pilot.eligible_challenger(incumbent, rows(1.2, pilot.PACE_LIMIT)))
        self.assertFalse(pilot.eligible_challenger(incumbent, rows(1.01, 0)))
        self.assertTrue(pilot.eligible_challenger(slow, rows(2.0, 0)))
        self.assertFalse(pilot.eligible_challenger(slow, rows(0.1, 0)))
        self.assertEqual(pilot.p90_gap(rows(0.0, 0)[:9] + slow[-1:]), 0)

    def test_training_replacements_cannot_be_scored_as_rebirths(self):
        with self.assertRaises(ValueError):
            pilot.measure(arena.ArenaMatch("Pkunk", "Utwig", seed=0))
        with self.assertRaises(ValueError):
            pilot.measure(arena.ArenaMatch("Orz", "Pkunk", seed=0))
