import random
import unittest
from unittest import mock

import pygame

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Objects.Ships.registry import create_ship
from src.audio import NullAudioService
from src.resources import HeadlessAssetManager
from src.training.baseline.engine import ACTION_INDEX_BY_MASK
from src.training.baseline.shot_outcomes import ShotTracker, format_tallies
from src.training.contracts import ACTION_INDEX_TABLE

pygame.init()

_IDLE = ACTION_INDEX_TABLE[0]
_A1 = ACTION_INDEX_TABLE[ACTION_INDEX_BY_MASK[8]]
_A2 = ACTION_INDEX_TABLE[ACTION_INDEX_BY_MASK[16]]


def _battle(target, gap):
    """A Supox (player 2) facing ``target`` (player 1) from ``gap`` away."""

    resources = HeadlessAssetManager()
    audio = NullAudioService()
    simulation = BattleSimulation(
        None,
        create_ship(target, 1, resources=resources, audio_service=audio),
        create_ship("Supox", 2, resources=resources, audio_service=audio),
        audio_service=audio,
        rng=random.Random(3),
        include_stars=False,
        resources=resources,
    )
    simulation.world.objects[:] = [
        obj for obj in simulation.world.objects if type(obj).__name__ != "Asteroid"
    ]
    me, enemy = simulation.player1, simulation.player2
    me.position = [2000.0, 2000.0]
    enemy.position = [2000.0, 2000.0 + gap]
    for ship in (me, enemy):
        ship.velocity = [0.0, 0.0]
        ship.heading = 0
        ship.rotation = 0.0
    return simulation


def _supox_shot_outcome(target, gap, target_action, frames=30):
    simulation = _battle(target, gap)
    tracker = ShotTracker()
    tracker.on_step(simulation)
    # The target acts only once the shot exists, so a defence isn't spent
    # (or cooling down) before it.
    simulation.step(actions={1: _IDLE, 2: _A1})
    tracker.on_step(simulation)
    for _ in range(frames):
        simulation.step(actions={1: target_action, 2: _IDLE})
        tracker.on_step(simulation)
    tracker.finish()
    shots = [shot for shot in tracker.finished if shot.shooter == "Supox"]
    return [shot.outcome for shot in shots], tracker


class ShotTrackerTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(const, "PERCEPTION_DELAY_FRAMES", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_shot_that_takes_crew_is_a_hit(self):
        outcomes, _ = _supox_shot_outcome("Mycon", 300.0, _IDLE)

        self.assertEqual(outcomes, ["hit"])

    def test_a_shot_into_a_raised_shield_is_absorbed(self):
        outcomes, _ = _supox_shot_outcome("Yehat", 300.0, _A2)

        self.assertEqual(outcomes, ["absorbed"])

    def test_a_shot_the_target_destroys_is_shot_down(self):
        # A held Earthling point defence destroys the 1 HP shot.
        outcomes, _ = _supox_shot_outcome("Earthling", 300.0, _A2)

        self.assertEqual(outcomes, ["shot_down"])

    def test_tallies_group_by_shooter_ability_and_target(self):
        _, tracker = _supox_shot_outcome("Mycon", 300.0, _IDLE)

        tallies = tracker.tallies()

        tally = tallies[("Supox", "SupoxA1", "Mycon")]
        self.assertEqual(tally.total, 1)
        self.assertEqual(tally.share("hit"), 1.0)
        self.assertIn("SupoxA1", format_tallies(tallies))


if __name__ == "__main__":
    unittest.main()
