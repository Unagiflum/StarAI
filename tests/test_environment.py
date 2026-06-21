import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

from src.Battle.environment import (
    BattleObservation,
    EntityObservation,
    HeadlessBattleEnvironment,
)


class HeadlessBattleEnvironmentTests(unittest.TestCase):
    def test_same_seed_and_actions_produce_identical_episodes(self):
        first = HeadlessBattleEnvironment("Arilou", "Earthling", seed=41)
        second = HeadlessBattleEnvironment("Arilou", "Earthling", seed=41)

        self.assertEqual(first.reset(), second.reset())
        actions = {1: {"action2": True}, 2: {}}
        first_transition = first.step(actions)
        second_transition = second.step(actions)

        self.assertEqual(first_transition, second_transition)

    def test_seeded_environments_are_isolated_from_global_randomness(self):
        first = HeadlessBattleEnvironment(seed=17)
        second = HeadlessBattleEnvironment(seed=17)

        first_observation = first.reset()
        random.seed(999)
        for _ in range(100):
            random.random()
        second_observation = second.reset()

        self.assertEqual(first_observation, second_observation)
        self.assertEqual(first.step({}), second.step({}))

    def test_different_seeds_change_initial_state(self):
        first = HeadlessBattleEnvironment(seed=1).reset()
        second = HeadlessBattleEnvironment(seed=2).reset()

        self.assertNotEqual(first.ships, second.ships)

    def test_observation_contains_only_immutable_plain_values(self):
        environment = HeadlessBattleEnvironment(seed=3)
        observation = environment.reset()

        self.assertIsInstance(observation, BattleObservation)
        self.assertEqual(len(observation.ships), 2)
        self.assertTrue(all(
            isinstance(entity, EntityObservation)
            for entity in observation.entities
        ))
        self.assertNotIn("star", {entity.kind for entity in observation.entities})
        self.assertEqual({ship.player for ship in observation.ships}, {1, 2})
        self.assertEqual(environment.simulation.world.stars, [])

    def test_terminal_transition_assigns_zero_sum_win_rewards(self):
        environment = HeadlessBattleEnvironment(seed=5)
        environment.reset()
        environment.simulation.player2.current_hp = 0

        transition = environment.step({})

        self.assertTrue(transition.terminated)
        self.assertFalse(transition.truncated)
        self.assertEqual(transition.rewards, {1: 1.0, 2: -1.0})
        self.assertEqual(transition.info["winner"], 1)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            environment.step({})

    def test_max_steps_truncates_a_live_episode(self):
        environment = HeadlessBattleEnvironment(seed=7, max_steps=1)
        environment.reset()

        transition = environment.step({})

        self.assertFalse(transition.terminated)
        self.assertTrue(transition.truncated)
        self.assertEqual(transition.rewards, {1: 0.0, 2: 0.0})

    def test_step_requires_reset(self):
        with self.assertRaisesRegex(RuntimeError, "not been reset"):
            HeadlessBattleEnvironment().step({})

    def test_fresh_process_needs_no_pygame_initialization_or_display(self):
        script = """
import pygame
from src.Battle.environment import HeadlessBattleEnvironment
assert not pygame.get_init()
assert pygame.display.get_surface() is None
environment = HeadlessBattleEnvironment(seed=23)
observation = environment.reset()
environment.step({1: {}, 2: {}})
assert len(observation.ships) == 2
assert not pygame.get_init()
assert pygame.display.get_surface() is None
print('headless-ok')
"""
        environment = os.environ.copy()
        environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "headless-ok")


if __name__ == "__main__":
    unittest.main()
