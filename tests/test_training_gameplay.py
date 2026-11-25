import os
import random
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.effects import BattleEffect
from src.Objects.Ships.registry import create_ship
from src.audio import RecordingAudioService
from src.training.event_ledger import BattleEventLedger


class TrainingGameplayTests(unittest.TestCase):
    def create_simulation(self, trainee="Earthling", opponent="Earthling", seed=7):
        audio = RecordingAudioService()
        simulation = BattleSimulation(
            None,
            create_ship(trainee, 1, audio_service=audio),
            create_ship(opponent, 2, audio_service=audio),
            audio_service=audio,
            rng=random.Random(seed),
            include_stars=False,
            training_event_ledger=BattleEventLedger(),
        )
        return simulation, audio

    def test_training_death_immediately_respawns_without_stopping_music(self):
        simulation, audio = self.create_simulation()
        old_trainee = simulation.player1
        surviving_opponent = simulation.player2
        old_trainee.current_hp = 0

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_deaths"], (1,))
        self.assertIsNot(simulation.player1, old_trainee)
        self.assertIs(simulation.player2, surviving_opponent)
        self.assertGreater(simulation.player1.current_hp, 0)
        self.assertTrue(simulation.player1.currently_alive)
        self.assertIs(simulation.player1.opponent, surviving_opponent)
        self.assertIs(surviving_opponent.opponent, simulation.player1)
        self.assertNotIn(old_trainee, simulation.world.objects)
        self.assertTrue(any(isinstance(obj, BattleEffect) for obj in simulation.world))
        operation_names = [operation[0] for operation in audio.operations]
        self.assertIn("play_effect", operation_names)
        self.assertNotIn("stop_music", operation_names)
        self.assertNotIn("play_victory_ditty", operation_names)

    def test_training_pkunk_uses_normal_immediate_respawn(self):
        simulation, _audio = self.create_simulation(trainee="Pkunk")
        old_pkunk = simulation.player1
        old_pkunk.attempt_rebirth = mock.Mock(return_value=True)
        old_pkunk.current_hp = 0

        simulation.step(actions={1: {}, 2: {}})

        old_pkunk.attempt_rebirth.assert_not_called()
        self.assertIsNot(simulation.player1, old_pkunk)
        self.assertEqual(simulation.training_episode_deaths, (1,))

    def _rebirth_timeline(self, simulation, pkunk):
        """Frames from death to wreck hidden, to rebirth, to controllable."""
        pkunk.attempt_rebirth = mock.Mock(return_value=True)
        for _ in range(3):
            simulation.step(actions={1: {}, 2: {}})
        pkunk.current_hp = 0
        start = simulation.frame_id
        hidden = reborn = None
        for _ in range(300):
            simulation.step(actions={1: {}, 2: {}})
            elapsed = simulation.frame_id - start
            if hidden is None and pkunk not in simulation.world.objects:
                hidden = elapsed
            if reborn is None and pkunk.currently_alive:
                reborn = elapsed
            entry = simulation.entry
            if reborn is not None and (entry is None or pkunk not in entry.entering_ships):
                return hidden, reborn, elapsed
        self.fail("the Pkunk was not reborn")

    def test_training_pkunk_rebirth_matches_a_real_battle(self):
        audio = RecordingAudioService()
        real = BattleSimulation(
            None,
            create_ship("Pkunk", 1, audio_service=audio),
            create_ship("Earthling", 2, audio_service=audio),
            audio_service=audio,
            rng=random.Random(5),
            include_stars=False,
        )
        training, _audio = self.create_simulation(trainee="Pkunk", seed=5)
        training.enable_training_rebirth()

        real_timeline = self._rebirth_timeline(real, real.player1)
        training_pkunk = training.player1
        training_timeline = self._rebirth_timeline(training, training_pkunk)

        self.assertEqual(training_timeline, real_timeline)
        hidden, reborn, _controllable = training_timeline
        self.assertEqual(reborn - hidden, const.PKUNK_REBIRTH_PAUSE_FRAMES)
        self.assertIs(training.player1, training_pkunk)
        self.assertEqual(training_pkunk.current_hp, training_pkunk.max_hp)
        self.assertEqual(training_pkunk.current_energy, training_pkunk.max_energy)

    def test_training_trainee_rebirth_is_not_a_death_and_keeps_its_opponent(self):
        simulation, _audio = self.create_simulation(trainee="Pkunk")
        simulation.enable_training_rebirth()
        pkunk = simulation.player1
        opponent = simulation.player2
        pkunk.attempt_rebirth = mock.Mock(return_value=True)
        pkunk.current_hp = 0

        state = simulation.step(actions={1: {}, 2: {}})

        pkunk.attempt_rebirth.assert_called_once_with()
        self.assertEqual(state["training_episode_deaths"], ())
        self.assertIs(simulation.player1, pkunk)
        self.assertIs(simulation.player2, opponent)
        self.assertFalse(pkunk.currently_alive)
        self.assertTrue(simulation.training_rebirth_pending)
        self.assertTrue(simulation.training_ship_absent(1))
        self.assertFalse(simulation.training_ship_absent(2))

    def test_training_opponent_pkunk_rebirth_skips_the_replacement_pause(self):
        simulation, _audio = self.create_simulation(opponent="Pkunk")
        simulation.enable_training_rebirth()
        trainee = simulation.player1
        pkunk = simulation.player2
        pkunk.attempt_rebirth = mock.Mock(return_value=True)
        pkunk.current_hp = 0
        pkunk.last_lethal_damage_source = SimpleNamespace(parent=trainee)

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_deaths"], ())
        self.assertEqual(state["training_episode_kills"], ())
        self.assertFalse(state["training_opponent_absent"])
        self.assertIs(simulation.player2, pkunk)
        self.assertTrue(simulation.training_ship_absent(2))

    def test_training_pkunk_failed_rebirth_roll_respawns_normally(self):
        simulation, _audio = self.create_simulation(trainee="Pkunk")
        simulation.enable_training_rebirth()
        old_pkunk = simulation.player1
        old_pkunk.attempt_rebirth = mock.Mock(return_value=False)
        old_pkunk.current_hp = 0

        state = simulation.step(actions={1: {}, 2: {}})

        old_pkunk.attempt_rebirth.assert_called_once_with()
        self.assertEqual(state["training_episode_deaths"], (1,))
        self.assertIsNot(simulation.player1, old_pkunk)
        self.assertFalse(simulation.training_ship_absent(1))

    def test_opponent_death_during_trainee_rebirth_keeps_the_trainee(self):
        simulation, _audio = self.create_simulation(trainee="Pkunk")
        simulation.enable_training_rebirth()
        pkunk = simulation.player1
        old_opponent = simulation.player2
        pkunk.attempt_rebirth = mock.Mock(return_value=True)
        pkunk.current_hp = 0
        simulation.step(actions={1: {}, 2: {}})
        old_opponent.current_hp = 0

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_deaths"], (2,))
        self.assertIs(simulation.player1, pkunk)
        self.assertIsNot(simulation.player2, old_opponent)
        self.assertTrue(simulation.training_ship_absent(1))

    def test_training_kill_requires_trainee_owned_lethal_source(self):
        simulation, _audio = self.create_simulation()
        trainee = simulation.player1
        old_opponent = simulation.player2
        old_opponent.current_hp = 0
        old_opponent.last_lethal_damage_source = SimpleNamespace(parent=trainee)

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_kills"], (1,))
        self.assertEqual(state["training_episode_deaths"], (2,))

    def test_opponent_respawn_waits_for_three_targetless_seconds(self):
        simulation, _audio = self.create_simulation()
        trainee = simulation.player1
        trainee.position = [1000.0, 1000.0]
        trainee.previous_position = trainee.position.copy()
        trainee.velocity = [0.0, 0.0]
        trainee.current_hp = 1000
        old_opponent = simulation.player2
        old_opponent.current_hp = 0
        old_opponent.last_lethal_damage_source = SimpleNamespace(parent=trainee)

        death_state = simulation.step(actions={1: {}, 2: {}})
        replacement = simulation.player2

        self.assertTrue(death_state["training_opponent_absent"])
        self.assertFalse(replacement.currently_alive)
        self.assertNotIn(replacement, simulation.world.objects)
        self.assertIsNone(trainee.opponent)

        for _ in range(const.TRAINING_POST_OPPONENT_DEATH_FRAMES - 1):
            state = simulation.step(actions={1: {}, 2: {}})

        self.assertTrue(state["training_opponent_absent"])
        completion = simulation.step(actions={1: {}, 2: {}})
        self.assertFalse(completion["training_opponent_absent"])
        self.assertTrue(completion["training_post_death_completed"])
        self.assertTrue(replacement.currently_alive)
        self.assertIn(replacement, simulation.world.objects)
        self.assertIs(trainee.opponent, replacement)
        self.assertIs(replacement.opponent, trainee)

    def test_opponent_self_destruct_does_not_credit_trainee_kill(self):
        simulation, _audio = self.create_simulation(opponent="Shofixti")
        old_opponent = simulation.player2
        old_opponent.current_hp = 0
        old_opponent.last_lethal_damage_source = SimpleNamespace(
            name="ShofixtiA2",
            parent=old_opponent,
        )

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_kills"], ())
        self.assertEqual(state["training_episode_deaths"], (2,))

    def test_planet_death_counts_without_trainee_kill(self):
        simulation, _audio = self.create_simulation()
        old_trainee = simulation.player1
        old_trainee.current_hp = 0
        old_trainee.last_lethal_damage_source = SimpleNamespace(name="Planet")

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_kills"], ())
        self.assertEqual(state["training_episode_deaths"], (1,))

    def test_training_chmmr_satellite_spawn_is_resolved_before_first_step(self):
        simulation, _audio = self.create_simulation(trainee="Chmmr", seed=11)

        satellites = [
            obj
            for obj in simulation.world
            if getattr(obj, "name", None) == "ChmmrSatellite"
            and getattr(obj, "parent", None) is simulation.player1
        ]

        self.assertTrue(simulation.player1._satellites_spawned)
        self.assertLessEqual(len(satellites), 3)
        self.assertEqual(simulation.player1.drain_spawned_objects(), [])
        simulation.player1.update()
        self.assertEqual(simulation.player1.drain_spawned_objects(), [])
        self.assertEqual(simulation.frame_id, 0)


if __name__ == "__main__":
    unittest.main()
