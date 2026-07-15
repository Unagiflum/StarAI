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

    def test_training_kill_requires_trainee_owned_lethal_source(self):
        simulation, _audio = self.create_simulation()
        trainee = simulation.player1
        old_opponent = simulation.player2
        old_opponent.current_hp = 0
        old_opponent.last_lethal_damage_source = SimpleNamespace(parent=trainee)

        state = simulation.step(actions={1: {}, 2: {}})

        self.assertEqual(state["training_episode_kills"], (1,))
        self.assertEqual(state["training_episode_deaths"], (2,))

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
