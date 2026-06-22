import os
import importlib
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.battle_aftermath import AftermathState
from src.Battle.world import World
from src.Objects.Ships.registry import create_ship
from src.audio import RecordingAudioService


class PkunkRebirthTests(unittest.TestCase):
    def test_rebirth_chance_uses_configured_decay_after_each_success(self):
        ship = create_ship("Pkunk", 1)
        self.assertEqual(ship.rebirth_chance_decay, 0.333)
        ship.audio_service = RecordingAudioService()
        ship.rng = mock.Mock()
        ship.rng.random.side_effect = [0.74, 0.24, 0.08]

        chances = []
        for _ in range(3):
            chances.append(ship.current_rebirth_chance)
            ship.current_hp = 0
            self.assertTrue(ship.attempt_rebirth())
            self.assertEqual(ship.current_hp, 0)
            ship.complete_rebirth()

        self.assertEqual(chances[0], 0.75)
        self.assertEqual(chances[1], 0.75 * 0.333)
        self.assertAlmostEqual(chances[2], 0.75 * 0.333 ** 2)
        self.assertEqual(ship.rebirth_count, 3)
        self.assertEqual(ship.current_hp, ship.max_hp)
        self.assertEqual(
            [operation[0] for operation in ship.audio_service.operations],
            ["play_effect"] * 3,
        )
        self.assertEqual(
            ship.audio_service.operations[0][1].name,
            "Pkunk-rebirth.wav",
        )

    def test_failed_roll_does_not_restore_ship_or_play_sound(self):
        ship = create_ship("Pkunk", 1)
        ship.current_hp = 0
        ship.audio_service = RecordingAudioService()
        ship.rng = mock.Mock()
        ship.rng.random.return_value = 0.75

        self.assertFalse(ship.attempt_rebirth())

        self.assertEqual(ship.current_hp, 0)
        self.assertEqual(ship.rebirth_count, 0)
        self.assertEqual(ship.audio_service.operations, [])

    def test_battle_rebirth_repositions_ship_and_starts_entry(self):
        ship = create_ship("Pkunk", 1)
        opponent = create_ship("Earthling", 2)
        ship.current_hp = 0
        ship.position = [100, 100]
        opponent.position = [4000, 4000]
        audio = RecordingAudioService()
        rng = mock.Mock()
        rng.random.return_value = 0
        rng.randint.return_value = 7
        rng.uniform.return_value = 0
        ship.audio_service = audio
        ship.rng = rng

        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.player1 = ship
        simulation.player2 = opponent
        simulation.world = World([ship, opponent])
        simulation.frame_id = 20
        simulation.aftermath = None
        simulation.sound_enabled = True
        simulation.audio = audio
        simulation.rng = rng
        simulation.entry = None
        simulation.entry_animations_enabled = True
        simulation.needs_selection = False

        battle_module = importlib.import_module("src.Battle.battle")
        with mock.patch.object(
            battle_module,
            "random_position_away_from",
            return_value=(2000, 2500),
        ):
            simulation._update_aftermath()

            self.assertEqual(ship.current_hp, 0)
            self.assertFalse(ship.currently_alive)
            self.assertIsNone(simulation.entry)
            self.assertIsNotNone(simulation.aftermath)
            final_death_frame = max(
                item.frame
                for item in simulation.aftermath.pending_explosions
                if item.ship is ship
            )

            while ship in simulation.pending_rebirths:
                simulation.frame_id += 1
                simulation._update_aftermath()

        self.assertEqual(ship.position, [2000, 2500])
        self.assertEqual(ship.heading, 7)
        self.assertEqual(ship.current_hp, ship.max_hp)
        self.assertTrue(ship.currently_alive)
        self.assertFalse(ship.trackable)
        self.assertIsNone(simulation.aftermath)
        self.assertEqual(simulation.entry.entering_ships, (ship,))
        self.assertEqual(simulation.entry.diagonal_trail_ships, frozenset((ship,)))
        self.assertEqual(
            simulation.frame_id,
            final_death_frame + const.PKUNK_REBIRTH_PAUSE_FRAMES,
        )
        played_effects = [
            operation[1].name
            for operation in audio.operations
            if operation[0] == "play_effect"
        ]
        self.assertEqual(played_effects[0], "shipdies.wav")
        self.assertEqual(played_effects[-2:], ["shipdies.wav", "Pkunk-rebirth.wav"])
        self.assertFalse(any(
            operation[0] == "play_victory_ditty"
            for operation in audio.operations
        ))

    def test_winning_resets_the_initial_chance(self):
        ship = create_ship("Pkunk", 1)
        ship.rebirth_count = 3

        ship.on_battle_won()

        self.assertEqual(ship.rebirth_count, 0)
        self.assertEqual(ship.current_rebirth_chance, 0.75)

    def test_staggered_pending_rebirth_does_not_count_as_a_win(self):
        first = create_ship("Pkunk", 1)
        second = create_ship("Pkunk", 2)
        first.rebirth_count = 2
        second.rebirth_count = 1
        first.current_hp = 0
        second.current_hp = 0
        first.currently_alive = False
        second.currently_alive = False

        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.player1 = first
        simulation.player2 = second
        simulation.world = World([])
        simulation.frame_id = const.PKUNK_REBIRTH_PAUSE_FRAMES
        simulation.aftermath = AftermathState(started_frame=0, latest_death_frame=0)
        simulation.pending_rebirths = {first, second}
        simulation.rebirth_pause_start_frames = {
            first: 0,
            second: simulation.frame_id,
        }
        simulation.entry = None
        simulation.entry_animations_enabled = False
        simulation.audio = RecordingAudioService()
        simulation.rng = mock.Mock()
        simulation.rng.randint.return_value = 0
        simulation.rng.uniform.return_value = 0

        simulation._complete_ready_rebirths()

        self.assertTrue(first.currently_alive)
        self.assertEqual(first.rebirth_count, 2)
        self.assertEqual(simulation.pending_rebirths, {second})


if __name__ == "__main__":
    unittest.main()
