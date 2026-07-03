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
from src.Battle.battle_aftermath import (
    AftermathState,
    PendingRebirth,
    aftermath_camera_targets,
)
from src.Battle.world import World
from src.Objects.Ships.registry import create_ship
from src.audio import RecordingAudioService


class LongDurationRecordingAudioService(RecordingAudioService):
    def play_effect(self, path, volume=const.SOUND_EFFECT_VOLUME):
        super().play_effect(path, volume)
        return 999.0


class PkunkRebirthTests(unittest.TestCase):
    def test_rebirth_chance_uses_configured_decay_after_each_success(self):
        ship = create_ship("Pkunk", 1)
        self.assertEqual(ship.rebirth_chance_decay, 1.0)
        ship.audio_service = RecordingAudioService()
        ship.rng = mock.Mock()
        ship.rng.random.side_effect = [0.49, 0.24, 0.08]

        chances = []
        for _ in range(3):
            chances.append(ship.current_rebirth_chance)
            ship.current_hp = 0
            self.assertTrue(ship.attempt_rebirth())
            self.assertEqual(ship.current_hp, 0)
            ship.complete_rebirth()

        self.assertEqual(chances, [0.5, 0.5, 0.5])
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

    def test_successful_rebirth_restores_full_battery(self):
        ship = create_ship("Pkunk", 1)
        ship.current_hp = 0
        ship.current_energy = 1
        ship.energy_timer = 7
        ship.audio_service = RecordingAudioService()

        ship.complete_rebirth()

        self.assertEqual(ship.current_energy, ship.max_energy)
        self.assertEqual(ship.energy_timer, 0)

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
        audio = LongDurationRecordingAudioService()
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
            self.assertEqual(
                aftermath_camera_targets(
                    simulation.aftermath,
                    ship,
                    opponent,
                    simulation.frame_id,
                ),
                [opponent, ship],
            )
            final_death_frame = max(
                item.frame
                for item in simulation.aftermath.pending_explosions
                if item.ship is ship
            )

            while (
                simulation.aftermath is not None
                and ship in simulation.aftermath.pending_rebirths
            ):
                simulation.frame_id += 1
                simulation._update_aftermath()
                if simulation.aftermath is not None:
                    self.assertEqual(
                        aftermath_camera_targets(
                            simulation.aftermath,
                            ship,
                            opponent,
                            simulation.frame_id,
                        ),
                        [opponent, ship],
                    )

        self.assertEqual(ship.position, [2000, 2500])
        self.assertEqual(ship.heading, 7)
        self.assertEqual(ship.current_hp, ship.max_hp)
        self.assertTrue(ship.currently_alive)
        self.assertFalse(ship.trackable)
        self.assertIsNone(simulation.aftermath)
        self.assertEqual(simulation.entry.entering_ships, (ship,))
        self.assertEqual(
            simulation.entry.trail_styles,
            {ship: ship.rebirth_entry_trail_style()},
        )
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

    def test_round_start_resets_the_initial_chance(self):
        ship = create_ship("Pkunk", 1)
        ship.rebirth_count = 3

        ship.on_round_started()

        self.assertEqual(ship.rebirth_count, 0)
        self.assertEqual(ship.current_rebirth_chance, 0.5)

    def test_preserved_winner_resets_when_next_opponent_is_committed(self):
        ship = create_ship("Pkunk", 1)
        first_opponent = create_ship("Earthling", 2)
        simulation = BattleSimulation(
            None,
            ship,
            first_opponent,
            audio_service=RecordingAudioService(),
            include_stars=False,
        )
        ship.rebirth_count = 2
        next_opponent = create_ship("Vux", 2)

        simulation.select_next_round((ship, next_opponent))

        self.assertIs(simulation.player1, ship)
        self.assertIs(ship.opponent, next_opponent)
        self.assertEqual(ship.rebirth_count, 0)
        self.assertEqual(ship.current_rebirth_chance, 0.5)

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
        simulation.aftermath = AftermathState(
            started_frame=0,
            latest_death_frame=0,
            pending_rebirths={
                first: PendingRebirth(first, ready_frame=simulation.frame_id),
                second: PendingRebirth(
                    second,
                    ready_frame=(
                        simulation.frame_id + const.PKUNK_REBIRTH_PAUSE_FRAMES
                    ),
                ),
            },
        )
        simulation.entry = None
        simulation.entry_animations_enabled = False
        simulation.audio = RecordingAudioService()
        simulation.rng = mock.Mock()
        simulation.rng.randint.return_value = 0
        simulation.rng.uniform.return_value = 0

        simulation._complete_ready_rebirths()

        self.assertTrue(first.currently_alive)
        self.assertEqual(first.rebirth_count, 2)
        self.assertEqual(set(simulation.aftermath.pending_rebirths), {second})


if __name__ == "__main__":
    unittest.main()
