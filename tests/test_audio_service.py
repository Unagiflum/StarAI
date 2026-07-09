import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.audio import (
    DisplayGatedAudioService,
    NullAudioService,
    PygameAudioService,
    RecordingAudioService,
    initialize_pygame_audio,
)
from src.Battle.battle import BattleSimulation
from src.Battle.effects import BattleEffect
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.registry import create_ship


class AudioServiceIntegrationTests(unittest.TestCase):
    def test_display_gated_audio_only_plays_when_enabled(self):
        enabled = [False]
        base = RecordingAudioService()
        audio = DisplayGatedAudioService(base, lambda: enabled[0])

        effect = audio.load_effect("test.wav")
        audio.start_battle_music()
        effect.play()
        enabled[0] = True
        audio.start_battle_music()
        effect.play()
        enabled[0] = False
        audio.stop_music()

        self.assertEqual(
            [operation[0] for operation in base.operations],
            ["start_battle_music", "play_effect", "stop_music"],
        )

    def test_audio_initialization_falls_back_to_null_service(self):
        with mock.patch(
            "pygame.mixer.init", side_effect=pygame.error("no audio device")
        ):
            service = initialize_pygame_audio()

        self.assertIsInstance(service, NullAudioService)

    def test_audio_initialization_returns_pygame_service_on_success(self):
        resources = object()
        with (
            mock.patch("pygame.mixer.init") as mixer_init,
            mock.patch("pygame.mixer.set_num_channels") as set_num_channels,
        ):
            service = initialize_pygame_audio(resources)

        mixer_init.assert_called_once_with()
        set_num_channels.assert_called_once_with(32)
        self.assertIsInstance(service, PygameAudioService)
        self.assertIs(service.resources, resources)

    def test_disabled_headless_simulation_never_touches_pygame_audio(self):
        audio = NullAudioService()
        with (
            mock.patch("pygame.mixer.init") as mixer_init,
            mock.patch("pygame.mixer.Sound") as sound,
            mock.patch("pygame.mixer.music.load") as music_load,
            mock.patch("pygame.mixer.music.play") as music_play,
            mock.patch("pygame.mixer.music.set_volume") as music_volume,
            mock.patch("pygame.mixer.music.stop") as music_stop,
        ):
            first = create_ship("Earthling", 1, audio_service=audio)
            second = create_ship("Earthling", 2, audio_service=audio)
            simulation = BattleSimulation(
                None, first, second, sound_enabled=False, audio_service=audio
            )
            simulation.step(actions={1: {"action1": True}, 2: {}})
            simulation.select_next_round(None)

        mixer_init.assert_not_called()
        sound.assert_not_called()
        music_load.assert_not_called()
        music_play.assert_not_called()
        music_volume.assert_not_called()
        music_stop.assert_not_called()

    def test_simulations_keep_audio_enablement_and_effects_isolated(self):
        first_audio = RecordingAudioService()
        second_audio = RecordingAudioService()
        original_ability_flag = Ability.sound_enabled
        original_effect_flag = BattleEffect.sound_enabled

        first = BattleSimulation(
            None,
            create_ship("Earthling", 1, audio_service=first_audio),
            create_ship("Earthling", 2, audio_service=first_audio),
            audio_service=first_audio,
        )
        second = BattleSimulation(
            None,
            create_ship("Earthling", 1, audio_service=second_audio),
            create_ship("Earthling", 2, audio_service=second_audio),
            audio_service=second_audio,
        )
        first_audio.operations.clear()
        second_audio.operations.clear()

        first.step(actions={1: {"action1": True}, 2: {}})
        self.assertTrue(any(op[0] == "play_effect" for op in first_audio.operations))
        self.assertEqual(second_audio.operations, [])

        second.step(actions={1: {"action1": True}, 2: {}})
        self.assertTrue(any(op[0] == "play_effect" for op in second_audio.operations))
        self.assertIs(first.player1.audio_service, first_audio)
        self.assertIs(second.player1.audio_service, second_audio)
        self.assertEqual(Ability.sound_enabled, original_ability_flag)
        self.assertEqual(BattleEffect.sound_enabled, original_effect_flag)

    def test_disabled_recording_service_does_not_inherit_enabled_state(self):
        enabled = RecordingAudioService(enabled=True)
        disabled = RecordingAudioService(enabled=False)

        enabled_simulation = BattleSimulation(
            None,
            create_ship("Earthling", 1, audio_service=enabled),
            create_ship("Earthling", 2, audio_service=enabled),
            audio_service=enabled,
        )
        disabled_simulation = BattleSimulation(
            None,
            create_ship("Earthling", 1, audio_service=disabled),
            create_ship("Earthling", 2, audio_service=disabled),
            audio_service=disabled,
        )

        self.assertTrue(enabled_simulation.sound_enabled)
        self.assertFalse(disabled_simulation.sound_enabled)
        self.assertEqual(enabled.operations, [("start_battle_music",)])
        self.assertEqual(disabled.operations, [])


if __name__ == "__main__":
    unittest.main()
