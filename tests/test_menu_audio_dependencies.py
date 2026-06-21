import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.UI import ui, ui_button


class RecordingSoundManager:
    def __init__(self, trace=None):
        self.played = []
        self.trace = trace

    def play_sound(self, sound_name):
        self.played.append(sound_name)
        if self.trace is not None:
            self.trace.append(("sound", sound_name))


def click(position=(5, 5)):
    return pygame.event.Event(
        pygame.MOUSEBUTTONDOWN,
        {"button": 1, "pos": position},
    )


class ButtonMenuAudioTests(unittest.TestCase):
    def test_button_plays_passed_manager_before_callback(self):
        trace = []
        sound_manager = RecordingSoundManager(trace)
        button = ui_button.Button(
            0, 0, 10, 10, "Click", lambda: trace.append(("callback", None))
        )

        button.handle_event(click(), sound_manager)

        self.assertEqual(sound_manager.played, ["menu"])
        self.assertEqual(
            trace, [("sound", "menu"), ("callback", None)]
        )

    def test_buttons_keep_passed_sound_managers_isolated(self):
        first_manager = RecordingSoundManager()
        second_manager = RecordingSoundManager()
        button = ui_button.Button(0, 0, 10, 10, "Click", lambda: None)

        button.handle_event(click(), first_manager)

        self.assertEqual(first_manager.played, ["menu"])
        self.assertEqual(second_manager.played, [])
        self.assertFalse(hasattr(ui, "sound_manager"))


class ToggleButtonMenuAudioTests(unittest.TestCase):
    def test_toggle_plays_passed_manager_before_state_callback(self):
        trace = []
        sound_manager = RecordingSoundManager(trace)
        toggle = ui_button.ToggleButton(
            0,
            0,
            10,
            10,
            "Toggle",
            callback=lambda state: trace.append(("callback", state)),
        )

        toggle.handle_event(click(), sound_manager)

        self.assertTrue(toggle.value)
        self.assertEqual(sound_manager.played, ["menu"])
        self.assertEqual(
            trace, [("sound", "menu"), ("callback", True)]
        )

    def test_toggle_without_manager_still_changes_state(self):
        states = []
        toggle = ui_button.ToggleButton(
            0, 0, 10, 10, "Toggle", callback=states.append
        )

        toggle.handle_event(click(), None)

        self.assertTrue(toggle.value)
        self.assertEqual(states, [True])


if __name__ == "__main__":
    unittest.main()
