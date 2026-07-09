import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

import src.const as const
from src.Menus.train_ai import (
    EPSILON_VALUES,
    LEARNING_RATE_VALUES,
    MATCH_TIME_LIMIT_VALUES,
    MOVEMENT_BEHAVIORS,
    REWARD_LABELS,
    REWARD_VALUES,
    ROUNDS_PER_BATCH_VALUES,
    RewardSlider,
    TrainingUIState,
    training_layout,
)
from src.UI.ui_button import Checkbox


class TrainingUIStateTests(unittest.TestCase):
    def test_defaults_match_training_specification(self):
        state = TrainingUIState()

        self.assertEqual(state.selected_slot, 1)
        self.assertEqual(state.slot_labels, ["", "", "", ""])
        self.assertEqual(state.rewards, {label: 0.0 for label in REWARD_LABELS})
        self.assertEqual(state.opponent_mode, "simple")
        self.assertEqual(state.movement_behaviors, set())
        self.assertEqual(state.turning_behavior, "none")
        self.assertEqual(state.rounds_per_batch, 10)
        self.assertEqual(state.match_time_limit, 2400)
        self.assertEqual(state.learning_rate, 0.001)
        self.assertEqual(state.epsilon, 0.1)
        self.assertFalse(state.display_on)
        self.assertFalse(state.running)

    def test_all_existing_ai_mode_disables_simple_behavior_controls(self):
        state = TrainingUIState()
        self.assertTrue(state.simple_behavior_controls_enabled)

        state.opponent_mode = "all"

        self.assertFalse(state.simple_behavior_controls_enabled)

    def test_behavior_choices_are_the_three_independent_checkboxes(self):
        self.assertEqual(
            MOVEMENT_BEHAVIORS,
            (
                "Move forward continuously",
                "Hold A1 continuously",
                "Hold A2 continuously",
            ),
        )


class TrainingLayoutTests(unittest.TestCase):
    def test_arena_uses_the_full_height_at_the_right_edge(self):
        layout = training_layout()

        self.assertEqual(layout.arena_rect.size, (const.SCREEN_HEIGHT,) * 2)
        self.assertEqual(layout.arena_rect.right, const.SCREEN_WIDTH)
        self.assertEqual(layout.control_rect.right, layout.arena_rect.left)

    def test_hud_placeholders_are_equal_width_and_side_by_side(self):
        first, second = training_layout().hud_rects

        self.assertEqual(first.width, second.width)
        self.assertEqual(first.y, second.y)
        self.assertLess(first.right, second.left)


class RewardSliderTests(unittest.TestCase):
    def test_reward_scale_has_all_23_doubling_values(self):
        self.assertEqual(len(REWARD_VALUES), 23)
        self.assertEqual(REWARD_VALUES[0], -10.24)
        self.assertEqual(REWARD_VALUES[10:13], (-0.01, 0.0, 0.01))
        self.assertEqual(REWARD_VALUES[-1], 10.24)

    def test_slider_snaps_to_discrete_reward_values(self):
        slider = RewardSlider((0, 0, 550, 40), "Reward")

        slider.set_from_x(slider.line_rect.left)
        self.assertEqual(slider.value, -10.24)
        slider.set_from_x(slider.line_rect.centerx)
        self.assertEqual(slider.value, 0.0)
        slider.set_from_x(slider.line_rect.right)
        self.assertEqual(slider.value, 10.24)


class RegimenSliderTests(unittest.TestCase):
    def test_regimen_sliders_expose_the_requested_discrete_values(self):
        self.assertEqual(ROUNDS_PER_BATCH_VALUES, (1, 2, 5, 10, 20, 50))
        self.assertEqual(
            MATCH_TIME_LIMIT_VALUES,
            (240, 480, 1200, 2400, 4800, 12000),
        )
        self.assertEqual(
            LEARNING_RATE_VALUES,
            (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01),
        )
        self.assertEqual(
            EPSILON_VALUES,
            (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.1, 0.2, 0.5),
        )


class DisabledCheckboxTests(unittest.TestCase):
    def test_disabled_checkbox_does_not_toggle(self):
        checkbox = Checkbox(0, 0, 200, 40, "Disabled")
        checkbox.enabled = False
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (10, 10)}
        )

        checkbox.handle_event(event)

        self.assertFalse(checkbox.value)


if __name__ == "__main__":
    unittest.main()
