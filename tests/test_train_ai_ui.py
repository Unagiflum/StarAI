import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

pygame.font.init()

import src.const as const
from src.Menus.train_ai import (
    EPSILON_VALUES,
    GAMMA_VALUES,
    LEARNING_RATE_VALUES,
    MATCH_TIME_LIMIT_VALUES,
    MOVEMENT_BEHAVIORS,
    REWARD_LABELS,
    REWARD_VALUES,
    ROUNDS_PER_BATCH_VALUES,
    BATCH_GROUPING_VALUES,
    RewardSlider,
    TrainingBatchLogBox,
    TrainingUIState,
    _display_off_console_lines,
    _draw_training_huds,
    _draw_training_battle,
    _progress_for_model_update,
    _set_slider_value,
    training_config_from_state,
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
        self.assertEqual(state.batch_grouping, 250)
        self.assertEqual(state.match_time_limit, 2400)
        self.assertEqual(state.learning_rate, 0.001)
        self.assertEqual(state.epsilon, 0.1)
        self.assertEqual(state.gamma, 0.99)
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

    def test_saved_reward_value_loads_without_position_api(self):
        slider = RewardSlider((0, 0, 550, 40), "Reward")

        self.assertTrue(_set_slider_value(slider, 2.56))

        self.assertEqual(slider.value, 2.56)


class RegimenSliderTests(unittest.TestCase):
    def test_regimen_sliders_expose_the_requested_discrete_values(self):
        self.assertEqual(ROUNDS_PER_BATCH_VALUES, (1, 2, 5, 10, 20, 50))
        self.assertEqual(BATCH_GROUPING_VALUES, (50, 100, 250, 500, 1000))
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
        self.assertEqual(GAMMA_VALUES, (0.9, 0.95, 0.98, 0.99, 0.995, 0.999))


class DisabledCheckboxTests(unittest.TestCase):
    def test_disabled_checkbox_does_not_toggle(self):
        checkbox = Checkbox(0, 0, 200, 40, "Disabled")
        checkbox.enabled = False
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (10, 10)}
        )

        checkbox.handle_event(event)

        self.assertFalse(checkbox.value)


class TrainingConfigAdapterTests(unittest.TestCase):
    def test_training_config_from_state_carries_ui_values(self):
        state = TrainingUIState(selected_ship="Earthling")
        state.rewards["Kill enemy"] = 2.56
        state.movement_behaviors = {"Move forward continuously"}
        state.rounds_per_batch = 2
        state.gamma = 0.995
        state.hidden_layer_size = 64
        state.hidden_layer_count = 1

        config = training_config_from_state(state)

        self.assertEqual(config.trainee_ship, "Earthling")
        self.assertEqual(config.reward_weights["Kill enemy"], 2.56)
        self.assertEqual(config.movement_behaviors, frozenset({"Move forward continuously"}))
        self.assertEqual(config.rounds_per_batch, 2)
        self.assertEqual(config.gamma, 0.995)
        self.assertEqual(config.hidden_layer_width, 64)
        self.assertEqual(config.hidden_layer_count, 1)

    def test_non_reset_settings_update_preserves_existing_progress(self):
        progress = _progress_for_model_update(
            {"progress": {"completed_batches": 42}},
            reset_checkpoint=False,
        )

        self.assertEqual(progress, {"completed_batches": 42})

    def test_checkpoint_reset_clears_existing_progress(self):
        progress = _progress_for_model_update(
            {"progress": {"completed_batches": 42}},
            reset_checkpoint=True,
        )

        self.assertEqual(progress, {"completed_batches": 0})


class TrainingBatchLogBoxTests(unittest.TestCase):
    def test_log_box_keeps_selectable_text(self):
        box = TrainingBatchLogBox()
        box.set_lines(["first", "second", "third"])
        box.selection_anchor = 0
        box.selection_focus = 1

        self.assertEqual(box.selected_text, "first\nsecond")


class TrainingConsoleTests(unittest.TestCase):
    def _status(self, **overrides):
        values = {
            "running": True,
            "stopping": False,
            "completed_batches": 2,
            "current_round": 3,
            "total_rounds": 25,
            "current_opponent": "Earthling",
            "current_frame": 42,
            "replay_size": 99,
            "last_action_exploratory": True,
            "weighted_total_return": 12.5,
            "recent_loss": None,
            "component_totals": {"Kill enemy": 2.0},
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_display_off_console_includes_batch_logs_before_live_status(self):
        status = self._status()

        lines = _display_off_console_lines(status, ("Batch      1 | summary",))

        self.assertIn("Current batch", lines)
        self.assertIn("Round:    3/  25", lines)
        self.assertEqual(lines[0], "Completed batches")
        self.assertLess(
            lines.index("Batch      1 | summary"),
            lines.index("Current batch"),
        )
        self.assertIn("Batch:      3", lines)
        self.assertIn("Frame:       42 | Replay:     99", lines)
        self.assertIn("Action: explore | Return:     12.50", lines)
        self.assertIn("Loss:          -", lines)

    def test_display_off_console_keeps_current_batch_block_height_stable(self):
        lines_with_component = _display_off_console_lines(
            self._status(), ("Batch      1 | summary",)
        )
        lines_without_components = _display_off_console_lines(
            self._status(component_totals={}), ("Batch      1 | summary",)
        )

        self.assertEqual(len(lines_with_component), len(lines_without_components))
        placeholder_rows = lines_without_components[-6:]
        self.assertEqual(len(set(placeholder_rows)), 1)
        self.assertTrue(placeholder_rows[0].startswith("-"))
        self.assertTrue(placeholder_rows[0].endswith(":        -"))


class TrainingBatchLogBoxTests(unittest.TestCase):
    def _box_with_lines(self, line_count=20):
        rect = pygame.Rect(0, 0, 300, 100)
        surface = pygame.Surface(rect.size)
        font = pygame.font.Font(None, 20)
        box = TrainingBatchLogBox()
        box.set_lines(tuple(f"line {index}" for index in range(line_count)))
        box.draw(surface, rect, font)
        return box, rect, font

    def test_mousewheel_scrolls_up_from_bottom_page(self):
        box, rect, font = self._box_with_lines()
        bottom_line = box.scroll_line

        with mock.patch("pygame.mouse.get_pos", return_value=rect.center):
            box.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, {"y": 1}), rect, font)

        self.assertLess(box.scroll_line, bottom_line)

    def test_legacy_wheel_scrolls_up_from_bottom_page(self):
        box, rect, font = self._box_with_lines()
        bottom_line = box.scroll_line

        box.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 4, "pos": rect.center},
            ),
            rect,
            font,
        )

        self.assertLess(box.scroll_line, bottom_line)

    def test_new_lines_do_not_force_scroll_when_user_reading_history(self):
        box, rect, font = self._box_with_lines()

        with mock.patch("pygame.mouse.get_pos", return_value=rect.center):
            box.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, {"y": 1}), rect, font)
        scrolled_line = box.scroll_line

        box.set_lines(tuple(f"line {index}" for index in range(24)))

        self.assertEqual(box.scroll_line, scrolled_line)

    def test_new_lines_follow_when_view_is_at_bottom(self):
        box, rect, font = self._box_with_lines()

        box.set_lines(tuple(f"line {index}" for index in range(24)))

        self.assertEqual(box.scroll_line, 24 - box.visible_count)


class TrainingBattleDisplayTests(unittest.TestCase):
    def test_display_on_draws_cropped_battle_surface_into_arena(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        screen.fill((0, 0, 255))
        battle_surface = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        rect = training_layout().arena_rect
        status = SimpleNamespace(
            battle_view={
                "game_objects": (),
                "border_rect": pygame.Rect(
                    const.SCREEN_LEFT,
                    0,
                    const.SCREEN_HEIGHT,
                    const.SCREEN_HEIGHT,
                ),
                "border_color": (50, 50, 50),
                "frame_id": 1,
                "original_ships": (),
                "camera_targets": (),
                "entry_state": None,
            }
        )

        def fake_draw(surface, *args, **kwargs):
            surface.fill((255, 0, 0))
            source = pygame.Rect(
                const.SCREEN_LEFT,
                0,
                const.SCREEN_HEIGHT,
                const.SCREEN_HEIGHT,
            )
            pygame.draw.rect(surface, (0, 255, 0), source)

        with (
            mock.patch("src.Menus.train_ai.draw_battle_arena", side_effect=fake_draw),
            mock.patch("pygame.display.flip") as display_flip,
        ):
            _draw_training_battle(screen, rect, status, battle_surface, object())

        self.assertEqual(screen.get_at((rect.left + 10, rect.top + 10))[:3], (0, 255, 0))
        self.assertEqual(screen.get_at((rect.left - 10, rect.top + 10))[:3], (0, 0, 255))
        display_flip.assert_not_called()

    def test_display_on_hud_draws_live_ship_values(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        screen.fill((0, 0, 0))
        sprite = pygame.Surface((24, 24), pygame.SRCALPHA)
        sprite.fill((255, 255, 255))
        trainee = SimpleNamespace(
            player=1,
            name="Earthling",
            sprites=(sprite,),
            set_sprite=lambda: sprite,
            current_hp=8,
            max_hp=10,
            current_energy=6,
            max_energy=12,
        )
        opponent = SimpleNamespace(
            player=2,
            name="Chenjesu",
            sprites=(sprite,),
            set_sprite=lambda: sprite,
            current_hp=4,
            max_hp=10,
            current_energy=2,
            max_energy=8,
        )
        status = SimpleNamespace(
            battle_view={
                "original_ships": (trainee, opponent),
                "game_objects": (),
            }
        )

        _draw_training_huds(
            screen,
            training_layout().hud_rects,
            status,
            pygame.font.SysFont(None, 24),
            pygame.font.SysFont(None, 18),
        )

        first, second = training_layout().hud_rects
        self.assertEqual(screen.get_at((first.left + 1, first.top + 1))[:3], const.P1_COLOR)
        self.assertEqual(screen.get_at((second.left + 1, second.top + 1))[:3], const.P2_COLOR)


if __name__ == "__main__":
    unittest.main()
