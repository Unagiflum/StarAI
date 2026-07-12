import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

pygame.font.init()

import src.const as const
import src.Menus.train_ai as train_ai
from src.UI import ui, ui_slider
from src.Menus.train_ai import (
    ACTION_TOP,
    EPSILON_DECAY_VALUES,
    EPSILON_FRAME_SPAN_VALUES,
    EPSILON_VALUES,
    DISPLAY_TOP,
    FOOTER_CONTROL_HEIGHT,
    GAMMA_VALUES,
    INSTANCE_CONTROL_HEIGHT,
    INSTANCE_ADD_WIDTH,
    INSTANCE_CLOSE_WIDTH,
    INSTANCE_GAP,
    INSTANCE_POSITION_WIDTH,
    INSTANCE_RUNNING_WIDTH,
    INSTANCE_SEPARATOR_HEIGHT,
    INSTANCE_TOP,
    LEARNING_RATE_VALUES,
    MATCH_TIME_LIMIT_VALUES,
    MINIBATCH_SIZE_VALUES,
    REWARD_LABELS,
    REWARD_VALUES,
    REPLAY_UPDATES_PER_BATCH_VALUES,
    ROUNDS_PER_BATCH_VALUES,
    BATCH_GROUPING_VALUES,
    AI_OPPONENT_PERCENT_VALUES,
    SIMPLE_ACTIVITY_VALUES,
    TRAINING_INSTANCE_SOFT_MAX,
    TRAINING_INSTANCE_SUPPORTED_MAX,
    RewardSlider,
    SliderRow,
    TRAINING_HUD_HEIGHT,
    TrainingBatchLogBox,
    TrainingInstanceManager,
    TrainingUIState,
    UI_TOP_MARGIN,
    _clear_reset_model_artifacts,
    _display_off_console_lines,
    _draw_training_huds,
    _draw_training_battle,
    _epsilon_for_model_update,
    _format_short_count,
    _format_training_duration,
    _instance_row_parts,
    _instance_status_text,
    _training_settings_match,
    _progress_for_model_update,
    _set_slider_value,
    training_config_from_state,
    training_layout,
)
from src.Battle.battle_draw import (
    BAR_WIDTH,
    BattleDrawController,
    HUD_BOTTOM_PADDING,
    MARINE_REGION_HEIGHT,
    RenderSnapshot,
    VIEWPORT_MARGIN,
    VIEWPORT_SIZE,
)
from src.UI.ui_button import Checkbox


class TrainingUIStateTests(unittest.TestCase):
    def test_defaults_match_training_specification(self):
        state = TrainingUIState()

        self.assertEqual(state.selected_slot, 1)
        self.assertEqual(state.slot_labels, ["", "", "", ""])
        self.assertEqual(state.rewards, {label: 0.0 for label in REWARD_LABELS})
        self.assertEqual(state.opponent_mode, "simple")
        self.assertEqual(state.ai_opponent_chance, 0.0)
        self.assertEqual(state.forward_activity, 0.0)
        self.assertEqual(state.a1_activity, 0.0)
        self.assertEqual(state.a2_activity, 0.0)
        self.assertEqual(state.face_opponent_activity, 0.0)
        self.assertEqual(state.rounds_per_batch, 1)
        self.assertEqual(state.batch_grouping, 50)
        self.assertEqual(state.match_time_limit, 1200)
        self.assertEqual(state.learning_rate, 0.0001)
        self.assertEqual(state.starting_epsilon, 0.1)
        self.assertEqual(state.current_epsilon, 0.1)
        self.assertEqual(state.epsilon_decay, 0.998)
        self.assertEqual(state.epsilon_frame_span, 8)
        self.assertEqual(state.gamma, 0.99)
        self.assertEqual(state.minibatch_size, 64)
        self.assertEqual(state.replay_updates_per_batch, 500)
        self.assertFalse(state.display_on)
        self.assertFalse(state.running)

    def test_ai_opponent_mode_keeps_simple_behavior_controls_available(self):
        state = TrainingUIState()
        self.assertTrue(state.simple_behavior_controls_enabled)

        state.opponent_mode = "all"

        self.assertTrue(state.simple_behavior_controls_enabled)

        state.running = True

        self.assertFalse(state.simple_behavior_controls_enabled)

    def test_simple_activity_values_are_five_percent_steps(self):
        self.assertEqual(
            SIMPLE_ACTIVITY_VALUES,
            tuple(float(value) for value in range(0, 101, 5)),
        )

    def test_ai_opponent_percent_values_are_five_percent_steps(self):
        self.assertEqual(
            AI_OPPONENT_PERCENT_VALUES,
            tuple(float(value) for value in range(0, 101, 5)),
        )


class TrainingInstanceManagerTests(unittest.TestCase):
    class FakeSession:
        def __init__(self, *, running=True, stopping=False, error=""):
            self.status = SimpleNamespace(
                running=running,
                stopping=stopping,
                error=error,
            )
            self.stop_requested = False
            self.display_events = []

        def request_stop(self):
            self.stop_requested = True
            self.status.stopping = True

        def set_display_on(self, enabled):
            self.display_events.append(enabled)

    def test_default_manager_creates_one_active_instance(self):
        manager = TrainingInstanceManager()

        self.assertEqual(len(manager.instances), 1)
        self.assertEqual(manager.active_instance.instance_id, 1)
        self.assertEqual(manager.active_instance.label, "Instance 1")
        self.assertIs(manager.active_state, manager.active_instance.state)
        self.assertIsNone(manager.active_session)

    def test_active_session_and_continuity_are_instance_scoped(self):
        manager = TrainingInstanceManager()
        session = SimpleNamespace()

        manager.set_active_session(session)

        self.assertIs(manager.active_session, session)

        manager.active_instance.last_running = True
        manager.clear_active_session_continuity()

        self.assertIsNone(manager.active_session)
        self.assertFalse(manager.active_instance.last_running)

    def test_add_instance_makes_new_default_instance_active(self):
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"

        instance = manager.add_instance()

        self.assertEqual(len(manager.instances), 2)
        self.assertIs(manager.active_instance, instance)
        self.assertEqual(instance.label, "Instance 2")
        self.assertIsNone(manager.active_state.selected_ship)
        self.assertEqual(manager.active_position_text(), "02/02")
        self.assertEqual(manager.instances[0].state.selected_ship, "Earthling")

    def test_select_instance_switches_active_state(self):
        manager = TrainingInstanceManager()
        first_id = manager.active_instance.instance_id
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"

        manager.select_instance(first_id)

        self.assertIs(manager.active_instance, manager.instances[0])
        self.assertIsNone(manager.active_state.selected_ship)

        manager.select_instance(second.instance_id)

        self.assertEqual(manager.active_state.selected_ship, "Androsynth")

    def test_remove_active_stopped_instance_selects_neighbor(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        second = manager.add_instance()

        self.assertTrue(manager.remove_active_stopped_instance())

        self.assertEqual(manager.instances, [first])
        self.assertIs(manager.active_instance, first)
        self.assertEqual(manager.active_position_text(), "01/01")
        self.assertNotIn(second, manager.instances)

    def test_manager_selects_through_supported_instance_count(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SUPPORTED_MAX:
            manager.add_instance()

        self.assertEqual(len(manager.instances), 25)
        for index, instance in enumerate(manager.instances, start=1):
            manager.select_instance(instance.instance_id)
            self.assertIs(manager.active_instance, instance)
            self.assertEqual(manager.active_position_text(), f"{index:02d}/25")

    def test_manager_enforces_supported_instance_count(self):
        manager = TrainingInstanceManager()
        while manager.can_add_instance():
            manager.add_instance()

        self.assertFalse(manager.can_add_instance())
        with self.assertRaises(ValueError):
            manager.add_instance()

    def test_manager_reports_soft_limit_confirmation_need(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SOFT_MAX:
            self.assertFalse(manager.add_requires_confirmation())
            manager.add_instance()

        self.assertTrue(manager.add_requires_confirmation())

    def test_instance_labels_and_rows_remain_unique_for_selection(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SUPPORTED_MAX:
            manager.add_instance()

        labels = [instance.label for instance in manager.instances]
        row_prefixes = [
            _instance_row_parts(index, instance)[0]
            for index, instance in enumerate(manager.instances, start=1)
        ]

        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(row_prefixes), len(set(row_prefixes)))

    def test_position_and_running_count_are_zero_padded(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.session = self.FakeSession(running=False, stopping=True)
        third = manager.add_instance()
        third.session = self.FakeSession(running=False)

        self.assertEqual(manager.active_position_text(), "03/03")
        self.assertEqual(manager.running_count(), 2)
        self.assertEqual(manager.running_count_text(), "02>")

    def test_remove_active_instance_refuses_last_or_running_instance(self):
        manager = TrainingInstanceManager()

        self.assertFalse(manager.remove_active_stopped_instance())

        manager.add_instance()
        manager.active_instance.session = SimpleNamespace(
            status=SimpleNamespace(running=True, stopping=False)
        )

        self.assertFalse(manager.remove_active_stopped_instance())
        self.assertEqual(len(manager.instances), 2)

    def test_instance_row_uses_ship_slot_and_status(self):
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 1
        manager.active_instance.session = SimpleNamespace(
            status=SimpleNamespace(running=True, stopping=False, error=None)
        )

        prefix, status = _instance_row_parts(1, manager.active_instance)

        self.assertEqual(status, "Running")
        self.assertIn("01]", prefix)
        self.assertIn("Earthling-01", prefix)
        self.assertEqual(_instance_status_text(manager.active_instance), "Running")

    def test_writer_reservation_blocks_same_running_slot(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        self.assertTrue(manager.reserve_writer(first, "Earthling", 1))

        second = manager.add_instance()
        second.session = self.FakeSession(running=True)

        self.assertFalse(manager.reserve_writer(second, "Earthling", 1))
        self.assertEqual(manager.writer_owner("Earthling", 1), first.instance_id)

    def test_writer_reservation_allows_distinct_running_slots(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        self.assertTrue(manager.reserve_writer(first, "Earthling", 1))

        second = manager.add_instance()
        second.session = self.FakeSession(running=True)

        self.assertTrue(manager.reserve_writer(second, "Androsynth", 1))
        self.assertEqual(manager.writer_owner("Earthling", 1), first.instance_id)
        self.assertEqual(manager.writer_owner("Androsynth", 1), second.instance_id)

    def test_stopped_session_releases_writer_reservation(self):
        manager = TrainingInstanceManager()
        instance = manager.active_instance
        instance.session = self.FakeSession(running=True)
        self.assertTrue(manager.reserve_writer(instance, "Earthling", 1))

        instance.session.status.running = False
        manager.release_stopped_writers()

        self.assertIsNone(manager.writer_owner("Earthling", 1))
        self.assertIsNone(instance.writer_key)

    def test_stop_active_instance_leaves_other_running(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        manager.select_instance(first.instance_id)

        manager.request_stop_active()

        self.assertTrue(first.session.stop_requested)
        self.assertFalse(second.session.stop_requested)

    def test_enabling_active_display_disables_other_instances(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        first.session.display_events.clear()
        first.state.display_on = True

        manager.set_active_display(True)

        self.assertFalse(first.state.display_on)
        self.assertTrue(second.state.display_on)
        self.assertEqual(first.session.display_events, [False])
        self.assertEqual(second.session.display_events, [True])

    def test_select_instance_turns_display_off_for_old_and_new_active(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        first.session.display_events.clear()
        first.state.display_on = True
        second.state.display_on = True

        manager.select_instance(first.instance_id)

        self.assertIs(manager.active_instance, first)
        self.assertFalse(first.state.display_on)
        self.assertFalse(second.state.display_on)
        self.assertEqual(first.session.display_events, [False])
        self.assertEqual(second.session.display_events, [False])

    def test_back_action_stops_all_when_non_active_instance_is_running(self):
        manager = TrainingInstanceManager()
        manager.active_instance.session = self.FakeSession(running=True)

        self.assertEqual(manager.back_action(), "active_running")

        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        manager.select_instance(manager.instances[0].instance_id)

        self.assertEqual(manager.back_action(), "stop_all")
        self.assertTrue(manager.background_instances_running())

    def test_stop_all_running_requests_stop_without_touching_stopped_instances(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.display_on = True
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.state.display_on = True
        second.session = self.FakeSession(running=False)
        first.session.display_events.clear()

        manager.request_stop_all_running()

        self.assertTrue(first.session.stop_requested)
        self.assertFalse(second.session.stop_requested)
        self.assertFalse(first.state.display_on)
        self.assertEqual(first.session.display_events, [False])

    def test_stop_all_running_includes_selected_instance(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        manager.select_instance(first.instance_id)

        manager.request_stop_all_running()

        self.assertTrue(first.session.stop_requested)
        self.assertTrue(second.session.stop_requested)

    def test_close_running_active_instance_disables_display_before_stop(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.display_on = True
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        manager.select_instance(first.instance_id)
        first.session.display_events.clear()

        result = manager.request_close_active_instance()

        self.assertEqual(result, "pending")
        self.assertTrue(first.pending_removal)
        self.assertEqual(first.session.display_events, [False])
        self.assertTrue(first.session.stop_requested)
        self.assertIs(manager.active_instance, second)

    def test_close_running_instance_remains_graceful_at_supported_count(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SUPPORTED_MAX:
            manager.add_instance()
        first = manager.instances[0]
        second = manager.instances[1]
        first.session = self.FakeSession(running=True)
        first.state.display_on = True
        manager.select_instance(first.instance_id)
        first.session.display_events.clear()

        result = manager.request_close_active_instance()

        self.assertEqual(result, "pending")
        self.assertEqual(len(manager.instances), TRAINING_INSTANCE_SUPPORTED_MAX)
        self.assertTrue(first.pending_removal)
        self.assertTrue(first.session.stop_requested)
        self.assertEqual(first.session.display_events, [False])
        self.assertIs(manager.active_instance, second)

    def test_close_last_running_instance_creates_replacement(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)

        result = manager.request_close_active_instance()

        self.assertEqual(result, "pending")
        self.assertEqual(len(manager.instances), 2)
        self.assertTrue(first.pending_removal)
        self.assertIsNot(manager.active_instance, first)

    def test_cleanup_removes_stopped_pending_instance(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        manager.select_instance(first.instance_id)
        manager.request_close_active_instance()

        first.session.status.running = False
        first.session.status.stopping = False
        manager.cleanup_stopped_pending_removals()

        self.assertEqual(manager.instances, [second])
        self.assertIs(manager.active_instance, second)


class TrainingLayoutTests(unittest.TestCase):
    def test_instance_strip_sits_above_tabs_and_content(self):
        layout = training_layout()
        instance_rect = pygame.Rect(
            8,
            INSTANCE_TOP,
            layout.control_rect.width - 16,
            INSTANCE_CONTROL_HEIGHT,
        )
        tab_rect = pygame.Rect(8, UI_TOP_MARGIN, layout.control_rect.width - 16, 40)

        self.assertLessEqual(instance_rect.bottom, tab_rect.top)
        self.assertLessEqual(tab_rect.bottom, layout.content_rect.top)
        self.assertFalse(instance_rect.colliderect(layout.content_rect))
        self.assertGreaterEqual(UI_TOP_MARGIN - instance_rect.bottom, INSTANCE_SEPARATOR_HEIGHT)

    def test_instance_buttons_leave_room_for_dropdown_label(self):
        layout = training_layout()
        dropdown_width = (
            layout.control_rect.width
            - 2 * 8
            - INSTANCE_POSITION_WIDTH
            - INSTANCE_RUNNING_WIDTH
            - INSTANCE_CLOSE_WIDTH
            - INSTANCE_ADD_WIDTH
            - 4 * INSTANCE_GAP
        )

        self.assertGreaterEqual(dropdown_width, 320)

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

    def test_hud_rectangles_hold_full_shared_hud_and_sit_flush_bottom(self):
        expected_height = MARINE_REGION_HEIGHT + VIEWPORT_SIZE + HUD_BOTTOM_PADDING
        layout = training_layout()

        self.assertEqual(TRAINING_HUD_HEIGHT, expected_height)
        for hud_rect in layout.hud_rects:
            self.assertEqual(hud_rect.height, expected_height)
            self.assertEqual(hud_rect.bottom, const.SCREEN_HEIGHT)
            self.assertGreaterEqual(
                hud_rect.width,
                BAR_WIDTH * 2 + VIEWPORT_SIZE + 2 * VIEWPORT_MARGIN,
            )

    def test_bottom_controls_do_not_overlap_full_size_huds(self):
        layout = training_layout()
        display_rect = pygame.Rect(
            8,
            DISPLAY_TOP,
            layout.control_rect.width - 16,
            FOOTER_CONTROL_HEIGHT,
        )
        action_gap = 10
        action_width = (layout.control_rect.width - 16 - action_gap) // 2
        start_rect = pygame.Rect(
            8,
            ACTION_TOP,
            action_width,
            FOOTER_CONTROL_HEIGHT,
        )
        back_rect = pygame.Rect(
            8 + action_width + action_gap,
            ACTION_TOP,
            action_width,
            FOOTER_CONTROL_HEIGHT,
        )

        for control_rect in (display_rect, start_rect, back_rect):
            for hud_rect in layout.hud_rects:
                self.assertFalse(control_rect.colliderect(hud_rect))

    def test_display_off_log_region_remains_the_training_arena(self):
        layout = training_layout()

        self.assertEqual(layout.arena_rect.size, (const.SCREEN_HEIGHT,) * 2)
        self.assertEqual(layout.arena_rect.right, const.SCREEN_WIDTH)
        for hud_rect in layout.hud_rects:
            self.assertFalse(layout.arena_rect.colliderect(hud_rect))


class RewardSliderTests(unittest.TestCase):
    def test_reward_scale_has_all_27_doubling_values(self):
        self.assertEqual(len(REWARD_VALUES), 27)
        self.assertEqual(REWARD_VALUES[0], -40.96)
        self.assertEqual(REWARD_VALUES[12:15], (-0.01, 0.0, 0.01))
        self.assertEqual(REWARD_VALUES[-1], 40.96)

    def test_slider_snaps_to_discrete_reward_values(self):
        slider = RewardSlider((0, 0, 550, 40), "Reward")

        slider.set_from_x(slider.line_rect.left)
        self.assertEqual(slider.value, -40.96)
        slider.set_from_x(slider.line_rect.centerx)
        self.assertEqual(slider.value, 0.0)
        slider.set_from_x(slider.line_rect.right)
        self.assertEqual(slider.value, 40.96)

    def test_saved_reward_value_loads_without_position_api(self):
        slider = RewardSlider((0, 0, 550, 40), "Reward")

        self.assertTrue(_set_slider_value(slider, 2.56))

        self.assertEqual(slider.value, 2.56)

    def test_disabled_reward_slider_uses_gray_background(self):
        slider = RewardSlider((0, 0, 550, 40), "Reward")
        slider.enabled = False
        surface = pygame.Surface((550, 40), pygame.SRCALPHA)
        font = pygame.font.Font(None, 20)

        slider.draw(surface, font, mouse_pos=(-1, -1))

        self.assertEqual(surface.get_at((2, 2))[:3], ui.DARK_GREY)


class SliderRowTests(unittest.TestCase):
    def test_label_value_slider_uses_fixed_right_track_region(self):
        first = SliderRow(
            (16, 20, 544, 34),
            "Replay size, batch=30k",
            0,
            100,
            50,
            layout=SliderRow.LABEL_VALUE_SLIDER,
            slider_width=184,
        )
        second = SliderRow(
            (16, 60, 544, 34),
            "Gradient steps, UTD=1.1",
            0,
            100,
            50,
            layout=SliderRow.LABEL_VALUE_SLIDER,
            slider_width=184,
        )

        self.assertEqual(first.line_rect.x, second.line_rect.x)
        self.assertEqual(first.line_rect.width, second.line_rect.width)
        self.assertEqual(first.line_rect.right, second.line_rect.right)

    def test_short_count_formats_regimen_context_values(self):
        self.assertEqual(_format_short_count(30000), "30k")
        self.assertEqual(_format_short_count(15000000), "15M")

    def test_disabled_slider_value_remains_yellow(self):
        slider = SliderRow(
            (0, 0, 240, 34),
            "Value",
            0,
            100,
            50,
            value_formatter=lambda value: str(int(value)),
        )
        slider.enabled = False
        surface = pygame.Surface((240, 34), pygame.SRCALPHA)
        font = pygame.font.Font(None, 20)

        slider.draw(surface, font, mouse_pos=(-1, -1))

        pixels = pygame.surfarray.pixels3d(surface)
        self.assertTrue(((pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 0)).any())


class TrainingSettingsComparisonTests(unittest.TestCase):
    def test_decayed_current_epsilon_does_not_change_settings_match(self):
        saved = {
            "regimen": {
                "starting_epsilon": 0.2,
                "current_epsilon": 0.2,
                "epsilon": 0.2,
                "epsilon_decay": 0.997,
            }
        }
        live = {
            "regimen": {
                "starting_epsilon": 0.2,
                "current_epsilon": 0.125,
                "epsilon": 0.125,
                "epsilon_decay": 0.997,
            }
        }

        self.assertTrue(_training_settings_match(saved, live))

    def test_starting_epsilon_still_changes_settings_match(self):
        saved = {
            "regimen": {
                "starting_epsilon": 0.2,
                "current_epsilon": 0.125,
                "epsilon": 0.125,
            }
        }
        live = {
            "regimen": {
                "starting_epsilon": 0.3,
                "current_epsilon": 0.125,
                "epsilon": 0.125,
            }
        }

        self.assertFalse(_training_settings_match(saved, live))


class GenericSliderTests(unittest.TestCase):
    def test_disabled_slider_uses_gray_background(self):
        slider = ui_slider.Slider(
            0,
            0,
            200,
            0.0,
            100.0,
            50.0,
            "Value",
            values=tuple(float(value) for value in range(0, 101, 5)),
            height=44,
        )
        slider.enabled = False
        surface = pygame.Surface((200, 44), pygame.SRCALPHA)
        font = pygame.font.Font(None, 20)

        slider.draw(surface, font)

        self.assertEqual(surface.get_at((slider.rect.centerx, 4))[:3], ui.DARK_GREY)


class RegimenSliderTests(unittest.TestCase):
    def test_regimen_sliders_expose_the_requested_discrete_values(self):
        self.assertEqual(ROUNDS_PER_BATCH_VALUES, tuple(range(1, 51, 1)))
        self.assertEqual(BATCH_GROUPING_VALUES, tuple(range(25, 1001, 25)))
        self.assertEqual(
            MATCH_TIME_LIMIT_VALUES,
            tuple(range(240, 12001, 240)),
        )
        self.assertEqual(MINIBATCH_SIZE_VALUES, (16, 32, 64, 128, 256, 512))
        self.assertEqual(
            REPLAY_UPDATES_PER_BATCH_VALUES,
            tuple(range(100, 5001, 100)),
        )
        self.assertEqual(
            LEARNING_RATE_VALUES,
            (0.00001, 0.00003, 0.00010, 0.00030, 0.00100, 0.00300, 0.01000),
        )
        self.assertEqual(
            EPSILON_VALUES,
            tuple(round(i * 0.025, 3) for i in range(41)),
        )
        self.assertEqual(
            EPSILON_DECAY_VALUES,
            tuple(round(0.950 + i * 0.001, 3) for i in range(51)),
        )
        self.assertEqual(EPSILON_FRAME_SPAN_VALUES, tuple(range(1, 49)))
        self.assertEqual(
            GAMMA_VALUES,
            tuple(round(0.950 + i * 0.001, 3) for i in range(51)),
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


class TrainingConfigAdapterTests(unittest.TestCase):
    def test_training_config_from_state_carries_ui_values(self):
        state = TrainingUIState(selected_ship="Earthling")
        state.rewards["Kill enemy"] = 2.56
        state.ai_opponent_chance = 50.0
        state.opponent_mode = "all"
        state.forward_activity = 25.0
        state.a1_activity = 50.0
        state.a2_activity = 75.0
        state.face_opponent_activity = 100.0
        state.rounds_per_batch = 2
        state.gamma = 0.995
        state.minibatch_size = 128
        state.replay_updates_per_batch = 1000
        state.starting_epsilon = 0.2
        state.current_epsilon = 0.125
        state.epsilon_decay = 0.997
        state.epsilon_frame_span = 12
        state.hidden_layer_size = 64
        state.hidden_layer_count = 1

        config = training_config_from_state(state)

        self.assertEqual(config.trainee_ship, "Earthling")
        self.assertEqual(config.reward_weights["Kill enemy"], 2.56)
        self.assertEqual(config.opponent_mode, "all")
        self.assertEqual(config.ai_opponent_chance, 50.0)
        self.assertEqual(config.forward_activity, 25.0)
        self.assertEqual(config.a1_activity, 50.0)
        self.assertEqual(config.a2_activity, 75.0)
        self.assertEqual(config.face_opponent_activity, 100.0)
        self.assertEqual(config.rounds_per_batch, 2)
        self.assertEqual(config.gamma, 0.995)
        self.assertEqual(config.minibatch_size, 128)
        self.assertEqual(config.replay_updates_per_batch, 1000)
        self.assertEqual(config.starting_epsilon, 0.2)
        self.assertEqual(config.epsilon, 0.125)
        self.assertEqual(config.epsilon_decay, 0.997)
        self.assertEqual(config.epsilon_frame_span, 12)
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

    def test_checkpoint_reset_returns_to_starting_epsilon(self):
        epsilon = _epsilon_for_model_update(
            starting_epsilon=0.3,
            current_epsilon=0.125,
            reset_checkpoint=True,
        )

        self.assertEqual(epsilon, 0.3)

    def test_non_reset_update_preserves_current_epsilon(self):
        epsilon = _epsilon_for_model_update(
            starting_epsilon=0.3,
            current_epsilon=0.125,
            reset_checkpoint=False,
        )

        self.assertEqual(epsilon, 0.125)

    def test_checkpoint_reset_clears_existing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            pth_path = Path(directory) / "Earthling-01.pth"
            csv_path = Path(directory) / "Earthling-01.csv"
            pth_path.write_bytes(b"checkpoint")
            csv_path.write_text("old,csv\n", encoding="utf-8")

            _clear_reset_model_artifacts(SimpleNamespace(pth_path=pth_path))

            self.assertEqual(pth_path.read_bytes(), b"")
            self.assertFalse(csv_path.exists())


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
            "current_epsilon": 0.125,
            "epsilon_decay": 0.997,
            "gamma": 0.99,
            "component_totals": {"Kill enemy": 2.0},
            "previous_opponent": "Chenjesu",
            "batch_component_totals": {},
            "elapsed_training_seconds": 3723.0,
            "batches_per_hour": 12.5,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_display_off_console_includes_batch_logs_before_live_status(self):
        status = self._status()

        lines = _display_off_console_lines(status, ("Batch      1 | summary",))

        self.assertIn("Current batch", lines)
        self.assertIn("Round:       3/  25", lines)
        self.assertEqual(lines[0], "Completed batches")
        self.assertLess(
            lines.index("Batch      1 | summary"),
            lines.index("Current batch"),
        )
        self.assertIn("Batch:         3", lines)
        self.assertIn("Elapsed:   1:02:03", lines)
        self.assertIn("Rate:        12.50/h", lines)
        reward_name_width = max(len(label) for label in REWARD_LABELS)
        self.assertIn(f"{'Opponent:':<10}{'Earthling':<{reward_name_width}}", lines)
        self.assertIn("Replay:       99", lines)
        self.assertIn("Return:       12.50", lines)
        self.assertIn("Loss:              -", lines)
        self.assertIn(f"{'Gamma:':<10}{0.99:>10.3f}", lines)
        self.assertIn(f"{'Epsilon:':<10}{0.125:>10.5f}", lines)
        self.assertIn(f"{'Eps decay:':<10}{0.997:>10.3f}", lines)
        self.assertIn(
            f"{'Reward components':<{reward_name_width}}|   Chenjesu |  Batch -",
            lines,
        )
        self.assertIn(
            f"{'Kill enemy':<{reward_name_width}}|     2.0000 |        -",
            lines,
        )

    def test_display_off_console_keeps_current_batch_block_height_stable(self):
        lines_with_component = _display_off_console_lines(
            self._status(), ("Batch      1 | summary",)
        )
        lines_without_components = _display_off_console_lines(
            self._status(component_totals={}), ("Batch      1 | summary",)
        )

        self.assertEqual(len(lines_with_component), len(lines_without_components))
        placeholder_rows = lines_without_components[-len(REWARD_LABELS) :]
        self.assertEqual(len(placeholder_rows), len(REWARD_LABELS))
        self.assertTrue(all(row.endswith("|        -") for row in placeholder_rows))

    def test_training_duration_formats_for_benchmark_notes(self):
        self.assertEqual(_format_training_duration(0), "0:00:00")
        self.assertEqual(_format_training_duration(65.9), "0:01:05")
        self.assertEqual(_format_training_duration(3723), "1:02:03")


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
    def test_legacy_training_hud_helpers_are_removed(self):
        for name in (
            "_ship_for_player",
            "_draw_value_bar",
            "_draw_ship_icon",
            "_draw_training_hud_panel",
        ):
            self.assertFalse(hasattr(train_ai, name), name)

    def test_display_on_battle_uses_shared_controller_with_training_arena(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        rect = training_layout().arena_rect
        controller = mock.Mock()
        star_field_renderer = object()
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

        with mock.patch("pygame.display.flip") as display_flip:
            _draw_training_battle(
                screen,
                rect,
                status,
                star_field_renderer,
                controller,
            )

        args, kwargs = controller.draw.call_args
        layout = args[2]
        options = kwargs["options"]

        self.assertEqual(layout.arena_rect, rect)
        self.assertIsNone(layout.player1_hud_rect)
        self.assertIsNone(layout.player2_hud_rect)
        self.assertIs(args[4], star_field_renderer)
        self.assertFalse(options.draw_huds)
        display_flip.assert_not_called()

    def test_display_on_without_battle_view_draws_status_instead_of_stale_frame(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        rect = training_layout().arena_rect
        controller = mock.Mock()
        status = SimpleNamespace(
            running=True,
            completed_batches=2,
            current_round=1,
            total_rounds=1,
            current_opponent="Earthling",
            replay_size=128,
            weighted_total_return=0.0,
            recent_loss=None,
            component_totals={},
            battle_view=None,
            display_message="Applying gradient descent",
        )
        font = pygame.font.SysFont(None, 36)
        small_font = pygame.font.SysFont(None, 24)

        with mock.patch("src.Menus.train_ai._draw_training_status") as draw_status:
            _draw_training_battle(
                screen,
                rect,
                status,
                object(),
                controller,
                font,
                small_font,
            )

        controller.draw.assert_not_called()
        draw_status.assert_called_once_with(screen, rect, status, font, small_font)

    def test_display_on_huds_use_shared_controller_with_training_rects(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        hud_rects = training_layout().hud_rects
        controller = mock.Mock()
        star_field_renderer = object()
        status = SimpleNamespace(
            battle_view={
                "game_objects": (),
                "border_color": (50, 50, 50),
                "frame_id": 1,
                "original_ships": (),
                "camera_targets": (),
                "entry_state": None,
            }
        )

        _draw_training_huds(
            screen,
            hud_rects,
            status,
            star_field_renderer,
            controller,
        )

        args, kwargs = controller.draw.call_args
        layout = args[2]
        options = kwargs["options"]

        self.assertEqual(layout.player1_hud_rect, hud_rects[0])
        self.assertEqual(layout.player2_hud_rect, hud_rects[1])
        self.assertIs(args[4], star_field_renderer)
        self.assertFalse(options.draw_arena)

    def test_display_on_hud_draws_shared_live_ship_features(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        screen.fill((0, 0, 0))
        trainee = SimpleNamespace(
            player=1,
            name="Earthling",
            position=[500.0, 500.0],
            previous_position=[500.0, 500.0],
            current_hp=8,
            max_hp=10,
            current_energy=6,
            max_energy=12,
            boarded_marines=(),
            limpets_attached=0,
        )
        opponent = SimpleNamespace(
            player=2,
            name="Chenjesu",
            position=[700.0, 500.0],
            previous_position=[700.0, 500.0],
            current_hp=4,
            max_hp=10,
            current_energy=2,
            max_energy=8,
            boarded_marines=(),
            limpets_attached=0,
        )
        snapshot = RenderSnapshot(
            stars=(),
            planets=(),
            thrust_markers=(),
            asteroids=(),
            abilities=(),
            ships=(trainee, opponent),
            effects=(),
            live_ships=(trainee, opponent),
        )
        status = SimpleNamespace(
            battle_view={
                "game_objects": snapshot,
                "border_color": (50, 50, 50),
                "frame_id": 1,
                "original_ships": (trainee, opponent),
                "camera_targets": (trainee, opponent),
                "entry_state": None,
            }
        )

        with (
            mock.patch("src.Battle.battle_draw._render_world_to_surface"),
            mock.patch("pygame.display.flip") as display_flip,
        ):
            _draw_training_huds(
                screen,
                training_layout().hud_rects,
                status,
                object(),
                BattleDrawController(),
            )

        first, second = training_layout().hud_rects
        first_tint = screen.get_at((first.left + 1, first.top + 1))[:3]
        second_tint = screen.get_at((second.left + 1, second.top + 1))[:3]
        self.assertGreater(first_tint[1], first_tint[0])
        self.assertGreater(first_tint[2], first_tint[0])
        self.assertGreater(second_tint[0], second_tint[1])
        self.assertGreater(second_tint[2], second_tint[1])

        hud_content_width = BAR_WIDTH * 2 + VIEWPORT_SIZE + 2 * VIEWPORT_MARGIN
        draw_x_offset = (first.width - hud_content_width) // 2
        viewport_left = first.left + draw_x_offset + BAR_WIDTH + VIEWPORT_MARGIN
        viewport_top = first.top + MARINE_REGION_HEIGHT
        self.assertEqual(
            screen.get_at((viewport_left, viewport_top))[:3],
            const.HUD_VIEWPORT_BORDER,
        )
        display_flip.assert_not_called()


if __name__ == "__main__":
    unittest.main()
