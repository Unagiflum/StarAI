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
from src.training import torch_backend
from src.training.contracts import (
    REFLECTION_AUGMENTATION_METADATA_KEY,
    REFLECTION_AUGMENTATION_MODE,
)
from src.training.session import TrainingSessionStatus
from src.training.rewards import REWARD_POINT_A1, REWARD_SPAWN_A1, REWARD_SPAWN_A2
from src.Menus.train_ai import (
    ACTION_TOP,
    EPSILON_DECAY_VALUES,
    EPSILON_FLOOR_VALUES,
    EPSILON_FRAME_SPAN_VALUES,
    EPSILON_VALUES,
    DISPLAY_TOP,
    FOOTER_CONTROL_HEIGHT,
    GAMMA_VALUES,
    INSTANCE_CONTROL_HEIGHT,
    INSTANCE_ADD_WIDTH,
    INSTANCE_CLOSE_WIDTH,
    INSTANCE_GAP,
    INSTANCE_DROPDOWN_MAX_VISIBLE_ROWS,
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
    BATCH_CONTROLLED_FIELDS,
    BATCH_GROUPING_VALUES,
    InstanceDropdown,
    AI_OPPONENT_PERCENT_VALUES,
    SIMPLE_ACTIVITY_VALUES,
    TRAINING_INSTANCE_SOFT_MAX,
    TRAINING_INSTANCE_SUPPORTED_MAX,
    TRAINING_BATCH_LOG_FONT_SIZE,
    RewardSlider,
    SliderRow,
    apply_batch_settings,
    architecture_for_state,
    batch_settings_from_state,
    coordinated_architecture_signature,
    instances_with_different_batch_settings,
    validate_coordinated_batch_start,
    TRAINING_HUD_HEIGHT,
    TrainingBatchLogBox,
    TrainingInstanceManager,
    TrainingUIState,
    UI_TOP_MARGIN,
    load_training_ui_session,
    save_training_ui_session,
    training_instance_manager_from_json,
    training_instance_manager_to_json,
    _clear_reset_model_artifacts,
    _display_off_console_lines,
    _draw_training_huds,
    _draw_training_battle,
    _epsilon_for_model_update,
    _format_short_count,
    _format_replay_buffer_size,
    _format_update_to_data_ratio,
    _format_training_duration,
    _speedometer_console_lines,
    _instance_row_parts,
    _instance_status_text,
    _training_settings_match,
    _wheel_step,
    _progress_for_model_update,
    _set_slider_value,
    training_config_from_state,
    training_layout,
)
from src.training.model_registry import (
    SLOT_EMPTY,
    SLOT_USER,
    TrainingModelSlot,
    metadata_from_state,
    model_architecture_metadata,
    replay_checkpoint_path,
)
from src.training.opponent_cache import ModelSaveCoordinator, OpponentModelCache
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


class ReplayBufferSizeHintTests(unittest.TestCase):
    def test_hint_uses_packed_replay_payload_size(self):
        self.assertEqual(_format_replay_buffer_size(20_000), "~46MB")
        self.assertEqual(_format_replay_buffer_size(30_000), "~69MB")
        self.assertEqual(_format_replay_buffer_size(250_000), "~572MB")


class TrainingUIStateTests(unittest.TestCase):
    def test_defaults_match_training_specification(self):
        with mock.patch(
            "src.Menus.train_ai.torch_backend.cuda_available", return_value=False
        ):
            state = TrainingUIState()

        self.assertEqual(state.selected_slot, 1)
        self.assertEqual(state.slot_labels, ["", "", "", ""])
        expected_rewards = {label: 0.0 for label in REWARD_LABELS}
        expected_rewards.update(
            {
                "Kill enemy": 40.96,
                "Reduce enemy crew": 5.12,
                "Get debuffed": -1.28,
                "Lose crew": -2.56,
                "Die": -20.48,
            }
        )
        self.assertEqual(state.rewards, expected_rewards)
        self.assertEqual(state.opponent_mode, "all")
        self.assertEqual(state.ai_opponent_chance, 100.0)
        self.assertEqual(state.forward_activity, 0.0)
        self.assertEqual(state.a1_activity, 25.0)
        self.assertEqual(state.a2_activity, 0.0)
        self.assertEqual(state.face_opponent_activity, 100.0)
        self.assertEqual(state.rounds_per_batch, 1)
        self.assertEqual(state.batch_grouping, 10)
        self.assertEqual(state.match_time_limit, 1200)
        self.assertEqual(state.learning_rate, 0.0001)
        self.assertEqual(state.starting_epsilon, 0.5)
        self.assertEqual(state.current_epsilon, 0.5)
        self.assertEqual(state.epsilon_floor, 0.05)
        self.assertEqual(state.epsilon_decay, 0.998)
        self.assertEqual(state.epsilon_frame_span, 8)
        self.assertEqual(state.gamma, 0.99)
        self.assertEqual(state.minibatch_size, 2048)
        self.assertEqual(state.replay_updates_per_batch, 15)
        self.assertEqual(state.training_device, "cpu")
        self.assertFalse(state.display_on)
        self.assertFalse(state.running)

    def test_training_device_choices_only_include_cpu_and_gpu(self):
        self.assertEqual(
            train_ai.TRAINING_DEVICE_LABELS,
            (
                (torch_backend.DEVICE_CPU, "CPU"),
                (torch_backend.DEVICE_GPU, "GPU"),
            ),
        )

    def test_training_device_defaults_to_gpu_when_available(self):
        with mock.patch(
            "src.Menus.train_ai.torch_backend.cuda_available", return_value=True
        ):
            state = TrainingUIState()

        self.assertEqual(state.training_device, torch_backend.DEVICE_GPU)

    def test_legacy_auto_device_uses_available_gpu(self):
        payload = training_instance_manager_to_json(TrainingInstanceManager())
        payload["instances"][0]["state"]["training_device"] = (
            torch_backend.DEVICE_AUTO
        )

        with mock.patch(
            "src.Menus.train_ai.torch_backend.cuda_available", return_value=True
        ):
            restored = training_instance_manager_from_json(payload)

        self.assertEqual(restored.active_state.training_device, torch_backend.DEVICE_GPU)

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
            self.join_calls = 0

        def request_stop(self):
            self.stop_requested = True
            self.status.stopping = True

        def set_display_on(self, enabled):
            self.display_events.append(enabled)

        def join(self):
            self.join_calls += 1

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
        manager.active_tab = "opponent"

        manager.select_instance(first_id)

        self.assertIs(manager.active_instance, manager.instances[0])
        self.assertIsNone(manager.active_state.selected_ship)
        self.assertEqual(manager.active_tab, "opponent")

        manager.select_instance(second.instance_id)

        self.assertEqual(manager.active_state.selected_ship, "Androsynth")
        self.assertEqual(manager.active_tab, "opponent")

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

    def test_manager_does_not_require_confirmation_when_adding_many_instances(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SOFT_MAX:
            self.assertFalse(manager.add_requires_confirmation())
            manager.add_instance()

        self.assertFalse(manager.add_requires_confirmation())

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
        self.assertEqual(manager.instance_summary_text(), "02>/03")

    def test_select_relative_instance_wraps_through_instances(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        second = manager.add_instance()
        third = manager.add_instance()

        self.assertIs(manager.select_relative_instance(1), first)
        self.assertIs(manager.select_relative_instance(-1), third)
        self.assertIs(manager.select_relative_instance(-1), second)

    def test_instance_dropdown_uses_largest_row_count_that_fits_screen(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SUPPORTED_MAX:
            manager.add_instance()
        dropdown = InstanceDropdown(
            pygame.Rect(10, 300, 250, INSTANCE_CONTROL_HEIGHT),
            manager,
            lambda _instance_id: None,
        )

        expected_rows = min(
            INSTANCE_DROPDOWN_MAX_VISIBLE_ROWS,
            (
                const.SCREEN_HEIGHT
                - (dropdown.rect.bottom + 4)
                - INSTANCE_TOP
            )
            // INSTANCE_CONTROL_HEIGHT,
            len(manager.instances),
        )

        self.assertGreater(expected_rows, 8)
        self.assertEqual(dropdown.visible_row_count(), expected_rows)
        self.assertEqual(
            dropdown.list_rect().height,
            expected_rows * INSTANCE_CONTROL_HEIGHT,
        )

    def test_mouse_wheel_values_are_normalized_to_one_instance_step(self):
        self.assertEqual(_wheel_step(1), -1)
        self.assertEqual(_wheel_step(2), -1)
        self.assertEqual(_wheel_step(-1), 1)
        self.assertEqual(_wheel_step(-3), 1)
        self.assertEqual(_wheel_step(0), 0)

    def test_expanded_instance_dropdown_wheel_scrolls_one_row_per_event(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < TRAINING_INSTANCE_SUPPORTED_MAX:
            manager.add_instance()
        dropdown = InstanceDropdown(
            pygame.Rect(10, 300, 250, INSTANCE_CONTROL_HEIGHT),
            manager,
            lambda _instance_id: None,
        )
        dropdown.expanded = True
        dropdown.scroll_index = 3
        event = pygame.event.Event(
            pygame.MOUSEWHEEL,
            {"y": 2, "pos": dropdown.list_rect().center},
        )

        self.assertTrue(dropdown.handle_event(event))

        self.assertEqual(dropdown.scroll_index, 2)

    def test_training_session_json_round_trip_preserves_instances_and_settings(self):
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 2
        manager.active_state.learning_rate = 0.003
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 3
        second.state.match_time_limit = 2400
        manager.batch_scheduling.apply_to_all_open_instances = True
        manager.active_tab = "batch"

        restored = training_instance_manager_from_json(
            training_instance_manager_to_json(manager)
        )

        self.assertEqual(len(restored.instances), 2)
        self.assertEqual(restored.active_instance_id, second.instance_id)
        self.assertEqual(restored.instances[0].state.selected_ship, "Earthling")
        self.assertEqual(restored.instances[0].state.selected_slot, 2)
        self.assertEqual(restored.instances[0].state.learning_rate, 0.003)
        self.assertEqual(restored.active_state.selected_ship, "Androsynth")
        self.assertEqual(restored.active_state.match_time_limit, 2400)
        self.assertTrue(restored.batch_scheduling.apply_to_all_open_instances)
        self.assertEqual(restored.active_tab, "regimen")

    def test_version_one_session_migrates_active_instances_tab(self):
        payload = training_instance_manager_to_json(TrainingInstanceManager())
        payload.pop("active_tab")
        payload["version"] = 1
        payload["instances"][0]["state"]["active_tab"] = "rewards"

        restored = training_instance_manager_from_json(payload)

        self.assertEqual(restored.active_tab, "rewards")

    def test_training_session_save_and_load_uses_supplied_path(self):
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 4
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train_ai_session.json"

            save_training_ui_session(manager, path)
            restored = load_training_ui_session(path)

        self.assertEqual(restored.active_state.selected_ship, "Earthling")
        self.assertEqual(restored.active_state.selected_slot, 4)

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

    def test_enabling_display_after_coordinated_run_does_not_notify_stopped_proxy(self):
        manager = TrainingInstanceManager()
        instance = manager.active_instance
        instance.session = self.FakeSession(running=False)

        manager.set_active_display(True)

        self.assertTrue(instance.state.display_on)
        self.assertEqual(instance.session.display_events, [])

    def test_select_instance_transfers_global_display_to_new_active(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.session = self.FakeSession(running=True)
        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        first.session.display_events.clear()
        manager.set_active_display(True)
        first.session.display_events.clear()
        second.session.display_events.clear()

        manager.select_instance(first.instance_id)

        self.assertIs(manager.active_instance, first)
        self.assertTrue(manager.display_on)
        self.assertTrue(first.state.display_on)
        self.assertFalse(second.state.display_on)
        self.assertEqual(first.session.display_events, [True])
        self.assertEqual(second.session.display_events, [False])

    def test_back_action_is_blocked_when_any_instance_is_running(self):
        manager = TrainingInstanceManager()
        manager.active_instance.session = self.FakeSession(running=True)

        self.assertEqual(manager.back_action(), "blocked")

        second = manager.add_instance()
        second.session = self.FakeSession(running=True)
        manager.select_instance(manager.instances[0].instance_id)

        self.assertEqual(manager.back_action(), "blocked")
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

    def test_join_all_sessions_waits_for_each_distinct_owner_once(self):
        manager = TrainingInstanceManager()
        owner = mock.Mock()
        first = manager.active_instance
        second = manager.add_instance()
        first.session = SimpleNamespace(_scheduler=owner)
        second.session = SimpleNamespace(_scheduler=owner)
        manager.batch_scheduling.coordinated_session = owner

        manager.join_all_sessions()

        owner.join.assert_called_once_with()

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

    def test_batch_settings_helpers_copy_all_regimen_controlled_fields(self):
        source = TrainingUIState()
        target = TrainingUIState()
        source.match_time_limit = 2400
        source.rounds_per_batch = 3
        source.batch_grouping = 20
        source.minibatch_size = 512
        source.replay_updates_per_batch = 30
        source.learning_rate = 0.0003
        source.replay_buffer_size = 250000

        apply_batch_settings(source, target)

        self.assertEqual(
            batch_settings_from_state(target),
            batch_settings_from_state(source),
        )
        self.assertEqual(target.replay_buffer_size, source.replay_buffer_size)
        self.assertEqual(set(batch_settings_from_state(source)), set(BATCH_CONTROLLED_FIELDS))

    def test_apply_future_changes_does_not_copy_existing_values(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        second = manager.add_instance()
        manager.select_instance(first.instance_id)
        first.state.match_time_limit = 2400
        first.state.rounds_per_batch = 4
        first.state.batch_grouping = 25
        first.state.minibatch_size = 512
        first.state.replay_updates_per_batch = 40
        first.state.learning_rate = 0.0003

        original_second_batch = batch_settings_from_state(second.state)
        manager.set_apply_future_changes_to_all(True)

        self.assertTrue(manager.batch_scheduling.apply_to_all_open_instances)
        self.assertEqual(batch_settings_from_state(second.state), original_second_batch)

        first.state.rounds_per_batch = 8
        manager.propagate_future_changes(first, scalar_fields=("rounds_per_batch",))

        self.assertEqual(second.state.rounds_per_batch, 8)
        self.assertNotEqual(second.state.match_time_limit, first.state.match_time_limit)

    def test_new_instance_keeps_defaults_when_future_changes_is_checked(self):
        manager = TrainingInstanceManager()
        manager.active_state.gamma = 0.975
        manager.set_apply_future_changes_to_all(True)

        added = manager.add_instance()

        self.assertEqual(added.state.gamma, TrainingUIState().gamma)

    def test_future_reward_slot_and_label_changes_reach_nonempty_unique_ships(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        manager.select_instance(first.instance_id)
        manager.set_apply_future_changes_to_all(True)

        reward_label = REWARD_LABELS[0]
        first.state.rewards[reward_label] = 10.24
        first.state.selected_slot = 3
        first.state.slot_labels[2] = "changed"
        manager.propagate_future_changes(
            first,
            reward_labels=(reward_label,),
            slot_label_indices=(2,),
        )
        manager.propagate_selected_slot(first)

        self.assertEqual(second.state.rewards[reward_label], 10.24)
        self.assertEqual(second.state.selected_slot, 3)
        self.assertEqual(second.state.slot_labels[2], "changed")

    def test_ship_selection_limit_matches_ai_slot_count(self):
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        for _ in range(3):
            manager.add_instance().state.selected_ship = "Earthling"
        fifth = manager.add_instance()

        self.assertFalse(manager.can_select_ship("Earthling", instance=fifth))

    def test_running_instance_temporarily_disables_future_changes_without_clearing_it(self):
        manager = TrainingInstanceManager()
        manager.batch_scheduling.apply_to_all_open_instances = True
        manager.active_instance.session = self.FakeSession(running=True)

        validation = manager.coordinated_batch_validation(
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY)
        )

        self.assertFalse(validation.can_start_all)
        self.assertTrue(manager.batch_scheduling.apply_to_all_open_instances)
        self.assertFalse(manager.future_changes_effective())

    def test_duplicate_ship_disables_slot_and_label_propagation(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        second = manager.add_instance()
        second.state.selected_ship = "Earthling"
        second.state.selected_slot = 2
        manager.select_instance(first.instance_id)
        manager.set_apply_future_changes_to_all(True)

        first.state.selected_slot = 3
        first.state.slot_labels[2] = "changed"
        manager.propagate_selected_slot(first)
        manager.propagate_future_changes(first, slot_label_indices=(2,))

        self.assertEqual(second.state.selected_slot, 2)
        self.assertNotEqual(second.state.slot_labels[2], "changed")

    def test_slot_and_label_propagation_silently_skips_instance_without_ship(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        second = manager.add_instance()
        manager.select_instance(first.instance_id)
        manager.set_apply_future_changes_to_all(True)

        first.state.selected_slot = 3
        first.state.slot_labels[2] = "changed"
        manager.propagate_selected_slot(first)
        manager.propagate_future_changes(first, slot_label_indices=(2,))

        self.assertEqual(second.state.selected_slot, 1)
        self.assertEqual(second.state.slot_labels[2], "")

    def test_instances_with_different_batch_settings_reports_mismatches(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        second = manager.add_instance()
        third = manager.add_instance()
        second.state.rounds_per_batch = first.state.rounds_per_batch
        third.state.rounds_per_batch = first.state.rounds_per_batch + 1

        differing = instances_with_different_batch_settings(manager.instances, first.state)

        self.assertEqual(differing, (third,))

    def test_coordinated_architecture_signature_uses_required_fields(self):
        architecture = model_architecture_metadata(256, 2)
        architecture["optimizer"] = "sgd"
        signature = coordinated_architecture_signature(architecture)

        self.assertEqual(
            signature,
            (
                ("input_size", architecture["input_size"]),
                ("output_count", architecture["output_count"]),
                ("hidden_layer_width", 256),
                ("hidden_layer_count", 2),
            ),
        )

    def _coordinated_manager_with_two_user_slots(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        first.state.slot_labels[0] = "one"
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "two"
        return manager, first, second

    def test_start_all_validation_allows_two_gpu_user_slots(self):
        manager, first, second = self._coordinated_manager_with_two_user_slots()
        metadata1 = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="one",
            architecture=architecture_for_state(first.state),
            training={},
        )
        metadata2 = metadata_from_state(
            ship="Androsynth",
            slot=1,
            description="two",
            architecture=architecture_for_state(second.state),
            training={},
        )
        slots = {
            ("Earthling", 1): TrainingModelSlot(
                "Earthling",
                1,
                SLOT_USER,
                "one",
                metadata=metadata1,
            ),
            ("Androsynth", 1): TrainingModelSlot(
                "Androsynth",
                1,
                SLOT_USER,
                "two",
                metadata=metadata2,
            ),
        }

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: slots[(ship, slot)],
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertTrue(validation.can_start_all)
        self.assertEqual(validation.included_instances, (first, second))

    def test_start_all_validation_requires_reset_instead_of_blocking_old_schema(self):
        manager, first, second = self._coordinated_manager_with_two_user_slots()
        first_metadata = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="one",
            architecture=architecture_for_state(first.state),
            training={},
        )
        first_metadata["observation_schema_version"] -= 1
        second_metadata = metadata_from_state(
            ship="Androsynth",
            slot=1,
            description="two",
            architecture=architecture_for_state(second.state),
            training={},
        )
        slots = {
            ("Earthling", 1): TrainingModelSlot(
                "Earthling", 1, SLOT_USER, "one", metadata=first_metadata
            ),
            ("Androsynth", 1): TrainingModelSlot(
                "Androsynth", 1, SLOT_USER, "two", metadata=second_metadata
            ),
        }

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: slots[(ship, slot)],
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertTrue(validation.can_start_all)
        self.assertEqual(validation.reset_required_instances, (first,))

    def test_start_all_validation_requires_reset_for_saved_architecture_mismatch(self):
        manager, first, second = self._coordinated_manager_with_two_user_slots()
        first_metadata = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="one",
            architecture=model_architecture_metadata(
                first.state.hidden_layer_size * 2,
                first.state.hidden_layer_count,
            ),
            training={},
        )
        second_metadata = metadata_from_state(
            ship="Androsynth",
            slot=1,
            description="two",
            architecture=architecture_for_state(second.state),
            training={},
        )
        slots = {
            ("Earthling", 1): TrainingModelSlot(
                "Earthling", 1, SLOT_USER, "one", metadata=first_metadata
            ),
            ("Androsynth", 1): TrainingModelSlot(
                "Androsynth", 1, SLOT_USER, "two", metadata=second_metadata
            ),
        }

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: slots[(ship, slot)],
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertTrue(validation.can_start_all)
        self.assertEqual(validation.reset_required_instances, (first,))

    def test_long_reset_confirmation_uses_compact_instance_positions(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < 25:
            manager.add_instance()
        for instance in manager.instances:
            instance.state.selected_ship = "Earthling"
            instance.state.selected_slot = 1

        text = train_ai.coordinated_reset_confirmation_text(
            manager,
            manager.instances[:6],
        )
        all_text = train_ai.coordinated_reset_confirmation_text(
            manager,
            manager.instances,
        )

        self.assertIn("6 saved models in instances 01-06", text)
        self.assertNotIn("Earthling-01", text)
        self.assertIn("all 25 open instances", all_text)

    def test_bulk_delete_targets_are_eligible_and_deduplicated(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        second = manager.add_instance()
        second.state.selected_ship = "Earthling"
        third = manager.add_instance()
        third.state.selected_ship = "Androsynth"
        manager.add_instance().state.selected_ship = "Mycon"

        class Repository:
            def slot_for(self, ship, slot):
                source = SLOT_USER if ship in ("Earthling", "Androsynth") else SLOT_EMPTY
                return TrainingModelSlot(ship, int(slot), source)

        targets, affected = train_ai.eligible_user_model_deletion_targets(
            manager,
            Repository(),
            3,
        )

        self.assertEqual(
            [(target.ship, target.slot) for target in targets],
            [("Earthling", 3), ("Androsynth", 3)],
        )
        self.assertEqual(affected, (first, second, third))

    def test_bulk_delete_confirmation_compacts_twenty_five_instances(self):
        manager = TrainingInstanceManager()
        while len(manager.instances) < 25:
            manager.add_instance()

        text = train_ai.bulk_delete_confirmation_text(
            manager,
            manager.instances,
            target_count=25,
            slot=4,
        )

        self.assertIn("ALL 25 user models", text)
        self.assertIn("eligible instances 01-25", text)
        self.assertNotIn("01, 02", text)

    def test_deleted_model_references_are_cleared_across_open_instances(self):
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.slot_labels[1] = "First"
        first.state.loaded_ship = "Earthling"
        first.state.loaded_slot = 2
        first.state.loaded_architecture = {"hidden": 256}
        first.state.loaded_training = {"gamma": 0.99}
        target = TrainingModelSlot("Earthling", 2, SLOT_USER)

        second = manager.add_instance()
        second.state.selected_ship = "Earthling"
        second.state.slot_labels[1] = "Second"
        unaffected = manager.add_instance()
        unaffected.state.selected_ship = "Androsynth"
        unaffected.state.slot_labels[1] = "Keep"
        first.session = SimpleNamespace(slot=target)

        train_ai.clear_deleted_model_references(manager, (target,))

        self.assertEqual(first.state.slot_labels[1], "")
        self.assertEqual(second.state.slot_labels[1], "")
        self.assertEqual(unaffected.state.slot_labels[1], "Keep")
        self.assertIsNone(first.state.loaded_ship)
        self.assertIsNone(first.state.loaded_slot)
        self.assertIsNone(first.state.loaded_architecture)
        self.assertIsNone(first.state.loaded_training)
        self.assertIsNone(first.session)

    def test_start_all_validation_blocks_running_instances_and_retains_preference(self):
        manager, _first, _second = self._coordinated_manager_with_two_user_slots()
        manager.batch_scheduling.apply_to_all_open_instances = True
        manager.active_instance.session = self.FakeSession(running=True)

        validation = manager.coordinated_batch_validation(
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY)
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "running")
        self.assertTrue(manager.batch_scheduling.apply_to_all_open_instances)

    def test_start_all_validation_blocks_incomplete_instances(self):
        manager = TrainingInstanceManager()
        manager.add_instance()

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "incomplete")

    def test_start_all_validation_blocks_duplicate_writer_targets(self):
        manager, first, second = self._coordinated_manager_with_two_user_slots()
        second.state.selected_ship = first.state.selected_ship

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "duplicate_writer")

    def test_start_all_validation_blocks_different_batch_settings(self):
        manager, _first, second = self._coordinated_manager_with_two_user_slots()
        second.state.rounds_per_batch += 1

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "regimen")
        self.assertEqual(validation.blocking_reason, "Regimen settings differ")

    def test_start_all_validation_requires_matching_slots(self):
        manager, _first, second = self._coordinated_manager_with_two_user_slots()
        second.state.selected_slot = 2
        second.state.slot_labels[1] = "two"

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "slot")

    def test_start_all_validation_requires_matching_ai_frequency(self):
        manager, _first, second = self._coordinated_manager_with_two_user_slots()
        second.state.ai_opponent_chance = 50.0

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "opponent")

    def test_start_all_validation_averages_current_epsilon(self):
        manager, first, second = self._coordinated_manager_with_two_user_slots()
        first.state.current_epsilon = 0.2
        second.state.current_epsilon = 0.6

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertTrue(validation.can_start_all)
        self.assertAlmostEqual(validation.shared_current_epsilon, 0.4)

    def test_start_all_validation_blocks_mixed_architectures_with_required_status_text(self):
        manager, first, second = self._coordinated_manager_with_two_user_slots()
        second.state.hidden_layer_size = first.state.hidden_layer_size * 2

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "architecture")

    def test_start_all_validation_blocks_non_gpu_devices(self):
        manager, _first, _second = self._coordinated_manager_with_two_user_slots()

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "cpu",
        )

        self.assertFalse(validation.can_start_all)
        self.assertEqual(validation.blocking_code, "device")

    def test_start_all_validation_allows_display_on(self):
        manager, _first, second = self._coordinated_manager_with_two_user_slots()
        second.state.display_on = True

        validation = validate_coordinated_batch_start(
            manager,
            lambda ship, slot: TrainingModelSlot(ship, slot, SLOT_EMPTY),
            torch_module=object(),
            cuda_available=True,
            training_device_key_func=lambda _choice: "gpu",
        )

        self.assertTrue(validation.can_start_all)

    def test_coordinated_manager_attaches_proxies_and_releases_writers_after_stop(self):
        from src.training.coordinated import (
            CoordinatedRuntimeComponents,
            CoordinatedTrainingRecord,
            CoordinatedTrainingSession,
        )
        from src.training.replay import TrainingReplayBuffer

        manager, first, second = self._coordinated_manager_with_two_user_slots()
        slots = (
            (first, TrainingModelSlot("Earthling", 1, SLOT_USER)),
            (second, TrainingModelSlot("Androsynth", 1, SLOT_USER)),
        )
        records = []
        for instance, slot in slots:
            metadata = metadata_from_state(
                ship=slot.ship,
                slot=slot.slot,
                description=instance.state.slot_labels[slot.slot - 1],
                architecture=architecture_for_state(instance.state),
                training={},
            )
            records.append(
                CoordinatedTrainingRecord(
                    instance_id=instance.instance_id,
                    repository=SimpleNamespace(),
                    slot=slot,
                    metadata=metadata,
                    config=training_config_from_state(instance.state),
                    batch_grouping=1,
                )
            )

        scheduler = CoordinatedTrainingSession(
            tuple(records),
            component_builder=lambda _record: CoordinatedRuntimeComponents(
                object(),
                object(),
                TrainingReplayBuffer(4),
            ),
            idle_sleep_seconds=0.001,
        )

        self.assertTrue(manager.reserve_writers_for_slots(slots))
        manager.start_coordinated_session(scheduler)
        proxies = scheduler.proxies
        self.assertIs(first.session, proxies[first.instance_id])
        self.assertIs(second.session, proxies[second.instance_id])
        self.assertTrue(manager.coordinated_run_active())
        self.assertEqual(manager.writer_owner("Earthling", 1), first.instance_id)

        manager.request_stop_all_running()
        scheduler.join(1.0)
        manager.release_stopped_writers()
        manager.cleanup_coordinated_session()

        self.assertFalse(manager.coordinated_run_active())
        self.assertIsNone(manager.writer_owner("Earthling", 1))
        self.assertIsNone(manager.writer_owner("Androsynth", 1))


class TrainingUIRunWiringTests(unittest.TestCase):
    class StopRun(Exception):
        pass

    @staticmethod
    def trainee_start_position():
        return (train_ai.CONTROL_WIDTH // 2, train_ai.CONTENT_TOP + 535)

    @staticmethod
    def synced_action_position():
        action_gap = 10
        available = train_ai.CONTROL_WIDTH - 2 * train_ai.TAB_MARGIN - 2 * action_gap
        synced_width = min(240, available)
        side_width = (available - synced_width) // 2
        return (
            train_ai.TAB_MARGIN + side_width + action_gap + synced_width // 2,
            train_ai.ACTION_TOP + train_ai.FOOTER_CONTROL_HEIGHT // 2,
        )

    class FakeRepository:
        def __init__(self, *_args):
            self.user_dir = Path("unused")
            self.slot = TrainingModelSlot(
                "Earthling",
                1,
                SLOT_USER,
                description="Ready",
                metadata=metadata_from_state(
                    ship="Earthling",
                    slot=1,
                    description="Ready",
                    architecture=model_architecture_metadata(256, 2),
                    training={},
                ),
            )

        def slot_for(self, ship, slot):
            if ship == "Earthling" and int(slot) == 1:
                return self.slot
            return TrainingModelSlot(str(ship), int(slot), SLOT_EMPTY)

        def slots_for_ship(self, ship):
            return [self.slot_for(ship, slot) for slot in range(1, 5)]

    class FakeSession:
        created = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.slot = kwargs["slot"]
            self.history = ()
            self.log_lines = ()
            self.status = SimpleNamespace(
                running=True,
                stopping=False,
                error="",
                current_epsilon=0.1,
            )
            self.started = False
            self.__class__.created.append(self)

        def start(self):
            self.started = True

        def set_display_on(self, _enabled):
            pass

    class FakeCoordinatedRepository:
        def __init__(self, *_args):
            self.user_dir = Path("unused")
            self.slots = {}
            for ship in ("Earthling", "Androsynth"):
                metadata = metadata_from_state(
                    ship=ship,
                    slot=1,
                    description=f"{ship} Ready",
                    architecture=model_architecture_metadata(256, 2),
                    training={},
                )
                self.slots[(ship, 1)] = TrainingModelSlot(
                    ship,
                    1,
                    SLOT_USER,
                    description=f"{ship} Ready",
                    metadata=metadata,
                )

        def slot_for(self, ship, slot):
            return self.slots.get(
                (str(ship), int(slot)),
                TrainingModelSlot(str(ship), int(slot), SLOT_EMPTY),
            )

        def slots_for_ship(self, ship):
            return [self.slot_for(ship, slot) for slot in range(1, 5)]

        def create_or_update_user_model(self, metadata):
            slot = TrainingModelSlot(
                metadata["ship"],
                int(metadata["slot"]),
                SLOT_USER,
                description=metadata.get("description", ""),
                metadata=metadata,
            )
            self.slots[(slot.ship, slot.slot)] = slot
            return slot

    def test_window_close_waits_for_running_session_before_saving_and_exit(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        order = []

        class ClosingSession:
            def __init__(self):
                self.status = TrainingSessionStatus(ship="Earthling", running=True)
                self.history = ()
                self.log_lines = ()

            def request_stop(self):
                order.append("stop")
                self.status.stopping = True

            def set_display_on(self, _enabled):
                pass

            def join(self):
                order.append("join")

        session = ClosingSession()
        manager.active_instance.session = session
        close_event = pygame.event.Event(pygame.QUIT)
        event_calls = [0]

        def events():
            event_calls[0] += 1
            if event_calls[0] == 1:
                return [close_event]
            self.assertNotIn("save", order)
            if event_calls[0] == 2:
                return [close_event]
            session.status.running = False
            session.status.stopping = False
            return []

        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch("src.Menus.train_ai.fit_ship_sprites", return_value={}),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", side_effect=events),
            mock.patch("pygame.display.flip"),
            mock.patch(
                "src.Menus.train_ai.save_training_ui_session",
                side_effect=lambda *_args, **_kwargs: order.append("save"),
            ),
            mock.patch(
                "pygame.quit",
                side_effect=lambda: order.append("pygame_quit"),
            ),
            mock.patch(
                "src.Menus.train_ai.sys.exit",
                side_effect=lambda: (_ for _ in ()).throw(self.StopRun()),
            ),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertGreaterEqual(event_calls[0], 3)
        self.assertEqual(order, ["stop", "join", "save", "pygame_quit"])

    def test_window_close_exits_immediately_when_training_is_idle(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        order = []
        close_event = pygame.event.Event(pygame.QUIT)

        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch("src.Menus.train_ai.fit_ship_sprites", return_value={}),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[close_event]),
            mock.patch("pygame.display.flip"),
            mock.patch(
                "src.Menus.train_ai.save_training_ui_session",
                side_effect=lambda *_args, **_kwargs: order.append("save"),
            ),
            mock.patch(
                "pygame.quit",
                side_effect=lambda: order.append("pygame_quit"),
            ),
            mock.patch(
                "src.Menus.train_ai.sys.exit",
                side_effect=lambda: (_ for _ in ()).throw(self.StopRun()),
            ),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(order, ["save", "pygame_quit"])

    class FakeCoordinatedSession:
        created = []

        def __init__(self, records, **kwargs):
            self.records = records
            self.kwargs = kwargs
            self.active = True
            self.started = False
            self.proxies = {
                record.instance_id: SimpleNamespace(
                    slot=record.slot,
                    status=TrainingSessionStatus(
                        ship=record.slot.ship,
                        running=True,
                    ),
                    history=(),
                    log_lines=(),
                    set_display_on=lambda _enabled: None,
                )
                for record in records
            }
            self.__class__.created.append(self)

        def start(self):
            self.started = True

        def request_stop(self):
            for proxy in self.proxies.values():
                proxy.status.stopping = True

    def test_run_passes_shared_cache_and_save_coordinator_to_sessions(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 1
        self.FakeSession.created = []

        start_pos = self.trainee_start_position()
        start_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": start_pos},
        )
        footer_states = []
        trainee_action_states = []
        close_states = []

        def capture_footer(button, *_args, **_kwargs):
            if button.rect.y == train_ai.ACTION_TOP:
                footer_states.append((button.text, button.enabled))
            elif button.rect.y == 518:
                trainee_action_states.append((button.text, button.enabled))
            elif button.text == "Close":
                close_states.append((button.text, button.enabled))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.TrainingSession", self.FakeSession),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite},
            ),
            mock.patch(
                "src.Menus.train_ai._display_off_console_content",
                return_value=(("running",), (ui.WHITE,)),
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[start_event]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_footer,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(len(self.FakeSession.created), 1)
        created = self.FakeSession.created[0]
        cache = created.kwargs["opponent_model_cache"]
        coordinator = created.kwargs["save_coordinator"]

        self.assertTrue(created.started)
        self.assertIs(manager.active_session, created)
        self.assertEqual(manager.active_state.loaded_ship, "Earthling")
        self.assertEqual(manager.active_state.loaded_slot, 1)
        self.assertEqual(
            manager.active_state.loaded_architecture,
            architecture_for_state(manager.active_state),
        )
        self.assertIsNotNone(manager.active_state.loaded_training)
        self.assertIsInstance(cache, OpponentModelCache)
        self.assertIsInstance(coordinator, ModelSaveCoordinator)
        self.assertIs(cache._save_coordinator, coordinator)
        self.assertEqual(
            footer_states,
            [("Display", True), ("Stop all", True), ("Back", False)],
        )
        self.assertEqual(trainee_action_states, [("Stop", True)])
        self.assertEqual(close_states, [("Close", False)])

    def test_idle_start_is_disabled_for_selected_empty_slot_without_description(self):
        class EmptyRepository:
            def __init__(self, *_args):
                self.user_dir = Path("unused")

            def slot_for(self, ship, slot):
                return TrainingModelSlot(str(ship), int(slot), SLOT_EMPTY)

            def slots_for_ship(self, ship):
                return [self.slot_for(ship, slot) for slot in range(1, 5)]

        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 1
        footer_states = []
        trainee_action_states = []

        def capture_footer(button, *_args, **_kwargs):
            if button.rect.y == train_ai.ACTION_TOP:
                footer_states.append((button.text, button.enabled))
            elif button.rect.y == 518:
                trainee_action_states.append((button.text, button.enabled))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                EmptyRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_footer,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(
            footer_states,
            [("Display", True), ("Start synced", False), ("Back", True)],
        )
        self.assertEqual(trainee_action_states, [("Start", False)])

    def test_individual_start_confirms_and_resets_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
            pth_path = Path(directory) / "Earthling-01.pth"
            pth_path.write_bytes(b"old checkpoint")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Ready",
                architecture=model_architecture_metadata(256, 2),
                training={},
                progress={"completed_batches": 42},
            )
            current_schema_version = metadata["observation_schema_version"]
            metadata["observation_schema_version"] -= 1

            class ResettableRepository:
                def __init__(self):
                    self.user_dir = Path(directory)
                    self.slot = TrainingModelSlot(
                        "Earthling",
                        1,
                        SLOT_USER,
                        description="Ready",
                        pth_path=pth_path,
                        metadata=metadata,
                    )

                def slot_for(self, ship, slot):
                    if ship == "Earthling" and int(slot) == 1:
                        return self.slot
                    return TrainingModelSlot(str(ship), int(slot), SLOT_EMPTY)

                def slots_for_ship(self, ship):
                    return [self.slot_for(ship, slot) for slot in range(1, 5)]

                def create_or_update_user_model(self, updated_metadata):
                    self.slot = TrainingModelSlot(
                        "Earthling",
                        1,
                        SLOT_USER,
                        description=updated_metadata.get("description", ""),
                        pth_path=pth_path,
                        metadata=dict(updated_metadata),
                    )
                    return self.slot

            repository = ResettableRepository()
            manager = TrainingInstanceManager()
            manager.active_state.selected_ship = "Earthling"
            manager.active_state.selected_slot = 1
            self.FakeSession.created = []
            start_event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {
                    "button": 1,
                    "pos": self.trainee_start_position(),
                },
            )
            sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

            with (
                mock.patch(
                    "src.Menus.train_ai.TrainingInstanceManager",
                    return_value=manager,
                ),
                mock.patch(
                    "src.Menus.train_ai.load_training_ui_session",
                    return_value=manager,
                ),
                mock.patch(
                    "src.Menus.train_ai.TrainingModelRepository",
                    return_value=repository,
                ),
                mock.patch("src.Menus.train_ai.TrainingSession", self.FakeSession),
                mock.patch("src.Menus.train_ai.ConfirmationPrompt") as prompt_class,
                mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
                mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
                mock.patch(
                    "src.Menus.train_ai.fit_ship_sprites",
                    return_value={"Earthling": sprite},
                ),
                mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
                mock.patch("pygame.event.get", return_value=[start_event]),
                mock.patch("pygame.display.flip", side_effect=self.StopRun),
            ):
                with self.assertRaises(self.StopRun):
                    train_ai.run(screen)

                prompt_text, on_confirm = prompt_class.call_args.args
                self.assertIn("incompatible", prompt_text)
                self.assertEqual(self.FakeSession.created, [])
                on_confirm()

            self.assertEqual(pth_path.read_bytes(), b"")
            self.assertEqual(
                repository.slot.metadata["observation_schema_version"],
                current_schema_version,
            )
            self.assertEqual(
                repository.slot.metadata["progress"]["completed_batches"],
                0,
            )
            self.assertEqual(len(self.FakeSession.created), 1)
            self.assertTrue(self.FakeSession.created[0].started)

    def test_apply_all_loads_eligible_instances_and_warns_for_empty_ship(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        manager.add_instance()
        manager.select_instance(first.instance_id)
        manager.set_apply_future_changes_to_all(True)

        load_pos = (
            train_ai.CONTROL_WIDTH // 2,
            train_ai.CONTENT_TOP + 478 + 21,
        )
        load_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": load_pos},
        )
        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.InformationPrompt") as prompt_class,
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", side_effect=[[], [load_event]]),
            mock.patch(
                "pygame.display.flip",
                side_effect=[None, self.StopRun()],
            ),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(first.state.loaded_ship, "Earthling")
        self.assertIsNone(manager.instances[1].state.loaded_ship)
        prompt_class.assert_called_once_with(
            "Not all slots had eligible models to load"
        )

    def test_apply_all_delete_uses_typed_prompt_and_deletes_eligible_models(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        first.state.slot_labels[0] = "Earthling Ready"
        first.state.loaded_ship = "Earthling"
        first.state.loaded_slot = 1
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "Androsynth Ready"
        manager.select_instance(first.instance_id)
        manager.set_apply_future_changes_to_all(True)

        class DeletableRepository:
            def __init__(self):
                self.deleted = []
                self.slots = {
                    (ship, 1): TrainingModelSlot(
                        ship,
                        1,
                        SLOT_USER,
                        description=f"{ship} Ready",
                    )
                    for ship in ("Earthling", "Androsynth")
                }

            def slot_for(self, ship, slot):
                return self.slots.get(
                    (str(ship), int(slot)),
                    TrainingModelSlot(str(ship), int(slot), SLOT_EMPTY),
                )

            def slots_for_ship(self, ship):
                return [self.slot_for(ship, slot) for slot in range(1, 5)]

            def delete_user_model(self, ship, slot):
                key = (str(ship), int(slot))
                self.deleted.append(key)
                self.slots.pop(key, None)

        repository = DeletableRepository()
        layout = training_layout()
        delete_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {
                "button": 1,
                "pos": (
                    layout.content_rect.x + train_ai.CONTROL_WIDTH - 34,
                    layout.content_rect.y + 310,
                ),
            },
        )
        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                return_value=repository,
            ),
            mock.patch("src.Menus.train_ai.TypedDeleteConfirmationPrompt") as prompt_class,
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[delete_event]),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

            prompt_text, on_confirm = prompt_class.call_args.args
            self.assertIn("ALL 2 user models", prompt_text)
            self.assertIn("instances 01-02", prompt_text)
            self.assertEqual(repository.deleted, [])
            on_confirm()

        self.assertEqual(
            repository.deleted,
            [("Earthling", 1), ("Androsynth", 1)],
        )
        self.assertEqual(first.state.slot_labels[0], "")
        self.assertEqual(second.state.slot_labels[0], "")
        self.assertIsNone(first.state.loaded_ship)
        self.assertIsNone(first.state.loaded_slot)

    def test_start_all_automatically_enables_coordinated_cpu_workers(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        first = manager.active_instance
        manager.active_tab = "batch"
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        first.state.slot_labels[0] = "Earthling Ready"
        first.state.training_device = torch_backend.DEVICE_GPU
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "Androsynth Ready"
        second.state.training_device = torch_backend.DEVICE_GPU
        manager.select_instance(first.instance_id)
        self.FakeCoordinatedSession.created = []

        start_all_pos = self.synced_action_position()
        events = [
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": start_all_pos},
            ),
        ]
        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeCoordinatedRepository,
            ),
            mock.patch(
                "src.Menus.train_ai.CoordinatedTrainingSession",
                self.FakeCoordinatedSession,
            ),
            mock.patch("src.Menus.train_ai.torch_backend.get_torch", return_value=object()),
            mock.patch("src.Menus.train_ai.torch_backend.cuda_available", return_value=True),
            mock.patch(
                "src.Menus.train_ai.torch_backend.training_device_key",
                return_value=torch_backend.DEVICE_GPU,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=events),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(len(self.FakeCoordinatedSession.created), 1)
        created = self.FakeCoordinatedSession.created[0]
        self.assertTrue(created.started)
        self.assertTrue(
            created.kwargs["coordinated_cpu_workers_enabled"]
        )
        records_by_id = {record.instance_id: record for record in created.records}
        for instance in (first, second):
            record = records_by_id[instance.instance_id]
            self.assertEqual(instance.state.loaded_ship, record.slot.ship)
            self.assertEqual(instance.state.loaded_slot, record.slot.slot)
            self.assertEqual(
                instance.state.loaded_architecture,
                record.metadata["architecture"],
            )
            self.assertEqual(
                instance.state.loaded_training,
                record.metadata["training"],
            )

    def test_start_all_confirms_and_resets_only_incompatible_checkpoints(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        first.state.slot_labels[0] = "Earthling Ready"
        first.state.training_device = torch_backend.DEVICE_GPU
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "Androsynth Ready"
        second.state.training_device = torch_backend.DEVICE_GPU
        manager.select_instance(first.instance_id)

        repository = self.FakeCoordinatedRepository()
        for key, slot in tuple(repository.slots.items()):
            metadata = dict(slot.metadata)
            metadata["progress"] = {"completed_batches": 42}
            if key == ("Earthling", 1):
                metadata["observation_schema_version"] -= 1
            repository.slots[key] = TrainingModelSlot(
                slot.ship,
                slot.slot,
                SLOT_USER,
                description=slot.description,
                metadata=metadata,
            )

        start_all_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {
                "button": 1,
                "pos": self.synced_action_position(),
            },
        )
        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        self.FakeCoordinatedSession.created = []

        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                return_value=repository,
            ),
            mock.patch(
                "src.Menus.train_ai.CoordinatedTrainingSession",
                self.FakeCoordinatedSession,
            ),
            mock.patch("src.Menus.train_ai.ConfirmationPrompt") as prompt_class,
            mock.patch("src.Menus.train_ai.torch_backend.get_torch", return_value=object()),
            mock.patch("src.Menus.train_ai.torch_backend.cuda_available", return_value=True),
            mock.patch(
                "src.Menus.train_ai.torch_backend.training_device_key",
                return_value=torch_backend.DEVICE_GPU,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[start_all_event]),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

            prompt_text, on_confirm = prompt_class.call_args.args
            self.assertIn("01 (Earthling-01)", prompt_text)
            self.assertEqual(self.FakeCoordinatedSession.created, [])
            on_confirm()

        self.assertEqual(len(self.FakeCoordinatedSession.created), 1)
        created = self.FakeCoordinatedSession.created[0]
        records = {record.slot.ship: record for record in created.records}
        self.assertEqual(
            records["Earthling"].metadata["progress"]["completed_batches"],
            0,
        )
        self.assertEqual(
            records["Androsynth"].metadata["progress"]["completed_batches"],
            42,
        )

    def test_coordinated_stop_all_confirms_before_showing_stopping(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        first.state.slot_labels[0] = "Earthling Ready"
        first.state.training_device = torch_backend.DEVICE_GPU
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "Androsynth Ready"
        second.state.training_device = torch_backend.DEVICE_GPU
        manager.select_instance(first.instance_id)
        self.FakeCoordinatedSession.created = []

        start_all_pos = self.synced_action_position()
        click_start_all = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": start_all_pos},
        )
        click_stop_all = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": start_all_pos},
        )
        prompt = train_ai.ConfirmationPrompt("", lambda: None)
        confirm_stop_all = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": prompt.yes_button.rect.center},
        )
        footer_states = []

        def capture_footer(button, *_args, **_kwargs):
            if button.rect.y == train_ai.ACTION_TOP:
                footer_states.append((button.text, button.enabled))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeCoordinatedRepository,
            ),
            mock.patch(
                "src.Menus.train_ai.CoordinatedTrainingSession",
                self.FakeCoordinatedSession,
            ),
            mock.patch("src.Menus.train_ai.torch_backend.get_torch", return_value=object()),
            mock.patch("src.Menus.train_ai.torch_backend.cuda_available", return_value=True),
            mock.patch(
                "src.Menus.train_ai.torch_backend.training_device_key",
                return_value=torch_backend.DEVICE_GPU,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch(
                "pygame.event.get",
                side_effect=[
                    [click_start_all],
                    [click_stop_all],
                    [confirm_stop_all],
                ],
            ),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_footer,
            ),
            mock.patch(
                "pygame.display.flip",
                side_effect=[None, None, self.StopRun()],
            ),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(
            footer_states[:3],
            [("Display", True), ("Stop synced", True), ("Back", False)],
        )
        self.assertEqual(
            footer_states[3:6],
            [("Display", True), ("Stop synced", True), ("Back", False)],
        )
        self.assertEqual(
            footer_states[-3:],
            [("Display", True), ("Stopping synced", False), ("Back", False)],
        )

    def test_background_individual_run_keeps_selected_instance_start_available(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        selected = manager.active_instance
        selected.state.selected_ship = "Earthling"
        selected.state.selected_slot = 1
        selected.state.slot_labels[0] = "Ready"
        background = manager.add_instance()
        background.state.selected_ship = "Androsynth"
        background.state.selected_slot = 1
        background.state.slot_labels[0] = "Running"
        background.session = SimpleNamespace(
            status=TrainingSessionStatus(
                ship="Androsynth",
                running=True,
            ),
            history=(),
            log_lines=(),
            set_display_on=lambda _enabled: None,
        )
        manager.select_instance(selected.instance_id)
        footer_states = []
        trainee_action_states = []
        close_states = []

        def capture_footer(button, *_args, **_kwargs):
            if button.rect.y == train_ai.ACTION_TOP:
                footer_states.append((button.text, button.enabled))
            elif button.rect.y == 518:
                trainee_action_states.append((button.text, button.enabled))
            elif button.text == "Close":
                close_states.append((button.text, button.enabled))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_footer,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(
            footer_states,
            [("Display", True), ("Stop all", True), ("Back", False)],
        )
        self.assertEqual(trainee_action_states, [("Start", True)])
        self.assertEqual(close_states, [("Close", True)])

    def test_global_stop_all_stops_independent_runs_from_another_tab(self):
        class StoppableSession:
            def __init__(self):
                self.status = TrainingSessionStatus(ship="Earthling", running=True)
                self.history = ()
                self.log_lines = ()

            def request_stop(self):
                self.status.stopping = True

            def set_display_on(self, _enabled):
                pass

        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 1
        manager.active_state.slot_labels[0] = "Ready"
        manager.active_instance.session = StoppableSession()
        manager.active_tab = "opponent"
        stop_all_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": self.synced_action_position()},
        )
        footer_states = []

        def capture_footer(button, *_args, **_kwargs):
            if button.rect.y == train_ai.ACTION_TOP:
                footer_states.append((button.text, button.enabled))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ConfirmationPrompt") as prompt_class,
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[stop_all_event]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_footer,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

            prompt_text, on_confirm = prompt_class.call_args.args
            self.assertIn("stop all running", prompt_text)
            on_confirm()

        self.assertEqual(manager.active_tab, "opponent")
        self.assertEqual(
            footer_states,
            [("Display", True), ("Stop all", True), ("Back", False)],
        )
        self.assertTrue(manager.active_session.status.stopping)

    def test_apply_all_running_shows_stop_all_on_selected_stopped_instance(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        selected = manager.active_instance
        selected.state.selected_ship = "Earthling"
        selected.state.selected_slot = 1
        selected.state.slot_labels[0] = "Ready"
        background = manager.add_instance()
        background.state.selected_ship = "Androsynth"
        background.state.selected_slot = 1
        background.state.slot_labels[0] = "Running"
        background.session = SimpleNamespace(
            status=TrainingSessionStatus(ship="Androsynth", running=True),
            history=(),
            log_lines=(),
            set_display_on=lambda _enabled: None,
        )
        manager.select_instance(selected.instance_id)
        manager.set_apply_future_changes_to_all(True)
        trainee_action_states = []
        scope_states = []

        def capture_button(button, *_args, **_kwargs):
            if button.rect.y == 518:
                trainee_action_states.append((button.text, button.enabled))

        def capture_scope(checkbox, *_args, **_kwargs):
            scope_states.append(
                (checkbox.text, checkbox.is_checked, checkbox.enabled)
            )

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_button,
            ),
            mock.patch(
                "src.Menus.train_ai.TabScopeCheckbox.draw",
                new=capture_scope,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(trainee_action_states, [("Stop all", True)])
        self.assertEqual(
            scope_states,
            [("Apply actions to all instances", True, True)],
        )

    def test_apply_all_stop_from_stopped_instance_stops_every_running_instance(self):
        class StoppableSession:
            def __init__(self, ship):
                self.status = TrainingSessionStatus(ship=ship, running=True)
                self.history = ()
                self.log_lines = ()

            def request_stop(self):
                self.status.stopping = True

            def set_display_on(self, _enabled):
                pass

        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        selected = manager.active_instance
        selected.state.selected_ship = "Earthling"
        selected.state.selected_slot = 1
        selected.state.slot_labels[0] = "Ready"
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "Running"
        second.session = StoppableSession("Androsynth")
        manager.select_instance(selected.instance_id)
        manager.set_apply_future_changes_to_all(True)
        stop_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": self.trainee_start_position()},
        )
        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ConfirmationPrompt") as prompt_class,
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[stop_event]),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

            prompt_text, on_confirm = prompt_class.call_args.args
            self.assertIn("stop all running", prompt_text)
            on_confirm()

        self.assertTrue(second.session.status.stopping)
        self.assertIsNone(selected.session)

    def test_apply_all_idle_uses_bulk_load_start_and_delete_labels(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        manager.active_state.selected_ship = "Earthling"
        manager.active_state.selected_slot = 1
        manager.active_state.slot_labels[0] = "Ready"
        manager.set_apply_future_changes_to_all(True)
        trainee_action_states = []
        delete_states = []

        def capture_button(button, *_args, **_kwargs):
            if button.rect.y in (476, 518):
                trainee_action_states.append(
                    (button.text, button.enabled, button.rect.copy())
                )
            elif button.rect.width == train_ai.SLOT_DELETE_BUTTON_WIDTH:
                delete_states.append((button.text, button.enabled, button.rect.width))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_button,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        action_states = {
            text: (enabled, rect)
            for text, enabled, rect in trainee_action_states
        }
        self.assertTrue(action_states["Load all"][0])
        self.assertTrue(action_states["Start all eligible"][0])
        action_rects = tuple(rect for _enabled, rect in action_states.values())
        self.assertEqual(action_rects[0].width, action_rects[1].width)
        self.assertLess(action_rects[0].width, train_ai.CONTROL_WIDTH - 32)
        self.assertTrue(
            all(rect.centerx == train_ai.CONTROL_WIDTH // 2 for rect in action_rects)
        )
        body_font = train_ai.largest_fitting_font(
            train_ai.REWARD_LABELS,
            270,
            max_height=34,
            maximum=32,
        )
        longest_label_width = body_font.size(
            train_ai.TRAINEE_ACTION_LONGEST_LABEL
        )[0]
        self.assertGreaterEqual(
            action_rects[0].width - longest_label_width,
            2 * train_ai.TRAINEE_ACTION_HORIZONTAL_PADDING,
        )
        self.assertEqual(
            delete_states[0],
            ("X (ALL)", True, train_ai.SLOT_DELETE_BUTTON_WIDTH),
        )
        self.assertTrue(all(text == "X (ALL)" for text, _enabled, _width in delete_states))

    def test_apply_all_start_launches_every_eligible_independent_instance(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        first = manager.active_instance
        first.state.selected_ship = "Earthling"
        first.state.selected_slot = 1
        first.state.slot_labels[0] = "Earthling Ready"
        first.state.training_device = torch_backend.DEVICE_GPU
        second = manager.add_instance()
        second.state.selected_ship = "Androsynth"
        second.state.selected_slot = 1
        second.state.slot_labels[0] = "Androsynth Ready"
        second.state.training_device = torch_backend.DEVICE_GPU
        manager.select_instance(first.instance_id)
        manager.set_apply_future_changes_to_all(True)
        repository = self.FakeCoordinatedRepository()
        self.FakeSession.created = []
        start_event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": self.trainee_start_position()},
        )
        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)

        with (
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                return_value=repository,
            ),
            mock.patch("src.Menus.train_ai.TrainingSession", self.FakeSession),
            mock.patch("src.Menus.train_ai.ConfirmationPrompt") as prompt_class,
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite, "Androsynth": sprite},
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[start_event]),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

            prompt_text, on_confirm = prompt_class.call_args.args
            self.assertIn("Start all 2 eligible instances", prompt_text)
            on_confirm()

        self.assertEqual(len(self.FakeSession.created), 2)
        self.assertTrue(all(session.started for session in self.FakeSession.created))
        self.assertTrue(manager.is_running_or_stopping(first))
        self.assertTrue(manager.is_running_or_stopping(second))

    def test_individual_stopping_disables_stop_button(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        instance = manager.active_instance
        instance.state.selected_ship = "Earthling"
        instance.state.selected_slot = 1
        instance.state.slot_labels[0] = "Ready"
        instance.state.running = True
        instance.session = SimpleNamespace(
            status=TrainingSessionStatus(
                ship="Earthling",
                running=True,
                stopping=True,
            ),
            history=(),
            log_lines=(),
            set_display_on=lambda _enabled: None,
        )
        footer_states = []
        trainee_action_states = []

        def capture_footer(button, *_args, **_kwargs):
            if button.rect.y == train_ai.ACTION_TOP:
                footer_states.append((button.text, button.enabled))
            elif button.rect.y == 518:
                trainee_action_states.append((button.text, button.enabled))

        sprite = pygame.Surface((32, 32), pygame.SRCALPHA)
        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch(
                "src.Menus.train_ai.fit_ship_sprites",
                return_value={"Earthling": sprite},
            ),
            mock.patch(
                "src.Menus.train_ai._display_off_console_lines",
                return_value=("stopping",),
            ),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
            mock.patch("pygame.event.get", return_value=[]),
            mock.patch(
                "src.Menus.train_ai.ui_button.Button.draw",
                new=capture_footer,
            ),
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        self.assertEqual(
            footer_states,
            [("Display", True), ("Stopping all", False), ("Back", False)],
        )
        self.assertEqual(trainee_action_states, [("Stopping", False)])

    def test_disabled_start_all_shows_setup_settings_tooltip(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        manager = TrainingInstanceManager()
        start_all_pos = self.synced_action_position()

        draw_order = []
        with (
            mock.patch(
                "src.Menus.train_ai.TrainingInstanceManager",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.load_training_ui_session",
                return_value=manager,
            ),
            mock.patch(
                "src.Menus.train_ai.TrainingModelRepository",
                self.FakeRepository,
            ),
            mock.patch("src.Menus.train_ai.ui.load_background", return_value=None),
            mock.patch("src.Menus.train_ai.load_menu_ship_sprites", return_value={}),
            mock.patch("src.Menus.train_ai.fit_ship_sprites", return_value={}),
            mock.patch("pygame.mouse.get_pos", return_value=start_all_pos),
            mock.patch("pygame.event.get", return_value=[]),
            mock.patch(
                "src.Menus.train_ai._draw_hud_placeholders",
                side_effect=lambda *_args: draw_order.append("hud"),
            ),
            mock.patch(
                "src.Menus.train_ai.ui.draw_ship_tooltip",
                side_effect=lambda *_args: draw_order.append("tooltip"),
            ) as draw_tooltip,
            mock.patch("pygame.display.flip", side_effect=self.StopRun),
        ):
            with self.assertRaises(self.StopRun):
                train_ai.run(screen)

        draw_tooltip.assert_called_once()
        self.assertEqual(
            draw_tooltip.call_args.args[2],
            "At least two training instances are required",
        )
        self.assertLess(draw_order.index("hud"), draw_order.index("tooltip"))


class TrainingMenuClockTests(unittest.TestCase):
    def _clock(self, multiplier):
        clock = mock.Mock(multiplier=multiplier)
        clock.set_multiplier.side_effect = lambda value: setattr(
            clock,
            "multiplier",
            value,
        )
        clock.tick.return_value = 0.125
        return clock

    def test_display_off_paces_training_menu_at_base_24_fps(self):
        clock = self._clock(multiplier=const.VIDEO_FPS_MULTIPLIER)

        elapsed = train_ai._tick_training_menu_clock(clock, display_on=False)

        self.assertEqual(elapsed, 0.125)
        clock.set_multiplier.assert_called_once_with(1)
        clock.tick.assert_called_once_with()

    def test_display_on_restores_configured_display_multiplier(self):
        clock = self._clock(multiplier=1)

        elapsed = train_ai._tick_training_menu_clock(clock, display_on=True)

        self.assertEqual(elapsed, 0.125)
        clock.set_multiplier.assert_called_once_with(const.VIDEO_FPS_MULTIPLIER)
        clock.tick.assert_called_once_with()


class TrainingStartAllStyleTests(unittest.TestCase):
    def test_start_all_green_is_darker_with_matching_state_alpha(self):
        self.assertEqual(train_ai.START_ALL_GREEN, (0, 105, 0, ui.OK_GREEN[3]))
        self.assertEqual(train_ai.START_ALL_GREEN_HI[:3], (0, 105, 0))
        self.assertEqual(train_ai.START_ALL_GREEN_HI[3], ui.OK_GREEN_HI[3])


class TrainingLayoutTests(unittest.TestCase):
    def test_typed_delete_prompt_has_three_character_field_and_cursor(self):
        confirmed = []
        prompt = train_ai.TypedDeleteConfirmationPrompt(
            "Delete models? Type yes to confirm.",
            lambda: confirmed.append(True),
        )

        self.assertEqual(prompt.confirmation_field.max_length, 3)
        self.assertTrue(prompt.confirmation_field.active)
        self.assertFalse(prompt.delete_button.enabled)

        surface = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        font = pygame.font.SysFont(None, 32)
        button_font = pygame.font.SysFont(None, 28)
        with (
            mock.patch("pygame.time.get_ticks", return_value=100),
            mock.patch("pygame.mouse.get_pos", return_value=(0, 0)),
        ):
            prompt.draw(surface, font, button_font)
        cursor_pos = (
            prompt.confirmation_field.rect.left + 8,
            prompt.confirmation_field.rect.centery,
        )
        self.assertEqual(surface.get_at(cursor_pos)[:3], ui.WHITE)

        for key, character in (
            (pygame.K_y, "Y"),
            (pygame.K_e, "e"),
            (pygame.K_s, "S"),
            (pygame.K_x, "x"),
        ):
            prompt.handle_event(
                pygame.event.Event(
                    pygame.KEYDOWN,
                    {"key": key, "unicode": character},
                )
            )

        self.assertEqual(prompt.confirmation_field.text, "YeS")
        self.assertTrue(prompt.confirmation_ready)
        self.assertTrue(prompt.delete_button.enabled)
        prompt.handle_event(
            pygame.event.Event(
                pygame.KEYDOWN,
                {"key": pygame.K_RETURN, "unicode": "\r"},
            )
        )
        self.assertEqual(confirmed, [True])
        self.assertTrue(prompt.done)

    def test_idle_display_on_leaves_arena_black(self):
        surface = pygame.Surface((320, 240))
        surface.fill(ui.WHITE)
        arena = pygame.Rect(20, 20, 200, 120)
        state = TrainingUIState(display_on=True)

        train_ai._draw_arena_placeholder(
            surface,
            arena,
            state,
            pygame.font.SysFont(None, 24),
        )

        self.assertEqual(surface.get_at(arena.center)[:3], ui.BLACK)
        self.assertEqual(surface.get_at(arena.topleft)[:3], ui.GREY)

    def test_instance_strip_sits_above_tabs_and_content(self):
        layout = training_layout()
        instance_rect = pygame.Rect(
            8,
            INSTANCE_TOP,
            layout.control_rect.width - 16,
            INSTANCE_CONTROL_HEIGHT,
        )
        tab_rect = pygame.Rect(
            8,
            UI_TOP_MARGIN,
            layout.control_rect.width - 16,
            train_ai.TAB_HEIGHT + train_ai.TAB_GAP,
        )

        self.assertLessEqual(instance_rect.bottom, tab_rect.top)
        self.assertLessEqual(tab_rect.bottom, layout.content_rect.top)
        self.assertFalse(instance_rect.colliderect(layout.content_rect))
        self.assertGreaterEqual(UI_TOP_MARGIN - instance_rect.bottom, INSTANCE_SEPARATOR_HEIGHT)

    def test_apply_all_strip_sits_between_tab_content_and_action_row(self):
        layout = training_layout()
        strip = pygame.Rect(
            0,
            train_ai.APPLY_ALL_STRIP_TOP,
            train_ai.CONTROL_WIDTH,
            train_ai.APPLY_ALL_STRIP_HEIGHT,
        )

        self.assertEqual(strip.width, layout.control_rect.width)
        self.assertEqual(layout.content_rect.bottom, strip.top)
        self.assertEqual(strip.bottom, ACTION_TOP - train_ai.TAB_GAP)
        self.assertEqual(layout.tab_box_rect.left, strip.left)
        self.assertEqual(layout.tab_box_rect.right, strip.right)
        self.assertEqual(layout.tab_box_rect.bottom, strip.bottom)
        self.assertTrue(layout.tab_box_rect.contains(strip))

    def test_apply_all_strip_uses_square_tab_colors_and_new_label(self):
        checkbox = train_ai.TabScopeCheckbox(
            0,
            0,
            300,
            train_ai.APPLY_ALL_STRIP_HEIGHT,
            "Apply actions to all instances",
        )
        font = pygame.font.SysFont(None, 18)

        surface = pygame.Surface(checkbox.rect.size, pygame.SRCALPHA)
        checkbox.draw(surface, font, mouse_pos=(400, 400))
        self.assertEqual(
            surface.get_at((0, 0)),
            (*const.TAB_BUTTON_COLOR, const.TAB_BUTTON_NORMAL_ALPHA),
        )

        surface.fill((0, 0, 0, 0))
        checkbox.draw(surface, font, mouse_pos=(299, 29))
        self.assertEqual(
            surface.get_at((299, 29)),
            (*const.TAB_BUTTON_COLOR, const.TAB_BUTTON_HOVER_ALPHA),
        )

        surface.fill((0, 0, 0, 0))
        checkbox.is_checked = True
        checkbox.draw(surface, font, mouse_pos=(400, 400))
        self.assertEqual(
            surface.get_at((299, 29)),
            (*const.TAB_BUTTON_COLOR, const.TAB_BUTTON_SELECTED_ALPHA),
        )
        self.assertEqual(checkbox.text, "Apply actions to all instances")
        self.assertEqual(train_ai.TAB_BOX_BORDER_WIDTH, 3)

    def test_tab_box_uses_three_times_thicker_vertical_border(self):
        rect = pygame.Rect(0, 0, 60, 40)
        surface = pygame.Surface(rect.size)
        surface.fill(ui.BLACK)

        train_ai._draw_tab_box_border(surface, rect)

        self.assertEqual(
            train_ai.TAB_BOX_VERTICAL_BORDER_WIDTH,
            train_ai.TAB_BOX_BORDER_WIDTH * 3,
        )
        border_color = const.TAB_BUTTON_COLOR
        self.assertEqual(surface.get_at((8, rect.centery))[:3], border_color)
        self.assertEqual(surface.get_at((9, rect.centery))[:3], ui.BLACK)
        self.assertEqual(surface.get_at((rect.centerx, 2))[:3], border_color)
        self.assertEqual(surface.get_at((rect.centerx, 3))[:3], ui.BLACK)

    def test_tabs_use_an_opaque_tab_colored_border_in_every_state(self):
        tab = train_ai.TabButton(0, 0, 140, 34, "Trainee", lambda: None)
        font = pygame.font.SysFont(None, 18)
        expected_border = (*const.TAB_BUTTON_COLOR, 255)

        for active, mouse_pos in (
            (False, (300, 300)),
            (False, (70, 17)),
            (True, (300, 300)),
        ):
            surface = pygame.Surface(tab.rect.size, pygame.SRCALPHA)
            tab.active = active
            tab.draw(surface, font, mouse_pos=mouse_pos)
            self.assertEqual(surface.get_at((0, tab.rect.centery)), expected_border)

    def test_instance_buttons_leave_room_for_dropdown_label(self):
        layout = training_layout()
        summary_rect = pygame.Rect(
            8,
            INSTANCE_TOP,
            train_ai.INSTANCE_SUMMARY_WIDTH,
            INSTANCE_CONTROL_HEIGHT,
        )
        dropdown_width = (
            layout.control_rect.width
            - 2 * 8
            - summary_rect.width
            - INSTANCE_CLOSE_WIDTH
            - INSTANCE_ADD_WIDTH
            - 3 * INSTANCE_GAP
        )
        dropdown_rect = pygame.Rect(
            summary_rect.right + INSTANCE_GAP,
            INSTANCE_TOP,
            dropdown_width,
            INSTANCE_CONTROL_HEIGHT,
        )
        close_rect = pygame.Rect(
            dropdown_rect.right + INSTANCE_GAP,
            INSTANCE_TOP,
            INSTANCE_CLOSE_WIDTH,
            INSTANCE_CONTROL_HEIGHT,
        )
        add_rect = pygame.Rect(
            close_rect.right + INSTANCE_GAP,
            INSTANCE_TOP,
            INSTANCE_ADD_WIDTH,
            INSTANCE_CONTROL_HEIGHT,
        )

        self.assertGreaterEqual(dropdown_width, 320)
        self.assertEqual(add_rect.right, layout.control_rect.width - 8)

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

    def test_reward_values_use_hundredths_without_positive_sign(self):
        slider = RewardSlider((0, 0, 550, 40), REWARD_SPAWN_A1, 40.96)

        self.assertEqual(slider.format_value(), "40.96")
        slider.value = 0.0
        self.assertEqual(slider.format_value(), "0.00")
        slider.value = -5.12
        self.assertEqual(slider.format_value(), "-5.12")

    def test_ongoing_reward_uses_per_second_suffix_and_aligned_numeric_edge(self):
        font = pygame.font.Font(None, 20)
        discrete = RewardSlider((0, 0, 550, 40), REWARD_SPAWN_A1, 40.96)
        ongoing = RewardSlider((0, 40, 550, 40), REWARD_POINT_A1, 40.96)

        _, discrete_rect = discrete._rendered_value(font)
        _, ongoing_rect = ongoing._rendered_value(font)
        suffix_width = font.size("/s")[0]

        self.assertEqual(discrete.format_value(), "40.96")
        self.assertEqual(ongoing.format_value(), "40.96/s")
        self.assertEqual(discrete_rect.right, ongoing_rect.right - suffix_width)
        self.assertEqual(discrete.layout, SliderRow.LABEL_VALUE_SLIDER)

    def test_a2_reward_presentation_tracks_ship_without_changing_storage_key(self):
        slider = RewardSlider(
            (0, 0, 550, 40),
            REWARD_SPAWN_A2,
            10.24,
            ship_name="Ilwrath",
        )

        self.assertEqual(slider.reward_key, REWARD_SPAWN_A2)
        self.assertEqual(slider.label, "Maintain A2")
        self.assertEqual(slider.format_value(), "10.24/s")

        slider.set_ship("Earthling")

        self.assertEqual(slider.reward_key, REWARD_SPAWN_A2)
        self.assertEqual(slider.label, REWARD_SPAWN_A2)
        self.assertEqual(slider.format_value(), "10.24")


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

    def test_label_value_slider_right_justifies_values_before_handle(self):
        font = pygame.font.Font(None, 20)
        short = SliderRow(
            (16, 20, 544, 34),
            "Starting Epsilon",
            0,
            1,
            0.5,
            decimal_places=3,
            layout=SliderRow.LABEL_VALUE_SLIDER,
            slider_width=184,
        )
        long = SliderRow(
            (16, 60, 544, 34),
            "Starting Epsilon",
            0,
            1,
            0.5,
            decimal_places=3,
            value_suffix=" (0.223)",
            layout=SliderRow.LABEL_VALUE_SLIDER,
            slider_width=184,
        )

        _, short_rect = short._rendered_value(font)
        _, long_rect = long._rendered_value(font)
        expected_right = (
            short.line_rect.left
            - short.handle_radius
            - short.VALUE_HANDLE_GAP
        )

        self.assertEqual(short_rect.right, expected_right)
        self.assertEqual(long_rect.right, expected_right)
        self.assertLess(long_rect.right, long.line_rect.left - long.handle_radius)
        self.assertLess(long_rect.left, short_rect.left)

    def test_starting_epsilon_value_includes_current_epsilon(self):
        slider = SliderRow(
            (16, 20, 544, 34),
            "Starting Epsilon",
            0,
            1,
            0.5,
            decimal_places=3,
            value_suffix=" (0.223)",
            layout=SliderRow.LABEL_VALUE_SLIDER,
            slider_width=184,
        )

        self.assertEqual(slider.format_value(), "0.500 (0.223)")

    def test_short_count_formats_regimen_context_values(self):
        self.assertEqual(_format_short_count(30000), "30k")
        self.assertEqual(_format_short_count(15000000), "15M")

    def test_update_to_data_ratio_uses_replay_samples_without_augmentation(self):
        self.assertEqual(
            _format_update_to_data_ratio(4096, 10, 30000),
            "1.37",
        )

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
    def test_legacy_regimen_implies_current_reflection_augmentation(self):
        legacy = {"regimen": {"minibatch_size": 32}}
        current = {
            "regimen": {
                "minibatch_size": 32,
                REFLECTION_AUGMENTATION_METADATA_KEY: (
                    REFLECTION_AUGMENTATION_MODE
                ),
            }
        }

        self.assertTrue(_training_settings_match(legacy, current))

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
        self.assertEqual(BATCH_GROUPING_VALUES, tuple(range(5, 201, 5)))
        self.assertEqual(
            MATCH_TIME_LIMIT_VALUES,
            tuple(range(240, 12001, 240)),
        )
        self.assertEqual(
            MINIBATCH_SIZE_VALUES,
            (16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        )
        self.assertEqual(
            REPLAY_UPDATES_PER_BATCH_VALUES,
            (10, 15) + tuple(range(20, 501, 10)),
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
            EPSILON_FLOOR_VALUES,
            tuple(round(i * 0.005, 3) for i in range(31)),
        )
        self.assertEqual(
            EPSILON_DECAY_VALUES,
            tuple(round(0.950 + i * 0.001, 3) for i in range(51)),
        )
        self.assertEqual(EPSILON_FRAME_SPAN_VALUES, tuple(range(1, 49)))
        self.assertEqual(
            GAMMA_VALUES,
            tuple(round(0.950 + i * 0.001, 3) for i in range(50)),
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
        state.training_device = "cpu"
        state.starting_epsilon = 0.2
        state.current_epsilon = 0.125
        state.epsilon_floor = 0.075
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
        self.assertEqual(config.training_device, "cpu")
        self.assertEqual(config.starting_epsilon, 0.2)
        self.assertEqual(config.epsilon, 0.125)
        self.assertEqual(config.epsilon_floor, 0.075)
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
            replay_path = replay_checkpoint_path(pth_path)
            pth_path.write_bytes(b"checkpoint")
            csv_path.write_text("old,csv\n", encoding="utf-8")
            replay_path.write_bytes(b"replay")

            _clear_reset_model_artifacts(SimpleNamespace(pth_path=pth_path))

            self.assertEqual(pth_path.read_bytes(), b"")
            self.assertFalse(csv_path.exists())
            self.assertFalse(replay_path.exists())


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
            "ship": "Androsynth",
            "completed_batches": 15,
            "current_round": 12,
            "total_rounds": 25,
            "current_opponent": "Mycon",
            "current_frame": 100,
            "current_frame_limit": 1200,
            "replay_size": 30000,
            "last_action_exploratory": True,
            "weighted_total_return": -0.01,
            "recent_loss": 0.2849,
            "learning_rate": 0.0003,
            "current_epsilon": 0.36577,
            "epsilon_decay": 0.99,
            "gamma": 0.99,
            "component_totals": {"Kill enemy": 2.0},
            "previous_opponent": "Chenjesu",
            "batch_component_totals": {},
            "elapsed_training_seconds": 91266.0,
            "batches_per_hour": 5.69,
            "simulation_speed_multiplier": 20.25,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_display_off_console_includes_batch_logs_before_live_status(self):
        status = self._status()

        lines = _display_off_console_lines(status, ("Batch      1 | summary",))

        self.assertIn("Current batch", lines)
        self.assertEqual(lines[0], "Completed batches")
        self.assertLess(
            lines.index("Batch      1 | summary"),
            lines.index("Current batch"),
        )
        self.assertEqual(
            lines[lines.index("Batch      1 | summary") + 2],
            "|]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
            "----------------------------------------| "
            "20.25x Real time",
        )
        current_batch_index = lines.index("Current batch")
        self.assertEqual(
            lines[current_batch_index : current_batch_index + 16],
            (
                "Current batch",
                "Ship      | Androsynth",
                "Status    |    Running           ",
                "Opponent  |      Mycon             ",
                "Time      |         25h:21m:06s",
                "Replay    |      30000 frames",
                "Batch     |         16",
                "Round     |         12 / 25",
                "Frame     |        100 / 1200",
                "Batches/h |          5.69",
                "Reward    |         -0.0100",
                "Loss      |          0.2849",
                "Gamma     |          0.990",
                "Eps decay |          0.990",
                "Epsilon   |          0.36577",
                "Learning  |          0.00030",
            ),
        )
        reward_name_width = max(len(label) for label in REWARD_LABELS)
        self.assertIn(
            f"{'Reward components':<{reward_name_width}} |   Chenjesu |  Batch -",
            lines,
        )
        self.assertIn(
            f"{'Kill enemy':<{reward_name_width}} |     2.0000 |        -",
            lines,
        )

    def test_display_off_console_assigns_requested_section_colors(self):
        status = self._status()
        history = ("Batch      1 | first", "Batch      2 | second")

        lines, colors = train_ai._display_off_console_content(status, history)

        self.assertEqual(colors[lines.index("Completed batches")], (255, 255, 255))
        self.assertEqual(colors[lines.index(history[0])], (155, 255, 155))
        self.assertEqual(colors[lines.index(history[1])], (155, 155, 255))

        speed_line, speed_legend = _speedometer_console_lines(status)
        self.assertEqual(colors[lines.index(speed_line)], (0, 255, 0))
        self.assertEqual(colors[lines.index(speed_legend)], (255, 255, 255))

        current_heading = lines.index("Current batch")
        self.assertEqual(colors[current_heading], (255, 255, 255))
        self.assertEqual(colors[current_heading + 1], (155, 255, 155))
        self.assertEqual(colors[current_heading + 2], (155, 155, 255))

        reward_heading = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("Reward components")
        )
        self.assertEqual(colors[reward_heading], (255, 255, 255))
        self.assertEqual(colors[reward_heading + 1], (155, 255, 155))
        self.assertEqual(colors[reward_heading + 2], (155, 155, 255))

    def test_display_off_console_keeps_error_details_visible(self):
        status = self._status(
            running=False,
            error="worker 1 exited unexpectedly with code 1\nstartup traceback",
        )

        lines = _display_off_console_lines(status, ())

        self.assertIn("Training error", lines)
        self.assertIn("worker 1 exited unexpectedly with code 1", lines)
        self.assertIn("startup traceback", lines)
        self.assertIn("Current batch", lines)

    def test_batch_log_font_size_keeps_log_fitting(self):
        self.assertEqual(TRAINING_BATCH_LOG_FONT_SIZE, 11)

    def test_speedometer_caps_bar_without_capping_speed_label(self):
        self.assertEqual(
            _speedometer_console_lines(
                self._status(simulation_speed_multiplier=45.125)
            ),
            (
                "|]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
                "]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]| "
                "45.12x Real time",
                "0         5        10        15        20        25        30"
                "        35        40",
            ),
        )

    def test_speedometer_aligns_single_digit_speed_at_decimal(self):
        speed_line, _scale_line = _speedometer_console_lines(
            self._status(simulation_speed_multiplier=5.25)
        )

        self.assertTrue(speed_line.endswith("|  5.25x Real time"))

    def test_current_batch_ship_name_is_right_aligned(self):
        lines = _display_off_console_lines(self._status(ship="Mycon"), ())

        self.assertIn("Ship      |      Mycon", lines)

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
        self.assertEqual(_format_training_duration(0), "0h:00m:00s")
        self.assertEqual(_format_training_duration(65.9), "0h:01m:05s")
        self.assertEqual(_format_training_duration(3723), "1h:02m:03s")


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

    def test_up_and_down_keys_scroll_when_pointer_is_over_log(self):
        box, rect, font = self._box_with_lines()
        bottom_line = box.scroll_line

        with mock.patch("pygame.mouse.get_pos", return_value=rect.center):
            handled = box.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP}),
                rect,
                font,
            )

        self.assertTrue(handled)
        self.assertEqual(box.scroll_line, bottom_line - 1)

        with mock.patch("pygame.mouse.get_pos", return_value=(-1, -1)):
            handled = box.handle_event(
                pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN}),
                rect,
                font,
            )

        self.assertFalse(handled)
        self.assertEqual(box.scroll_line, bottom_line - 1)

    def test_scrollbar_track_pages_and_thumb_drags(self):
        box, rect, font = self._box_with_lines(40)
        track, thumb = box._scrollbar_geometry(rect, font)

        self.assertEqual(thumb.bottom, track.bottom)
        box.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": (track.centerx, track.top + 1)},
            ),
            rect,
            font,
        )
        self.assertLess(box.scroll_line, box._max_scroll_line())

        track, thumb = box._scrollbar_geometry(rect, font)
        box.handle_event(
            pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {"button": 1, "pos": thumb.center},
            ),
            rect,
            font,
        )
        box.handle_event(
            pygame.event.Event(
                pygame.MOUSEMOTION,
                {"pos": (track.centerx, track.top)},
            ),
            rect,
            font,
        )

        self.assertEqual(box.scroll_line, 0)

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

    def test_same_lines_are_noop(self):
        box = TrainingBatchLogBox()
        lines = tuple(f"line {index}" for index in range(3))
        box.set_lines(lines)
        original_lines = box.lines
        box.scroll_line = 2

        box.set_lines(tuple(lines))

        self.assertIs(box.lines, original_lines)
        self.assertEqual(box.scroll_line, 2)


class TrainingBattleDisplayTests(unittest.TestCase):
    def test_playback_interpolates_one_physics_frame_at_configured_ui_rate(self):
        playback = train_ai.TrainingDisplayPlayback()
        status = SimpleNamespace(battle_view={"frame_id": 10})

        self.assertEqual(playback.interpolation_for(1, status, 0.01), 0.0)
        self.assertAlmostEqual(
            playback.interpolation_for(1, status, 1.0 / const.VIDEO_FPS),
            const.FPS / const.VIDEO_FPS,
        )
        self.assertEqual(playback.interpolation_for(1, status, 1.0), 1.0)

        status.battle_view = {"frame_id": 11}
        self.assertEqual(playback.interpolation_for(1, status, 0.01), 0.0)

    def test_playback_resets_when_switching_instances(self):
        playback = train_ai.TrainingDisplayPlayback()
        status = SimpleNamespace(battle_view={"frame_id": 10})
        playback.interpolation_for(1, status, 0.0)
        playback.interpolation_for(1, status, 1.0)

        self.assertEqual(playback.interpolation_for(2, status, 0.1), 0.0)

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
                interp_t=0.5,
            )

        args, kwargs = controller.draw.call_args
        layout = args[2]
        options = kwargs["options"]

        self.assertEqual(layout.arena_rect, rect)
        self.assertIsNone(layout.player1_hud_rect)
        self.assertIsNone(layout.player2_hud_rect)
        self.assertIs(args[4], star_field_renderer)
        self.assertFalse(options.draw_huds)
        self.assertEqual(options.interp_t, 0.5)
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

    def test_worker_rendered_frame_blits_without_reconstructing_object_graph(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        frame = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        frame.fill((12, 34, 56))
        status = SimpleNamespace(
            battle_view={"frame_id": 1, "rendered_frames": (frame,)}
        )
        controller = mock.Mock()

        _draw_training_battle(
            screen,
            training_layout().arena_rect,
            status,
            object(),
            controller,
            interp_t=0.5,
        )

        controller.draw.assert_not_called()
        self.assertEqual(
            screen.get_at(training_layout().arena_rect.center)[:3],
            (12, 34, 56),
        )

    def test_worker_rendered_huds_crop_native_panel_height_before_scaling(self):
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        frame = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        frame.fill((0, 0, 255))
        hud_rects = training_layout().hud_rects
        source_height = hud_rects[0].height
        frame.fill((255, 0, 0), pygame.Rect(0, 0, const.SCREEN_LEFT, source_height))
        frame.fill(
            (0, 255, 0),
            pygame.Rect(
                const.SCREEN_LEFT + const.SCREEN_HEIGHT,
                0,
                const.SCREEN_WIDTH - const.SCREEN_LEFT - const.SCREEN_HEIGHT,
                source_height,
            ),
        )
        status = SimpleNamespace(
            battle_view={"frame_id": 1, "rendered_frames": (frame,)}
        )
        controller = mock.Mock()

        _draw_training_huds(
            screen,
            hud_rects,
            status,
            object(),
            controller,
        )

        controller.draw.assert_not_called()
        left_color = screen.get_at(hud_rects[0].center)[:3]
        right_color = screen.get_at(hud_rects[1].center)[:3]
        self.assertGreater(left_color[0], 240)
        self.assertLess(left_color[2], 5)
        self.assertGreater(right_color[1], 240)
        self.assertLess(right_color[2], 5)

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
            interp_t=0.5,
        )

        args, kwargs = controller.draw.call_args
        layout = args[2]
        options = kwargs["options"]

        self.assertEqual(layout.player1_hud_rect, hud_rects[0])
        self.assertEqual(layout.player2_hud_rect, hud_rects[1])
        self.assertIs(args[4], star_field_renderer)
        self.assertFalse(options.draw_arena)
        self.assertEqual(options.interp_t, 0.5)

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
