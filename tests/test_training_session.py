import csv
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.training import torch_backend
from src.audio import RecordingAudioService
from src.training.contracts import OBSERVATION_SCHEMA_VERSION
from src.training.coordinated_contracts import TrainingEpisodeResult
from src.training.model_registry import (
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
    replay_checkpoint_path,
)
from src.training.opponent_cache import (
    ModelSaveCoordinator,
    OpponentModelCache,
    OpponentModelKey,
)
from src.training.orchestration import (
    OPPONENT_MODE_EXISTING_AI,
    OpponentDiscoveryResult,
    OpponentSpec,
    TrainingBatchAborted,
    TrainingBatchResult,
    TrainingOrchestrationConfig,
    TrainingRoundResult,
)
from src.training.replay import TrainingReplayBuffer, save_training_checkpoint
from src.training.render_view import freeze_battle_view
from src.training.session import (
    BatchMetrics,
    TrainingSession,
    TrainingSessionStatus,
    append_grouped_metrics_csv,
    batch_metrics_history_from_metadata,
    batch_metrics_to_metadata,
    format_batch_summary_line,
    metrics_from_batch_result,
    rolling_metrics,
    validate_model_metadata,
)
from src.training.value_network import ValueNetworkConfig, build_value_network


def _episode_result(
    *,
    win=False,
    loss=False,
    draw=True,
    terminal_reason=None,
    kills=0,
    deaths=0,
):
    return TrainingEpisodeResult(
        opponent=OpponentSpec("Earthling"),
        frames=1,
        terminal_reason=terminal_reason or ("timeout" if draw else "resolved"),
        mature_samples=1,
        total_return=0.0,
        win=win,
        loss=loss,
        draw=draw,
        kills=kills,
        deaths=deaths,
    )


def _round_result(
    total_return=0.0,
    *,
    win=False,
    loss=False,
    draw=True,
    episode_results=(),
):
    return TrainingRoundResult(
        opponent=OpponentSpec("Earthling"),
        frames=1,
        terminal_reason="timeout",
        mature_samples=1,
        total_return=total_return,
        win=win,
        loss=loss,
        draw=draw,
        episode_results=tuple(episode_results),
    )


class _RecordingSaveSession(TrainingSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved_batches = []
        self.saved_replay_flags = []

    def _save_state(self, model, optimizer, replay_buffer, *, include_replay=True):
        super()._save_state(
            model,
            optimizer,
            replay_buffer,
            include_replay=include_replay,
        )
        self.saved_batches.append(self.status.completed_batches)
        self.saved_replay_flags.append(include_replay)


class _NoSaveSession(TrainingSession):
    def _save_state(self, model, optimizer, replay_buffer, *, include_replay=True):
        return None


class _MutableOpponentCache:
    def __init__(self, snapshot):
        self.snapshot_value = snapshot
        self.load_initial_calls = 0

    def load_initial(self, repository, *, device_choice=None):
        self.load_initial_calls += 1

    def snapshot(self, *, device_choice=None):
        return self.snapshot_value


class TrainingMetricsTests(unittest.TestCase):
    def test_batch_summary_line_uses_specified_format(self):
        metrics = BatchMetrics(
            batch=86,
            kills=2,
            deaths=4,
            batch_count=1,
            average_match_score=0.809,
            epsilon=0.26545,
            learning_rate=0.0003,
            average_loss=1.2658,
        )
        rolling = BatchMetrics(
            batch=86,
            kills=231,
            deaths=362,
            batch_count=100,
            average_match_score=0.819,
            epsilon=0.26545,
            learning_rate=0.0003,
            average_loss=1.2658,
        )

        self.assertEqual(
            format_batch_summary_line(metrics, rolling),
            "#     86 |    2 K (   2.31),    4 D (   3.62) | "
            "Score:   0.809 (  0.819) | Epsilon: 0.26545 | "
            "LR: 0.00030 | Loss: 1.2658 (1.2658)",
        )

    def test_batch_summary_keeps_negative_score_sign_adjacent(self):
        metrics = BatchMetrics(86, 2, 4, 1, -0.809, 0.26545, 0.0003, 1.2658)
        rolling = BatchMetrics(
            86,
            231,
            362,
            100,
            -0.819,
            0.26545,
            0.0003,
            1.2658,
        )

        summary = format_batch_summary_line(metrics, rolling)

        self.assertIn("Score:  -0.809 ( -0.819)", summary)
        self.assertNotIn("- 0.809", summary)
        self.assertNotIn("- 0.819", summary)

    def test_metrics_from_batch_result_counts_kills_deaths_and_average_score(self):
        result = TrainingBatchResult(
            completed_rounds=2,
            replay_size=10,
            optimization_losses=(0.2, 0.4),
            round_results=(
                _round_result(
                    10.0,
                    win=True,
                    loss=True,
                    draw=False,
                    episode_results=(
                        _episode_result(win=True, draw=False, kills=1),
                        _episode_result(loss=True, draw=False, deaths=1),
                        _episode_result(
                            terminal_reason="resolved",
                            kills=1,
                            deaths=1,
                        ),
                    ),
                ),
                _round_result(2.0, episode_results=(_episode_result(),)),
            ),
        )

        metrics = metrics_from_batch_result(
            result,
            batch=7,
            epsilon=0.1,
            learning_rate=0.001,
        )

        self.assertEqual((metrics.kills, metrics.deaths), (2, 2))
        self.assertEqual(metrics.batch_count, 1)
        self.assertAlmostEqual(metrics.average_match_score, 6.0)
        self.assertAlmostEqual(metrics.average_loss, 0.3)

    def test_metrics_do_not_infer_kills_or_deaths_from_outcomes(self):
        result = TrainingBatchResult(
            completed_rounds=1,
            replay_size=0,
            optimization_losses=(),
            round_results=(
                _round_result(
                    win=True,
                    loss=True,
                    draw=False,
                    episode_results=(
                        _episode_result(win=True, draw=False),
                        _episode_result(loss=True, draw=False, deaths=1),
                    ),
                ),
            ),
        )

        metrics = metrics_from_batch_result(
            result,
            batch=1,
            epsilon=0.0,
            learning_rate=0.001,
        )

        self.assertEqual((metrics.kills, metrics.deaths), (0, 1))

    def test_rolling_metrics_uses_available_window(self):
        history = (
            BatchMetrics(1, 1, 2, 1, 10.0, 0.1, 0.001, 1.0),
            BatchMetrics(2, 3, 0, 1, 20.0, 0.2, 0.002, 3.0),
        )

        rolling = rolling_metrics(history, grouping=5)

        self.assertEqual(rolling.batch, 2)
        self.assertEqual((rolling.kills, rolling.deaths), (4, 2))
        self.assertEqual(rolling.batch_count, 2)
        self.assertAlmostEqual(rolling.average_kills, 2.0)
        self.assertAlmostEqual(rolling.average_deaths, 1.0)
        self.assertAlmostEqual(rolling.average_match_score, 15.0)
        self.assertAlmostEqual(rolling.average_loss, 2.0)
        self.assertAlmostEqual(rolling.epsilon, 0.2)
        self.assertAlmostEqual(rolling.learning_rate, 0.002)

    def test_csv_append_writes_header_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.csv"

            append_grouped_metrics_csv(
                path,
                BatchMetrics(1000, 2, 4, 1, 34.3, 0.002, 0.0001, 0.05),
                BatchMetrics(1000, 231, 362, 100, 33.3, 0.002, 0.0001, 0.04),
            )
            append_grouped_metrics_csv(
                path,
                BatchMetrics(2000, 5, 3, 1, 35.0, 0.003, 0.0002, 0.06),
                BatchMetrics(2000, 250, 350, 100, 34.0, 0.003, 0.0002, 0.05),
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))

        self.assertEqual(
            rows,
            [
                [
                    "Batch", "Kills", "Average Kills", "Deaths",
                    "Average Deaths", "Score", "Average Score", "Epsilon",
                    "Learning Rate", "Loss", "Average Loss",
                ],
                [
                    "1000", "2", "2.31", "4", "3.62", "34.3", "33.3",
                    "0.00200", "0.000100", "0.0500", "0.0400",
                ],
                [
                    "2000", "5", "2.50", "3", "3.50", "35.0", "34.0",
                    "0.00300", "0.000200", "0.0600", "0.0500",
                ],
            ],
        )

    def test_csv_append_writes_current_counts_and_rolling_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.csv"

            append_grouped_metrics_csv(
                path,
                BatchMetrics(50, 4, 3, 1, 12.5, 0.100, 0.001, 0.25),
                BatchMetrics(50, 12, 9, 3, 11.5, 0.100, 0.001, 0.20),
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))

        self.assertEqual(
            rows[-1],
            [
                "50", "4", "4.00", "3", "3.00", "12.5", "11.5",
                "0.10000", "0.001000", "0.2500", "0.2000",
            ],
        )

    def test_batch_metrics_history_round_trips_through_metadata(self):
        metrics = BatchMetrics(12, 9, 10, 1, 42.5, 0.1, 0.001, 0.25)
        metadata = {
            "progress": {
                "completed_batches": 12,
                "recent_batch_metrics": [batch_metrics_to_metadata(metrics)],
            }
        }

        self.assertEqual(batch_metrics_history_from_metadata(metadata), (metrics,))

    def test_grouped_kill_death_averages_survive_metadata_round_trip(self):
        history = (
            BatchMetrics(12, 1, 1, 1, 42.5, 0.1, 0.001, 0.25),
            BatchMetrics(13, 3, 1, 1, 43.5, 0.1, 0.001, 0.35),
        )
        metadata = {
            "progress": {
                "completed_batches": 13,
                "recent_batch_metrics": [
                    batch_metrics_to_metadata(metrics) for metrics in history
                ],
            }
        }

        restored = batch_metrics_history_from_metadata(metadata)
        rolling = rolling_metrics(restored, grouping=2)

        self.assertEqual((rolling.kills, rolling.deaths), (4, 2))
        self.assertAlmostEqual(rolling.average_kills, 2.0)
        self.assertAlmostEqual(rolling.average_deaths, 1.0)


class TrainingCompatibilityTests(unittest.TestCase):
    def test_incompatible_schema_is_an_error(self):
        metadata = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="Test",
            architecture=model_architecture_metadata(8, 1),
            training={},
        )
        metadata["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION + 1

        report = validate_model_metadata(metadata)

        self.assertFalse(report.ok)
        self.assertIn("observation schema is incompatible", report.errors)

    def test_game_setting_difference_is_a_warning(self):
        metadata = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="Test",
            architecture=model_architecture_metadata(8, 1),
            training={},
            game_settings={
                "ship_directions": 16,
                "asteroid_count": 5,
                "repeat_key_delay": 3,
                "fps": 24,
            },
        )

        report = validate_model_metadata(
            metadata,
            game_settings={
                "ship_directions": 32,
                "asteroid_count": 5,
                "repeat_key_delay": 3,
                "fps": 24,
            },
        )

        self.assertTrue(report.ok)
        self.assertEqual(report.warnings, ("ship directions differs: model 16, current 32",))


class TrainingSessionDisplayTests(unittest.TestCase):
    def test_freeze_battle_view_copies_mutable_render_state_and_remaps_refs(self):
        ship = SimpleNamespace(
            position=[100.0, 200.0],
            previous_position=[90.0, 190.0],
            heading=3,
            previous_heading=2,
        )
        ability = SimpleNamespace(
            position=[105.0, 205.0],
            previous_position=[104.0, 204.0],
            parent=ship,
        )
        view = {
            "game_objects": (ship, ability),
            "original_ships": (ship,),
            "camera_targets": (ship,),
            "border_color": (50, 50, 50),
            "frame_id": 7,
        }

        frozen = freeze_battle_view(view)
        frozen_ship = frozen["game_objects"][0]
        frozen_ability = frozen["game_objects"][1]

        ship.position[0] = 999.0
        ship.previous_position[0] = 998.0
        ship.heading = 12

        self.assertIsNot(frozen_ship, ship)
        self.assertEqual(frozen_ship.position, [100.0, 200.0])
        self.assertEqual(frozen_ship.previous_position, [90.0, 190.0])
        self.assertEqual(frozen_ship.heading, 3)
        self.assertIs(frozen_ability.parent, frozen_ship)
        self.assertIs(frozen["original_ships"][0], frozen_ship)
        self.assertIs(frozen["camera_targets"][0], frozen_ship)

    def test_live_display_toggle_throttles_frame_progress(self):
        session = TrainingSession.__new__(TrainingSession)
        session._lock = threading.Lock()
        session._status = TrainingSessionStatus()
        session._display_on = threading.Event()
        session._display_on.set()
        session._next_display_frame_time = 100.0

        with (
            mock.patch("src.training.session.time.perf_counter", return_value=100.0),
            mock.patch("src.training.session.time.sleep") as sleep,
        ):
            session._on_progress({"event": "frame", "frame": 1})

        sleep.assert_called_once()

    def test_headless_frame_progress_updates_live_status_every_100_frames(self):
        session = TrainingSession.__new__(TrainingSession)
        session._lock = threading.Lock()
        session._status = TrainingSessionStatus()
        session._display_on = threading.Event()

        session._on_progress(
            {
                "event": "frame",
                "frame": 99,
                "replay_size": 99,
                "weighted_total_return": 9.9,
            }
        )

        self.assertEqual(session.status.current_frame, 0)
        self.assertEqual(session.status.replay_size, 0)
        self.assertEqual(session.status.weighted_total_return, 0.0)

        session._on_progress(
            {
                "event": "frame",
                "frame": 100,
                "replay_size": 100,
                "weighted_total_return": 10.0,
            }
        )

        self.assertEqual(session.status.current_frame, 100)
        self.assertEqual(session.status.replay_size, 100)
        self.assertEqual(session.status.weighted_total_return, 10.0)

    def test_headless_progress_drops_battle_view_without_freezing(self):
        session = TrainingSession.__new__(TrainingSession)
        session._lock = threading.Lock()
        session._status = TrainingSessionStatus()
        session._display_on = threading.Event()

        with mock.patch("src.training.session.freeze_battle_view") as freeze:
            session._on_progress(
                {
                    "event": "frame",
                    "frame": 1,
                    "battle_view": {"game_objects": ()},
                }
            )

        freeze.assert_not_called()
        self.assertIsNone(session.status.battle_view)

    def test_display_progress_stores_frozen_battle_view(self):
        session = TrainingSession.__new__(TrainingSession)
        session._lock = threading.Lock()
        session._status = TrainingSessionStatus()
        session._display_on = threading.Event()
        session._display_on.set()
        session._next_display_frame_time = 100.0
        frozen_view = {"game_objects": ("frozen",)}

        with (
            mock.patch(
                "src.training.session.freeze_battle_view",
                return_value=frozen_view,
            ) as freeze,
            mock.patch("src.training.session.time.perf_counter", return_value=100.0),
            mock.patch("src.training.session.time.sleep"),
        ):
            session._on_progress(
                {
                    "event": "frame",
                    "frame": 1,
                    "battle_view": {"game_objects": ("live",)},
                }
            )

        freeze.assert_called_once()
        self.assertIs(session.status.battle_view, frozen_view)
        self.assertEqual(session.status.current_frame, 1)

    def test_batch_optimization_progress_shows_display_message_and_clears_view(self):
        session = TrainingSession.__new__(TrainingSession)
        session._lock = threading.Lock()
        session._status = TrainingSessionStatus(
            battle_view={"game_objects": ("last-frame",)}
        )
        session._display_on = threading.Event()
        session._display_on.set()

        session._on_progress({"event": "batch_optimization_start"})

        self.assertEqual(session.status.display_message, "Applying gradient descent")
        self.assertIsNone(session.status.battle_view)

    def test_live_display_toggle_starts_and_stops_training_audio(self):
        session = TrainingSession.__new__(TrainingSession)
        session._status = TrainingSessionStatus(battle_view={"game_objects": ()})
        session._display_on = threading.Event()
        session.audio_service = RecordingAudioService()
        session._next_display_frame_time = 0.0

        session.set_display_on(True)
        session.set_display_on(False)

        self.assertEqual(
            [operation[0] for operation in session.audio_service.operations],
            ["start_battle_music", "stop_music"],
        )
        self.assertIsNone(session._status.battle_view)


class TrainingSessionTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def test_build_components_loads_checkpoint_on_preferred_device(self):
        torch = torch_backend.require_torch()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            slot.pth_path.write_bytes(b"checkpoint")
            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=1,
            )
            device = torch.device("cpu")

            with (
                mock.patch(
                    "src.training.session.torch_backend.preferred_device",
                    return_value=device,
                ) as preferred_device,
                mock.patch(
                    "src.training.session.load_training_checkpoint",
                ) as load_checkpoint,
                mock.patch(
                    "src.training.session.torch_backend.move_optimizer_state_to_device",
                ) as move_optimizer_state,
            ):
                model, optimizer, _ = session._build_components()

        preferred_device.assert_called_once_with()
        self.assertEqual(next(model.parameters()).device, device)
        load_checkpoint.assert_called_once()
        self.assertEqual(load_checkpoint.call_args.kwargs["map_location"], device)
        move_optimizer_state.assert_called_once_with(optimizer, device)

    def test_build_components_respects_explicit_cpu_training_device(self):
        torch = torch_backend.require_torch()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                ),
                batch_grouping=1,
            )

            with mock.patch(
                "src.training.session.torch_backend.preferred_device",
                return_value=torch.device("cuda"),
            ) as preferred_device:
                model, _optimizer, _replay_buffer = session._build_components()

        preferred_device.assert_not_called()
        self.assertEqual(next(model.parameters()).device, torch.device("cpu"))

    def test_session_accepts_existing_batch_history_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
                progress={"completed_batches": 7},
            )
            slot = repository.create_or_update_user_model(metadata)
            history = (BatchMetrics(7, 1, 0, 1, 5.0, 0.1, 0.001, 0.25),)
            log_lines = ("Batch      7 | summary",)

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=1,
                initial_history=history,
                initial_log_lines=log_lines,
            )

        self.assertEqual(session.status.completed_batches, 7)
        self.assertEqual(session.history, history)
        self.assertEqual(session.log_lines, log_lines)

    def test_session_runs_batch_saves_progress_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            def batch_runner(**kwargs):
                replay = kwargs["replay_buffer"]
                self.assertIsInstance(replay, TrainingReplayBuffer)
                kwargs["progress_callback"](
                    {
                        "event": "round_start",
                        "round_index": 1,
                        "total_rounds": 1,
                        "opponent": OpponentSpec("Earthling"),
                    }
                )
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=len(replay),
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0, win=True, draw=False),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=1,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=1)

            saved_slot = repository.slot_for("Earthling", 1)
            self.assertEqual(saved_slot.metadata["progress"]["completed_batches"], 1)
            self.assertGreater(saved_slot.pth_path.stat().st_size, 0)
            self.assertTrue((root / "user" / "Earthling-01.csv").exists())

        self.assertEqual(len(session.log_lines), 1)

    def test_session_saves_checkpoint_on_group_boundary_and_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            def batch_runner(**kwargs):
                if len(session.saved_batches) == 1:
                    self.assertFalse(replay_checkpoint_path(slot.pth_path).exists())
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0),),
                )

            session = _RecordingSaveSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=3,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=5)

            saved_slot = repository.slot_for("Earthling", 1)
            replay_exists = replay_checkpoint_path(saved_slot.pth_path).exists()

        self.assertEqual(session.saved_batches, [3, 5])
        self.assertEqual(session.saved_replay_flags, [False, True])
        self.assertEqual(saved_slot.metadata["progress"]["completed_batches"], 5)
        self.assertFalse(replay_exists)

    def test_session_skips_exit_save_after_group_boundary_model_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            def batch_runner(**kwargs):
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0),),
                )

            session = _RecordingSaveSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=3,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=6)
            replay_exists = replay_checkpoint_path(slot.pth_path).exists()

        self.assertEqual(session.saved_batches, [3, 6])
        self.assertEqual(session.saved_replay_flags, [False, False])
        self.assertFalse(replay_exists)

    def test_save_state_coordinates_and_notifies_cache_after_metadata_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            coordinator = ModelSaveCoordinator()
            key = OpponentModelKey("Earthling", 1)
            notifications = []

            def notify_model_saved(repository, ship, slot, *, device_choice=None):
                saved_slot = repository.slot_for(ship, slot)
                notifications.append(
                    (
                        ship,
                        slot,
                        device_choice,
                        saved_slot.metadata["progress"]["completed_batches"],
                        coordinator.is_saving(key),
                    )
                )

            cache = SimpleNamespace(notify_model_saved=notify_model_saved)
            original_save = save_training_checkpoint

            def saving_checkpoint(*args, **kwargs):
                self.assertTrue(coordinator.is_saving(key))
                return original_save(*args, **kwargs)

            def batch_runner(**kwargs):
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=99,
                batch_runner=batch_runner,
                opponent_model_cache=cache,
                save_coordinator=coordinator,
            )

            with mock.patch(
                "src.training.session.save_training_checkpoint",
                side_effect=saving_checkpoint,
            ):
                session.run_synchronously(max_batches=1)

        self.assertEqual(notifications, [("Earthling", 1, "auto", 1, False)])

    def test_failed_save_does_not_notify_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            coordinator = ModelSaveCoordinator()
            key = OpponentModelKey("Earthling", 1)
            cache = SimpleNamespace(notify_model_saved=mock.Mock())
            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=1,
                opponent_model_cache=cache,
                save_coordinator=coordinator,
            )
            model = build_value_network(ValueNetworkConfig(8, 1))
            replay_buffer = TrainingReplayBuffer(10)

            with (
                mock.patch(
                    "src.training.session.save_training_checkpoint",
                    side_effect=RuntimeError("save failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "save failed"),
            ):
                session._save_state(
                    model,
                    None,
                    replay_buffer,
                    include_replay=False,
                )

        cache.notify_model_saved.assert_not_called()
        self.assertFalse(coordinator.is_saving(key))

    def test_session_records_current_run_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
                progress={"completed_batches": 7},
            )
            slot = repository.create_or_update_user_model(metadata)

            def batch_runner(**kwargs):
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=1,
                batch_runner=batch_runner,
            )

            times = iter((100.0, 100.0, 130.0, 130.0, 130.0))
            current_time = [130.0]

            def fake_perf_counter():
                try:
                    current_time[0] = next(times)
                except StopIteration:
                    pass
                return current_time[0]

            with mock.patch(
                "src.training.session.time.perf_counter",
                side_effect=fake_perf_counter,
            ):
                session.run_synchronously(max_batches=1)

        status = session.status
        self.assertEqual(status.completed_batches, 8)
        self.assertEqual(status.elapsed_training_seconds, 30.0)
        self.assertEqual(status.last_batch_seconds, 30.0)
        self.assertEqual(status.average_batch_seconds, 30.0)
        self.assertEqual(status.batches_per_hour, 120.0)

    def test_session_decays_current_epsilon_after_each_completed_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            seen_epsilons = []

            def batch_runner(**kwargs):
                seen_epsilons.append(kwargs["config"].epsilon)
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    starting_epsilon=0.4,
                    epsilon=0.4,
                    epsilon_floor=0.05,
                    epsilon_decay=0.5,
                    epsilon_frame_span=12,
                ),
                batch_grouping=3,
                batch_runner=batch_runner,
            )

            session.run_synchronously(max_batches=2)

            saved_regimen = repository.slot_for("Earthling", 1).metadata["training"][
                "regimen"
            ]

        self.assertEqual(seen_epsilons, [0.4, 0.2])
        self.assertAlmostEqual(session.status.current_epsilon, 0.1)
        self.assertEqual([metrics.epsilon for metrics in session.history], [0.2, 0.1])
        self.assertEqual(saved_regimen["starting_epsilon"], 0.4)
        self.assertAlmostEqual(saved_regimen["current_epsilon"], 0.1)
        self.assertAlmostEqual(saved_regimen["epsilon"], 0.1)
        self.assertEqual(saved_regimen["epsilon_floor"], 0.05)
        self.assertEqual(saved_regimen["epsilon_decay"], 0.5)
        self.assertEqual(saved_regimen["epsilon_frame_span"], 12)

    def test_session_decay_stops_at_epsilon_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            def batch_runner(**kwargs):
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(),
                    round_results=(_round_result(1.0),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    starting_epsilon=0.4,
                    epsilon=0.4,
                    epsilon_floor=0.15,
                    epsilon_decay=0.5,
                ),
                batch_grouping=10,
                batch_runner=batch_runner,
            )

            session.run_synchronously(max_batches=3)

        self.assertAlmostEqual(session.status.current_epsilon, 0.15)
        self.assertEqual(
            [metrics.epsilon for metrics in session.history],
            [0.2, 0.15, 0.15],
        )

    def test_session_passes_display_predicate_to_batch_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            def batch_runner(**kwargs):
                self.assertTrue(kwargs["battle_view_enabled"]())
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(),
                    round_results=(_round_result(),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    display_on=True,
                ),
                batch_grouping=1,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=1)

    def test_session_caches_existing_ai_discovery_until_group_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            first = (OpponentSpec("Earthling", slot=1),)
            second = (OpponentSpec("Mycon", slot=2),)
            seen_opponents = []

            def batch_runner(**kwargs):
                seen_opponents.append(kwargs["discovered_opponents"])
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(),
                    round_results=(_round_result(),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    opponent_mode=OPPONENT_MODE_EXISTING_AI,
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=3,
                batch_runner=batch_runner,
            )

            with mock.patch(
                "src.training.session.discover_existing_ai_opponents",
                side_effect=(
                    OpponentDiscoveryResult(first),
                    OpponentDiscoveryResult(second),
                ),
            ) as discover:
                session.run_synchronously(max_batches=5)

        self.assertEqual(discover.call_count, 2)
        self.assertEqual(seen_opponents, [first, first, first, second, second])

    def test_sessions_with_shared_cache_receive_same_opponent_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            opponent_metadata = metadata_from_state(
                ship="Earthling",
                slot=2,
                description="Opponent",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            opponent_slot = repository.create_or_update_user_model(opponent_metadata)
            save_training_checkpoint(
                opponent_slot.pth_path,
                build_value_network(ValueNetworkConfig(8, 1)),
            )
            cache = OpponentModelCache()
            seen_opponents = []

            def batch_runner(**kwargs):
                seen_opponents.append(kwargs["discovered_opponents"])
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(),
                    round_results=(_round_result(),),
                )

            for slot_number in (1, 3):
                metadata = metadata_from_state(
                    ship="Earthling",
                    slot=slot_number,
                    description="Test",
                    architecture=model_architecture_metadata(8, 1),
                    training={"regimen": {"rounds_per_batch": 1}},
                )
                slot = repository.create_or_update_user_model(metadata)
                session = _NoSaveSession(
                    repository=repository,
                    slot=slot,
                    metadata=metadata,
                    config=TrainingOrchestrationConfig(
                        trainee_ship="Earthling",
                        opponent_mode=OPPONENT_MODE_EXISTING_AI,
                        hidden_layer_width=8,
                        hidden_layer_count=1,
                        epsilon_decay=1.0,
                    ),
                    batch_grouping=99,
                    batch_runner=batch_runner,
                    opponent_model_cache=cache,
                )
                session.run_synchronously(max_batches=1)

        self.assertEqual(len(seen_opponents), 2)
        self.assertEqual(len(seen_opponents[0]), 1)
        self.assertEqual(len(seen_opponents[1]), 1)
        self.assertEqual(seen_opponents[0][0].slot, 2)
        self.assertIs(seen_opponents[0][0].model, seen_opponents[1][0].model)

    def test_shared_cache_snapshot_is_captured_per_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            first = (OpponentSpec("Earthling", slot=1, model=object()),)
            second = (OpponentSpec("Mycon", slot=2, model=object()),)
            cache = _MutableOpponentCache(first)
            seen_opponents = []

            def batch_runner(**kwargs):
                seen_opponents.append(kwargs["discovered_opponents"])
                if len(seen_opponents) == 1:
                    cache.snapshot_value = second
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(),
                    round_results=(_round_result(),),
                )

            session = _NoSaveSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    opponent_mode=OPPONENT_MODE_EXISTING_AI,
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=99,
                batch_runner=batch_runner,
                opponent_model_cache=cache,
            )
            session.run_synchronously(max_batches=2)

        self.assertEqual(cache.load_initial_calls, 2)
        self.assertIs(seen_opponents[0], first)
        self.assertIs(seen_opponents[1], second)

    def test_session_resume_preserves_partial_grouping_average(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1, "batch_grouping": 3}},
            )
            slot = repository.create_or_update_user_model(metadata)

            scores = iter((10.0, 20.0))

            def first_runner(**kwargs):
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.1,),
                    round_results=(_round_result(next(scores)),),
                )

            first_session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=3,
                batch_runner=first_runner,
            )
            first_session.run_synchronously(max_batches=2)

            resumed_slot = repository.slot_for("Earthling", 1)
            resumed_session = TrainingSession(
                repository=repository,
                slot=resumed_slot,
                metadata=resumed_slot.metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=3,
                batch_runner=lambda **kwargs: TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.1,),
                    round_results=(_round_result(30.0),),
                ),
            )
            resumed_session.run_synchronously(max_batches=1)

            with (root / "user" / "Earthling-01.csv").open(
                newline="",
                encoding="utf-8",
            ) as file:
                rows = list(csv.reader(file))

        self.assertEqual(
            rows[-1],
            [
                "3", "0", "0.00", "0", "0.00", "30.0", "20.0",
                "0.10000", "0.001000", "0.1000", "0.1000",
            ],
        )

    def test_stop_abandons_current_batch_without_saving_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            session = None

            def batch_runner(**kwargs):
                self.assertFalse(kwargs["stop_requested"]())
                session.request_stop()
                self.assertTrue(kwargs["stop_requested"]())
                raise TrainingBatchAborted("training stop requested")

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                ),
                batch_grouping=1,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=1)

            saved_slot = repository.slot_for("Earthling", 1)
            completed = saved_slot.metadata.get("progress", {}).get(
                "completed_batches",
                0,
            )

        self.assertEqual(completed, 0)
        self.assertEqual(session.status.completed_batches, 0)
        self.assertEqual(session.log_lines, ())

    def test_stop_after_completed_batch_flushes_unsaved_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from src.training.model_registry import TrainingModelRepository

            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)

            session = None

            def batch_runner(**kwargs):
                session.request_stop()
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.125,),
                    round_results=(_round_result(5.0),),
                )

            session = _RecordingSaveSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    epsilon_decay=1.0,
                ),
                batch_grouping=3,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=3)

            saved_slot = repository.slot_for("Earthling", 1)

        self.assertEqual(session.saved_batches, [1])
        self.assertEqual(session.saved_replay_flags, [True])
        self.assertEqual(saved_slot.metadata["progress"]["completed_batches"], 1)
        self.assertEqual(session.status.completed_batches, 1)


if __name__ == "__main__":
    unittest.main()
