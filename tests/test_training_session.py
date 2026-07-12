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
from src.training.model_registry import (
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.orchestration import (
    OpponentSpec,
    TrainingBatchAborted,
    TrainingBatchResult,
    TrainingOrchestrationConfig,
    TrainingRoundResult,
)
from src.training.replay import TrainingReplayBuffer
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


def _round_result(total_return=0.0, *, win=False, loss=False, draw=True):
    return TrainingRoundResult(
        opponent=OpponentSpec("Earthling"),
        frames=1,
        terminal_reason="timeout",
        mature_samples=1,
        total_return=total_return,
        win=win,
        loss=loss,
        draw=draw,
    )


class TrainingMetricsTests(unittest.TestCase):
    def test_batch_summary_line_uses_specified_format(self):
        metrics = BatchMetrics(
            batch=45,
            match_count=250,
            wins=120,
            losses=30,
            draws=100,
            average_match_score=30.5,
            epsilon=0.002,
            learning_rate=0.0001,
            average_loss=0.06,
        )
        rolling = BatchMetrics(
            batch=45,
            match_count=250,
            wins=110,
            losses=40,
            draws=100,
            average_match_score=25.0,
            epsilon=0.002,
            learning_rate=0.0001,
            average_loss=0.05,
        )

        self.assertEqual(
            format_batch_summary_line(metrics, rolling),
            "Batch     45 |  48.0% W,  12.0% L,  40.0% D | ( 44.00% W) | "
            "Score:  30.500 ( 25.000) | Epsilon: 0.00200 | "
            "LR: 0.00010 | Loss: 0.0600 (0.0500)",
        )

    def test_metrics_from_batch_result_counts_outcomes_and_average_score(self):
        result = TrainingBatchResult(
            completed_rounds=3,
            replay_size=10,
            optimization_losses=(0.2, 0.4),
            round_results=(
                _round_result(10.0, win=True, draw=False),
                _round_result(-1.0, loss=True, draw=False),
                _round_result(3.0),
            ),
        )

        metrics = metrics_from_batch_result(
            result,
            batch=7,
            epsilon=0.1,
            learning_rate=0.001,
        )

        self.assertEqual(metrics.match_count, 3)
        self.assertEqual((metrics.wins, metrics.losses, metrics.draws), (1, 1, 1))
        self.assertAlmostEqual(metrics.average_match_score, 4.0)
        self.assertAlmostEqual(metrics.average_loss, 0.3)

    def test_rolling_metrics_uses_available_window(self):
        history = (
            BatchMetrics(1, 1, 1, 0, 0, 10.0, 0.1, 0.001, 1.0),
            BatchMetrics(2, 1, 0, 1, 0, 20.0, 0.2, 0.002, 3.0),
        )

        rolling = rolling_metrics(history, grouping=5)

        self.assertEqual(rolling.batch, 2)
        self.assertAlmostEqual(rolling.average_match_score, 15.0)
        self.assertAlmostEqual(rolling.average_loss, 2.0)
        self.assertAlmostEqual(rolling.epsilon, 0.2)
        self.assertAlmostEqual(rolling.learning_rate, 0.002)

    def test_rolling_metrics_keeps_grouped_outcome_totals_for_win_rate(self):
        history = (
            BatchMetrics(1, 3, 1, 1, 1, 10.0, 0.1, 0.001, 1.0),
            BatchMetrics(2, 3, 2, 1, 0, 20.0, 0.2, 0.002, 3.0),
        )

        rolling = rolling_metrics(history, grouping=2)

        self.assertEqual(rolling.match_count, 6)
        self.assertEqual((rolling.wins, rolling.losses, rolling.draws), (3, 2, 1))
        self.assertIn("( 50.00% W)", format_batch_summary_line(history[-1], rolling))

    def test_csv_append_writes_header_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.csv"

            append_grouped_metrics_csv(
                path,
                BatchMetrics(1000, 250, 0, 0, 0, 34.3, 0.002, 0.0001, 0.05),
            )
            append_grouped_metrics_csv(
                path,
                BatchMetrics(2000, 250, 0, 0, 0, 35.0, 0.003, 0.0002, 0.06),
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))

        self.assertEqual(
            rows,
            [
                ["Batch", "Win %", "Score", "Epsilon", "Learning Rate", "Loss"],
                ["1000", "0.0", "34.3", "0.00200", "0.000100", "0.0500"],
                ["2000", "0.0", "35.0", "0.00300", "0.000200", "0.0600"],
            ],
        )

    def test_csv_append_writes_grouped_win_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.csv"

            append_grouped_metrics_csv(
                path,
                BatchMetrics(50, 10, 4, 3, 3, 12.5, 0.100, 0.001, 0.25),
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.reader(file))

        self.assertEqual(rows[-1], ["50", "40.0", "12.5", "0.10000", "0.001000", "0.2500"])

    def test_batch_metrics_history_round_trips_through_metadata(self):
        metrics = BatchMetrics(12, 25, 9, 10, 6, 42.5, 0.1, 0.001, 0.25)
        metadata = {
            "progress": {
                "completed_batches": 12,
                "recent_batch_metrics": [batch_metrics_to_metadata(metrics)],
            }
        }

        self.assertEqual(batch_metrics_history_from_metadata(metadata), (metrics,))

    def test_grouped_win_rate_survives_metadata_round_trip(self):
        history = (
            BatchMetrics(12, 3, 1, 1, 1, 42.5, 0.1, 0.001, 0.25),
            BatchMetrics(13, 3, 2, 1, 0, 43.5, 0.1, 0.001, 0.35),
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

        self.assertEqual(rolling.match_count, 6)
        self.assertEqual(rolling.wins, 3)
        self.assertIn("( 50.00% W)", format_batch_summary_line(restored[-1], rolling))


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
            history = (BatchMetrics(7, 1, 1, 0, 0, 5.0, 0.1, 0.001, 0.25),)
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
        self.assertEqual(saved_regimen["epsilon_decay"], 0.5)
        self.assertEqual(saved_regimen["epsilon_frame_span"], 12)

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

        self.assertEqual(rows[-1], ["3", "0.0", "20.0", "0.10000", "0.001000", "0.1000"])

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


if __name__ == "__main__":
    unittest.main()
