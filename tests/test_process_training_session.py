import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from src.Menus import train_ai
from src.training import torch_backend
from src.training.cpu_contracts import TrainingOrchestrationConfig
from src.training.model_registry import (
    SLOT_EMPTY,
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.orchestration import OpponentSpec
from src.training.process_session import (
    ProcessModelSaveCoordinator,
    ProcessTrainingSession,
    _ProcessTrainingEngine,
)


class _PublisherStub:
    def apply_controls(self, _session):
        pass

    def publish_status(self, _session, **_kwargs):
        pass

    def publish_saved(self, _session):
        pass


class ProcessModelSaveCoordinatorTests(unittest.TestCase):
    def test_tracks_saving_and_advances_generation_only_after_commit(self):
        coordinator = ProcessModelSaveCoordinator()
        before_saving, before_generations = coordinator.snapshot()

        with coordinator.saving(("Earthling", 1)):
            saving, generations = coordinator.snapshot()
            self.assertIn(("Earthling", 1), saving)
            self.assertEqual(generations, before_generations)

        saving, generations = coordinator.snapshot()
        self.assertNotIn(("Earthling", 1), saving)
        self.assertEqual(sum(generations), sum(before_generations) + 1)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            with coordinator.saving(("Earthling", 1)):
                raise RuntimeError("failed")
        _saving, failed_generations = coordinator.snapshot()
        self.assertEqual(failed_generations, generations)


class ProcessOpponentDiscoveryTests(unittest.TestCase):
    def test_cached_saving_opponent_is_retained_then_reloaded_next_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Test",
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            slot = repository.create_or_update_user_model(metadata)
            coordinator = ProcessModelSaveCoordinator()
            context = multiprocessing.get_context("spawn")
            engine = _ProcessTrainingEngine(
                publisher=_PublisherStub(),
                stop_event=context.Event(),
                display_event=context.Event(),
                save_coordinator=coordinator,
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    opponent_mode="all",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                ),
                batch_grouping=10,
            )
            old = OpponentSpec(
                ship="Earthling",
                mode="all",
                slot=1,
                model="old-model",
            )
            engine._cached_existing_ai_opponents = (old,)
            engine._cached_existing_ai_opponents_at = 0
            _saving, engine._discovery_generations = coordinator.snapshot()

            def discover(repository_view, **_kwargs):
                visible = repository_view.slot_for("Earthling", 1)
                opponents = ()
                if visible.source != SLOT_EMPTY:
                    opponents = (
                        OpponentSpec(
                            ship="Earthling",
                            mode="all",
                            slot=1,
                            model="new-model",
                        ),
                    )
                return type("Discovery", (), {"opponents": opponents})()

            with mock.patch(
                "src.training.process_session.discover_existing_ai_opponents",
                side_effect=discover,
            ):
                with coordinator.saving(("Earthling", 1)):
                    engine._discovery_deferred = True
                    during_save = engine._existing_ai_opponents_for_batch()
                    self.assertEqual(during_save, (old,))

                after_commit = engine._existing_ai_opponents_for_batch()

        self.assertEqual(after_commit[0].model, "new-model")


class ProcessTrainingResourceTests(unittest.TestCase):
    def test_worker_engine_supplies_display_free_assets_to_batch_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Umgah",
                slot=1,
                description="Resource test",
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            slot = repository.create_or_update_user_model(metadata)
            context = multiprocessing.get_context("spawn")
            seen = []

            def batch_runner(**kwargs):
                seen.append(kwargs["resources"])
                raise RuntimeError("stop after resource capture")

            engine = _ProcessTrainingEngine(
                publisher=_PublisherStub(),
                stop_event=context.Event(),
                display_event=context.Event(),
                save_coordinator=ProcessModelSaveCoordinator(),
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Umgah",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                ),
                batch_grouping=1,
                batch_runner=batch_runner,
            )

            with self.assertRaisesRegex(RuntimeError, "resource capture"):
                engine._run_batch()

        assets = seen[0].ability("UmgahA1")
        self.assertEqual(len(assets.anchor_offsets), 3)
        self.assertFalse(seen[0]._asset_errors)


class ProcessTrainingSessionTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def test_cpu_session_reuses_spawn_worker_past_first_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Process test",
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            slot = repository.create_or_update_user_model(metadata)
            session = ProcessTrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    rounds_per_batch=1,
                    match_time_limit=2,
                    replay_updates_per_batch=0,
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                ),
                batch_grouping=1,
                save_coordinator=ProcessModelSaveCoordinator(),
            )
            try:
                session.start()
                worker_pid = session._process.pid
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    status = session.status
                    if status.error:
                        self.fail(status.error)
                    if status.completed_batches >= 2:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("CPU process did not continue past its first batch")

                self.assertTrue(session._process.is_alive())
                self.assertEqual(session._process.pid, worker_pid)
                session.request_stop()
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline and session.status.running:
                    time.sleep(0.02)
                session.join(5.0)

                self.assertFalse(session.status.running)
                self.assertGreaterEqual(session.status.completed_batches, 2)
                self.assertGreater(slot.pth_path.stat().st_size, 0)
            finally:
                session.request_stop()
                session.join(5.0)

    def test_display_frames_are_rendered_in_worker_shared_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Display process test",
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            slot = repository.create_or_update_user_model(metadata)
            session = ProcessTrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    rounds_per_batch=1,
                    match_time_limit=20,
                    replay_updates_per_batch=0,
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                    display_on=True,
                ),
                batch_grouping=1,
                save_coordinator=ProcessModelSaveCoordinator(),
                display_frame_count=1,
            )
            try:
                session.start()
                deadline = time.monotonic() + 30.0
                battle_view = None
                while time.monotonic() < deadline:
                    status = session.status
                    if status.error:
                        self.fail(status.error)
                    battle_view = status.battle_view
                    if battle_view and battle_view.get("rendered_frames"):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("CPU process did not publish a rendered frame")

                self.assertEqual(
                    len(battle_view["rendered_frames"]),
                    1,
                )
            finally:
                session.request_stop()
                session.join(10.0)

    def test_worker_initialization_failure_is_reported_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Invalid process test",
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            slot = repository.create_or_update_user_model(metadata)
            invalid_metadata = dict(metadata)
            invalid_metadata["observation_schema_version"] -= 1
            session = ProcessTrainingSession(
                repository=repository,
                slot=slot,
                metadata=invalid_metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                ),
                batch_grouping=1,
                save_coordinator=ProcessModelSaveCoordinator(),
            )

            session.start()
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                status = session.status
                if not status.running:
                    break
                time.sleep(0.02)
            else:
                self.fail("CPU process initialization failure was not reported")

            self.assertIn("observation schema is incompatible", status.error)
            while time.monotonic() < deadline and session._process is not None:
                session.status
                time.sleep(0.01)
            self.assertIsNone(session._process)


class ProcessTrainingRoutingTests(unittest.TestCase):
    def test_selected_cpu_uses_process_session(self):
        self.assertIs(
            train_ai.independent_session_class(torch_backend.DEVICE_CPU),
            ProcessTrainingSession,
        )

    def test_auto_and_gpu_keep_thread_session(self):
        self.assertIs(
            train_ai.independent_session_class(torch_backend.DEVICE_AUTO),
            train_ai.TrainingSession,
        )
        self.assertIs(
            train_ai.independent_session_class(torch_backend.DEVICE_GPU),
            train_ai.TrainingSession,
        )


if __name__ == "__main__":
    unittest.main()
