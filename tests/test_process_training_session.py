import multiprocessing
from multiprocessing import shared_memory
from dataclasses import replace
from pathlib import Path
import queue
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

import pygame

import src.const as const
from src.audio import RecordingAudioService
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
from src.training.session import TrainingSessionStatus
from src.training.process_session import (
    CpuBatchPacingGroup,
    ProcessModelSaveCoordinator,
    ProcessTrainingSession,
    _DISPLAY_BUFFER_COUNT,
    _DISPLAY_SEQUENCE_BYTES,
    _ProcessTrainingEngine,
    _SharedDisplayRenderer,
)


class _PublisherStub:
    def apply_controls(self, _session):
        pass

    def publish_status(self, _session, **_kwargs):
        pass

    def publish_saved(self, _session):
        pass


def _pacing_process(group, participant_index, completed_batches, results):
    group.wait_after_batch(
        participant_index,
        completed_batches,
        stop_requested=lambda: False,
    )
    results.put(participant_index)


class CpuBatchPacingGroupTests(unittest.TestCase):
    def test_default_pacing_is_a_zero_lead_barrier(self):
        group = CpuBatchPacingGroup(2)

        self.assertEqual(group.max_batch_lead, 0)

    def test_fast_participant_waits_until_lead_is_within_limit(self):
        group = CpuBatchPacingGroup(2, max_batch_lead=1)
        group.wait_after_batch(0, 1, stop_requested=lambda: False)
        finished = threading.Event()

        def report_second_fast_batch():
            group.wait_after_batch(0, 2, stop_requested=lambda: False)
            finished.set()

        thread = threading.Thread(target=report_second_fast_batch)
        thread.start()
        self.assertFalse(finished.wait(0.05))

        group.wait_after_batch(1, 1, stop_requested=lambda: False)
        thread.join(1.0)

        self.assertTrue(finished.is_set())

    def test_deactivating_slow_participant_releases_waiter(self):
        group = CpuBatchPacingGroup(2, max_batch_lead=1)
        finished = threading.Event()

        def report_fast_progress():
            group.wait_after_batch(0, 2, stop_requested=lambda: False)
            finished.set()

        thread = threading.Thread(target=report_fast_progress)
        thread.start()
        self.assertFalse(finished.wait(0.05))

        group.deactivate(1)
        thread.join(1.0)

        self.assertTrue(finished.is_set())

    def test_target_stopped_participant_does_not_block_later_peer_batch(self):
        group = CpuBatchPacingGroup(2)
        first_batch_finished = threading.Event()

        def report_stopping_participant_first_batch():
            group.wait_after_batch(0, 1, stop_requested=lambda: False)
            first_batch_finished.set()

        first_thread = threading.Thread(
            target=report_stopping_participant_first_batch
        )
        first_thread.start()
        group.wait_after_batch(1, 1, stop_requested=lambda: False)
        first_thread.join(1.0)
        self.assertTrue(first_batch_finished.is_set())
        finished = threading.Event()

        def report_surviving_participant_next_batch():
            group.wait_after_batch(1, 2, stop_requested=lambda: False)
            finished.set()

        thread = threading.Thread(target=report_surviving_participant_next_batch)
        thread.start()
        self.assertFalse(finished.wait(0.05))

        group.deactivate(0)
        thread.join(1.0)

        self.assertTrue(finished.is_set())

    def test_pacing_primitives_are_shared_by_spawned_processes(self):
        context = multiprocessing.get_context("spawn")
        group = CpuBatchPacingGroup(2, max_batch_lead=1, context=context)
        results = context.Queue()
        fast = context.Process(
            target=_pacing_process,
            args=(group, 0, 2, results),
        )
        slower = context.Process(
            target=_pacing_process,
            args=(group, 1, 1, results),
        )
        fast.start()
        try:
            with self.assertRaises(queue.Empty):
                results.get(timeout=0.1)
            slower.start()
            completed = {results.get(timeout=5.0), results.get(timeout=5.0)}
            fast.join(5.0)
            slower.join(5.0)
        finally:
            group.deactivate(0)
            group.deactivate(1)
            if fast.is_alive():
                fast.terminate()
            if slower.pid is not None and slower.is_alive():
                slower.terminate()
            fast.join(1.0)
            if slower.pid is not None:
                slower.join(1.0)
            results.close()

        self.assertEqual(completed, {0, 1})
        self.assertEqual(fast.exitcode, 0)
        self.assertEqual(slower.exitcode, 0)


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


class SharedDisplayRendererTests(unittest.TestCase):
    def test_prepared_renderer_writes_rgb_backing_buffer_without_reencoding(self):
        width = 4
        height = 2
        frame_count = 1
        frame_bytes = width * height * 3
        memory = shared_memory.SharedMemory(
            create=True,
            size=(
                _DISPLAY_SEQUENCE_BYTES
                + _DISPLAY_BUFFER_COUNT * frame_bytes
            ),
        )
        renderer = _SharedDisplayRenderer(frame_count)

        try:
            with (
                mock.patch.object(const, "SCREEN_WIDTH", width),
                mock.patch.object(const, "SCREEN_HEIGHT", height),
                mock.patch.object(const, "SCREEN_LEFT", 1),
                mock.patch(
                    "src.training.process_session.DisplayStarField",
                    return_value=object(),
                ),
            ):
                renderer.prepare()
                renderer._renderer = mock.Mock()
                renderer._renderer.draw.side_effect = (
                    lambda surface, *_args, **_kwargs: surface.fill((17, 17, 17))
                )
                renderer.attach(memory.name)
                with mock.patch(
                    "src.training.process_session.pygame.image.tobytes",
                    side_effect=AssertionError("RGB render target was reencoded"),
                ):
                    metadata = renderer.render({"frame_id": 1})

            buffer_index = metadata["shared_sequence"] % _DISPLAY_BUFFER_COUNT
            offset = _DISPLAY_SEQUENCE_BYTES + buffer_index * frame_bytes
            self.assertEqual(
                bytes(memory.buf[offset : offset + frame_bytes]),
                b"\x11" * frame_bytes,
            )
        finally:
            renderer.close()
            memory.close()
            memory.unlink()

    def test_reader_keeps_committed_frames_while_worker_writes_inactive_buffer(self):
        width = 4
        height = 2
        frame_count = 1
        frame_bytes = width * height * 3
        frame_set_bytes = frame_count * frame_bytes
        memory = shared_memory.SharedMemory(
            create=True,
            size=(
                _DISPLAY_SEQUENCE_BYTES
                + _DISPLAY_BUFFER_COUNT * frame_set_bytes
            ),
        )
        renderer = _SharedDisplayRenderer(frame_count)
        renderer._memory = memory
        renderer._memory_name = memory.name
        renderer._surface = pygame.Surface((width, height))
        renderer._renderer = mock.Mock()
        renderer._star_field = object()
        reader = object.__new__(ProcessTrainingSession)
        reader._display_memory = memory
        battle_view = {"frame_id": 7}

        try:
            with (
                mock.patch.object(const, "SCREEN_WIDTH", width),
                mock.patch.object(const, "SCREEN_HEIGHT", height),
                mock.patch.object(const, "SCREEN_LEFT", 1),
                mock.patch(
                    "src.training.process_session.pygame.image.tobytes",
                    return_value=b"\x11" * frame_bytes,
                ),
            ):
                metadata = renderer.render(battle_view)

                self.assertEqual(metadata["shared_sequence"], 1)
                self.assertEqual(
                    struct.unpack_from("<Q", memory.buf, 0)[0],
                    1,
                )

                # Simulate the worker beginning its next render in the other
                # slot without committing it yet. The published slot remains
                # stable and readable throughout that write.
                memory.buf[
                    _DISPLAY_SEQUENCE_BYTES:
                    _DISPLAY_SEQUENCE_BYTES + frame_set_bytes
                ] = b"\x22" * frame_set_bytes
                original_frombuffer = pygame.image.frombuffer
                with (
                    mock.patch(
                        "src.training.process_session.pygame.image.frombuffer",
                        wraps=original_frombuffer,
                    ) as frombuffer,
                    mock.patch(
                        "src.training.process_session.pygame.image.frombytes",
                        side_effect=AssertionError("copied frame pixels twice"),
                    ),
                ):
                    frames = reader._read_display_frames(metadata)

            self.assertIsNotNone(frames)
            frombuffer.assert_called_once()
            self.assertEqual(frames[0].get_at((0, 0))[:3], (17, 17, 17))
        finally:
            renderer.close()
            memory.unlink()


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
    def test_process_worker_renders_live_view_without_redundant_object_clone(self):
        engine = object.__new__(_ProcessTrainingEngine)
        battle_view = {"game_objects": (object(),)}

        self.assertIs(engine._battle_view_for_display(battle_view), battle_view)

    def test_worker_sound_effect_gate_follows_shared_display_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Audio gate test",
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            slot = repository.create_or_update_user_model(metadata)
            context = multiprocessing.get_context("spawn")
            display_event = context.Event()
            audio = RecordingAudioService()
            engine = _ProcessTrainingEngine(
                publisher=_PublisherStub(),
                stop_event=context.Event(),
                display_event=display_event,
                save_coordinator=ProcessModelSaveCoordinator(),
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                    display_on=False,
                ),
                batch_grouping=1,
                audio_service=audio,
            )
            effect = engine.audio_service.load_effect(Path("effect.wav"), 0.25)

            effect.play()
            display_event.set()
            effect.play()
            display_event.clear()
            effect.play()

        self.assertEqual(
            audio.operations,
            [("play_effect", Path("effect.wav"), 0.25)],
        )

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

    def test_worker_reports_run_relative_batches_to_pacing_group(self):
        waiting_messages = []
        waiting_views = []

        class PacingStub:
            def __init__(self):
                self.calls = []

            def wait_after_batch(
                self,
                participant_index,
                completed_batches,
                *,
                stop_requested,
            ):
                self.calls.append(
                    (participant_index, completed_batches, stop_requested())
                )
                waiting_messages.append(engine._status.display_message)
                waiting_views.append(engine._status.battle_view)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Pacing test",
                architecture=model_architecture_metadata(8, 1),
                training={},
                progress={"completed_batches": 40},
            )
            slot = repository.create_or_update_user_model(metadata)
            context = multiprocessing.get_context("spawn")
            pacing = PacingStub()
            engine = _ProcessTrainingEngine(
                publisher=_PublisherStub(),
                stop_event=context.Event(),
                display_event=context.Event(),
                save_coordinator=ProcessModelSaveCoordinator(),
                batch_pacing_group=pacing,
                batch_pacing_index=1,
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                ),
                batch_grouping=10,
            )
            engine._completed_batches_at_run_start = 40
            engine._batch_accounting_started_at = 10.0
            last_view = {"game_objects": ("last-frame",)}
            engine._status.battle_view = last_view

            with (
                mock.patch(
                    "src.training.process_session.TrainingSession._record_completed_batch",
                    return_value=43,
                ),
                mock.patch.object(
                    engine,
                    "_finalize_completed_batch_metrics",
                ) as finalize,
                mock.patch(
                    "src.training.process_session.time.perf_counter",
                    return_value=15.0,
                ),
            ):
                batch_number = engine._record_completed_batch(object())

        self.assertEqual(batch_number, 43)
        self.assertEqual(pacing.calls, [(1, 3, False)])
        self.assertEqual(waiting_messages, ["Waiting for synchronized CPU runs"])
        self.assertEqual(waiting_views, [last_view])
        self.assertIs(engine._status.battle_view, last_view)
        finalize.assert_called_once_with(batch_number=43, batch_seconds=5.0)

    def test_grouped_pacing_barrier_runs_after_save(self):
        events = []

        class PacingStub:
            def wait_after_batch(
                self,
                participant_index,
                completed_batches,
                *,
                stop_requested,
            ):
                events.append(
                    ("wait", participant_index, completed_batches, stop_requested())
                )

        engine = object.__new__(_ProcessTrainingEngine)
        engine._batch_pacing_group = PacingStub()
        engine._batch_pacing_index = 1
        engine._batch_accounting_started_at = 10.0
        engine._pending_paced_batch_number = None
        engine._completed_batches_at_run_start = 40
        engine.batch_grouping = 10
        engine._stop_requested = mock.Mock()
        engine._stop_requested.is_set.return_value = False
        engine._publisher = _PublisherStub()
        engine._lock = threading.Lock()
        engine._status = TrainingSessionStatus()

        with mock.patch(
            "src.training.process_session.TrainingSession._record_completed_batch",
            return_value=50,
        ):
            batch_number = engine._record_completed_batch(object())

        self.assertEqual(batch_number, 50)
        self.assertEqual(events, [])

        with (
            mock.patch(
                "src.training.process_session.TrainingSession._save_state",
                side_effect=lambda *_args, **_kwargs: events.append(("save",)),
            ),
            mock.patch.object(
                engine,
                "_finalize_completed_batch_metrics",
                side_effect=lambda **_kwargs: events.append(("finalize",)),
            ),
            mock.patch(
                "src.training.process_session.time.perf_counter",
                return_value=25.0,
            ),
        ):
            engine._save_state(object(), object(), object(), include_replay=False)

        self.assertEqual(
            events,
            [
                ("save",),
                ("wait", 1, 10, False),
                ("finalize",),
            ],
        )


class ProcessTrainingSessionTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

    def _make_session(
        self,
        root: Path,
        *,
        audio_service=None,
    ) -> ProcessTrainingSession:
        repository = TrainingModelRepository(root / "bundled", root / "user")
        metadata = metadata_from_state(
            ship="Earthling",
            slot=1,
            description="Process test",
            architecture=model_architecture_metadata(8, 1),
            training={},
        )
        slot = repository.create_or_update_user_model(metadata)
        return ProcessTrainingSession(
            repository=repository,
            slot=slot,
            metadata=metadata,
            config=TrainingOrchestrationConfig(
                trainee_ship="Earthling",
                training_device="cpu",
            ),
            batch_grouping=1,
            audio_service=audio_service,
            save_coordinator=ProcessModelSaveCoordinator(),
        )

    def test_hidden_process_session_cannot_stop_watched_session_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = RecordingAudioService()
            watched = self._make_session(root, audio_service=audio)
            hidden = self._make_session(root, audio_service=audio)

            with (
                mock.patch.object(watched, "_ensure_display_memory"),
                mock.patch.object(hidden, "_ensure_display_memory"),
            ):
                watched.set_display_on(True)
                hidden.set_display_on(False)

                self.assertEqual(audio.operations, [("start_battle_music",)])

                watched.set_display_on(False)
                hidden.set_display_on(False)

        self.assertEqual(
            audio.operations,
            [("start_battle_music",), ("stop_music",)],
        )

    def test_cached_running_status_projects_elapsed_time_and_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._make_session(Path(directory))
            session._status = TrainingSessionStatus(
                ship="Earthling",
                running=True,
                completed_batches=12,
                elapsed_training_seconds=20.0,
                batches_per_hour=360.0,
                display_message="Waiting for synchronized CPU runs",
            )
            session._completed_batches_at_run_start = 10
            session._elapsed_anchor_seconds = 20.0
            session._elapsed_anchor_at = 100.0

            with (
                mock.patch.object(session, "_drain_messages"),
                mock.patch(
                    "src.training.process_session.time.perf_counter",
                    return_value=110.0,
                ),
            ):
                status = session.status

        self.assertEqual(status.elapsed_training_seconds, 30.0)
        self.assertEqual(status.batches_per_hour, 240.0)

    def test_phase_only_worker_status_retains_last_transferred_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._make_session(Path(directory))
            last_view = {
                "frame_id": 7,
                "rendered_frames": (object(),),
            }
            session._status = TrainingSessionStatus(
                ship="Earthling",
                running=True,
                battle_view=last_view,
            )
            session._display_event.set()

            session._accept_status(
                {
                    "status": TrainingSessionStatus(
                        ship="Earthling",
                        running=True,
                        display_message="Waiting for synchronized CPU runs",
                    ),
                    "slot": session.slot,
                }
            )

        self.assertIs(session.status.battle_view, last_view)
        self.assertEqual(
            session.status.display_message,
            "Waiting for synchronized CPU runs",
        )

    def test_starting_epsilon_reset_updates_cached_stopped_status(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._make_session(Path(directory))
            session._status = replace(
                session._status,
                running=False,
                current_epsilon=0.125,
            )

            session.set_starting_epsilon(0.3)

            self.assertEqual(session.config.starting_epsilon, 0.3)
            self.assertEqual(session.config.epsilon, 0.3)
            self.assertEqual(session.status.current_epsilon, 0.3)

    def test_stopping_status_keeps_time_running_and_zeros_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._make_session(Path(directory))
            session._status = TrainingSessionStatus(
                ship="Earthling",
                running=True,
                completed_batches=11,
                simulation_speed_multiplier=8.0,
            )
            session._completed_batches_at_run_start = 10
            session._elapsed_anchor_seconds = 20.0
            session._elapsed_anchor_at = 100.0

            with mock.patch(
                "src.training.process_session.time.perf_counter",
                return_value=110.0,
            ):
                session.request_stop()
            with mock.patch(
                "src.training.process_session.time.perf_counter",
                return_value=112.0,
            ):
                session._accept_status(
                    {
                        "status": TrainingSessionStatus(
                            ship="Earthling",
                            running=True,
                            completed_batches=11,
                            elapsed_training_seconds=25.0,
                            simulation_speed_multiplier=9.0,
                        )
                    }
                )
            with (
                mock.patch.object(session, "_drain_messages"),
                mock.patch(
                    "src.training.process_session.time.perf_counter",
                    return_value=115.0,
                ),
            ):
                status = session.status

            self.assertTrue(status.stopping)
            self.assertEqual(status.elapsed_training_seconds, 35.0)
            self.assertEqual(status.batches_per_hour, 3600.0 / 35.0)
            self.assertEqual(status.simulation_speed_multiplier, 0.0)

    def test_final_stopped_status_freezes_projected_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            session = self._make_session(Path(directory))
            session._status = TrainingSessionStatus(
                ship="Earthling",
                running=True,
                completed_batches=11,
            )
            session._completed_batches_at_run_start = 10
            session._elapsed_anchor_seconds = 20.0
            session._elapsed_anchor_at = 100.0

            with mock.patch(
                "src.training.process_session.time.perf_counter",
                return_value=112.0,
            ):
                session._accept_status(
                    {
                        "status": TrainingSessionStatus(
                            ship="Earthling",
                            running=False,
                            completed_batches=11,
                            elapsed_training_seconds=25.0,
                        )
                    }
                )
            with (
                mock.patch.object(session, "_drain_messages"),
                mock.patch(
                    "src.training.process_session.time.perf_counter",
                    return_value=200.0,
                ),
            ):
                status = session.status

            self.assertFalse(status.running)
            self.assertFalse(status.stopping)
            self.assertEqual(status.elapsed_training_seconds, 32.0)
            self.assertEqual(status.batches_per_hour, 112.5)

    def test_running_worker_status_does_not_clear_parent_stop_request(self):
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
                    training_device="cpu",
                ),
                batch_grouping=1,
                save_coordinator=ProcessModelSaveCoordinator(),
            )
            session._status = TrainingSessionStatus(ship="Earthling", running=True)

            session.request_stop()
            session._accept_status(
                {
                    "status": TrainingSessionStatus(
                        ship="Earthling",
                        running=True,
                        stopping=False,
                    )
                }
            )

            self.assertTrue(session.status.running)
            self.assertTrue(session.status.stopping)

            session._accept_status(
                {
                    "status": TrainingSessionStatus(
                        ship="Earthling",
                        running=False,
                        stopping=False,
                    )
                }
            )

            self.assertFalse(session.status.running)
            self.assertFalse(session.status.stopping)

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

    def test_synced_cpu_runs_with_different_targets_exit_without_deadlock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            sessions = []
            pacing_group = CpuBatchPacingGroup(2)
            save_coordinator = ProcessModelSaveCoordinator()
            targets = (
                ("Earthling", {"stop_at_batch": 1}),
                ("Androsynth", {"stop_at_epsilon": 0.125}),
            )
            for pacing_index, (ship, target_kwargs) in enumerate(targets):
                metadata = metadata_from_state(
                    ship=ship,
                    slot=1,
                    description="Mixed target pacing test",
                    architecture=model_architecture_metadata(8, 1),
                    training={},
                )
                slot = repository.create_or_update_user_model(metadata)
                sessions.append(
                    ProcessTrainingSession(
                        repository=repository,
                        slot=slot,
                        metadata=metadata,
                        config=TrainingOrchestrationConfig(
                            trainee_ship=ship,
                            rounds_per_batch=1,
                            match_time_limit=2,
                            replay_updates_per_batch=0,
                            hidden_layer_width=8,
                            hidden_layer_count=1,
                            training_device="cpu",
                            epsilon=0.5,
                            epsilon_floor=0.05,
                            epsilon_decay=0.5,
                        ),
                        batch_grouping=1,
                        save_coordinator=save_coordinator,
                        batch_pacing_group=pacing_group,
                        batch_pacing_index=pacing_index,
                        **target_kwargs,
                    )
                )

            try:
                for session in sessions:
                    session.start()
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    statuses = tuple(session.status for session in sessions)
                    for status in statuses:
                        if status.error:
                            self.fail(status.error)
                    if all(not status.running for status in statuses):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("mixed CPU targets did not stop without deadlock")
                for session in sessions:
                    session.join(5.0)

                self.assertEqual(
                    tuple(session.status.completed_batches for session in sessions),
                    (1, 2),
                )
                self.assertEqual(sessions[1].status.current_epsilon, 0.125)
            finally:
                for session in sessions:
                    session.request_stop()
                    session.join(5.0)

    def test_display_frames_advance_continuously_in_worker_shared_memory(self):
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
                    match_time_limit=120,
                    replay_updates_per_batch=0,
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                    display_on=True,
                ),
                batch_grouping=1,
                save_coordinator=ProcessModelSaveCoordinator(),
            )
            try:
                session.start()
                deadline = time.monotonic() + 30.0
                battle_view = None
                frame_ids = set()
                while time.monotonic() < deadline:
                    status = session.status
                    if status.error:
                        self.fail(status.error)
                    battle_view = status.battle_view
                    if battle_view and battle_view.get("rendered_frames"):
                        frame_ids.add(int(battle_view["frame_id"]))
                    if len(frame_ids) >= 6:
                        break
                    time.sleep(0.02)
                else:
                    self.fail(
                        "CPU process did not publish six distinct display frames; "
                        f"received {sorted(frame_ids)}"
                    )

                self.assertEqual(
                    len(battle_view["rendered_frames"]),
                    const.VIDEO_FPS_MULTIPLIER,
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

    def test_auto_uses_cpu_process_and_gpu_keeps_thread_session(self):
        self.assertIs(
            train_ai.independent_session_class(torch_backend.DEVICE_AUTO),
            ProcessTrainingSession,
        )
        self.assertIs(
            train_ai.independent_session_class(torch_backend.DEVICE_GPU),
            train_ai.TrainingSession,
        )


if __name__ == "__main__":
    unittest.main()
