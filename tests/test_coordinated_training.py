import csv
import multiprocessing
from multiprocessing import shared_memory
import pickle
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.const as const
from src.audio import RecordingAudioService
from src.training import torch_backend
from src.training import batched_value_network
from src.training.coordinated import CPU_WORKER_FALLBACK_NOTICE
from src.training.batched_value_network import BatchedValueNetworkParameterCache
from src.training.contracts import ACTION_OUTPUT_SIZE, OBSERVATION_INPUT_SIZE
from src.training.contracts import action_for_index
from src.training.observation_transfer import (
    PACKED_OBSERVATION_BYTES,
    PackedObservation,
    pack_observation,
    unpack_observation,
    unpack_observation_array,
)
from src.training.coordinated import (
    CoordinatedActionRequest,
    CoordinatedFixedFrameWindowResult,
    CoordinatedRuntimeComponents,
    TrainingEpisodeResult,
    CoordinatedTrainingRecord,
    CoordinatedTrainingSession,
    _ProcessWorkerClient,
    append_coordinated_batch_timing_csv,
    select_actions_for_records,
    select_opponent_controls_for_windows,
    run_coordinated_fixed_frame_window,
)
from src.training.process_worker import (
    COMMAND_REQUEST_OBSERVATION,
    CoordinatedSimulationWorker,
    DisplayBufferSpec,
    FinishWindowCommand,
    FrameSteppedResult,
    RequestObservationCommand,
    ShutdownCommand,
    StartRunCommand,
    StartWindowCommand,
    StepFrameCommand,
    WindowFinishedResult,
    WindowObservationResult,
    WindowStartedResult,
    WorkerErrorResult,
    WorkerReadyResult,
    WorkerStoppedResult,
    start_worker_process,
)
from src.training.rewards import RewardDecisionFrame, RewardFrameOutcome
from src.training.model_registry import (
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.orchestration import TrainingOrchestrationConfig
from src.training.orchestration import OpponentSpec
from src.training.orchestration import TrainingBatchAborted
from src.training.orchestration import ValueNetworkPolicy
from src.training.replay import ActionSelection, ReplaySample, TrainingReplayBuffer
from src.training.value_network import ValueNetworkConfig, build_optimizer, build_value_network
from src.resources import HeadlessAssetManager


def _record(instance_id, ship, batch_grouping=1, **config_overrides):
    metadata = metadata_from_state(
        ship=ship,
        slot=1,
        description=f"{ship} test",
        architecture=model_architecture_metadata(8, 1),
        training={},
    )
    config_kwargs = {
        "trainee_ship": ship,
        "hidden_layer_width": 8,
        "hidden_layer_count": 1,
        "training_device": "cpu",
    }
    config_kwargs.update(config_overrides)
    return CoordinatedTrainingRecord(
        instance_id=instance_id,
        repository=TrainingModelRepository(Path("unused"), Path("unused")),
        slot=TrainingModelSlot(ship, 1, SLOT_USER, metadata=metadata),
        metadata=metadata,
        config=TrainingOrchestrationConfig(**config_kwargs),
        batch_grouping=batch_grouping,
    )


class CoordinatedTrainingSessionTests(unittest.TestCase):
    def _session(self, *, component_builder=None, **kwargs):
        return CoordinatedTrainingSession(
            (_record(1, "Earthling"), _record(2, "Androsynth")),
            component_builder=component_builder or self._component_builder([]),
            run_batches=False,
            idle_sleep_seconds=0.001,
            **kwargs,
        )

    def _component_builder(self, built):
        def build(record):
            built.append(record.instance_id)
            return CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=TrainingReplayBuffer(4),
            )

        return build

    def test_start_builds_records_and_stop_all_marks_every_proxy_stopped(self):
        built = []
        session = self._session(component_builder=self._component_builder(built))

        session.start()
        deadline = time.time() + 1.0
        while len(built) < 2 and time.time() < deadline:
            time.sleep(0.005)

        proxies = session.proxies
        self.assertEqual(built, [1, 2])
        self.assertTrue(proxies[1].status.running)
        self.assertTrue(proxies[2].status.running)
        self.assertEqual(proxies[1].status.display_message, "Coordinated scheduler idle")

        proxies[1].request_stop()
        session.join(1.0)

        self.assertFalse(session.active)
        self.assertFalse(proxies[1].status.running)
        self.assertFalse(proxies[1].status.stopping)
        self.assertFalse(proxies[2].status.running)

    def test_record_progress_updates_live_status_every_100_frames(self):
        session = self._session()
        state = session._states[1]

        session._on_record_progress(
            state,
            {
                "event": "frame",
                "frame": 99,
                "opponent": OpponentSpec("Earthling"),
                "replay_size": 99,
                "weighted_total_return": 9.9,
            },
        )

        status = session.status_for_instance(1)
        self.assertEqual(status.current_frame, 0)
        self.assertEqual(status.replay_size, 0)
        self.assertEqual(status.weighted_total_return, 0.0)

        session._on_record_progress(
            state,
            {
                "event": "frame",
                "frame": 100,
                "opponent": OpponentSpec("Earthling"),
                "replay_size": 100,
                "weighted_total_return": 10.0,
            },
        )

        status = session.status_for_instance(1)
        self.assertEqual(status.current_frame, 100)
        self.assertEqual(status.replay_size, 100)
        self.assertEqual(status.weighted_total_return, 10.0)

    def test_display_proxy_selects_instance_and_paces_at_physics_rate(self):
        session = self._session()
        session.proxies[1].set_display_on(True)
        session._display_changed.clear()
        session._next_display_frame_time = 100.0

        with (
            mock.patch("src.training.coordinated.time.perf_counter", return_value=100.0),
            mock.patch.object(session._display_changed, "wait", return_value=False) as wait,
        ):
            session._throttle_display_frame()

        wait.assert_called_once()
        self.assertAlmostEqual(wait.call_args.args[0], 1.0 / 24.0)

    def test_display_progress_updates_selected_instance_every_frame(self):
        session = self._session()
        session.set_display_on(1, True)
        rendered_frames = (object(),)

        session._on_record_progress(
            session._states[1],
            {
                "event": "frame",
                "frame": 1,
                "opponent": OpponentSpec("Earthling"),
                "battle_view": {
                    "frame_id": 1,
                    "rendered_frames": rendered_frames,
                },
            },
        )

        status = session.status_for_instance(1)
        self.assertEqual(status.current_frame, 1)
        self.assertIs(status.battle_view["rendered_frames"], rendered_frames)

    def test_display_starts_and_stops_coordinated_music_once(self):
        audio = RecordingAudioService()
        session = self._session(audio_service=audio)

        session.set_display_on(1, True)
        session.set_display_on(1, True)
        session.set_display_on(1, False)

        self.assertEqual(
            audio.operations,
            [("start_battle_music",), ("stop_music",)],
        )

    def test_coordinated_effects_follow_only_the_displayed_instance(self):
        audio = RecordingAudioService()
        session = self._session(audio_service=audio)
        session.set_display_on(1, True)
        audio.operations.clear()

        session._relay_audio_events(2, (("play_effect", "hidden.wav", 0.5),))
        session._relay_audio_events(1, (("play_effect", "visible.wav", 0.25),))

        self.assertEqual(
            audio.operations,
            [("play_effect", Path("visible.wav"), 0.25)],
        )

    def test_component_build_error_marks_all_records_and_exits(self):
        def fail(_record):
            raise RuntimeError("component build failed")

        session = self._session(component_builder=fail)

        session.start()
        session.join(1.0)

        self.assertFalse(session.active)
        self.assertEqual(session.status_for_instance(1).error, "component build failed")
        self.assertEqual(session.status_for_instance(2).error, "component build failed")


class FixedPolicy:
    def __init__(self, action_index=0):
        self.action_index = int(action_index)
        self.selection_count = 0
        self.reset_count = 0

    def select_action(self, _observation):
        self.selection_count += 1
        return ActionSelection(self.action_index, exploratory=False)

    def reset_exploration_span(self):
        self.reset_count += 1


class SequencePolicy:
    def __init__(self, selections):
        self.selections = list(selections)
        self.observations = []

    def select_action(self, observation):
        self.observations.append(tuple(observation))
        return self.selections.pop(0)


class ScriptedSimulation:
    def __init__(
        self,
        _screen,
        player1,
        player2,
        *,
        terminal_after=None,
        pending_rebirth=False,
        audio_service=None,
        resources=None,
        rng=None,
        include_stars=False,
        training_event_ledger=None,
    ):
        self.player1 = player1
        self.player2 = player2
        self.frame_id = 0
        self.world = []
        self.aftermath = None
        self.terminal_after = terminal_after
        self.pending_rebirth = bool(pending_rebirth)

    def step(self, actions=None):
        self.frame_id += 1
        if self.pending_rebirth:
            self.player1.current_hp = 0
            self.player1.currently_alive = False
            self.aftermath = SimpleNamespace(pending_rebirths=[self.player1])
        elif self.terminal_after is not None and self.frame_id >= self.terminal_after:
            self.player2.current_hp = 0
            self.player2.currently_alive = False
        return {"frame_id": self.frame_id}


class StepLoggingSimulation(ScriptedSimulation):
    def __init__(self, *args, step_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_log = step_log

    def step(self, actions=None):
        state = super().step(actions=actions)
        if self.step_log is not None:
            self.step_log.append((self.player1.name, self.frame_id))
        return state


class ScriptedSimulationFactory:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.created = []

    def __call__(self, *args, **kwargs):
        script = self.scripts.pop(0) if self.scripts else {}
        simulation = ScriptedSimulation(*args, **script, **kwargs)
        self.created.append(simulation)
        return simulation


class StepLoggingSimulationFactory(ScriptedSimulationFactory):
    def __init__(self, scripts, step_log):
        super().__init__(scripts)
        self.step_log = step_log

    def __call__(self, *args, **kwargs):
        script = self.scripts.pop(0) if self.scripts else {}
        simulation = StepLoggingSimulation(
            *args,
            step_log=self.step_log,
            **script,
            **kwargs,
        )
        self.created.append(simulation)
        return simulation


class RandomSequence:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        if not self.values:
            return 1.0
        return self.values.pop(0)


def fake_ship(name, _player_id, *, resources=None, audio_service=None):
    return SimpleNamespace(
        name=name,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        rotation=0.0,
        start_hp=10,
        current_hp=1,
        currently_alive=True,
        current_energy=0.0,
        max_thrust=1.0,
    )


def fake_decision_frame(**kwargs):
    return RewardDecisionFrame(
        frame_id=kwargs["frame_id"],
        observation=tuple(kwargs["observation"]),
        action_index=kwargs["action_index"],
    )


def fake_frame_outcome(**kwargs):
    return RewardFrameOutcome(
        frame_id=kwargs["frame_id"],
        terminal=kwargs["terminal"],
    )


class CoordinatedFixedFrameWindowTests(unittest.TestCase):
    def run_window(self, *, config, policy=None, simulation_factory=None, rng=None):
        replay = TrainingReplayBuffer(32)
        events = []
        policy = policy or FixedPolicy(0)
        simulation_factory = simulation_factory or ScriptedSimulationFactory([{}])
        with (
            mock.patch(
                "src.training.coordinated.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.coordinated.decision_frame_from_battle_state",
                side_effect=fake_decision_frame,
            ),
            mock.patch(
                "src.training.coordinated.frame_outcome_from_battle_state",
                side_effect=fake_frame_outcome,
            ),
        ):
            result = run_coordinated_fixed_frame_window(
                opponent=OpponentSpec("Earthling"),
                trainee_policy=policy,
                replay_buffer=replay,
                config=config,
                rng=rng,
                simulation_factory=simulation_factory,
                progress_callback=events.append,
                ship_factory=fake_ship,
            )
        return result, replay, events, policy, simulation_factory

    def test_window_consumes_exact_configured_frame_budget(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=3,
            gamma=0.0,
        )

        result, replay, events, policy, _factory = self.run_window(config=config)

        self.assertEqual(result.frames, 3)
        self.assertEqual([event["frame"] for event in events], [1, 2, 3])
        self.assertEqual(policy.selection_count, 3)
        self.assertEqual(len(replay), 3)
        self.assertEqual(len(result.episode_results), 1)
        self.assertEqual(result.episode_results[0].terminal_reason, "timeout")
        self.assertEqual(result.episode_results[0].frames, 3)

    def test_terminal_reset_continues_inside_same_fixed_window(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=5,
            gamma=0.0,
        )
        factory = ScriptedSimulationFactory(
            (
                {"terminal_after": 2},
                {},
            )
        )

        result, replay, _events, policy, factory = self.run_window(
            config=config,
            simulation_factory=factory,
        )

        self.assertEqual(result.frames, 5)
        self.assertEqual(policy.selection_count, 5)
        self.assertEqual(len(factory.created), 2)
        self.assertEqual(len(replay), 5)
        self.assertEqual(
            [episode.terminal_reason for episode in result.episode_results],
            ["resolved", "timeout"],
        )
        self.assertEqual(
            [episode.frames for episode in result.episode_results],
            [2, 3],
        )

    def test_pending_rebirth_consumes_budget_without_reset(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=4,
            gamma=0.0,
        )
        factory = ScriptedSimulationFactory(({"pending_rebirth": True},))

        result, replay, _events, policy, factory = self.run_window(
            config=config,
            simulation_factory=factory,
        )

        self.assertEqual(result.frames, 4)
        self.assertEqual(policy.selection_count, 4)
        self.assertEqual(len(factory.created), 1)
        self.assertEqual(len(replay), 4)
        self.assertEqual(len(result.episode_results), 1)
        self.assertEqual(result.episode_results[0].terminal_reason, "timeout")

    def test_new_window_randomizes_ship_start_hp(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=1,
            gamma=0.0,
        )
        factory = ScriptedSimulationFactory([{}])

        self.run_window(
            config=config,
            simulation_factory=factory,
            rng=RandomSequence([0.25, 0.01]),
        )

        self.assertEqual(factory.created[0].player1.current_hp, 3)
        self.assertEqual(factory.created[0].player2.current_hp, 1)


class CoordinatedProcessWorkerProtocolTests(unittest.TestCase):
    def test_command_and_result_dataclasses_are_picklable(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=3,
        )
        command = StartWindowCommand(
            record_id=1,
            round_index=2,
            config=config,
            opponent=OpponentSpec("Androsynth"),
            rng_seed=123,
        )
        result = WindowObservationResult(
            record_id=1,
            round_index=2,
            frame_count=0,
            trainee_observation=pack_observation(
                (0.0,) * OBSERVATION_INPUT_SIZE
            ),
            simple_opponent_controls={"forward": True},
        )

        self.assertEqual(pickle.loads(pickle.dumps(command)), command)
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)

    def test_worker_rejects_model_objects_in_start_window_commands(self):
        config = TrainingOrchestrationConfig(trainee_ship="Earthling")

        with self.assertRaises(ValueError):
            StartWindowCommand(
                record_id=1,
                round_index=1,
                config=config,
                opponent=OpponentSpec("Earthling", model=object()),
                rng_seed=1,
            )

    def test_worker_process_starts_and_shuts_down(self):
        try:
            context = multiprocessing.get_context("spawn")
        except ValueError:
            self.skipTest("spawn multiprocessing context is unavailable")
        process, connection = start_worker_process(context=context)
        try:
            connection.send(StartRunCommand(worker_id=3, record_id=7, base_seed=11))
            ready = connection.recv()
            self.assertIsInstance(ready, WorkerReadyResult)
            self.assertEqual(ready.worker_id, 3)
            self.assertEqual(ready.record_id, 7)
            self.assertFalse(ready.torch_imported)

            connection.send(ShutdownCommand())
            stopped = connection.recv()
            self.assertIsInstance(stopped, WorkerStoppedResult)
            self.assertEqual(stopped.worker_id, 3)
            self.assertEqual(stopped.record_id, 7)
        finally:
            connection.close()
            process.join(2.0)
            if process.is_alive():
                process.terminate()
                process.join(2.0)
        self.assertFalse(process.is_alive())

    def test_worker_process_writes_selected_display_frame_to_shared_memory(self):
        try:
            context = multiprocessing.get_context("spawn")
        except ValueError:
            self.skipTest("spawn multiprocessing context is unavailable")
        frame_bytes = const.SCREEN_WIDTH * const.SCREEN_HEIGHT * 3
        memory = shared_memory.SharedMemory(create=True, size=frame_bytes)
        process, connection = start_worker_process(context=context)
        try:
            connection.send(StartRunCommand(1, 1, 1, video_fps_multiplier=1))
            ready = connection.recv()
            self.assertIsInstance(ready, WorkerReadyResult)
            self.assertFalse(ready.torch_imported)
            connection.send(
                StartWindowCommand(
                    1,
                    1,
                    TrainingOrchestrationConfig(
                        trainee_ship="Earthling",
                        match_time_limit=1,
                    ),
                    OpponentSpec("Androsynth"),
                    1,
                )
            )
            started = connection.recv()
            self.assertIsInstance(started, WindowStartedResult)
            self.assertFalse(started.torch_imported)
            connection.send(
                StepFrameCommand(
                    1,
                    1,
                    0,
                    False,
                    opponent_controls={},
                    display_buffer=DisplayBufferSpec(memory.name, frame_count=1),
                )
            )
            result = connection.recv()
            self.assertIsInstance(result, FrameSteppedResult)
            self.assertFalse(result.torch_imported)
            self.assertEqual(result.display_frames_ready, 1)
            self.assertTrue(any(memory.buf[:1000]))
            connection.send(ShutdownCommand())
            self.assertIsInstance(connection.recv(), WorkerStoppedResult)
        finally:
            connection.close()
            process.join(3.0)
            if process.is_alive():
                process.terminate()
                process.join(2.0)
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                pass


class CoordinatedProcessWorkerWindowTests(unittest.TestCase):
    def _worker(self, *, simulation_factory=None):
        return CoordinatedSimulationWorker(
            simulation_factory=simulation_factory or ScriptedSimulationFactory([{}]),
            ship_factory=fake_ship,
        )

    def _start_window(self, worker, *, config=None, opponent=None, rng_seed=123):
        config = config or TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            match_time_limit=3,
            gamma=0.0,
        )
        worker.handle(StartRunCommand(worker_id=1, record_id=9, base_seed=1))
        return worker.handle(
            StartWindowCommand(
                record_id=9,
                round_index=1,
                config=config,
                opponent=opponent or OpponentSpec("Androsynth"),
                rng_seed=rng_seed,
            )
        )

    def test_start_window_and_observation_use_worker_local_state(self):
        worker = self._worker()
        with mock.patch(
            "src.training.process_worker.encode_observation",
            return_value=[0.0] * OBSERVATION_INPUT_SIZE,
        ):
            started = self._start_window(worker)
            observation = worker.handle(RequestObservationCommand(9, 1))

        self.assertEqual(started.name, "WINDOW_STARTED")
        self.assertEqual(started.frame_limit, 3)
        self.assertIsInstance(observation, WindowObservationResult)
        self.assertEqual(observation.frame_count, 0)
        self.assertEqual(
            tuple(unpack_observation(observation.trainee_observation)),
            (0.0,) * OBSERVATION_INPUT_SIZE,
        )
        self.assertIsNotNone(observation.simple_opponent_controls)

    def test_worker_constructs_battles_with_headless_resources(self):
        constructed_resources = []

        def recording_ship(name, player_id, *, resources=None, audio_service=None):
            constructed_resources.append(resources)
            return fake_ship(
                name,
                player_id,
                resources=resources,
                audio_service=audio_service,
            )

        worker = CoordinatedSimulationWorker(
            simulation_factory=ScriptedSimulationFactory([{}]),
            ship_factory=recording_ship,
        )

        self._start_window(worker)

        self.assertIsInstance(worker._resources, HeadlessAssetManager)
        self.assertEqual(constructed_resources, [worker._resources, worker._resources])

    def test_worker_encodes_each_simple_decision_state_once(self):
        worker = self._worker()
        encoded = [0.0] * OBSERVATION_INPUT_SIZE
        with (
            mock.patch(
                "src.training.process_worker.encode_observation",
                return_value=encoded,
            ) as encode,
            mock.patch(
                "src.training.coordinated_simulation.SimpleOpponentController."
                "direct_controls_for_frame",
                return_value=action_for_index(0),
            ) as simple_controls,
        ):
            self._start_window(
                worker,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    match_time_limit=3,
                    gamma=0.0,
                ),
            )
            # The compatibility/debug request returns the cache and does not encode.
            worker.handle(RequestObservationCommand(9, 1))
            first = worker.handle(StepFrameCommand(9, 1, 0, False, {}))
            second = worker.handle(StepFrameCommand(9, 1, 0, False, {}))
            third = worker.handle(StepFrameCommand(9, 1, 0, False, {}))

        self.assertEqual(encode.call_count, 3)
        self.assertEqual(simple_controls.call_count, 3)
        self.assertFalse(first.complete)
        self.assertFalse(second.complete)
        self.assertTrue(third.complete)
        self.assertIsNone(third.next_trainee_observation)
        self.assertIsNone(third.next_opponent_observation)
        self.assertIsNone(third.next_simple_opponent_controls)

    def test_model_opponent_observations_are_encoded_once_per_decision_state(self):
        worker = self._worker()
        with mock.patch(
            "src.training.process_worker.encode_observation",
            return_value=[0.0] * OBSERVATION_INPUT_SIZE,
        ) as encode:
            started = self._start_window(
                worker,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    match_time_limit=2,
                    gamma=0.0,
                ),
                opponent=OpponentSpec(
                    "Androsynth",
                    mode="all",
                    slot=1,
                ),
            )
            worker.handle(RequestObservationCommand(9, 1))
            first = worker.handle(StepFrameCommand(9, 1, 0, False, {}))
            second = worker.handle(StepFrameCommand(9, 1, 0, False, {}))

        self.assertIsNotNone(started.opponent_observation)
        self.assertIsNotNone(first.next_opponent_observation)
        self.assertIsNone(second.next_opponent_observation)
        self.assertEqual(encode.call_count, 4)

    def test_terminal_reset_returns_first_observation_of_new_battle(self):
        factory = ScriptedSimulationFactory(({"terminal_after": 1}, {}))
        worker = self._worker(simulation_factory=factory)

        def encode_for_battle(self_ship, _enemy_ship, **_kwargs):
            value = 0.0 if self_ship is factory.created[0].player1 else 1.0
            return [value] * OBSERVATION_INPUT_SIZE

        with mock.patch(
            "src.training.process_worker.encode_observation",
            side_effect=encode_for_battle,
        ):
            self._start_window(
                worker,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    match_time_limit=2,
                    gamma=0.0,
                ),
            )
            result = worker.handle(StepFrameCommand(9, 1, 0, False, {}))

        self.assertIsNotNone(result.terminal_episode)
        self.assertEqual(len(factory.created), 2)
        self.assertEqual(
            tuple(unpack_observation(result.next_trainee_observation)),
            (1.0,) * OBSERVATION_INPUT_SIZE,
        )

    def test_step_frame_returns_mature_samples_for_parent_replay_insertion(self):
        parent_replay = TrainingReplayBuffer(16)
        worker = self._worker()
        with (
            mock.patch(
                "src.training.coordinated.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.process_worker.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.coordinated.decision_frame_from_battle_state",
                side_effect=fake_decision_frame,
            ),
            mock.patch(
                "src.training.coordinated.frame_outcome_from_battle_state",
                side_effect=fake_frame_outcome,
            ),
        ):
            self._start_window(worker)
            step = worker.handle(
                StepFrameCommand(
                    record_id=9,
                    round_index=1,
                    trainee_action_index=3,
                    trainee_exploratory=False,
                    opponent_controls={"forward": False},
                )
            )

        self.assertIsInstance(step, FrameSteppedResult)
        self.assertEqual(step.frame_count, 1)
        self.assertEqual(len(step.mature_samples), 1)
        self.assertEqual(len(parent_replay), 0)
        parent_replay.extend(step.mature_samples)
        self.assertEqual(len(parent_replay), 1)
        self.assertEqual(parent_replay[0].action_index, 3)

    def test_step_frame_captures_interpolated_frames_when_requested(self):
        worker = self._worker()
        self._start_window(worker)
        display_buffer = DisplayBufferSpec("unused", frame_count=3)

        with mock.patch.object(
            worker,
            "_capture_display_frames",
            return_value=3,
        ) as capture:
            step = worker.handle(
                StepFrameCommand(
                    9,
                    1,
                    0,
                    False,
                    opponent_controls={},
                    display_buffer=display_buffer,
                )
            )

        self.assertEqual(step.display_frames_ready, 3)
        capture.assert_called_once_with(worker._runtime, display_buffer)

    def test_step_frame_returns_audio_events_when_display_capture_is_enabled(self):
        worker = self._worker()
        self._start_window(worker)
        worker._audio_service.play_effect(Path("effect.wav"), 0.25)

        step = worker.handle(
            StepFrameCommand(
                9,
                1,
                0,
                False,
                opponent_controls={},
                capture_audio=True,
            )
        )

        self.assertEqual(
            step.audio_events,
            (("play_effect", str(Path("effect.wav")), 0.25),),
        )

    def test_terminal_reset_and_finish_window_preserve_fixed_window_semantics(self):
        factory = ScriptedSimulationFactory(
            (
                {"terminal_after": 1},
                {},
            )
        )
        worker = self._worker(simulation_factory=factory)
        with (
            mock.patch(
                "src.training.coordinated.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.process_worker.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.coordinated.decision_frame_from_battle_state",
                side_effect=fake_decision_frame,
            ),
            mock.patch(
                "src.training.coordinated.frame_outcome_from_battle_state",
                side_effect=fake_frame_outcome,
            ),
        ):
            self._start_window(
                worker,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    match_time_limit=2,
                    gamma=0.0,
                ),
            )
            first = worker.handle(
                StepFrameCommand(9, 1, 0, False, opponent_controls={})
            )
            second = worker.handle(
                StepFrameCommand(9, 1, 0, False, opponent_controls={})
            )
            display_memory = mock.Mock()
            worker._display_memory = display_memory
            finished = worker.handle(FinishWindowCommand(9, 1))

        self.assertEqual(len(factory.created), 2)
        self.assertIsNotNone(first.terminal_episode)
        self.assertEqual(first.terminal_episode.terminal_reason, "resolved")
        self.assertTrue(second.complete)
        self.assertIsInstance(finished, WindowFinishedResult)
        self.assertEqual(finished.result.frames, 2)
        self.assertEqual(
            [episode.terminal_reason for episode in finished.result.episode_results],
            ["resolved", "timeout"],
        )
        self.assertEqual(len(first.mature_samples), 1)
        self.assertEqual(len(second.mature_samples), 1)
        self.assertIsNone(worker._runtime)
        self.assertIsNone(worker._collector)
        self.assertIsNone(worker._round_index)
        self.assertIsNone(worker._display_memory)
        display_memory.close.assert_called_once_with()

    def test_worker_process_reports_command_errors(self):
        try:
            context = multiprocessing.get_context("spawn")
        except ValueError:
            self.skipTest("spawn multiprocessing context is unavailable")
        process, connection = start_worker_process(context=context)
        try:
            connection.send(RequestObservationCommand(99, 1))
            error = connection.recv()
            self.assertIsInstance(error, WorkerErrorResult)
            self.assertEqual(error.command_name, COMMAND_REQUEST_OBSERVATION)
            self.assertIn("has not been started", error.exception_message)

            connection.send(ShutdownCommand())
            stopped = connection.recv()
            self.assertIsInstance(stopped, WorkerStoppedResult)
        finally:
            connection.close()
            process.join(2.0)
            if process.is_alive():
                process.terminate()
                process.join(2.0)
        self.assertFalse(process.is_alive())


class CoordinatedActionSelectionTests(unittest.TestCase):
    def test_batched_inference_accepts_packed_observation_numeric_views(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")
        torch = torch_backend.require_torch()
        models = (
            build_value_network(ValueNetworkConfig(8, 1)),
            build_value_network(ValueNetworkConfig(8, 1)),
        )
        with torch.no_grad():
            for model in models:
                for parameter in model.parameters():
                    parameter.zero_()
                model[-1].bias.copy_(torch.arange(ACTION_OUTPUT_SIZE).float())
        requests = tuple(
            CoordinatedActionRequest(
                record_id=index,
                policy=ValueNetworkPolicy(model, epsilon=0.0),
                observation=unpack_observation_array(
                    pack_observation([float(index)] * OBSERVATION_INPUT_SIZE)
                ),
            )
            for index, model in enumerate(models, start=1)
        )

        result = select_actions_for_records(requests)

        self.assertEqual(result.inference_mode, "batched_value_network")
        self.assertEqual(
            tuple(result.selections[index].action_index for index in (1, 2)),
            (ACTION_OUTPUT_SIZE - 1, ACTION_OUTPUT_SIZE - 1),
        )

    def test_select_actions_for_records_routes_epsilon_policy_results(self):
        policy1 = SequencePolicy((ActionSelection(3, exploratory=False),))
        policy2 = SequencePolicy((ActionSelection(7, exploratory=True),))

        result = select_actions_for_records(
            (
                CoordinatedActionRequest(1, policy1, (1.0, 2.0)),
                CoordinatedActionRequest(2, policy2, (3.0, 4.0)),
            )
        )

        self.assertEqual(result.inference_mode, "sequential_fallback")
        self.assertEqual(result.request_count, 2)
        self.assertEqual(result.exploratory_count, 1)
        self.assertEqual(result.selections[1].action_index, 3)
        self.assertEqual(result.selections[2].action_index, 7)
        self.assertEqual(policy1.observations, [(1.0, 2.0)])
        self.assertEqual(policy2.observations, [(3.0, 4.0)])

    def test_select_actions_for_records_rejects_duplicate_record_ids(self):
        policy = SequencePolicy(
            (
                ActionSelection(1, exploratory=False),
                ActionSelection(2, exploratory=False),
            )
        )

        with self.assertRaises(ValueError):
            select_actions_for_records(
                (
                    CoordinatedActionRequest(1, policy, ()),
                    CoordinatedActionRequest(1, policy, ()),
                )
            )

    def test_select_actions_for_records_batches_value_network_greedy_requests(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")
        torch = torch_backend.require_torch()
        models = (
            build_value_network(ValueNetworkConfig(8, 1)),
            build_value_network(ValueNetworkConfig(8, 1)),
        )
        with torch.no_grad():
            for model, action_index in zip(models, (3, 5)):
                for parameter in model.parameters():
                    parameter.zero_()
                model[-1].bias.copy_(torch.arange(ACTION_OUTPUT_SIZE).float())
                model[-1].bias[action_index] = 100.0
        policies = tuple(
            ValueNetworkPolicy(model, epsilon=0.0)
            for model in models
        )

        result = select_actions_for_records(
            (
                CoordinatedActionRequest(
                    1,
                    policies[0],
                    [0.0] * OBSERVATION_INPUT_SIZE,
                ),
                CoordinatedActionRequest(
                    2,
                    policies[1],
                    [1.0] * OBSERVATION_INPUT_SIZE,
                ),
            )
        )

        self.assertEqual(result.inference_mode, "batched_value_network")
        self.assertEqual(result.exploratory_count, 0)
        self.assertEqual(result.selections[1].action_index, 3)
        self.assertEqual(result.selections[2].action_index, 5)
        self.assertIsNone(result.selections[1].action_values)
        self.assertIsNone(result.selections[2].action_values)

    def test_select_actions_for_records_reuses_cached_batched_parameters(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")
        torch = torch_backend.require_torch()
        models = (
            build_value_network(ValueNetworkConfig(8, 1)),
            build_value_network(ValueNetworkConfig(8, 1)),
        )
        policies = tuple(
            ValueNetworkPolicy(model, epsilon=0.0)
            for model in models
        )
        requests = (
            CoordinatedActionRequest(
                1,
                policies[0],
                [0.0] * OBSERVATION_INPUT_SIZE,
            ),
            CoordinatedActionRequest(
                2,
                policies[1],
                [1.0] * OBSERVATION_INPUT_SIZE,
            ),
        )
        cache = BatchedValueNetworkParameterCache()
        stack_parameters = batched_value_network._stack_linear_parameters

        with mock.patch(
            "src.training.batched_value_network._stack_linear_parameters",
            wraps=stack_parameters,
        ) as stack_mock:
            first = select_actions_for_records(requests, parameter_cache=cache)
            second = select_actions_for_records(requests, parameter_cache=cache)

        self.assertEqual(stack_mock.call_count, 1)
        self.assertEqual(first.inference_mode, "batched_value_network")
        self.assertEqual(second.inference_mode, "batched_value_network")

    def test_select_actions_cache_uses_only_changing_greedy_subset(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")
        models = (
            build_value_network(ValueNetworkConfig(8, 1)),
            build_value_network(ValueNetworkConfig(8, 1)),
        )
        policies = []
        prepared_results = (
            (
                ActionSelection(action_index=1, exploratory=True),
                None,
            ),
            (
                None,
                ActionSelection(action_index=2, exploratory=True),
            ),
        )
        for model, results in zip(models, prepared_results):
            policy = mock.Mock()
            policy.model = model
            policy.prepare_action_selection.side_effect = results
            policy.complete_greedy_selection.side_effect = (
                lambda action_index: ActionSelection(
                    action_index=int(action_index),
                    exploratory=False,
                )
            )
            policies.append(policy)
        requests = tuple(
            CoordinatedActionRequest(
                index,
                policy,
                [float(index)] * OBSERVATION_INPUT_SIZE,
            )
            for index, policy in enumerate(policies, start=1)
        )
        cache = BatchedValueNetworkParameterCache(max_entries=2)
        stack_parameters = batched_value_network._stack_linear_parameters

        with mock.patch(
            "src.training.batched_value_network._stack_linear_parameters",
            wraps=stack_parameters,
        ) as stack_mock:
            first = select_actions_for_records(requests, parameter_cache=cache)
            second = select_actions_for_records(requests, parameter_cache=cache)

        self.assertEqual(stack_mock.call_count, 2)
        self.assertEqual(stack_mock.call_args_list[0].args[0], (models[1],))
        self.assertEqual(stack_mock.call_args_list[1].args[0], (models[0],))
        self.assertEqual(len(cache), 2)
        self.assertTrue(first.selections[1].exploratory)
        self.assertFalse(first.selections[2].exploratory)
        self.assertFalse(second.selections[1].exploratory)
        self.assertTrue(second.selections[2].exploratory)

    def test_opponent_controls_batch_only_ai_backed_windows(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")
        torch = torch_backend.require_torch()
        model = build_value_network(ValueNetworkConfig(8, 1))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model[-1].bias.copy_(torch.arange(ACTION_OUTPUT_SIZE).float())
            model[-1].bias[4] = 100.0
        simple_controls = {"forward": True}
        simple_controller = mock.Mock()
        simple_controller.controls_for_frame.return_value = simple_controls
        simple_window = SimpleNamespace(
            opponent=OpponentSpec("Earthling"),
            simulation=SimpleNamespace(
                player1=SimpleNamespace(),
                player2=SimpleNamespace(),
                frame_id=0,
                world=(),
            ),
            simple_controller=simple_controller,
        )
        ai_window = SimpleNamespace(
            opponent=OpponentSpec("Androsynth", model=model),
            simulation=SimpleNamespace(
                player1=SimpleNamespace(),
                player2=SimpleNamespace(),
                frame_id=0,
                world=(),
            ),
            simple_controller=mock.Mock(),
        )

        with mock.patch(
            "src.training.coordinated.encode_observation",
            return_value=[0.0] * OBSERVATION_INPUT_SIZE,
        ):
            controls = select_opponent_controls_for_windows(
                (simple_window, ai_window)
        )

        self.assertIs(controls[id(simple_window)], simple_controls)
        self.assertEqual(controls[id(ai_window)]["right"], True)
        simple_controller.controls_for_frame.assert_called_once()
        ai_window.simple_controller.controls_for_frame.assert_not_called()


class CoordinatedTimingCsvTests(unittest.TestCase):
    def test_append_coordinated_batch_timing_csv_writes_header_and_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.coordinated.csv"
            append_coordinated_batch_timing_csv(
                path,
                batch_number=3,
                instance_id=7,
                ship="Earthling",
                slot=1,
                instance_count=2,
                rounds=1,
                instance_frames=1200,
                coordinated_record_frames=2400,
                action_requests=2400,
                exploratory_actions=12,
                inference_mode="sequential_fallback:1200",
                batch_seconds=180.0,
                batches_per_hour=20.0,
                metrics=SimpleNamespace(
                    wins=1,
                    match_count=2,
                    average_match_score=4.25,
                    epsilon=0.5,
                    learning_rate=0.001,
                    average_loss=0.125,
                ),
                timing_seconds={
                    "observation": 1.0,
                    "trainee_inference": 2.0,
                    "opponent_inference": 3.0,
                    "simulation": 4.0,
                    "simulation_collision": 1.25,
                    "reward": 5.0,
                    "reward_pipeline": 0.75,
                    "optimization": 6.0,
                    "save": 7.0,
                    "collision_candidate_pairs": 123,
                    "collision_spatial_queries": 45,
                },
            )

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["Batch"], "3")
        self.assertEqual(row["Instance ID"], "7")
        self.assertEqual(row["Ship"], "Earthling")
        self.assertEqual(row["Instance Count"], "2")
        self.assertEqual(row["Instance Frames"], "1200")
        self.assertEqual(row["Coordinated Record Frames"], "2400")
        self.assertEqual(row["Inference Mode"], "sequential_fallback:1200")
        self.assertEqual(row["Win %"], "50.0")
        self.assertEqual(row["Timed Total Seconds"], "28.000000")
        self.assertEqual(row["Simulation Collision Seconds"], "1.250000")
        self.assertEqual(row["Reward Pipeline Seconds"], "0.750000")
        self.assertEqual(row["Collision Candidate Pairs"], "123")
        self.assertEqual(row["Collision Spatial Queries"], "45")


class FakeWorkerClient:
    def __init__(self, *, worker_id, record_id, fail_on_step=False):
        self.worker_id = int(worker_id)
        self.record_id = int(record_id)
        self.fail_on_step = bool(fail_on_step)
        self.started = False
        self.shutdown_called = False
        self.frame = 0
        self.frame_limit = 0
        self.round_index = 0
        self.opponent = OpponentSpec("Earthling")
        self.sent_commands = []
        self.results = []

    def start(self, *, base_seed, stop_requested=None):
        self.started = True
        self.base_seed = int(base_seed)

    def send(self, command):
        self.sent_commands.append(command)
        name = getattr(command, "name", "")
        if name == "START_WINDOW":
            self.frame = 0
            self.round_index = int(command.round_index)
            self.frame_limit = int(command.frame_limit)
            self.opponent = command.opponent
            self.results.append(
                WindowStartedResult(
                    record_id=self.record_id,
                    round_index=self.round_index,
                    frame_limit=self.frame_limit,
                    trainee_observation=pack_observation(
                        (0.0,) * OBSERVATION_INPUT_SIZE
                    ),
                    simple_opponent_controls={"forward": False},
                )
            )
        elif name == "REQUEST_OBSERVATION":
            raise AssertionError("steady-state observation request was issued")
        elif name == "STEP_FRAME":
            if self.fail_on_step:
                self.results.append(
                    WorkerErrorResult(
                        record_id=self.record_id,
                        command_name=name,
                        exception_type="RuntimeError",
                        exception_message="worker step failed",
                        traceback_text="traceback text",
                    )
                )
                return
            self.frame += 1
            sample = ReplaySample(
                observation=(0.0,) * OBSERVATION_INPUT_SIZE,
                action_index=int(command.trainee_action_index),
                return_value=1.0,
            )
            self.results.append(
                FrameSteppedResult(
                    record_id=self.record_id,
                    round_index=self.round_index,
                    frame_count=self.frame,
                    complete=self.frame >= self.frame_limit,
                    progress_payload={
                        "event": "frame",
                        "frame": self.frame,
                        "opponent": self.opponent,
                        "action_index": int(command.trainee_action_index),
                        "exploratory": bool(command.trainee_exploratory),
                        "weighted_total_return": 1.0,
                        "component_totals": {},
                    },
                    mature_samples=(sample,),
                    next_trainee_observation=(
                        pack_observation((0.0,) * OBSERVATION_INPUT_SIZE)
                        if self.frame < self.frame_limit
                        else None
                    ),
                    next_simple_opponent_controls=(
                        {"forward": False}
                        if self.frame < self.frame_limit
                        else None
                    ),
                )
            )
        elif name == "FINISH_WINDOW":
            episode = TrainingEpisodeResult(
                opponent=self.opponent,
                frames=self.frame,
                terminal_reason="timeout",
                mature_samples=self.frame,
                total_return=1.0,
                win=False,
                loss=False,
                draw=True,
                component_totals={},
            )
            result = CoordinatedFixedFrameWindowResult(
                opponent=self.opponent,
                frames=self.frame,
                mature_samples=self.frame,
                episode_results=(episode,),
                total_return=1.0,
                win=False,
                loss=False,
                draw=True,
                component_totals={},
            )
            self.results.append(
                WindowFinishedResult(
                    record_id=self.record_id,
                    round_index=self.round_index,
                    result=result,
                )
            )

    def recv(self, **_kwargs):
        return self.results.pop(0)

    def shutdown(self):
        self.shutdown_called = True


class CoordinatedWorkerBackedFrameLoopTests(unittest.TestCase):
    def test_process_worker_startup_wait_observes_stop_requests(self):
        process = mock.Mock()
        connection = mock.Mock()
        stop_requested = mock.Mock(return_value=False)
        client = _ProcessWorkerClient(
            worker_id=1,
            record_id=7,
            process_starter=mock.Mock(return_value=(process, connection)),
        )
        client.recv = mock.Mock(side_effect=TrainingBatchAborted("stop requested"))

        with self.assertRaises(TrainingBatchAborted):
            client.start(base_seed=123, stop_requested=stop_requested)

        client.recv.assert_called_once_with(stop_requested=stop_requested)

    def test_stop_during_worker_startup_aborts_without_fallback(self):
        clients = []
        session = None

        class StartupClient(FakeWorkerClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.abort_startup_called = False

            def start(self, *, base_seed, stop_requested=None):
                super().start(
                    base_seed=base_seed,
                    stop_requested=stop_requested,
                )
                session.request_stop()
                if stop_requested is not None and stop_requested():
                    raise TrainingBatchAborted("stop requested")

            def abort_startup(self):
                self.abort_startup_called = True

        def worker_factory(**kwargs):
            client = StartupClient(**kwargs)
            clients.append(client)
            return client

        session = CoordinatedTrainingSession(
            (
                _record(1, "Earthling"),
                _record(2, "Androsynth"),
            ),
            coordinated_cpu_workers_enabled=True,
            worker_client_factory=worker_factory,
        )

        with mock.patch.object(
            session,
            "_run_one_in_process_coordinated_batch",
            return_value=True,
        ) as in_process:
            self.assertFalse(session._run_one_coordinated_batch())

        self.assertEqual(len(clients), 1)
        self.assertTrue(clients[0].abort_startup_called)
        self.assertFalse(clients[0].shutdown_called)
        in_process.assert_not_called()
        self.assertTrue(session.coordinated_cpu_workers_enabled)
        self.assertIsNone(session.consume_notice())

    def test_worker_startup_failure_falls_back_and_posts_one_notice(self):
        session = CoordinatedTrainingSession(
            (
                _record(1, "Earthling"),
                _record(2, "Androsynth"),
            ),
            coordinated_cpu_workers_enabled=True,
        )

        with (
            mock.patch(
                "src.training.coordinated.simple_opponent_schedule",
                return_value=(OpponentSpec("Earthling"),),
            ),
            mock.patch.object(
                session,
                "_start_cpu_workers",
                side_effect=RuntimeError("startup failed"),
            ),
            mock.patch.object(
                session,
                "_run_one_in_process_coordinated_batch",
                return_value=True,
            ) as in_process,
        ):
            self.assertTrue(session._run_one_coordinated_batch())

        in_process.assert_called_once_with()
        self.assertFalse(session.coordinated_cpu_workers_enabled)
        self.assertEqual(session.consume_notice(), CPU_WORKER_FALLBACK_NOTICE)
        self.assertIsNone(session.consume_notice())

    def test_partial_worker_startup_failure_shuts_down_every_created_client(self):
        clients = []

        def worker_factory(**kwargs):
            client = FakeWorkerClient(**kwargs)
            if kwargs["record_id"] == 2:
                client.start = mock.Mock(side_effect=RuntimeError("startup failed"))
            clients.append(client)
            return client

        session = CoordinatedTrainingSession(
            (
                _record(1, "Earthling"),
                _record(2, "Androsynth"),
            ),
            worker_client_factory=worker_factory,
        )

        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            session._start_cpu_workers()

        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.shutdown_called for client in clients))

    def test_worker_backed_batches_reuse_processes_and_insert_replay(self):
        clients = []
        replay_buffers = {}

        def worker_factory(**kwargs):
            client = FakeWorkerClient(**kwargs)
            clients.append(client)
            return client

        def component_builder(record):
            replay = TrainingReplayBuffer(32)
            replay_buffers[record.instance_id] = replay
            return CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=replay,
            )

        session = CoordinatedTrainingSession(
            (
                _record(
                    1,
                    "Earthling",
                    batch_grouping=99,
                    match_time_limit=2,
                    epsilon=1.0,
                    gamma=0.0,
                ),
                _record(
                    2,
                    "Androsynth",
                    batch_grouping=99,
                    match_time_limit=2,
                    epsilon=1.0,
                    gamma=0.0,
                ),
            ),
            component_builder=component_builder,
            run_batches=True,
            coordinated_cpu_workers_enabled=True,
            worker_client_factory=worker_factory,
        )
        for state in session._states.values():
            state.components = component_builder(state.record)

        with (
            mock.patch(
                "src.training.coordinated.simple_opponent_schedule",
                return_value=(OpponentSpec("Earthling"),),
            ),
            mock.patch("src.training.coordinated.append_coordinated_batch_timing_csv"),
        ):
            self.assertTrue(session._run_one_coordinated_batch())
            self.assertTrue(session._run_one_coordinated_batch())

        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.started for client in clients))
        self.assertTrue(all(not client.shutdown_called for client in clients))
        self.assertEqual(len(replay_buffers[1]), 4)
        self.assertEqual(len(replay_buffers[2]), 4)
        self.assertEqual(session.status_for_instance(1).completed_batches, 2)
        self.assertEqual(session.status_for_instance(2).completed_batches, 2)
        self.assertEqual(session.status_for_instance(1).current_frame, 0)
        self.assertEqual(session.status_for_instance(2).current_frame, 0)
        stats = session.inference_stats
        self.assertEqual(stats.request_count, 8)
        self.assertEqual(stats.exploratory_count, 8)
        self.assertEqual(stats.mode_counts["exploration_only"], 4)
        self.assertFalse(
            any(
                command.name == COMMAND_REQUEST_OBSERVATION
                for client in clients
                for command in client.sent_commands
            )
        )

        session._shutdown_persistent_cpu_workers()

        self.assertTrue(all(client.shutdown_called for client in clients))

    def test_worker_error_aborts_batch_and_shuts_down_workers(self):
        clients = []

        def worker_factory(**kwargs):
            client = FakeWorkerClient(
                **kwargs,
                fail_on_step=kwargs["record_id"] == 1,
            )
            clients.append(client)
            return client

        session = CoordinatedTrainingSession(
            (
                _record(1, "Earthling", match_time_limit=1, epsilon=1.0),
                _record(2, "Androsynth", match_time_limit=1, epsilon=1.0),
            ),
            component_builder=lambda record: CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=TrainingReplayBuffer(8),
            ),
            coordinated_cpu_workers_enabled=True,
            worker_client_factory=worker_factory,
        )
        for state in session._states.values():
            state.components = CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=TrainingReplayBuffer(8),
            )

        with mock.patch(
            "src.training.coordinated.simple_opponent_schedule",
            return_value=(OpponentSpec("Earthling"),),
        ):
            with self.assertRaises(RuntimeError):
                session._run_one_coordinated_batch()

        self.assertTrue(all(client.shutdown_called for client in clients))
        self.assertEqual(session.status_for_instance(1).completed_batches, 0)
        self.assertEqual(session.status_for_instance(2).completed_batches, 0)


class CoordinatedFrameLoopTests(unittest.TestCase):
    def test_batch_advances_records_frame_by_frame(self):
        step_log = []
        replay_buffers = {}

        def component_builder(record):
            replay = TrainingReplayBuffer(32)
            replay_buffers[record.instance_id] = replay
            return CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=replay,
            )

        session = CoordinatedTrainingSession(
            (
                _record(
                    1,
                    "Earthling",
                    batch_grouping=99,
                    match_time_limit=3,
                    epsilon=1.0,
                    gamma=0.0,
                ),
                _record(
                    2,
                    "Androsynth",
                    batch_grouping=99,
                    match_time_limit=3,
                    epsilon=1.0,
                    gamma=0.0,
                ),
            ),
            component_builder=component_builder,
            simulation_factory=StepLoggingSimulationFactory((), step_log),
            run_batches=True,
        )
        for state in session._states.values():
            state.components = component_builder(state.record)

        with (
            mock.patch(
                "src.training.coordinated.simple_opponent_schedule",
                return_value=(OpponentSpec("Earthling"),),
            ),
            mock.patch(
                "src.training.coordinated.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.coordinated.decision_frame_from_battle_state",
                side_effect=fake_decision_frame,
            ),
            mock.patch(
                "src.training.coordinated.frame_outcome_from_battle_state",
                side_effect=fake_frame_outcome,
            ),
            mock.patch("src.training.coordinated.COORDINATED_TIMING_METRICS_ENABLED", True),
            mock.patch("src.training.coordinated.TRAINING_CSV_OUTPUT_ENABLED", True),
            mock.patch("src.training.coordinated.append_coordinated_batch_timing_csv") as timing_csv,
        ):
            self.assertTrue(session._run_one_coordinated_batch())

        self.assertEqual(
            step_log,
            [
                ("Earthling", 1),
                ("Androsynth", 1),
                ("Earthling", 2),
                ("Androsynth", 2),
                ("Earthling", 3),
                ("Androsynth", 3),
            ],
        )
        self.assertEqual(len(replay_buffers[1]), 3)
        self.assertEqual(len(replay_buffers[2]), 3)
        self.assertEqual(session.status_for_instance(1).completed_batches, 1)
        self.assertEqual(session.status_for_instance(2).completed_batches, 1)
        stats = session.inference_stats
        self.assertEqual(stats.last_mode, "exploration_only")
        self.assertEqual(stats.request_count, 6)
        self.assertEqual(stats.exploratory_count, 6)
        self.assertEqual(stats.mode_counts["exploration_only"], 3)
        timing = session.timing_stats
        self.assertEqual(timing.completed_batches, 1)
        self.assertEqual(timing.frame_count, 6)
        self.assertGreater(timing.observation_seconds, 0.0)
        self.assertGreater(timing.trainee_inference_seconds, 0.0)
        self.assertGreater(timing.opponent_inference_seconds, 0.0)
        self.assertGreater(timing.simulation_seconds, 0.0)
        self.assertGreater(timing.reward_seconds, 0.0)
        self.assertGreater(timing.optimization_seconds, 0.0)
        self.assertEqual(timing_csv.call_count, 2)
        first_timing_row = timing_csv.call_args_list[0].kwargs
        self.assertEqual(first_timing_row["batch_number"], 1)
        self.assertEqual(first_timing_row["instance_count"], 2)
        self.assertEqual(first_timing_row["rounds"], 1)
        self.assertEqual(first_timing_row["instance_frames"], 3)
        self.assertEqual(first_timing_row["coordinated_record_frames"], 6)
        self.assertEqual(first_timing_row["action_requests"], 6)
        self.assertEqual(first_timing_row["exploratory_actions"], 6)
        self.assertEqual(
            first_timing_row["inference_mode"],
            "exploration_only:3",
        )

    def test_stop_during_fallback_optimization_finishes_every_record(self):
        replay_buffers = {}

        def component_builder(record):
            replay = TrainingReplayBuffer(32)
            replay_buffers[record.instance_id] = replay
            return CoordinatedRuntimeComponents(
                model=object(),
                optimizer=object(),
                replay_buffer=replay,
            )

        session = CoordinatedTrainingSession(
            (
                _record(
                    1,
                    "Earthling",
                    batch_grouping=99,
                    match_time_limit=1,
                    epsilon=1.0,
                    gamma=0.0,
                    minibatch_size=1,
                    replay_updates_per_batch=2,
                ),
                _record(
                    2,
                    "Androsynth",
                    batch_grouping=99,
                    match_time_limit=1,
                    epsilon=1.0,
                    gamma=0.0,
                    minibatch_size=1,
                    replay_updates_per_batch=2,
                ),
            ),
            component_builder=component_builder,
            simulation_factory=StepLoggingSimulationFactory((), []),
            run_batches=True,
        )
        for state in session._states.values():
            state.components = component_builder(state.record)
        losses = iter((0.25, 0.75, 1.25, 1.75))
        optimize_calls = []

        def optimize(model, optimizer, replay_buffer, *, batch_size, rng=None):
            optimize_calls.append((model, optimizer, len(replay_buffer), batch_size))
            if len(optimize_calls) == 1:
                session.request_stop()
            return SimpleNamespace(loss=next(losses))

        with (
            mock.patch(
                "src.training.coordinated.simple_opponent_schedule",
                return_value=(OpponentSpec("Earthling"),),
            ),
            mock.patch(
                "src.training.coordinated.encode_observation",
                return_value=[0.0] * OBSERVATION_INPUT_SIZE,
            ),
            mock.patch(
                "src.training.coordinated.decision_frame_from_battle_state",
                side_effect=fake_decision_frame,
            ),
            mock.patch(
                "src.training.coordinated.frame_outcome_from_battle_state",
                side_effect=fake_frame_outcome,
            ),
            mock.patch("src.training.coordinated.optimize_from_replay", side_effect=optimize),
            mock.patch("src.training.coordinated.append_coordinated_batch_timing_csv"),
        ):
            self.assertTrue(session._run_one_coordinated_batch())

        self.assertEqual(len(optimize_calls), 4)
        self.assertTrue(session._stop_requested.is_set())
        self.assertEqual(session.status_for_instance(1).completed_batches, 1)
        self.assertEqual(session.status_for_instance(2).completed_batches, 1)
        self.assertEqual(session.status_for_instance(1).recent_loss, 0.5)
        self.assertEqual(session.status_for_instance(2).recent_loss, 1.5)
        self.assertEqual(
            session.status_for_instance(1).display_message,
            "Applying gradient descent",
        )

    def test_stop_during_batched_optimization_finishes_all_updates(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")

        def component_builder(_record):
            model = build_value_network(ValueNetworkConfig(8, 1))
            replay = TrainingReplayBuffer(4)
            replay.extend(
                (
                    ReplaySample(
                        observation=tuple([0.0] * OBSERVATION_INPUT_SIZE),
                        action_index=0,
                        return_value=1.0,
                    ),
                    ReplaySample(
                        observation=tuple([1.0] * OBSERVATION_INPUT_SIZE),
                        action_index=1,
                        return_value=0.5,
                    ),
                )
            )
            return CoordinatedRuntimeComponents(
                model=model,
                optimizer=build_optimizer(model, learning_rate=0.001),
                replay_buffer=replay,
            )

        session = CoordinatedTrainingSession(
            (
                _record(
                    1,
                    "Earthling",
                    minibatch_size=1,
                    replay_updates_per_batch=2,
                ),
                _record(
                    2,
                    "Androsynth",
                    minibatch_size=1,
                    replay_updates_per_batch=2,
                ),
            ),
            component_builder=component_builder,
            run_batches=True,
        )
        for state in session._states.values():
            state.components = component_builder(state.record)

        def train_batched_records(*args, **kwargs):
            session.request_stop()
            return (0.25, 0.75)

        with mock.patch(
            "src.training.coordinated.train_selected_action_regression_batched",
            side_effect=train_batched_records,
        ) as train_batched:
            losses = session._optimize_records()

        self.assertEqual(train_batched.call_count, 2)
        self.assertTrue(session._stop_requested.is_set())
        self.assertEqual(losses[1], (0.25, 0.25))
        self.assertEqual(losses[2], (0.75, 0.75))

    def test_grouped_save_writes_metadata_and_notifies_cache(self):
        with self.subTest("grouped save"):
            with mock.patch("src.training.coordinated.save_training_checkpoint") as save:
                with mock.patch("src.training.coordinated.append_grouped_metrics_csv"):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        repository = TrainingModelRepository(
                            root / "bundled",
                            root / "user",
                        )
                        metadata = metadata_from_state(
                            ship="Earthling",
                            slot=1,
                            description="Earthling test",
                            architecture=model_architecture_metadata(8, 1),
                            training={},
                        )
                        slot = repository.create_or_update_user_model(metadata)
                        record = CoordinatedTrainingRecord(
                            instance_id=1,
                            repository=repository,
                            slot=slot,
                            metadata=metadata,
                            config=TrainingOrchestrationConfig(
                                trainee_ship="Earthling",
                                hidden_layer_width=8,
                                hidden_layer_count=1,
                                training_device="cpu",
                                epsilon_decay=1.0,
                            ),
                            batch_grouping=1,
                        )
                        session = CoordinatedTrainingSession(
                            (record, _record(2, "Androsynth")),
                            run_batches=False,
                            opponent_model_cache=SimpleNamespace(
                                notify_model_saved=mock.Mock()
                            ),
                        )
                        state = session._states[1]
                        state.components = CoordinatedRuntimeComponents(
                            model=object(),
                            optimizer=object(),
                            replay_buffer=TrainingReplayBuffer(4),
                        )
                        result = SimpleNamespace(
                            replay_size=0,
                            average_loss=0.5,
                            round_results=(SimpleNamespace(
                                total_return=4.0,
                                win=True,
                                loss=False,
                                draw=False,
                                component_totals={},
                            ),),
                            optimization_losses=(0.5,),
                        )

                        batch_number = session._record_completed_batch(
                            state,
                            result,
                            batch_seconds=1.0,
                        )
                        session._save_state(state, include_replay=False)

                        saved_slot = repository.slot_for("Earthling", 1)

        self.assertEqual(batch_number, 1)
        self.assertEqual(saved_slot.metadata["progress"]["completed_batches"], 1)
        self.assertEqual(state.last_saved_completed_batches, 1)
        save.assert_called_once()
        session.opponent_model_cache.notify_model_saved.assert_called_once_with(
            repository,
            "Earthling",
            1,
            device_choice="cpu",
        )


if __name__ == "__main__":
    unittest.main()
