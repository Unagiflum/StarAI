import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.training.contracts import OBSERVATION_INPUT_SIZE
from src.training.coordinated import (
    CoordinatedRuntimeComponents,
    CoordinatedTrainingRecord,
    CoordinatedTrainingSession,
    run_coordinated_fixed_frame_window,
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
from src.training.replay import ActionSelection, TrainingReplayBuffer


def _record(instance_id, ship, **config_overrides):
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
        batch_grouping=1,
    )


class CoordinatedTrainingSessionTests(unittest.TestCase):
    def _session(self, *, component_builder=None):
        return CoordinatedTrainingSession(
            (_record(1, "Earthling"), _record(2, "Androsynth")),
            component_builder=component_builder or self._component_builder([]),
            run_batches=False,
            idle_sleep_seconds=0.001,
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


def fake_ship(name, _player_id, *, audio_service=None):
    return SimpleNamespace(
        name=name,
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        rotation=0.0,
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
    def run_window(self, *, config, policy=None, simulation_factory=None):
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
                _record(1, "Earthling", match_time_limit=3, epsilon=1.0, gamma=0.0),
                _record(2, "Androsynth", match_time_limit=3, epsilon=1.0, gamma=0.0),
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


if __name__ == "__main__":
    unittest.main()
