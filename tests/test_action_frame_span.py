"""Action frame spans, dueling value heads, and greedy evaluation batches."""

import copy
import csv
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.Battle.battle_ai import TrainedModelController
from src.Menus.train_ai import (
    TRAINING_LOG_EVALUATION_COLOR,
    TrainingUIState,
    _display_off_console_content,
    _maximum_batch_samples,
    _training_settings_match,
    _training_ui_state_from_json,
    architecture_for_state,
)
from src.training import torch_backend
from src.training.batched_value_network import (
    can_batch_value_networks,
    predict_action_values_batched,
    train_selected_action_regression_batched,
)
from src.training.contracts import (
    ACTION_INDEX_TABLE,
    ACTION_OUTPUT_SIZE,
    OBSERVATION_INPUT_SIZE,
    VALUE_HEAD_DUELING,
    VALUE_HEAD_STANDARD,
    TrainingAction,
)
from src.training.coordinated import (
    CoordinatedActionRequest,
    CoordinatedExplorationSchedule,
    CoordinatedRuntimeComponents,
    CoordinatedTrainingRecord,
    CoordinatedTrainingSession,
    CoordinatedValueNetworkPolicy,
    select_actions_for_records,
)
from src.training.coordinated_contracts import (
    CoordinatedFixedFrameWindowResult,
    TrainingEpisodeResult,
)
from src.training.model_loader import load_inference_model
from src.training.model_registry import (
    SLOT_USER,
    TrainingModelRepository,
    TrainingModelSlot,
    metadata_from_state,
    model_action_frame_span,
    model_architecture_metadata,
    normalize_architecture_metadata,
)
from src.training.observation_transfer import pack_observation
from src.training.orchestration import (
    OpponentSpec,
    TrainingBatchResult,
    TrainingOrchestrationConfig,
    TrainingRoundResult,
    ValueNetworkPolicy,
    run_training_round,
)
from src.training.replay import TrainingReplayBuffer, save_training_checkpoint
from src.training.replay_contracts import ActionHold, ActionSelection
from src.training.rewards import (
    REWARD_HIGH_SPEED,
    RewardDecisionFrame,
    RewardFrameOutcome,
    StagedTrajectoryPipeline,
)
from src.training.session import (
    EVALUATION_SEED,
    BatchMetrics,
    TrainingSession,
    evaluation_metrics_from_batch_result,
    format_batch_summary_line,
    format_evaluation_summary_line,
    is_evaluation_batch,
    is_evaluation_log_line,
    training_metrics_row_for_evaluation_batch,
)
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
    is_dueling_value_network,
    train_selected_action_regression,
)
from src.training.worker_protocol import (
    FrameSteppedResult,
    WindowFinishedResult,
    WindowStartedResult,
)

ZERO_OBSERVATION = (0.0,) * OBSERVATION_INPUT_SIZE


def _requires_torch(test):
    if torch_backend.get_torch() is None:
        test.skipTest("PyTorch is not installed")
    return torch_backend.require_torch()


class ActionHoldTests(unittest.TestCase):
    def test_hold_repeats_action_for_remaining_span_frames(self):
        hold = ActionHold()
        hold.begin(5, 3)

        self.assertEqual(
            [hold.held_action(), hold.held_action(), hold.held_action()],
            [5, 5, None],
        )

    def test_hold_is_invalidated_by_key_change_and_reset(self):
        hold = ActionHold()
        hold.begin(2, 4, key="model-a")

        self.assertIsNone(hold.held_action("model-b"))
        self.assertEqual(hold.held_action("model-a"), 2)
        hold.reset()
        self.assertIsNone(hold.held_action("model-a"))

    def test_span_of_one_never_holds(self):
        hold = ActionHold()
        hold.begin(7, 1)

        self.assertIsNone(hold.held_action())


class DecisionFrameSampleTests(unittest.TestCase):
    def _decision(self, frame_id):
        return RewardDecisionFrame(frame_id, (float(frame_id),), 3)

    def _speed_outcome(self, frame_id, *, terminal=False):
        return RewardFrameOutcome(
            frame_id,
            self_speed=2.0,
            self_max_thrust=1.0,
            terminal=terminal,
        )

    def test_held_frames_add_rewards_but_only_decisions_become_samples(self):
        pipeline = StagedTrajectoryPipeline(
            gamma=0.5,
            reward_weights={REWARD_HIGH_SPEED: 24.0},
        )
        emitted = []
        for frame_id, decision in zip(range(1, 5), (True, False, False, True)):
            index = pipeline.stage_decision(
                self._decision(frame_id),
                emit_sample=decision,
            )
            emitted.extend(
                pipeline.add_frame(
                    self._decision(frame_id),
                    self._speed_outcome(frame_id, terminal=frame_id == 4),
                    staged_index=index,
                )
            )

        # Each frame earns 24 * (2 - 1) / 1 / 24 = 1.0 high-speed reward.
        self.assertEqual([sample.start_frame_id for sample in emitted], [1, 4])
        self.assertAlmostEqual(emitted[0].return_value, 1 + 0.5 + 0.25 + 0.125)
        self.assertAlmostEqual(emitted[1].return_value, 1.0)

    def test_first_frame_of_trajectory_is_always_sampled(self):
        pipeline = StagedTrajectoryPipeline(
            gamma=0.5,
            reward_weights={REWARD_HIGH_SPEED: 24.0},
        )
        index = pipeline.stage_decision(self._decision(1), emit_sample=False)
        samples = pipeline.add_frame(
            self._decision(1),
            self._speed_outcome(1, terminal=True),
            staged_index=index,
        )

        self.assertEqual([sample.start_frame_id for sample in samples], [1])

    def test_real_round_records_one_sample_per_action_decision(self):
        replay = TrainingReplayBuffer(capacity=64)
        policy = ValueNetworkPolicy(
            None,
            epsilon=1.0,
            action_frame_span=4,
            rng=random.Random(3),
        )

        result = run_training_round(
            opponent=OpponentSpec("Earthling"),
            trainee_policy=policy,
            replay_buffer=replay,
            config=TrainingOrchestrationConfig(
                trainee_ship="Earthling",
                match_time_limit=12,
                action_frame_span=4,
            ),
            rng=random.Random(5),
        )

        self.assertEqual(result.frames, 12)
        self.assertEqual(len(replay), 3)
        self.assertEqual(sum(result.action_counts), 3)


class CoordinatedActionHoldTests(unittest.TestCase):
    def _requests(self, schedule, models):
        policies = [
            CoordinatedValueNetworkPolicy(
                model,
                record_id=record_id,
                exploration_schedule=schedule,
            )
            for record_id, model in enumerate(models, start=1)
        ]
        requests = tuple(
            CoordinatedActionRequest(
                record_id=policy.record_id,
                policy=policy,
                observation=ZERO_OBSERVATION,
            )
            for policy in policies
        )
        return policies, requests

    def test_greedy_actions_are_held_without_inference(self):
        _requires_torch(self)
        schedule = CoordinatedExplorationSchedule(
            epsilon=0.0,
            frame_span=3,
            rng=random.Random(0),
        )
        models = (
            build_value_network(ValueNetworkConfig(8, 1)),
            build_value_network(ValueNetworkConfig(8, 1)),
        )
        _policies, requests = self._requests(schedule, models)

        results = [select_actions_for_records(requests) for _ in range(4)]

        self.assertEqual(
            [result.inference_mode for result in results],
            ["batched_value_network", "held_only", "held_only", "batched_value_network"],
        )
        for record_id in (1, 2):
            first = results[0].selections[record_id]
            self.assertFalse(first.held)
            for held in (results[1], results[2]):
                self.assertTrue(held.selections[record_id].held)
                self.assertEqual(
                    held.selections[record_id].action_index,
                    first.action_index,
                )

    def test_reset_record_redecides_one_record_while_other_stays_held(self):
        _requires_torch(self)
        schedule = CoordinatedExplorationSchedule(
            epsilon=0.0,
            frame_span=4,
            rng=random.Random(0),
        )
        models = (
            build_value_network(ValueNetworkConfig(8, 1)),
            build_value_network(ValueNetworkConfig(8, 1)),
        )
        policies, requests = self._requests(schedule, models)
        select_actions_for_records(requests)

        policies[0].reset_exploration_span()
        result = select_actions_for_records(requests)

        self.assertEqual(result.inference_mode, "batched_value_network")
        self.assertFalse(result.selections[1].held)
        self.assertTrue(result.selections[2].held)

    def test_evaluation_records_never_explore(self):
        schedule = CoordinatedExplorationSchedule(
            epsilon=1.0,
            frame_span=2,
            rng=random.Random(0),
            greedy_record_ids=(2,),
        )
        requests = tuple(
            CoordinatedActionRequest(
                record_id=record_id,
                policy=SimpleNamespace(),
                observation=ZERO_OBSERVATION,
            )
            for record_id in (1, 2)
        )

        selections, greedy = schedule.prepare_selections(requests)

        self.assertTrue(selections[1].exploratory)
        self.assertEqual([request.record_id for request in greedy], [2])


class DuelingValueNetworkTests(unittest.TestCase):
    def setUp(self):
        self.torch = _requires_torch(self)
        self.torch.manual_seed(7)

    def test_new_networks_default_to_dueling_with_action_shaped_output(self):
        model = build_value_network(ValueNetworkConfig(16, 2))
        values = model(self.torch.zeros((3, OBSERVATION_INPUT_SIZE)))

        self.assertTrue(is_dueling_value_network(model))
        self.assertEqual(tuple(values.shape), (3, ACTION_OUTPUT_SIZE))

    def test_every_sample_trains_the_shared_state_value(self):
        model = build_value_network(ValueNetworkConfig(8, 1))
        optimizer = self.torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.value_head.bias.detach().clone()

        train_selected_action_regression(
            model,
            optimizer,
            self.torch.zeros((1, OBSERVATION_INPUT_SIZE)),
            self.torch.tensor([6]),
            self.torch.tensor([5.0]),
        )

        self.assertFalse(self.torch.equal(before, model.value_head.bias))

    def test_batched_inference_and_training_match_scalar_dueling_models(self):
        models = [build_value_network(ValueNetworkConfig(8, 2)) for _ in range(2)]
        scalar_models = [copy.deepcopy(model) for model in models]
        observations = [
            [random.Random(seed).uniform(-1, 1) for _ in range(OBSERVATION_INPUT_SIZE)]
            for seed in (1, 2)
        ]

        batched = predict_action_values_batched(models, observations)
        for row, model, observation in zip(batched, scalar_models, observations):
            expected = model(self.torch.tensor([observation]))[0]
            self.assertTrue(self.torch.allclose(row, expected, atol=1e-5))

        optimizers = [self.torch.optim.SGD(m.parameters(), lr=0.05) for m in models]
        scalar_optimizers = [
            self.torch.optim.SGD(m.parameters(), lr=0.05) for m in scalar_models
        ]
        train_selected_action_regression_batched(
            models,
            optimizers,
            [[observation] for observation in observations],
            [[3], [9]],
            [[1.5], [-0.5]],
        )
        for model, optimizer, observation, action, target in zip(
            scalar_models,
            scalar_optimizers,
            observations,
            (3, 9),
            (1.5, -0.5),
        ):
            train_selected_action_regression(
                model,
                optimizer,
                self.torch.tensor([observation]),
                self.torch.tensor([action]),
                self.torch.tensor([target]),
            )
        for batched_model, scalar_model in zip(models, scalar_models):
            for left, right in zip(
                batched_model.parameters(),
                scalar_model.parameters(),
            ):
                self.assertTrue(self.torch.allclose(left, right, atol=1e-5))

    def test_mixed_heads_do_not_share_the_batched_path(self):
        models = (
            build_value_network(ValueNetworkConfig(8, 1, head=VALUE_HEAD_DUELING)),
            build_value_network(ValueNetworkConfig(8, 1, head=VALUE_HEAD_STANDARD)),
        )

        self.assertFalse(can_batch_value_networks(models))


class ModelMetadataCompatibilityTests(unittest.TestCase):
    def test_architecture_without_head_is_a_standard_model(self):
        normalized = normalize_architecture_metadata(
            {"hidden_layer_width": 8, "hidden_layer_count": 1, "loss": "huber"}
        )

        self.assertEqual(normalized["head"], VALUE_HEAD_STANDARD)
        self.assertEqual(normalized["loss"], "mse")
        self.assertEqual(model_architecture_metadata(8, 1)["head"], VALUE_HEAD_DUELING)

    def test_legacy_epsilon_span_does_not_set_inference_span(self):
        self.assertEqual(
            model_action_frame_span(
                {"training": {"regimen": {"epsilon_frame_span": 8}}}
            ),
            1,
        )
        self.assertEqual(
            model_action_frame_span(
                {"training": {"regimen": {"action_frame_span": 4}}}
            ),
            4,
        )

    def test_loader_rebuilds_legacy_standard_model_and_reports_span(self):
        _requires_torch(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Legacy",
                architecture={"hidden_layer_width": 8, "hidden_layer_count": 1},
                training={"regimen": {"action_frame_span": 3}},
            )
            slot = repository.create_or_update_user_model(metadata)
            standard = build_value_network(
                ValueNetworkConfig(8, 1, head=VALUE_HEAD_STANDARD)
            )
            save_training_checkpoint(slot.pth_path, standard)

            loaded = load_inference_model(slot, device_choice="cpu")

        self.assertFalse(is_dueling_value_network(loaded.model))
        self.assertEqual(loaded.action_frame_span, 3)


def _round_result(score, action_counts):
    return TrainingRoundResult(
        opponent=OpponentSpec("Earthling"),
        frames=10,
        terminal_reason="timeout",
        mature_samples=1,
        total_return=score,
        win=False,
        loss=False,
        draw=True,
        action_counts=tuple(action_counts),
    )


class EvaluationHelperTests(unittest.TestCase):
    def test_last_batch_of_each_group_is_evaluated(self):
        self.assertTrue(is_evaluation_batch(5, 5))
        self.assertTrue(is_evaluation_batch(10, 5))
        self.assertFalse(is_evaluation_batch(4, 5))
        self.assertFalse(is_evaluation_batch(3, 1))

    def test_evaluation_metrics_rank_decided_actions(self):
        counts = [0] * ACTION_OUTPUT_SIZE
        counts[7] = 6
        counts[1] = 3
        counts[0] = 1
        result = TrainingBatchResult(
            completed_rounds=1,
            replay_size=0,
            optimization_losses=(),
            round_results=(_round_result(0.25, counts),),
        )

        metrics = evaluation_metrics_from_batch_result(result, batch=10)
        line = format_evaluation_summary_line(metrics)

        self.assertEqual([index for index, _share in metrics.action_shares], [7, 1, 0])
        self.assertAlmostEqual(metrics.action_shares[0][1], 0.6)
        self.assertTrue(is_evaluation_log_line(line))
        self.assertIn("[↑__1_]  60%", line)

    def test_evaluation_line_aligns_with_batch_summary_columns(self):
        counts = [0] * ACTION_OUTPUT_SIZE
        counts[ACTION_INDEX_TABLE.index(TrainingAction.from_mask(8))] = 35
        counts[ACTION_INDEX_TABLE.index(TrainingAction.from_mask(17))] = 32
        counts[ACTION_INDEX_TABLE.index(TrainingAction.from_mask(16))] = 9
        counts[ACTION_INDEX_TABLE.index(TrainingAction.from_mask(27))] = 24
        result = TrainingBatchResult(
            completed_rounds=1,
            replay_size=0,
            optimization_losses=(),
            round_results=(_round_result(0.028, counts),),
        )
        metrics = replace(
            evaluation_metrics_from_batch_result(result, batch=200, batch_seconds=233.0),
            kills=19,
            deaths=11,
        )
        training = BatchMetrics(198, 11, 20, 1, 0.0, 0.1, 0.0003, 0.0359, 244.0)
        rolling = BatchMetrics(198, 111, 229, 10, -0.009, 0.1, 0.0003, 0.0292)

        evaluation_line = format_evaluation_summary_line(metrics)
        training_line = format_batch_summary_line(training, rolling)

        self.assertEqual(
            evaluation_line,
            "E      200 |   19K              11D KD:  1.73 | Score:   0.028 ||   "
            "Actions: [___1_]  35% | [↑___2]  32% | [↑←_12]  24%"
            "            ||   0h:03m:53s",
        )
        self.assertEqual(
            training_line,
            "#      198 |   11K (  11.10),   20D (  22.90) | Score:   0.000 ( -0.009) | "
            "Epsilon: 0.10000 | LR: 0.00030 | Loss:  0.0359 ( 0.0292) |   0h:04m:04s",
        )
        self.assertEqual(len(evaluation_line), len(training_line))
        for marker in ("|", "K", "D", "Score:"):
            self.assertEqual(evaluation_line.index(marker), training_line.index(marker))

    def test_training_row_averages_only_the_current_group(self):
        history = tuple(
            BatchMetrics(batch, batch, 0, 1, float(batch), 0.1, 0.001, 0.1)
            for batch in (4, 6, 7, 8, 9)
        )

        latest, rolling = training_metrics_row_for_evaluation_batch(
            history,
            batch_number=10,
            batch_grouping=5,
        )

        self.assertEqual(latest.batch, 10)
        self.assertEqual(latest.kills, 9)
        self.assertEqual(rolling.batch_count, 4)
        self.assertAlmostEqual(rolling.average_match_score, 7.5)


class TrainingSessionEvaluationTests(unittest.TestCase):
    def test_group_end_batch_is_greedy_simple_and_logged_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Eval",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {"rounds_per_batch": 1}},
            )
            slot = repository.create_or_update_user_model(metadata)
            seen = []
            counts = [0] * ACTION_OUTPUT_SIZE
            counts[6] = 4

            def batch_runner(**kwargs):
                seen.append(kwargs)
                return TrainingBatchResult(
                    completed_rounds=1,
                    replay_size=0,
                    optimization_losses=(0.2,),
                    round_results=(_round_result(1.0, counts),),
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    opponent_mode="all",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    starting_epsilon=0.5,
                    epsilon=0.5,
                    epsilon_floor=0.01,
                    epsilon_decay=0.5,
                ),
                batch_grouping=2,
                batch_runner=batch_runner,
            )
            with mock.patch.object(
                session,
                "_existing_ai_opponents_for_batch",
                return_value=(),
            ):
                session.run_synchronously(max_batches=2)

            with (root / "user" / "Earthling-01-eval.csv").open(
                newline="",
                encoding="utf-8",
            ) as file:
                evaluation_rows = list(csv.reader(file))
            with (root / "user" / "Earthling-01.csv").open(
                newline="",
                encoding="utf-8",
            ) as file:
                training_rows = list(csv.reader(file))

        training_config = seen[0]["config"]
        evaluation_config = seen[1]["config"]
        self.assertEqual(training_config.epsilon, 0.5)
        self.assertEqual(evaluation_config.epsilon, 0.0)
        self.assertEqual(evaluation_config.opponent_mode, "simple")
        self.assertEqual(
            seen[1]["rng"].random(),
            random.Random(EVALUATION_SEED).random(),
        )
        # Only the training batch decays epsilon.
        self.assertAlmostEqual(session.status.current_epsilon, 0.25)
        self.assertFalse(is_evaluation_log_line(session.log_lines[0]))
        self.assertTrue(is_evaluation_log_line(session.log_lines[1]))
        self.assertEqual(evaluation_rows[1][0], "2")
        self.assertEqual(evaluation_rows[1][5], "a1")
        self.assertEqual(training_rows[-1][0], "2")
        self.assertEqual(len(session.history), 1)


class CoordinatedEvaluationBatchTests(unittest.TestCase):
    def test_worker_backed_evaluation_batch_holds_greedy_actions(self):
        torch = _requires_torch(self)
        clients = []

        class SpanWorkerClient:
            def __init__(self, *, worker_id, record_id):
                self.record_id = int(record_id)
                self.frame = 0
                self.frame_limit = 0
                self.round_index = 0
                self.opponent = OpponentSpec("Earthling")
                self.sent = []
                self.results = []
                clients.append(self)

            def start(self, *, base_seed, stop_requested=None):
                self.base_seed = base_seed

            def send(self, command):
                self.sent.append(command)
                if command.name == "START_WINDOW":
                    self.frame = 0
                    self.round_index = int(command.round_index)
                    self.frame_limit = int(command.frame_limit)
                    self.opponent = command.opponent
                    self.results.append(
                        WindowStartedResult(
                            record_id=self.record_id,
                            round_index=self.round_index,
                            frame_limit=self.frame_limit,
                            trainee_observation=pack_observation(ZERO_OBSERVATION),
                            simple_opponent_controls={"forward": False},
                        )
                    )
                elif command.name == "STEP_FRAME":
                    self.frame += 1
                    self.results.append(
                        FrameSteppedResult(
                            record_id=self.record_id,
                            round_index=self.round_index,
                            frame_count=self.frame,
                            complete=self.frame >= self.frame_limit,
                            next_trainee_observation=(
                                pack_observation(ZERO_OBSERVATION)
                                if self.frame < self.frame_limit
                                else None
                            ),
                        )
                    )
                elif command.name == "FINISH_WINDOW":
                    episode = TrainingEpisodeResult(
                        opponent=self.opponent,
                        frames=self.frame,
                        terminal_reason="timeout",
                        mature_samples=0,
                        total_return=0.0,
                        win=False,
                        loss=False,
                        draw=True,
                    )
                    self.results.append(
                        WindowFinishedResult(
                            record_id=self.record_id,
                            round_index=self.round_index,
                            result=CoordinatedFixedFrameWindowResult(
                                opponent=self.opponent,
                                frames=self.frame,
                                mature_samples=0,
                                episode_results=(episode,),
                                total_return=0.0,
                                win=False,
                                loss=False,
                                draw=True,
                                action_counts=(1,) + (0,) * (ACTION_OUTPUT_SIZE - 1),
                            ),
                        )
                    )

            def recv(self, **_kwargs):
                return self.results.pop(0)

            def shutdown(self):
                pass

        def record(instance_id, ship):
            metadata = metadata_from_state(
                ship=ship,
                slot=1,
                description=ship,
                architecture=model_architecture_metadata(8, 1),
                training={},
            )
            return CoordinatedTrainingRecord(
                instance_id=instance_id,
                repository=TrainingModelRepository(Path("unused"), Path("unused")),
                slot=TrainingModelSlot(ship, 1, SLOT_USER, metadata=metadata),
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship=ship,
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    training_device="cpu",
                    match_time_limit=4,
                    action_frame_span=2,
                    starting_epsilon=1.0,
                    epsilon=1.0,
                    epsilon_floor=0.0,
                    epsilon_decay=0.5,
                ),
                batch_grouping=2,
            )

        def component_builder(record_):
            model = build_value_network(ValueNetworkConfig(8, 1))
            return CoordinatedRuntimeComponents(
                model=model,
                optimizer=build_optimizer(model, learning_rate=0.001),
                replay_buffer=TrainingReplayBuffer(32),
            )

        session = CoordinatedTrainingSession(
            (record(1, "Earthling"), record(2, "Androsynth")),
            component_builder=component_builder,
            run_batches=True,
            coordinated_cpu_workers_enabled=True,
            worker_client_factory=lambda **kwargs: SpanWorkerClient(
                worker_id=kwargs["worker_id"],
                record_id=kwargs["record_id"],
            ),
            reconcile_metrics_csv=False,
        )
        for state in session._states.values():
            state.components = component_builder(state.record)

        with (
            mock.patch(
                "src.training.coordinated.simple_opponent_schedule",
                return_value=(OpponentSpec("Earthling"),),
            ),
            mock.patch("src.training.coordinated.append_coordinated_batch_timing_csv"),
            mock.patch("src.training.coordinated.append_grouped_metrics_csv"),
            mock.patch("src.training.coordinated.append_evaluation_metrics_csv"),
            mock.patch.object(session, "_save_state"),
        ):
            self.assertTrue(session._run_one_coordinated_batch())
            self.assertTrue(session._run_one_coordinated_batch())
        session._shutdown_persistent_cpu_workers()
        del torch

        for client in clients:
            windows = [
                command for command in client.sent if command.name == "START_WINDOW"
            ]
            steps = [command for command in client.sent if command.name == "STEP_FRAME"]
            training_steps, evaluation_steps = steps[:4], steps[4:]
            self.assertEqual(
                [step.trainee_held for step in training_steps],
                [False, True, False, True],
            )
            self.assertTrue(all(step.trainee_exploratory for step in training_steps))
            self.assertEqual(
                [step.trainee_held for step in evaluation_steps],
                [False, True, False, True],
            )
            self.assertFalse(any(step.trainee_exploratory for step in evaluation_steps))
            self.assertEqual(windows[1].rng_seed, EVALUATION_SEED + 1)
        self.assertAlmostEqual(session._shared_current_epsilon, 0.5)
        for instance_id in (1, 2):
            lines = session.log_lines_for_instance(instance_id)
            self.assertEqual(len(lines), 2)
            self.assertTrue(is_evaluation_log_line(lines[1]))
            self.assertIn("[_____] 100%", lines[1])


class TrainingUISpanTests(unittest.TestCase):
    def test_saved_sessions_map_legacy_epsilon_frame_span(self):
        state = _training_ui_state_from_json({"epsilon_frame_span": 6})

        self.assertEqual(state.action_frame_span, 6)
        self.assertTrue(
            _training_settings_match(
                {"regimen": {"epsilon_frame_span": 6}},
                {"regimen": {"action_frame_span": 6}},
            )
        )

    def test_samples_per_batch_count_decisions(self):
        state = TrainingUIState(
            rounds_per_batch=1,
            match_time_limit=3600,
            action_frame_span=4,
        )

        self.assertEqual(_maximum_batch_samples(state), 3600 * 25 // 4)

    def test_existing_models_keep_their_head_until_reset(self):
        state = TrainingUIState(hidden_layer_size=8, hidden_layer_count=1)
        slot = TrainingModelSlot(
            "Earthling",
            1,
            SLOT_USER,
            metadata={"architecture": {"hidden_layer_width": 8, "hidden_layer_count": 1}},
        )

        with mock.patch(
            "src.Menus.train_ai.model_slot_has_checkpoint",
            return_value=True,
        ):
            kept = architecture_for_state(state, slot)
            reset = architecture_for_state(state, slot, reset_checkpoint=True)
        with mock.patch(
            "src.Menus.train_ai.model_slot_has_checkpoint",
            return_value=False,
        ):
            empty = architecture_for_state(state, slot)

        self.assertEqual(kept["head"], VALUE_HEAD_STANDARD)
        self.assertEqual(reset["head"], VALUE_HEAD_DUELING)
        self.assertEqual(empty["head"], VALUE_HEAD_DUELING)

    def test_evaluation_log_lines_use_their_own_colour(self):
        lines, colors = _display_off_console_content(
            None,
            ("# 1 training", "E        2 greedy", "# 3 training", "E        4 greedy"),
        )

        evaluation_index = lines.index("E        2 greedy")
        self.assertEqual(colors[evaluation_index], TRAINING_LOG_EVALUATION_COLOR)
        self.assertNotEqual(colors[lines.index("# 1 training")], TRAINING_LOG_EVALUATION_COLOR)
        # Evaluation rows get a blank line on each side, except at the end.
        self.assertEqual(
            list(lines[evaluation_index - 2 : evaluation_index + 5]),
            ["# 1 training", "", "E        2 greedy", "", "# 3 training", "", "E        4 greedy"],
        )
        self.assertTrue(is_evaluation_log_line("EVAL # 2 legacy"))


class BattleModeSpanTests(unittest.TestCase):
    def test_trained_controller_holds_each_inference_for_its_span(self):
        ship = SimpleNamespace()
        simulation = SimpleNamespace(player1=ship, player2=SimpleNamespace(), frame_id=0)
        controller = TrainedModelController(1, object(), action_frame_span=3)
        decisions = iter(ActionSelection(index, exploratory=False) for index in (1, 6))

        with (
            mock.patch(
                "src.Battle.battle_ai.encode_observation",
                return_value=ZERO_OBSERVATION,
            ),
            mock.patch(
                "src.training.replay.select_action_epsilon_greedy",
                side_effect=lambda *args, **kwargs: next(decisions),
            ) as select_action,
        ):
            controls = [controller.actions_for_frame(simulation) for _ in range(4)]

        self.assertEqual(select_action.call_count, 2)
        self.assertEqual([control["forward"] for control in controls], [True] * 3 + [False])
        self.assertEqual([control["action1"] for control in controls], [False] * 3 + [True])


if __name__ == "__main__":
    unittest.main()
