import random
import tempfile
import unittest
from pathlib import Path

from src.training import torch_backend
from src.training.contracts import ACTION_OUTPUT_SIZE, OBSERVATION_INPUT_SIZE
from src.training.replay import (
    PACKED_REPLAY_SAMPLE_BYTES,
    ReplaySample,
    TrainingReplayBuffer,
    load_training_checkpoint,
    optimize_from_replay,
    replay_checkpoint_path,
    save_training_checkpoint,
    select_action_epsilon_greedy,
)
from src.training.replay_contracts import ReplayTransferChunk
from src.training.value_network import (
    ValueNetworkConfig,
    build_optimizer,
    build_value_network,
    predict_action_values,
    predict_action_values_read_only,
    selected_action_regression_loss,
)


def sample(identifier, *, action_index=0, return_value=0.0):
    observation = [0.0] * OBSERVATION_INPUT_SIZE
    observation[0] = float(identifier)
    return ReplaySample(
        observation=tuple(observation),
        action_index=action_index,
        return_value=return_value,
    )


class TrainingReplayBufferTests(unittest.TestCase):
    def test_contiguous_transfer_chunk_matches_scalar_insertion_across_wrap(self):
        expected = TrainingReplayBuffer(capacity=5)
        actual = TrainingReplayBuffer(capacity=5)
        initial = [sample(index, action_index=index, return_value=index / 10) for index in range(3)]
        incoming = [
            sample(index, action_index=index, return_value=index / 10)
            for index in range(3, 8)
        ]
        expected.extend(initial)
        actual.extend(initial)
        expected.extend(incoming)
        actual.extend(ReplayTransferChunk.from_mature_samples(incoming))

        self.assertEqual(actual.samples, expected.samples)

    def test_contiguous_transfer_chunk_larger_than_capacity_keeps_suffix(self):
        replay = TrainingReplayBuffer(capacity=3)
        replay.add(sample(99))
        incoming = [
            sample(index, action_index=index, return_value=index)
            for index in range(6)
        ]

        replay.extend(ReplayTransferChunk.from_samples(incoming))

        self.assertEqual(
            [(entry.observation[0], entry.action_index, entry.return_value) for entry in replay],
            [(3.0, 3, 3.0), (4.0, 4, 4.0), (5.0, 5, 5.0)],
        )

    def test_replay_eviction_is_fifo_and_capacity_bound(self):
        replay = TrainingReplayBuffer(capacity=3)

        for identifier in range(5):
            replay.add(sample(identifier))

        self.assertEqual(len(replay), 3)
        self.assertEqual([entry.observation[0] for entry in replay.samples], [2.0, 3.0, 4.0])

    def test_replay_uses_packed_arrays_and_samples_them_directly(self):
        replay = TrainingReplayBuffer(capacity=3)
        replay.extend(sample(identifier, action_index=identifier) for identifier in range(5))

        observations, actions, returns = replay.sample_minibatch_arrays(
            2,
            rng=random.Random(4),
        )

        self.assertEqual(observations.shape, (2, OBSERVATION_INPUT_SIZE))
        self.assertEqual(observations.dtype.name, "float32")
        self.assertEqual(actions.dtype.name, "uint8")
        self.assertEqual(returns.dtype.name, "float32")
        self.assertEqual(replay.storage_bytes, 3 * PACKED_REPLAY_SAMPLE_BYTES)
        self.assertEqual([entry.observation[0] for entry in replay], [2.0, 3.0, 4.0])

    def test_seeded_minibatch_sampling_is_deterministic(self):
        replay = TrainingReplayBuffer(capacity=10)
        replay.extend(sample(identifier) for identifier in range(6))

        first = replay.sample_minibatch(3, rng=random.Random(12))
        second = replay.sample_minibatch(3, rng=random.Random(12))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_replay_state_round_trips(self):
        replay = TrainingReplayBuffer(capacity=2)
        replay.extend(sample(identifier, return_value=identifier) for identifier in range(4))
        restored = TrainingReplayBuffer(capacity=9)

        restored.load_state(replay.to_state())

        self.assertEqual(restored.capacity, 2)
        self.assertEqual([entry.return_value for entry in restored.samples], [2.0, 3.0])


class TrainingPolicyAndOptimizationTests(unittest.TestCase):
    def setUp(self):
        if torch_backend.get_torch() is None:
            self.skipTest("PyTorch is not installed")
        self.torch = torch_backend.require_torch()
        self.torch.manual_seed(1234)

    def test_epsilon_zero_selects_greedy_action(self):
        model = build_value_network(ValueNetworkConfig(8, 1))
        with self.torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model[-1].bias.copy_(self.torch.arange(ACTION_OUTPUT_SIZE, dtype=self.torch.float32))

        selection = select_action_epsilon_greedy(
            model,
            [0.0] * OBSERVATION_INPUT_SIZE,
            epsilon=0.0,
            rng=random.Random(1),
        )

        self.assertFalse(selection.exploratory)
        self.assertEqual(selection.action_index, ACTION_OUTPUT_SIZE - 1)
        self.assertEqual(len(selection.action_values), ACTION_OUTPUT_SIZE)

    def test_read_only_prediction_does_not_change_model_mode(self):
        model = build_value_network(ValueNetworkConfig(8, 1))
        model.train()

        values = predict_action_values_read_only(
            model,
            [[0.0] * OBSERVATION_INPUT_SIZE],
        )

        self.assertTrue(model.training)
        self.assertEqual(tuple(values.shape), (1, ACTION_OUTPUT_SIZE))

    def test_epsilon_one_uses_controlled_exploration_rng(self):
        model = build_value_network(ValueNetworkConfig(8, 1))

        first = select_action_epsilon_greedy(
            model,
            [0.0] * OBSERVATION_INPUT_SIZE,
            epsilon=1.0,
            rng=random.Random(4),
        )
        second = select_action_epsilon_greedy(
            model,
            [0.0] * OBSERVATION_INPUT_SIZE,
            epsilon=1.0,
            rng=random.Random(4),
        )

        self.assertTrue(first.exploratory)
        self.assertEqual(first, second)
        self.assertIsNone(first.action_values)

    def test_synthetic_replay_problem_reduces_selected_action_loss(self):
        model = build_value_network(ValueNetworkConfig(16, 1))
        optimizer = build_optimizer(model, learning_rate=0.05)
        replay = TrainingReplayBuffer(capacity=32)
        replay.extend(
            sample(identifier, action_index=5, return_value=4.0)
            for identifier in range(16)
        )
        observations = [entry.observation for entry in replay.samples]
        actions = [entry.action_index for entry in replay.samples]
        returns = [entry.return_value for entry in replay.samples]
        initial_loss = float(
            selected_action_regression_loss(model, observations, actions, returns)
            .detach()
            .cpu()
            .item()
        )

        for _ in range(40):
            result = optimize_from_replay(
                model,
                optimizer,
                replay,
                batch_size=8,
                rng=random.Random(7),
            )

        final_loss = float(
            selected_action_regression_loss(model, observations, actions, returns)
            .detach()
            .cpu()
            .item()
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.batch_size, 8)
        self.assertLess(final_loss, initial_loss)

    def test_checkpoint_round_trips_predictions_and_optimizer_without_replay(self):
        model = build_value_network(ValueNetworkConfig(8, 1))
        optimizer = build_optimizer(model, learning_rate=0.001)
        replay = TrainingReplayBuffer(capacity=4)
        replay.extend(sample(identifier, action_index=identifier % ACTION_OUTPUT_SIZE) for identifier in range(3))
        observation = [[0.25] * OBSERVATION_INPUT_SIZE]
        before = predict_action_values(model, observation).detach().cpu()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.pth"
            save_training_checkpoint(
                path,
                model,
                optimizer=optimizer,
                replay_buffer=replay,
                extra_state={"completed_batches": 2},
            )
            self.assertFalse(replay_checkpoint_path(path).exists())

            restored_model = build_value_network(ValueNetworkConfig(8, 1))
            restored_optimizer = build_optimizer(restored_model, learning_rate=0.001)
            restored_replay = TrainingReplayBuffer(capacity=1)
            loaded = load_training_checkpoint(
                path,
                restored_model,
                optimizer=restored_optimizer,
                replay_buffer=restored_replay,
            )

        after = predict_action_values(restored_model, observation).detach().cpu()
        self.assertTrue(self.torch.allclose(before, after))
        self.assertTrue(loaded.has_optimizer_state)
        self.assertIsNone(loaded.replay_sample_count)
        self.assertEqual(loaded.extra_state["completed_batches"], 2)
        self.assertEqual(len(restored_replay), 0)

    def test_checkpoint_main_file_does_not_embed_replay_buffer(self):
        torch = torch_backend.require_torch()
        model = build_value_network(ValueNetworkConfig(8, 1))
        replay = TrainingReplayBuffer(capacity=4)
        replay.extend(sample(identifier) for identifier in range(3))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.pth"
            save_training_checkpoint(path, model, replay_buffer=replay)
            try:
                payload = torch.load(path, weights_only=False)
            except TypeError:
                payload = torch.load(path)

        self.assertNotIn("replay_buffer", payload)
        self.assertFalse(replay_checkpoint_path(path).exists())

    def test_checkpoint_save_removes_stale_replay_sidecar(self):
        model = build_value_network(ValueNetworkConfig(8, 1))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Earthling-01.pth"
            replay_checkpoint_path(path).write_bytes(b"stale replay")
            self.assertTrue(replay_checkpoint_path(path).exists())

            save_training_checkpoint(path, model)

            self.assertFalse(replay_checkpoint_path(path).exists())


if __name__ == "__main__":
    unittest.main()
