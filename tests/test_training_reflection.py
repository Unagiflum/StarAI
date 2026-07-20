import unittest

import numpy as np

from src.training import torch_backend
from src.training.contracts import (
    ACTION_INDEX_TABLE,
    OBSERVATION_FIELD_NAMES,
    OBSERVATION_INPUT_SIZE,
)
from src.training.reflection import (
    REFLECTED_ACTION_INDEX_TABLE,
    reflect_action_indices,
    reflect_observations,
)


_FIELD_INDEX = {
    field_name: index
    for index, field_name in enumerate(OBSERVATION_FIELD_NAMES)
}


class TrainingReflectionTests(unittest.TestCase):
    def test_reflection_transforms_every_directional_field_family(self):
        observation = np.zeros((1, OBSERVATION_INPUT_SIZE), dtype=np.float32)
        assignments = {
            "self.absolute_heading_sine": 0.6,
            "self.absolute_heading_cosine": -0.8,
            "enemy.absolute_heading_sine": -0.3,
            "enemy.absolute_heading_cosine": 0.95,
            "self.absolute_x_velocity": 0.4,
            "self.absolute_y_velocity": -0.3,
            "self.local_forward_velocity": 0.25,
            "self.local_right_velocity": -0.5,
            "enemy.opponent_bearing_sine": 0.75,
            "enemy.opponent_bearing_cosine": -0.25,
            "enemy.orz_turret_relative_sine": 0.6,
            "enemy.orz_turret_relative_cosine": -0.8,
            "self.left_repeat_countdown": 0.1,
            "self.right_repeat_countdown": 0.2,
            "enemy.left_held": 1.0,
            "enemy.right_held": 0.0,
            "object.planet.0.relative_bearing_sine": 0.7,
            "object.planet.0.relative_bearing_cosine": -0.4,
            "object.asteroid.3.relative_forward_velocity": 0.2,
            "object.asteroid.3.relative_right_velocity": -0.9,
            "object.asteroid.3.relative_closing_speed": 0.3,
            "object.asteroid.3.relative_lateral_velocity": -0.7,
        }
        for field_name, value in assignments.items():
            observation[0, _FIELD_INDEX[field_name]] = value

        mirrored = reflect_observations(observation)

        expected = {
            "self.absolute_heading_sine": -0.6,
            "self.absolute_heading_cosine": -0.8,
            "enemy.absolute_heading_sine": 0.3,
            "enemy.absolute_heading_cosine": 0.95,
            "self.absolute_x_velocity": -0.4,
            "self.absolute_y_velocity": -0.3,
            "self.local_forward_velocity": 0.25,
            "self.local_right_velocity": 0.5,
            "enemy.opponent_bearing_sine": -0.75,
            "enemy.opponent_bearing_cosine": -0.25,
            "enemy.orz_turret_relative_sine": -0.6,
            "enemy.orz_turret_relative_cosine": -0.8,
            "self.left_repeat_countdown": 0.2,
            "self.right_repeat_countdown": 0.1,
            "enemy.left_held": 0.0,
            "enemy.right_held": 1.0,
            "object.planet.0.relative_bearing_sine": -0.7,
            "object.planet.0.relative_bearing_cosine": -0.4,
            "object.asteroid.3.relative_forward_velocity": 0.2,
            "object.asteroid.3.relative_right_velocity": 0.9,
            "object.asteroid.3.relative_closing_speed": 0.3,
            "object.asteroid.3.relative_lateral_velocity": 0.7,
        }
        for field_name, value in expected.items():
            with self.subTest(field=field_name):
                self.assertAlmostEqual(
                    float(mirrored[0, _FIELD_INDEX[field_name]]),
                    value,
                    places=6,
                )

    def test_reflection_is_an_involution_for_numpy_batches(self):
        rng = np.random.default_rng(42)
        observations = rng.uniform(
            -1.0,
            1.0,
            size=(2, 3, OBSERVATION_INPUT_SIZE),
        ).astype(np.float32)
        restored = reflect_observations(reflect_observations(observations))

        np.testing.assert_allclose(restored, observations, rtol=0.0, atol=1e-7)
        self.assertFalse(np.shares_memory(observations, restored))

    def test_action_reflection_swaps_left_and_right_for_all_actions(self):
        actions = np.arange(len(ACTION_INDEX_TABLE), dtype=np.uint8)

        mirrored = reflect_action_indices(actions)
        restored = reflect_action_indices(mirrored)

        np.testing.assert_array_equal(
            mirrored,
            np.asarray(REFLECTED_ACTION_INDEX_TABLE, dtype=np.uint8),
        )
        np.testing.assert_array_equal(restored, actions)
        for original_index, mirrored_index in enumerate(mirrored):
            original = ACTION_INDEX_TABLE[original_index]
            reflected = ACTION_INDEX_TABLE[int(mirrored_index)]
            with self.subTest(action=original_index):
                self.assertEqual(reflected.thrust, original.thrust)
                self.assertEqual(reflected.turn_left, original.turn_right)
                self.assertEqual(reflected.turn_right, original.turn_left)
                self.assertEqual(reflected.a1, original.a1)
                self.assertEqual(reflected.a2, original.a2)

    def test_reflection_supports_training_device_tensors(self):
        torch = torch_backend.get_torch()
        if torch is None:
            self.skipTest("PyTorch is not installed")
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=str(device)):
                observations = torch.zeros(
                    (2, OBSERVATION_INPUT_SIZE),
                    device=device,
                )
                observations[
                    :, _FIELD_INDEX["self.absolute_x_velocity"]
                ] = 0.5
                actions = torch.tensor([2, 4], dtype=torch.long, device=device)

                mirrored_observations = reflect_observations(observations)
                mirrored_actions = reflect_action_indices(actions)

                self.assertEqual(
                    mirrored_observations.device.type,
                    device.type,
                )
                self.assertTrue(
                    torch.all(
                        mirrored_observations[
                            :, _FIELD_INDEX["self.absolute_x_velocity"]
                        ]
                        == -0.5
                    )
                )
                self.assertEqual(mirrored_actions.cpu().tolist(), [4, 2])


if __name__ == "__main__":
    unittest.main()
