import math
import pickle
import struct
import unittest

import numpy as np

from src.training.contracts import OBSERVATION_INPUT_SIZE
from src.training.observation_transfer import (
    PACKED_OBSERVATION_BYTES,
    PackedObservation,
    pack_observation,
    unpack_observation,
    unpack_observation_array,
)
from src.training.worker_protocol import (
    FrameSteppedResult,
    StartRunCommand,
)


class PackedObservationTests(unittest.TestCase):
    def test_payload_is_exact_float32_size_and_parent_view_is_float32(self):
        packed = pack_observation([0.25] * OBSERVATION_INPUT_SIZE)

        self.assertEqual(len(packed.data), OBSERVATION_INPUT_SIZE * 4)
        self.assertEqual(len(packed.data), PACKED_OBSERVATION_BYTES)
        values = unpack_observation_array(packed)
        self.assertEqual(values.dtype, np.dtype("float32"))
        self.assertEqual(values.shape, (OBSERVATION_INPUT_SIZE,))

    def test_round_trip_preserves_representative_values_at_float32_precision(self):
        source = [
            (index - 200) / 37.0
            for index in range(OBSERVATION_INPUT_SIZE)
        ]
        packed = pack_observation(source)
        result = np.asarray(unpack_observation(packed), dtype=np.float32)

        np.testing.assert_array_equal(result, np.asarray(source, dtype=np.float32))

    def test_malformed_payloads_and_non_finite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            PackedObservation(b"\0" * (PACKED_OBSERVATION_BYTES - 1))
        with self.assertRaises(ValueError):
            pack_observation([0.0] * (OBSERVATION_INPUT_SIZE - 1))
        with self.assertRaises(ValueError):
            pack_observation(
                [math.nan] + [0.0] * (OBSERVATION_INPUT_SIZE - 1)
            )
        nan_payload = struct.pack("<f", math.nan) + b"\0" * (
            PACKED_OBSERVATION_BYTES - 4
        )
        with self.assertRaises(ValueError):
            unpack_observation(PackedObservation(nan_payload))

    def test_worker_wire_dataclasses_remain_picklable(self):
        command = StartRunCommand(1, 2, 3)
        result = FrameSteppedResult(
            record_id=2,
            round_index=4,
            frame_count=1,
            complete=False,
            next_trainee_observation=pack_observation(
                [0.0] * OBSERVATION_INPUT_SIZE
            ),
            next_simple_opponent_controls={"forward": False},
        )

        self.assertEqual(pickle.loads(pickle.dumps(command)), command)
        self.assertEqual(pickle.loads(pickle.dumps(result)), result)


if __name__ == "__main__":
    unittest.main()
