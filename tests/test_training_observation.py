import math
import unittest
from types import SimpleNamespace

import src.const as const
from src.training.contracts import (
    ENEMY_SHIP_BLOCK_OFFSET,
    OBJECT_SLOT_OFFSET,
    OBSERVATION_FIELD_NAMES,
    OBSERVATION_INPUT_SIZE,
    SELF_SHIP_BLOCK_OFFSET,
    SHIP_BLOCK_FIELDS,
    SHIP_TYPE_CATALOG_ORDER,
)
from src.training.observation import encode_observation


def _field(prefix, name):
    return OBSERVATION_FIELD_NAMES.index(f"{prefix}.{name}")


def _ship(name="Earthling", **overrides):
    values = {
        "name": name,
        "max_hp": 18,
        "max_energy": 18,
        "current_hp": 12,
        "current_energy": 9,
        "thrust_wait": 4,
        "thrust_timer": 3,
        "turn_wait": 2,
        "turn_timer": 1,
        "thrust_increment": 5,
        "a1_wait": 6,
        "action1_timer": 5,
        "a2_wait": 7,
        "action2_timer": 4,
        "a3_wait": 8,
        "action3_timer": 3,
        "energy_timer": 2,
        "rotation": 90,
        "velocity": [30.0, -40.0],
        "trackable": True,
        "thrust_active": False,
        "turn_left_active": False,
        "turn_right_active": False,
        "action1_active": False,
        "action2_active": False,
        "input_pressed_frames": {},
        "newly_pressed_controls": set(),
        "limpets_attached": 0,
        "boarded_marines": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingObservationTests(unittest.TestCase):
    def test_encoder_produces_533_finite_values_with_zero_filled_objects(self):
        trainee = _ship("Earthling")
        enemy = _ship("Mycon")

        observation = encode_observation(trainee, enemy, frame_id=100)

        self.assertEqual(len(observation), OBSERVATION_INPUT_SIZE)
        self.assertTrue(all(math.isfinite(value) for value in observation))
        self.assertEqual(
            observation[SHIP_TYPE_CATALOG_ORDER.index("Mycon")],
            1.0,
        )
        self.assertEqual(sum(observation[: len(SHIP_TYPE_CATALOG_ORDER)]), 1.0)
        self.assertTrue(all(value == 0.0 for value in observation[OBJECT_SLOT_OFFSET:]))

    def test_ship_block_encodes_general_runtime_values(self):
        trainee = _ship(
            max_hp=50,
            max_energy=40,
            current_hp=25,
            current_energy=10,
            thrust_wait=6,
            thrust_timer=12,
            turn_wait=3,
            turn_timer=9,
            thrust_increment=7,
            rotation=450,
            velocity=[60, 80],
            trackable=False,
        )
        enemy = _ship("Mycon")

        observation = encode_observation(trainee, enemy)

        self.assertEqual(observation[_field("self", "maximum_crew")], 1.0)
        self.assertEqual(observation[_field("self", "maximum_battery")], 0.8)
        self.assertEqual(observation[_field("self", "current_crew")], 0.5)
        self.assertEqual(observation[_field("self", "current_battery")], 0.2)
        self.assertEqual(observation[_field("self", "thrust_wait")], 6 / const.FPS)
        self.assertEqual(observation[_field("self", "thrust_timer")], 12 / const.FPS)
        self.assertEqual(observation[_field("self", "turn_wait")], 3 / const.FPS)
        self.assertEqual(observation[_field("self", "turn_timer")], 9 / const.FPS)
        self.assertEqual(observation[_field("self", "thrust_increment")], 0.7)
        self.assertEqual(observation[_field("self", "absolute_angle")], 0.25)
        self.assertEqual(observation[_field("self", "absolute_speed")], 1.0)
        self.assertEqual(observation[_field("self", "absolute_x_velocity")], 0.6)
        self.assertEqual(observation[_field("self", "absolute_y_velocity")], 0.8)
        self.assertEqual(observation[_field("self", "trackable")], 0.0)

    def test_held_flags_disambiguate_zero_repeat_countdown(self):
        trainee = _ship(
            thrust_active=False,
            action1_active=True,
            action2_active=True,
            input_pressed_frames={"action1": 10, "action2": 10},
            newly_pressed_controls={"action2"},
        )
        enemy = _ship("Mycon")

        observation = encode_observation(trainee, enemy, frame_id=11)

        self.assertEqual(observation[_field("self", "thrust_repeat_countdown")], 0.0)
        self.assertEqual(observation[_field("self", "thrust_held")], 0.0)
        self.assertEqual(
            observation[_field("self", "a1_repeat_countdown")],
            const.INPUT_REPEAT_DELAY_FRAMES - 1,
        )
        self.assertEqual(observation[_field("self", "a1_held")], 1.0)
        self.assertEqual(observation[_field("self", "a2_repeat_countdown")], 0.0)
        self.assertEqual(observation[_field("self", "a2_held")], 1.0)

    def test_limpet_and_form_adjusted_movement_values_are_read_from_ship(self):
        trainee = _ship(
            "Androsynth",
            form="A2",
            limpets_attached=3,
            thrust_wait=11,
            turn_wait=9,
            thrust_increment=4,
        )
        enemy = _ship("Mmrnmrhm", form="YWing")

        observation = encode_observation(trainee, enemy)

        self.assertEqual(observation[_field("self", "thrust_wait")], 11 / const.FPS)
        self.assertEqual(observation[_field("self", "turn_wait")], 9 / const.FPS)
        self.assertEqual(observation[_field("self", "thrust_increment")], 0.4)
        self.assertEqual(observation[_field("self", "androsynth_blazer_form")], 1.0)
        self.assertEqual(observation[_field("self", "limpet_count")], 3 / 64)
        self.assertEqual(observation[_field("enemy", "mmrnmrhm_alternate_form")], 1.0)

    def test_missing_optional_state_defaults_to_zero(self):
        trainee = SimpleNamespace(name="Earthling")
        enemy = SimpleNamespace(name="Unknown")

        observation = encode_observation(trainee, enemy)

        self.assertEqual(sum(observation[: len(SHIP_TYPE_CATALOG_ORDER)]), 0.0)
        self.assertEqual(len(observation[SELF_SHIP_BLOCK_OFFSET:ENEMY_SHIP_BLOCK_OFFSET]), len(SHIP_BLOCK_FIELDS))
        self.assertEqual(len(observation[ENEMY_SHIP_BLOCK_OFFSET:OBJECT_SLOT_OFFSET]), len(SHIP_BLOCK_FIELDS))
        self.assertEqual(observation[_field("self", "maximum_crew")], 0.0)
        self.assertEqual(observation[_field("self", "trackable")], 1.0)
        self.assertEqual(observation[_field("self", "orz_turret_relative_cosine")], 0.0)


if __name__ == "__main__":
    unittest.main()
