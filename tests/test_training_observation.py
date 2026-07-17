import math
import unittest
from types import SimpleNamespace

import src.const as const
from src.training.contracts import (
    ENEMY_SHIP_BLOCK_OFFSET,
    OBJECT_SLOT_FIELDS,
    OBJECT_SLOT_GROUPS,
    OBJECT_SLOT_OFFSET,
    OBSERVATION_FIELD_NAMES,
    OBSERVATION_INPUT_SIZE,
    SELF_SHIP_BLOCK_OFFSET,
    SHIP_BLOCK_FIELDS,
    SHIP_TYPE_CATALOG_ORDER,
)
from src.training.observation import build_observation_context, encode_observation


def _field(prefix, name):
    return OBSERVATION_FIELD_NAMES.index(f"{prefix}.{name}")


def _object_field(group, slot, name):
    return OBSERVATION_FIELD_NAMES.index(f"object.{group}.{slot}.{name}")


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


def _obj(name, **overrides):
    values = {
        "name": name,
        "position": [0.0, 0.0],
        "velocity": [0.0, 0.0],
        "currently_alive": True,
        "current_hp": 1,
        "current_damage": 0,
        "can_expire": False,
        "expiration_timer": 0,
        "type": "projectile",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrainingObservationTests(unittest.TestCase):
    def test_shared_frame_context_matches_independent_perspective_encoding(self):
        trainee = _ship("Earthling", player=1, position=[100.0, 150.0])
        enemy = _ship("Mycon", player=2, position=[450.0, 500.0])
        objects = [
            trainee,
            enemy,
            _obj("EarthlingA1", parent=trainee, player=1, position=[120.0, 150.0]),
            _obj("MyconA1", parent=enemy, player=2, position=[430.0, 500.0]),
            _obj("Asteroid", type="space", position=[300.0, 300.0]),
        ]
        context = build_observation_context(trainee, enemy, objects)

        trainee_shared = encode_observation(
            trainee,
            enemy,
            frame_id=7,
            game_objects=objects,
            context=context,
        )
        enemy_shared = encode_observation(
            enemy,
            trainee,
            frame_id=7,
            game_objects=objects,
            context=context,
        )

        self.assertEqual(
            trainee_shared,
            encode_observation(trainee, enemy, frame_id=7, game_objects=objects),
        )
        self.assertEqual(
            enemy_shared,
            encode_observation(enemy, trainee, frame_id=7, game_objects=objects),
        )

    def test_encoder_produces_finite_values_with_zero_filled_objects(self):
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

    def test_object_slots_encode_enemy_planet_and_toroidal_geometry(self):
        trainee = _ship(
            "Earthling",
            position=[const.ARENA_SIZE - 10.0, 500.0],
            rotation=90,
            velocity=[2.0, 0.0],
        )
        enemy = _ship("Mycon", position=[10.0, 500.0], velocity=[2.0, 5.0])
        planet = _obj(
            "Planet",
            position=[const.ARENA_SIZE - 10.0, 480.0],
            gravity=1,
            diameter=100,
        )

        observation = encode_observation(
            trainee,
            enemy,
            game_objects=[trainee, enemy, planet],
        )

        self.assertEqual(observation[_object_field("enemy_ship", 0, "present")], 1.0)
        self.assertAlmostEqual(
            observation[_object_field("enemy_ship", 0, "relative_bearing_sine")],
            0.0,
        )
        self.assertAlmostEqual(
            observation[_object_field("enemy_ship", 0, "relative_bearing_cosine")],
            1.0,
        )
        self.assertEqual(observation[_object_field("enemy_ship", 0, "inverse_distance")], 5.0)
        self.assertAlmostEqual(
            observation[_object_field("enemy_ship", 0, "relative_velocity_sine")],
            0.0,
        )
        self.assertAlmostEqual(
            observation[_object_field("enemy_ship", 0, "relative_velocity_cosine")],
            -1.0,
        )
        self.assertEqual(observation[_object_field("planet", 0, "present")], 1.0)
        self.assertEqual(observation[_object_field("planet", 0, "remaining_timer")], 5.0)

    def test_object_slot_groups_order_and_zero_masking_are_stable(self):
        trainee = _ship(
            "Earthling",
            player=1,
            position=[const.ARENA_SIZE - 10.0, 500.0],
            rotation=90,
        )
        enemy = _ship("Mycon", player=2, position=[500.0, 500.0])
        beta = _obj(
            "BetaA1",
            parent=enemy,
            player=2,
            position=[const.ARENA_SIZE - 20.0, 500.0],
        )
        alpha = _obj("AlphaA1", parent=enemy, player=2, position=[0.0, 500.0])
        friendly = _obj(
            "FriendlyA1",
            parent=trainee,
            player=1,
            position=[const.ARENA_SIZE - 10.0, 520.0],
            current_damage=3,
            can_expire=True,
            expiration_timer=30,
        )

        observation = encode_observation(
            trainee,
            enemy,
            game_objects=[trainee, enemy, beta, alpha, friendly],
        )

        self.assertEqual(observation[_object_field("enemy_a1", 0, "present")], 1.0)
        self.assertAlmostEqual(
            observation[_object_field("enemy_a1", 0, "relative_bearing_cosine")],
            1.0,
        )
        self.assertEqual(observation[_object_field("enemy_a1", 1, "present")], 1.0)
        self.assertAlmostEqual(
            observation[_object_field("enemy_a1", 1, "relative_bearing_cosine")],
            -1.0,
        )
        self.assertEqual(observation[_object_field("friendly_a1", 0, "expires")], 1.0)
        self.assertEqual(
            observation[_object_field("friendly_a1", 0, "remaining_timer")],
            30 / const.FPS,
        )
        self.assertEqual(observation[_object_field("friendly_a1", 0, "expected_crew_effect")], -0.3)
        self.assertTrue(
            all(
                observation[_object_field("friendly_a1", 1, field)] == 0.0
                for field in OBJECT_SLOT_FIELDS
            )
        )

    def test_special_object_classification_tracks_satellites_and_laser(self):
        trainee = _ship(
            "Earthling",
            player=1,
            position=[100.0, 100.0],
            velocity=[0.0, 0.0],
        )
        enemy = _ship("Chmmr", player=2, position=[500.0, 500.0])
        enemy_satellite = _obj(
            "ChmmrSatellite",
            parent=enemy,
            player=2,
            type="special_object",
            position=[110.0, 100.0],
            previous_position=[107.0, 96.0],
            velocity=[99.0, 99.0],
            current_hp=7,
        )
        friendly_satellite = _obj(
            "ChmmrSatellite",
            parent=trainee,
            player=1,
            type="special_object",
            position=[100.0, 90.0],
            previous_position=[103.0, 86.0],
            velocity=[99.0, 99.0],
            current_hp=4,
        )
        satellite_laser = _obj(
            "ChmmrSatelliteLaser",
            parent=enemy_satellite,
            player=2,
            type="laser",
            position=[120.0, 100.0],
            current_damage=2,
        )
        syreen_crew = _obj(
            "SyreenCrew",
            parent=enemy,
            player=2,
            type="special_object",
            position=[130.0, 100.0],
        )

        observation = encode_observation(
            trainee,
            enemy,
            game_objects=[
                trainee,
                enemy,
                enemy_satellite,
                friendly_satellite,
                satellite_laser,
                syreen_crew,
            ],
        )

        self.assertEqual(observation[_object_field("enemy_non_a1", 0, "present")], 1.0)
        self.assertAlmostEqual(
            observation[_object_field("enemy_non_a1", 0, "relative_velocity_sine")],
            0.6,
        )
        self.assertAlmostEqual(
            observation[_object_field("enemy_non_a1", 0, "relative_velocity_cosine")],
            -0.8,
        )
        self.assertEqual(observation[_object_field("enemy_non_a1", 0, "relative_speed")], 0.05)
        self.assertEqual(observation[_object_field("enemy_non_a1", 0, "current_hit_points")], 0.14)
        self.assertEqual(observation[_object_field("enemy_non_a1", 1, "present")], 1.0)
        self.assertEqual(observation[_object_field("enemy_non_a1", 1, "expected_crew_effect")], -0.2)
        self.assertEqual(observation[_object_field("enemy_non_a1", 2, "present")], 0.0)
        self.assertEqual(observation[_object_field("friendly_non_a1", 0, "present")], 1.0)
        self.assertAlmostEqual(
            observation[_object_field("friendly_non_a1", 0, "relative_velocity_sine")],
            -0.6,
        )
        self.assertAlmostEqual(
            observation[_object_field("friendly_non_a1", 0, "relative_velocity_cosine")],
            -0.8,
        )
        self.assertEqual(observation[_object_field("friendly_non_a1", 0, "relative_speed")], 0.05)
        self.assertEqual(observation[_object_field("friendly_non_a1", 0, "current_hit_points")], 0.08)
        self.assertEqual(observation[_object_field("syreen_crew", 0, "present")], 1.0)
        self.assertEqual(observation[_object_field("syreen_crew", 0, "expected_crew_effect")], 0.1)

    def test_satellite_frame_velocity_uses_wrapped_displacement(self):
        trainee = _ship(
            "Earthling",
            player=1,
            position=[0.0, 100.0],
            velocity=[0.0, 0.0],
        )
        enemy = _ship("Chmmr", player=2, position=[500.0, 500.0])
        satellite = _obj(
            "ChmmrSatellite",
            parent=enemy,
            player=2,
            type="special_object",
            position=[3.0, 100.0],
            previous_position=[const.ARENA_SIZE - 2.0, 100.0],
            velocity=[-99.0, 0.0],
        )

        observation = encode_observation(
            trainee,
            enemy,
            game_objects=[trainee, enemy, satellite],
        )

        self.assertAlmostEqual(
            observation[_object_field("enemy_non_a1", 0, "relative_velocity_sine")],
            1.0,
        )
        self.assertAlmostEqual(
            observation[_object_field("enemy_non_a1", 0, "relative_velocity_cosine")],
            0.0,
            places=12,
        )
        self.assertEqual(
            observation[_object_field("enemy_non_a1", 0, "relative_speed")],
            0.05,
        )

    def test_ship_specific_live_counts_use_world_objects(self):
        trainee = _ship(
            "Orz",
            player=1,
            position=[100.0, 100.0],
            rotation=0,
            turret_heading=90,
        )
        enemy = _ship("Mycon", player=2, position=[200.0, 100.0], boarded_marines=[])
        floating_marine = _obj("OrzA3", parent=trainee, player=1, mode="outbound")
        boarded_marine = _obj("OrzA3", parent=trainee, player=1, mode="boarded")
        enemy.boarded_marines.append(boarded_marine)
        objects = [
            trainee,
            enemy,
            floating_marine,
            boarded_marine,
            _obj("KzerZaA2", parent=trainee, player=1),
            _obj("ChmmrSatellite", parent=trainee, player=1),
            _obj("ChenjesuA2", parent=trainee, player=1),
            _obj("KohrAhA1", parent=trainee, player=1),
        ]

        observation = encode_observation(trainee, enemy, game_objects=objects)

        self.assertAlmostEqual(observation[_field("self", "orz_turret_relative_sine")], 1.0)
        self.assertAlmostEqual(observation[_field("self", "orz_turret_relative_cosine")], 0.0)
        self.assertEqual(observation[_field("self", "orz_marines_floating")], 1 / 8)
        self.assertEqual(observation[_field("self", "orz_marines_boarded_on_enemy")], 1 / 8)
        self.assertEqual(observation[_field("self", "ur_quan_fighters")], 1 / 25)
        self.assertEqual(observation[_field("self", "chmmr_satellites")], 1 / 3)
        self.assertEqual(observation[_field("self", "chenjesu_dogis")], 1 / 4)
        self.assertEqual(observation[_field("self", "kohr_ah_saws")], 1 / 8)

    def test_object_slot_group_contract_still_totals_38_slots(self):
        self.assertEqual(sum(count for _, count in OBJECT_SLOT_GROUPS), 38)


if __name__ == "__main__":
    unittest.main()
