import math
import os
import random
import unittest
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle import collision_responses, collisions, status_bar
from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.Slylandro.A1.SlylandroA1 import SlylandroA1
from src.Objects.Ships.Slylandro.A2.SlylandroA2 import SlylandroA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ship


class SlylandroTests(unittest.TestCase):
    def setUp(self):
        self.ship = create_ship("Slylandro", 1)
        self.ship.initialize_in_battle([500.0, 500.0], 0)
        self.ship.in_battle = True
        self.target = create_ship("Earthling", 2)
        self.target.initialize_in_battle([500.0, 200.0], 0)
        self.target.in_battle = True
        self.ship.opponent = self.target
        self.target.opponent = self.ship

    def test_catalog_and_non_sentient_crew_configuration(self):
        definition = SHIP_DEFINITIONS["Slylandro"]

        self.assertEqual(definition.ship_type, "Probe")
        self.assertTrue(definition.immune_to_psychic)
        self.assertTrue(self.ship.durability_capabilities.immune_to_psychic)
        self.assertEqual(self.ship.crew_bar_color, const.HUD_NONSENTIENT_HP_COLOR)
        self.assertEqual(ABILITY_DEFINITIONS["SlylandroA1"].number_of_bolts, 2)

    def test_status_bar_draws_probe_crew_in_gray(self):
        hp_bar = mock.Mock(height=20)
        energy_bar = mock.Mock(height=20)
        with mock.patch.object(
            status_bar,
            "_get_status_bar",
            side_effect=(hp_bar, energy_bar),
        ):
            status_bar.draw_player_status(mock.Mock(), self.ship, 0, 100, 10, 2)

        self.assertEqual(
            hp_bar.draw.call_args.args[4],
            const.HUD_NONSENTIENT_HP_COLOR,
        )

    def test_lightning_segment_count_grows_and_decays_over_a1_cooldown(self):
        self.assertEqual(
            [SlylandroA1.segment_count(frame, 5) for frame in range(1, 6)],
            [2, 3, 3, 2, 0],
        )
        self.assertEqual(
            [SlylandroA1.segment_count(frame, 6) for frame in range(1, 7)],
            [2, 3, 4, 3, 2, 0],
        )

        plan = self.ship.plan_action1()
        self.assertEqual(len(plan.spawned_objects), 2)
        bolt = plan.spawned_objects[0]
        for expected_segments in (2, 3, 4):
            self.assertTrue(bolt.update())
            self.assertEqual(len(bolt.collision_segments()), expected_segments)
            self.assertEqual(bolt.points[0], self.ship.position)

    def test_first_interception_suppresses_subsequent_generations(self):
        bolts = self.ship.plan_action1().spawned_objects
        for bolt in bolts:
            bolt.update()
        first = bolts[0]
        contact = first.points[1]

        collision_responses.resolve_laser_hit(
            first,
            self.target,
            [],
            [0.0, 1.0],
            contact,
            lambda *args: None,
            segment_index=0,
        )

        self.assertTrue(first.session.suppressed)
        self.assertTrue(bolts[1].session.suppressed)
        for bolt in bolts:
            self.assertTrue(bolt.update())
            self.assertEqual(bolt.points, [])

    def test_each_bolt_can_damage_the_same_intercepting_target(self):
        self.ship.position = [100.0, 500.0]
        self.target.position = [200.0, 500.0]
        bolts = self.ship.plan_action1().spawned_objects
        for bolt in bolts:
            bolt.points = [[100.0, 500.0], [300.0, 500.0]]
            bolt.calculate_end_position()
        starting_hp = self.target.current_hp

        collisions._handle_laser_collisions(
            list(bolts),
            [self.ship, self.target],
            [],
            [],
            [],
            [],
            [],
        )

        self.assertEqual(self.target.current_hp, starting_hp - 2)
        self.assertTrue(bolts[0].session.suppressed)

    def test_dead_or_cloaked_target_uses_random_segment_directions(self):
        bolt = self.ship.plan_action1().spawned_objects[0]
        bolt.rng = mock.Mock()
        bolt.rng.choice.side_effect = lambda values: values[0]
        bolt.rng.randrange.side_effect = [4, 8]
        bolt.rng.uniform.return_value = 20
        self.target.cloaked = True
        self.target.trackable = False

        bolt.update()

        first_delta = (
            bolt.points[1][0] - bolt.points[0][0],
            bolt.points[1][1] - bolt.points[0][1],
        )
        second_delta = (
            bolt.points[2][0] - bolt.points[1][0],
            bolt.points[2][1] - bolt.points[1][1],
        )
        self.assertAlmostEqual(first_delta[0], 20)
        self.assertAlmostEqual(first_delta[1], 0)
        self.assertAlmostEqual(second_delta[0], 0)
        self.assertAlmostEqual(second_delta[1], 20)

    def test_probe_always_moves_at_max_thrust_and_thrust_reverses_once(self):
        self.assertEqual(self.ship.velocity, [0.0, -60.0])

        self.ship.set_control_state("thrust", True, frame_id=1)
        self.ship.process_controls(frame_id=1)
        self.assertEqual(self.ship.heading, const.SHIP_DIRECTIONS // 2)
        self.assertAlmostEqual(math.hypot(*self.ship.velocity), 60)

        self.ship.process_controls(frame_id=2)
        self.assertEqual(self.ship.heading, const.SHIP_DIRECTIONS // 2)

    def test_collision_velocity_is_quantized_and_returns_to_max_thrust(self):
        angle = math.radians(31)
        self.ship.collision_velocity = [
            math.sin(angle) * 42,
            -math.cos(angle) * 42,
        ]

        self.ship.update_physics()

        self.assertEqual(self.ship.heading, 1)
        self.assertAlmostEqual(math.hypot(*self.ship.velocity), 60)
        self.assertEqual(self.ship.rotation, const.TURN_ANGLE)

    def test_trajectory_prediction_preserves_inertialess_max_speed(self):
        predicted = self.ship.predict_unhindered_trajectory(3)
        actual = []
        for _ in range(3):
            self.ship.update_physics()
            actual.append(list(self.ship.position))

        self.assertEqual(predicted, actual)

    def test_animation_advance_is_blocked_by_overlap(self):
        self.ship.animation_frame = 3
        with mock.patch.object(self.ship, "_sprite_would_overlap", return_value=True):
            self.ship.update()
        self.assertEqual(self.ship.animation_frame, 3)

    def test_a2_recharges_once_and_only_destroys_in_range_asteroids(self):
        nearby = Asteroid(rng=random.Random(1))
        nearby.position = [600.0, 500.0]
        nearby.previous_position = nearby.position.copy()
        distant = Asteroid(rng=random.Random(2))
        distant.position = [900.0, 500.0]
        distant.previous_position = distant.position.copy()
        self.target.position = [550.0, 500.0]
        target_hp = self.target.current_hp
        self.ship.asteroids = [nearby, distant]
        self.ship.current_energy = 3

        pulse = self.ship.perform_action2()

        self.assertIsInstance(pulse, SlylandroA2)
        self.assertEqual(self.ship.current_energy, self.ship.max_energy)

        collisions._handle_area_damage(
            [pulse, nearby, distant, self.target], []
        )

        self.assertFalse(nearby.currently_alive)
        self.assertTrue(distant.currently_alive)
        self.assertEqual(self.target.current_hp, target_hp)

    def test_a2_without_an_in_range_asteroid_does_not_activate(self):
        distant = Asteroid(rng=random.Random(3))
        distant.position = [900.0, 500.0]
        distant.previous_position = distant.position.copy()
        self.ship.asteroids = [distant]
        self.ship.current_energy = 3

        result = self.ship.perform_action2()

        self.assertIsNone(result)
        self.assertEqual(self.ship.current_energy, 3)
        self.assertEqual(self.ship.action2_timer, 0)

    def test_valid_a2_plays_sound_and_starts_cooldown(self):
        nearby = Asteroid(rng=random.Random(4))
        nearby.position = [600.0, 500.0]
        self.ship.asteroids = [nearby]
        sound = mock.Mock()
        self.ship.audio_service = mock.Mock()
        self.ship.audio_service.load_effect.return_value = sound

        result = self.ship.perform_action2()

        self.assertIsInstance(result, SlylandroA2)
        sound.play.assert_called_once_with()
        self.assertEqual(
            self.ship.action2_timer,
            const.cooldown_frames(self.ship.a2_wait),
        )


if __name__ == "__main__":
    unittest.main()
