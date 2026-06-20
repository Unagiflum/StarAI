import math
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

import src.const as const
from src.Battle import collisions
from src.Battle.battle import (
    BattleSimulation,
    aftermath_camera_targets,
    aftermath_ready_for_selection,
    start_or_update_aftermath,
)
from src.Battle.battle_draw import calculate_view_parameters
from src.Battle.battle_init import validate_ship_positions
from src.Objects.object import PlayerObject
from src.Objects.Ships.ability import Ability, wrapped_endpoint
from src.Objects.Ships.space_ship import SpaceShip
from src.toroidal import (
    nearest_position,
    view_center_and_size,
    wrapped_delta,
    wrapped_distance,
    wrapped_midpoint,
)


class Positioned:
    def __init__(self, position, size=(10, 10)):
        self.position = list(position)
        self.previous_position = list(position)
        self.size = list(size)
        self.can_move = True


class ShipState:
    def reset_controls(self):
        self.thrust_active = False
        self.turn_left_active = False
        self.turn_right_active = False
        self.action1_active = False
        self.action2_active = False
        self.input_pressed_frames.clear()
        self.newly_pressed_controls.clear()
        self.released_controls.clear()


class ToroidalMechanicsTests(unittest.TestCase):
    def test_central_geometry_operations_agree_at_wrapped_edges(self):
        first = [const.ARENA_SIZE - 100, 100]
        second = [100, const.ARENA_SIZE - 100]

        self.assertEqual(wrapped_delta(first, second), [200, -200])
        self.assertAlmostEqual(wrapped_distance(first, second), math.hypot(200, 200))
        self.assertEqual(nearest_position(second, first), [const.ARENA_SIZE + 100, -100])
        self.assertEqual(wrapped_midpoint(first, second), [0, 0])

    def test_view_size_has_minimum_and_arena_bounds(self):
        center, near_size = view_center_and_size([[0, 0], [10, 0]])
        _, far_size = view_center_and_size([[0, 0], [const.ARENA_SIZE / 2, 0]])

        self.assertEqual(center, [5.0, 0.0])
        self.assertEqual(near_size, const.SCREEN_HEIGHT / const.MAX_ZOOM)
        self.assertEqual(far_size, const.ARENA_SIZE / 2)

    def test_distance_uses_shortest_path_across_arena_edge(self):
        player = PlayerObject("player", None, [10, 10], 1, 1.0)
        player.position = [const.ARENA_SIZE - 10, 20]
        target = Positioned([10, 5])

        delta, distance = player.distance_to(target)

        self.assertEqual(delta, [20, -15])
        self.assertAlmostEqual(distance, 25.0)

    def test_exactly_half_an_arena_keeps_the_unwrapped_direction(self):
        start = [0, 0]
        end = [const.ARENA_SIZE / 2, -const.ARENA_SIZE / 2]

        self.assertEqual(wrapped_endpoint(start, end), end)

    def test_wrapped_endpoint_returns_nearest_image(self):
        start = [const.ARENA_SIZE - 25, 20]
        end = [15, const.ARENA_SIZE - 10]

        self.assertEqual(
            wrapped_endpoint(start, end),
            [const.ARENA_SIZE + 15, -10],
        )

    def test_ship_position_validation_uses_wrapped_distance(self):
        near_edge_pair = ([const.ARENA_SIZE - 100, 0], [100, 0])
        far_pair = ([0, 0], [const.MIN_SHIP_SEPARATION, 0])

        self.assertFalse(validate_ship_positions(*near_edge_pair))
        self.assertTrue(validate_ship_positions(*far_pair))

    def test_camera_centers_on_edge_crossing_targets(self):
        targets = [
            Positioned([const.ARENA_SIZE - 100, const.ARENA_SIZE / 2]),
            Positioned([100, const.ARENA_SIZE / 2]),
        ]

        scale, translation = calculate_view_parameters([], targets)

        self.assertEqual(scale, const.MAX_ZOOM)
        self.assertEqual(
            translation,
            [const.SCREEN_HEIGHT / 2, const.SCREEN_HEIGHT / 2 - const.ARENA_SIZE / 2],
        )

    def test_gravity_uses_wrapped_direction(self):
        player = PlayerObject("player", None, [10, 10], 1, 1.0)
        player.position = [const.ARENA_SIZE - 10, 0]
        player.planet = SimpleNamespace(
            position=[10, 0],
            diameter=10,
            gravity=4,
        )

        self.assertEqual(player.get_gravity(), [2.0, 0.0])

    def test_collision_normal_uses_nearest_wrapped_image(self):
        left = Positioned([const.ARENA_SIZE - 5, 100], size=(20, 20))
        right = Positioned([5, 100], size=(20, 20))

        normal, distance, overlap = collisions._collision_info(left, right)

        self.assertEqual(normal, [-1.0, 0.0])
        self.assertEqual(distance, 10.0)
        self.assertEqual(overlap, 10.0)
        self.assertTrue(collisions._objects_overlap(left, right, overlap))


class InputTimingTests(unittest.TestCase):
    @staticmethod
    def make_ship():
        ship = SpaceShip.__new__(SpaceShip)
        ship.thrust_active = False
        ship.turn_left_active = False
        ship.turn_right_active = False
        ship.action1_active = False
        ship.action2_active = False
        ship.input_pressed_frames = {}
        ship.newly_pressed_controls = set()
        ship.released_controls = set()
        ship.thrust_timer = 2
        ship.turn_timer = 2
        ship.action1_timer = 2
        ship.action2_timer = 2
        ship.action3_timer = 2
        ship.energy_timer = 0
        ship.energy_wait = 100
        ship.current_energy = 0
        ship.max_energy = 0
        ship.energy_regen = 0
        ship.inertia = True
        return ship

    def test_timers_advance_once_during_frame_processing(self):
        ship = self.make_ship()

        ship.process_controls(frame_id=10)

        self.assertEqual(ship.thrust_timer, 1)
        self.assertEqual(ship.action1_timer, 1)

    def test_new_action_press_is_immediate_and_not_repeated_same_frame(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        calls = []
        ship.perform_action1 = lambda: calls.append("action1")

        ship.set_control_state("action1", True, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(calls, ["action1"])

    def test_held_action_observes_repeat_delay(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        calls = []
        ship.perform_action1 = lambda: calls.append("action1")

        ship.set_control_state("action1", True, frame_id=10)
        ship.process_controls(frame_id=10)
        ship.process_controls(frame_id=11)
        ship.process_controls(frame_id=12)
        ship.process_controls(frame_id=13)

        self.assertEqual(calls, ["action1", "action1"])

    def test_action_release_hook_runs_once(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        releases = []
        ship.perform_action1 = lambda: None
        ship.perform_action1_release = lambda: releases.append("release")

        ship.set_control_state("action1", True, frame_id=10)
        ship.process_controls(frame_id=10)
        ship.set_control_state("action1", False, frame_id=11)
        ship.process_controls(frame_id=11)

        self.assertEqual(releases, ["release"])

    def test_press_and_release_in_same_frame_keeps_both_edges(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        calls = []
        ship.perform_action1 = lambda: calls.append("action1")
        ship.perform_action1_release = lambda: calls.append("release")

        ship.set_control_state("action1", True, frame_id=10)
        ship.set_control_state("action1", False, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(calls, ["action1", "release"])

    def test_invalid_combined_action_falls_back_to_both_actions(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        ship.action2_timer = 0
        calls = []
        ship.perform_action1 = lambda: calls.append("action1")
        ship.perform_action2 = lambda: calls.append("action2")
        ship.perform_action3 = lambda: (calls.append("action3"), False)

        ship.set_control_state("action1", True, frame_id=10)
        ship.set_control_state("action2", True, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(calls, ["action3", "action1", "action2"])

    def test_valid_combined_action_runs_without_individual_actions(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        ship.action2_timer = 0
        calls = []
        ship.perform_action1 = lambda: calls.append("action1")
        ship.perform_action2 = lambda: calls.append("action2")
        ship.perform_action3 = lambda: (calls.append("action3"), True)

        ship.set_control_state("action1", True, frame_id=10)
        ship.set_control_state("action2", True, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(calls, ["action3"])


class SimulationInputTests(unittest.TestCase):
    class CountingShip:
        def __init__(self):
            self.currently_alive = True
            self.current_hp = 1
            self.processed_frames = []
            self.control_changes = []

        def set_control_state(self, control, pressed, frame_id):
            self.control_changes.append((control, pressed, frame_id))

        def process_controls(self, frame_id):
            self.processed_frames.append(frame_id)
            return []

    @staticmethod
    def make_simulation():
        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.running = True
        simulation.frame_id = 0
        simulation.needs_selection = False
        simulation.aftermath = None
        simulation.player1 = SimulationInputTests.CountingShip()
        simulation.player2 = SimulationInputTests.CountingShip()
        simulation.game_objects = []
        simulation.settings = {
            f"Player {player}: {control}": player * 10 + index
            for player in (1, 2)
            for index, control in enumerate(("Forward", "Left", "Right", "Action 1", "Action 2"))
        }
        simulation.key_states = {
            key: False for key in simulation.settings.values()
        }
        return simulation

    def test_each_living_ship_processes_input_once_per_step(self):
        simulation = self.make_simulation()

        simulation.step()
        simulation.step()

        self.assertEqual(simulation.player1.processed_frames, [1, 2])
        self.assertEqual(simulation.player2.processed_frames, [1, 2])

    def test_key_change_is_ingested_before_frame_processing(self):
        simulation = self.make_simulation()
        forward_key = simulation.settings["Player 1: Forward"]

        simulation.step(key_changes=[(forward_key, True)])

        self.assertEqual(simulation.player1.control_changes, [("thrust", True, 1)])
        self.assertEqual(simulation.player1.processed_frames, [1])


class CollisionCharacterizationTests(unittest.TestCase):
    @staticmethod
    def make_ship():
        ship = SpaceShip.__new__(SpaceShip)
        ship.name = "Target"
        ship.player = 2
        ship.position = [5, 100]
        ship.previous_position = ship.position.copy()
        ship.size = [20, 20]
        ship.current_hp = 10
        ship.currently_alive = True
        return ship

    @staticmethod
    def make_projectile(parent):
        projectile = Ability.__new__(Ability)
        projectile.name = "TestProjectile"
        projectile.projectile_name = "TestProjectile"
        projectile.type = "projectile"
        projectile.player = 1
        projectile.parent = parent
        projectile.position = [const.ARENA_SIZE - 5, 100]
        projectile.previous_position = projectile.position.copy()
        projectile.size = [20, 20]
        projectile.masks = None
        projectile.heading = 0
        projectile.frames = 1
        projectile.can_collide = True
        projectile.currently_alive = True
        projectile.current_hp = 1
        projectile.current_damage = 4
        projectile.hit_parent = False
        projectile.hit_self = False
        projectile.death_animation = []
        projectile.velocity = [1, 0]
        return projectile

    def test_projectile_damages_ship_across_wrapped_boundary(self):
        parent = self.make_ship()
        parent.player = 1
        target = self.make_ship()
        projectile = self.make_projectile(parent)
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast", return_value=sentinel_effect),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_ship_collisions([projectile], [target], effects)

        self.assertEqual(target.current_hp, 6)
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(effects, [sentinel_effect])


class AftermathCharacterizationTests(unittest.TestCase):
    def test_death_initializes_aftermath_and_holds_camera(self):
        random.seed(7)
        dead = ShipState()
        dead.player = 1
        dead.current_hp = 0
        dead.currently_alive = True
        dead.size = [70, 70]
        dead.rotation = 0
        dead.position = [100, 200]
        dead.thrust_active = True
        dead.turn_left_active = False
        dead.turn_right_active = False
        dead.action1_active = False
        dead.action2_active = False
        dead.input_pressed_frames = {}
        dead.newly_pressed_controls = set()
        dead.released_controls = set()
        survivor = ShipState()
        survivor.player = 2
        survivor.current_hp = 5
        survivor.currently_alive = True

        aftermath = start_or_update_aftermath(
            None, [dead], dead, survivor, [dead, survivor], 30, sound_enabled=False
        )

        self.assertFalse(dead.currently_alive)
        self.assertEqual(aftermath["dead_players"], {1})
        self.assertEqual(aftermath_camera_targets(aftermath, dead, survivor, 30), [survivor, dead])
        self.assertFalse(aftermath_ready_for_selection(aftermath, 30, sound_enabled=False))
        self.assertTrue(
            aftermath_ready_for_selection(
                aftermath, 30 + const.POST_DEATH_CONTROL_FRAMES, sound_enabled=False
            )
        )


if __name__ == "__main__":
    unittest.main()
