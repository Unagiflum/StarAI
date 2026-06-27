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
from src.Battle.battle import BattleSimulation, reset_round_objects
from src.Battle.battle_aftermath import (
    AftermathState,
    ScheduledExplosion,
    aftermath_camera_targets,
    aftermath_ready_for_selection,
    start_or_update_aftermath,
    update_aftermath,
)
from src.Battle.battle_draw import calculate_view_parameters
from src.Battle.battle_init import validate_ship_positions
from src.Objects.object import PlayerObject
from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.ability import Ability, wrapped_endpoint
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
from src.Objects.Ships.action_transaction import ActionPlan
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

        self.assertEqual(player.get_gravity(), [4.0, 0.0])



class InputTimingTests(unittest.TestCase):
    class TimingShip(SpaceShip):
        def _observed_action(self, action_number):
            return ActionPlan(
                action_number=action_number,
                valid=True,
                side_effects=(
                    lambda: self.observed_calls.append(f"action{action_number}"),
                ),
            )

        def plan_action1(self):
            return self._observed_action(1)

        def plan_action2(self):
            return self._observed_action(2)

        def plan_action3(self):
            return self._observed_action(3)

        def handles_combined_action(self):
            return self.combined_action_handled

        def perform_action1_release(self):
            self.observed_calls.append("release")

    @staticmethod
    def make_ship(combined_action_handled=False):
        ship = InputTimingTests.TimingShip.__new__(InputTimingTests.TimingShip)
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
        ship.current_hp = 1
        ship.max_energy = 0
        ship.energy_regen = 0
        ship.inertia = True
        ship.observed_calls = []
        ship.combined_action_handled = combined_action_handled
        ship.camera_freeze_timer = 0
        ship.frozen_camera_position = [0.0, 0.0]
        return ship

    def test_timers_advance_once_during_frame_processing(self):
        ship = self.make_ship()

        ship.process_controls(frame_id=10)

        self.assertEqual(ship.thrust_timer, 1)
        self.assertEqual(ship.action1_timer, 1)

    def test_new_action_press_is_immediate_and_not_repeated_same_frame(self):
        ship = self.make_ship()
        ship.action1_timer = 0

        ship.set_control_state("action1", True, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(ship.observed_calls, ["action1"])

    def test_held_action_observes_repeat_delay(self):
        ship = self.make_ship()
        ship.action1_timer = 0

        ship.set_control_state("action1", True, frame_id=10)
        ship.process_controls(frame_id=10)
        ship.process_controls(frame_id=11)
        ship.process_controls(frame_id=12)
        ship.process_controls(frame_id=13)

        self.assertEqual(ship.observed_calls, ["action1", "action1"])

    def test_action_release_hook_runs_once(self):
        ship = self.make_ship()
        ship.action1_timer = 0

        ship.set_control_state("action1", True, frame_id=10)
        ship.process_controls(frame_id=10)
        ship.set_control_state("action1", False, frame_id=11)
        ship.process_controls(frame_id=11)

        self.assertEqual(ship.observed_calls, ["action1", "release"])

    def test_press_and_release_in_same_frame_keeps_both_edges(self):
        ship = self.make_ship()
        ship.action1_timer = 0

        ship.set_control_state("action1", True, frame_id=10)
        ship.set_control_state("action1", False, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(ship.observed_calls, ["action1", "release"])

    def test_invalid_combined_action_falls_back_to_both_actions(self):
        ship = self.make_ship()
        ship.action1_timer = 0
        ship.action2_timer = 0

        ship.set_control_state("action1", True, frame_id=10)
        ship.set_control_state("action2", True, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(
            ship.observed_calls,
            ["action3", "action1", "action2"],
        )

    def test_valid_combined_action_runs_without_individual_actions(self):
        ship = self.make_ship(combined_action_handled=True)
        ship.action1_timer = 0
        ship.action2_timer = 0

        ship.set_control_state("action1", True, frame_id=10)
        ship.set_control_state("action2", True, frame_id=10)
        ship.process_controls(frame_id=10)

        self.assertEqual(ship.observed_calls, ["action3"])


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


class ObjectLifecycleCharacterizationTests(unittest.TestCase):
    class UpdatingObject:
        def __init__(self, name, events, survives=True, spawned=None):
            self.name = name
            self.events = events
            self.survives = survives
            self.spawned = list(spawned or [])

        def update(self):
            self.events.append(self.name)
            return self.survives

        def drain_spawned_objects(self):
            spawned, self.spawned = self.spawned, []
            return spawned

    def test_updates_use_a_snapshot_and_append_spawned_objects_in_source_order(self):
        events = []
        first_spawn = self.UpdatingObject("first spawn", events)
        second_spawn = self.UpdatingObject("second spawn", events)
        first = self.UpdatingObject("first", events, spawned=[first_spawn])
        removed = self.UpdatingObject("removed", events, survives=False)
        second = self.UpdatingObject("second", events, spawned=[second_spawn])
        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.game_objects = [first, removed, second]

        simulation._update_objects()

        self.assertEqual(events, ["first", "removed", "second"])
        self.assertEqual(
            simulation.game_objects,
            [first, second, first_spawn, second_spawn],
        )

    def test_state_exposes_the_authoritative_game_objects_list(self):
        simulation = SimulationInputTests.make_simulation()
        authoritative = simulation.game_objects

        state = simulation.state()

        self.assertIs(state["game_objects"], authoritative)


    def test_round_reset_preserves_group_and_intra_group_order(self):
        retained_ship = SpaceShip.__new__(SpaceShip)
        retained_ship.name = "test"
        retained_ship.battles_fought = 0
        retained_ship.currently_alive = True
        retained_ship.current_hp = 1
        replaced_ship = SpaceShip.__new__(SpaceShip)
        replaced_ship.name = "test"
        replaced_ship.battles_fought = 0
        replacement_ship = SpaceShip.__new__(SpaceShip)
        replacement_ship.name = "test"
        replacement_ship.battles_fought = 0
        preserved_ability = Ability.__new__(Ability)
        preserved_ability.parent = retained_ship
        preserved_ability.currently_alive = True
        preserved_ability.current_hp = 1
        discarded_ability = Ability.__new__(Ability)
        discarded_ability.parent = replaced_ship
        discarded_ability.currently_alive = True
        discarded_ability.current_hp = 1
        first_persistent = object()
        second_persistent = object()
        game_objects = [
            retained_ship,
            first_persistent,
            preserved_ability,
            discarded_ability,
            second_persistent,
            replaced_ship,
        ]

        with (
            mock.patch("src.Battle.battle.initialize_new_round_ships"),
            mock.patch("src.Battle.battle.update_preserved_abilities"),
        ):
            reset_round_objects(
                game_objects,
                retained_ship,
                replacement_ship,
                retained_ship,
                replaced_ship,
            )

        self.assertEqual(
            game_objects,
            [
                first_persistent,
                second_persistent,
                preserved_ability,
                retained_ship,
                replacement_ship,
            ],
        )

    def test_final_aftermath_effect_appends_after_survivors_when_ship_is_hidden(self):
        dead_ship = ShipState()
        dead_ship.player = 1
        dead_ship.currently_alive = False
        dead_ship.current_hp = 0
        survivor = ShipState()
        survivor.player = 2
        survivor.currently_alive = True
        survivor.current_hp = 1
        first = object()
        last = object()
        effect = object()
        game_objects = [first, dead_ship, last]
        aftermath = AftermathState(
            started_frame=10,
            latest_death_frame=10,
            pending_explosions=[ScheduledExplosion(
                frame=10,
                ship=dead_ship,
                position=[1, 2],
                scale=1.0,
                is_final=True,
            )],
            death_effects={1: []},
            ships_pending_hide={dead_ship},
        )

        with mock.patch(
            "src.Battle.battle_aftermath.BattleEffect.ship_explosion",
            return_value=effect,
        ):
            update_aftermath(
                aftermath,
                dead_ship,
                survivor,
                game_objects,
                10,
                sound_enabled=False,
            )

        self.assertEqual(game_objects, [first, last, effect])
        self.assertEqual(aftermath.pending_explosions, [])
        self.assertEqual(aftermath.ships_pending_hide, set())


class AftermathCharacterizationTests(unittest.TestCase):
    @staticmethod
    def make_ship(player, hp=5, alive=True, size=(70, 70)):
        ship = ShipState()
        ship.player = player
        ship.current_hp = hp
        ship.currently_alive = alive
        ship.size = list(size)
        ship.rotation = 0
        ship.position = [100 * player, 200]
        ship.thrust_active = False
        ship.turn_left_active = False
        ship.turn_right_active = False
        ship.action1_active = False
        ship.action2_active = False
        ship.input_pressed_frames = {}
        ship.newly_pressed_controls = set()
        ship.released_controls = set()
        return ship

    def test_death_initializes_aftermath_and_holds_camera(self):
        random.seed(7)
        dead = self.make_ship(1, hp=0)
        dead.thrust_active = True
        survivor = self.make_ship(2)

        aftermath = start_or_update_aftermath(
            None, [dead], dead, survivor, [dead, survivor], 30, sound_enabled=False
        )

        self.assertIsInstance(aftermath, AftermathState)
        self.assertFalse(dead.currently_alive)
        self.assertFalse(dead.thrust_active)
        self.assertEqual(aftermath.dead_players, {1})
        self.assertEqual(aftermath.started_frame, 30)
        self.assertIsInstance(aftermath.pending_explosions[0], ScheduledExplosion)
        self.assertEqual(aftermath.pending_explosions[0].frame, 30)
        self.assertEqual(aftermath_camera_targets(aftermath, dead, survivor, 30), [survivor, dead])
        self.assertFalse(aftermath_ready_for_selection(aftermath, 30, sound_enabled=False))
        conclusion_frame = aftermath.death_sequence_ready_frame
        update_aftermath(
            aftermath,
            dead,
            survivor,
            [dead, survivor],
            conclusion_frame,
            sound_enabled=False,
        )
        self.assertTrue(
            aftermath_ready_for_selection(
                aftermath,
                conclusion_frame + const.VICTORY_DITTY_VIEW_FRAMES,
                sound_enabled=False,
            )
        )

    def test_death_releases_tracking_projectiles_and_notifies_fighters(self):
        dead = self.make_ship(1, hp=0)
        dead.trackable = True
        survivor = self.make_ship(2)
        survivor.opponent = dead

        projectile = Ability.__new__(Ability)
        projectile.opponent = dead
        projectile.omnidirectional = False
        projectile.heading = 4
        projectile.tracking = True
        projectile.rotation = 90
        projectile.velocity = [12, 0]

        special_object = KzerZaA2.__new__(KzerZaA2)
        special_object.opponent = dead
        special_object.parent = survivor
        special_object.mode = special_object.ATTACKING

        start_or_update_aftermath(
            None,
            [dead],
            dead,
            survivor,
            [dead, survivor, projectile, special_object],
            30,
            sound_enabled=False,
        )
        projectile.update_heading()

        self.assertIsNone(survivor.opponent)
        self.assertIsNone(projectile.opponent)
        self.assertEqual(projectile.rotation, 90)
        self.assertEqual(projectile.velocity, [12, 0])
        self.assertIsNone(special_object.opponent)
        self.assertEqual(special_object.mode, special_object.RETURNING)

    def test_simultaneous_deaths_are_registered_once_in_player_order(self):
        random.seed(11)
        first = self.make_ship(1, hp=0)
        second = self.make_ship(2, hp=0)
        first.shofixti_self_destruct = True
        simulation = BattleSimulation.__new__(BattleSimulation)
        simulation.player1 = first
        simulation.player2 = second
        simulation.game_objects = [first, second]
        simulation.frame_id = 40
        simulation.aftermath = None
        simulation.sound_enabled = False
        simulation.needs_selection = False

        with (
            mock.patch(
                "src.Battle.battle_aftermath.start_or_update_aftermath",
                wraps=start_or_update_aftermath,
            ) as register_deaths,
            mock.patch(
                "src.Battle.battle_aftermath.BattleEffect.ship_explosion",
                side_effect=[object(), object()],
            ),
        ):
            simulation._update_aftermath()

        register_deaths.assert_called_once()
        self.assertEqual(register_deaths.call_args.args[1], [first, second])
        self.assertEqual(simulation.aftermath.dead_players, {1, 2})
        self.assertEqual(simulation.aftermath.camera_hold_targets, [first, second])
        self.assertIs(simulation.aftermath.tie_break_ship, first)
        self.assertEqual(simulation.aftermath.choose_second_player, 1)
        self.assertEqual(
            [effect.ship for effect in simulation.aftermath.pending_explosions],
            [first] * 4 + [second] * 4,
        )

    def test_scheduled_explosions_keep_frame_and_object_order(self):
        random.seed(13)
        dead = self.make_ship(1, hp=0)
        survivor = self.make_ship(2)
        game_objects = [dead, survivor]
        aftermath = start_or_update_aftermath(
            None, [dead], dead, survivor, game_objects, 10, sound_enabled=False
        )

        self.assertEqual(
            [item.frame for item in aftermath.pending_explosions],
            [10, 13, 16, 19, 22],
        )
        effects = [object() for _ in range(5)]
        with mock.patch(
            "src.Battle.battle_aftermath.BattleEffect.ship_explosion",
            side_effect=effects,
        ):
            update_aftermath(
                aftermath, dead, survivor, game_objects, 16, sound_enabled=False
            )
            self.assertEqual(game_objects, [dead, survivor, *effects[:3]])
            self.assertEqual(
                [item.frame for item in aftermath.pending_explosions], [19, 22]
            )
            update_aftermath(
                aftermath, dead, survivor, game_objects, 22, sound_enabled=False
            )

        self.assertEqual(game_objects, [survivor, *effects])
        self.assertEqual(aftermath.death_effects[1], effects)
        self.assertEqual(aftermath.pending_explosions, [])
        self.assertEqual(aftermath.ships_pending_hide, set())

    def test_camera_releases_on_animation_view_boundary(self):
        dead = self.make_ship(1, hp=0)
        survivor = self.make_ship(2)
        aftermath = start_or_update_aftermath(
            None, [dead], dead, survivor, [dead, survivor], 20, sound_enabled=False
        )

        release_frame = aftermath.death_sequence_ready_frame
        self.assertEqual(
            aftermath_camera_targets(
                aftermath,
                dead,
                survivor,
                release_frame - 1,
            ),
            [survivor, dead],
        )
        self.assertIsNone(
            aftermath_camera_targets(
                aftermath,
                dead,
                survivor,
                release_frame,
            )
        )

    def test_victory_audio_starts_once_at_view_boundary_and_honors_sound_setting(self):
        dead = self.make_ship(1, hp=0, alive=False)
        survivor = self.make_ship(2)
        start_frame = 50
        conclusion_frame = start_frame + const.POST_DEATH_EFFECT_FRAMES
        aftermath = AftermathState(
            start_frame,
            start_frame,
            death_sequence_ready_frame=conclusion_frame,
        )

        with mock.patch("src.Battle.battle_aftermath.play_victory_ditty") as play_ditty:
            update_aftermath(
                aftermath,
                dead,
                survivor,
                [],
                conclusion_frame - 1,
                sound_enabled=True,
            )
            play_ditty.assert_not_called()
            update_aftermath(
                aftermath,
                dead,
                survivor,
                [],
                conclusion_frame,
                sound_enabled=True,
            )
            update_aftermath(
                aftermath,
                dead,
                survivor,
                [],
                conclusion_frame + 1,
                sound_enabled=True,
            )
            play_ditty.assert_called_once_with(survivor)

        muted_aftermath = AftermathState(
            start_frame,
            start_frame,
            death_sequence_ready_frame=conclusion_frame,
        )
        with mock.patch("src.Battle.battle_aftermath.play_victory_ditty") as play_ditty:
            update_aftermath(
                muted_aftermath,
                dead,
                survivor,
                [],
                conclusion_frame,
                sound_enabled=False,
            )
        play_ditty.assert_not_called()
        self.assertTrue(muted_aftermath.ditty_started)

        other_dead = self.make_ship(2, hp=0, alive=False)
        tie_aftermath = AftermathState(
            start_frame,
            start_frame,
            tie_break_ship=dead,
            choose_second_player=dead.player,
            death_sequence_ready_frame=conclusion_frame,
        )
        with mock.patch("src.Battle.battle_aftermath.play_victory_ditty") as play_ditty:
            update_aftermath(
                tie_aftermath,
                dead,
                other_dead,
                [],
                conclusion_frame,
                sound_enabled=True,
            )
        play_ditty.assert_called_once_with(dead)

    def test_selection_becomes_ready_on_exact_control_boundary(self):
        aftermath = AftermathState(
            started_frame=20,
            latest_death_frame=20,
            conclusion_started_frame=75,
        )

        self.assertFalse(
            aftermath_ready_for_selection(
                aftermath,
                75 + const.VICTORY_DITTY_VIEW_FRAMES - 1,
                sound_enabled=True,
            )
        )
        self.assertTrue(
            aftermath_ready_for_selection(
                aftermath,
                75 + const.VICTORY_DITTY_VIEW_FRAMES,
                sound_enabled=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
