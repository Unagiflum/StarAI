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
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    FighterCollisionCapabilities,
    LaserTargetCapabilities,
    ShipImpactContext,
    ShipImpactResult,
)
from src.Battle.battle import (
    AftermathState,
    BattleSimulation,
    ScheduledExplosion,
    aftermath_camera_targets,
    aftermath_ready_for_selection,
    reset_round_objects,
    start_or_update_aftermath,
    update_aftermath,
)
from src.Battle.battle_draw import calculate_view_parameters
from src.Battle.battle_init import validate_ship_positions
from src.Objects.object import PlayerObject
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability, wrapped_endpoint
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.Druuge.A1.DruugeA1 import DruugeA1
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
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

    def test_collision_cleanup_preserves_survivor_order_and_list_identity(self):
        first = object()
        live_ability = Ability.__new__(Ability)
        live_ability.currently_alive = True
        dead_ability = Ability.__new__(Ability)
        dead_ability.currently_alive = False
        live_asteroid = Asteroid.__new__(Asteroid)
        live_asteroid.currently_alive = True
        dead_asteroid = Asteroid.__new__(Asteroid)
        dead_asteroid.currently_alive = False
        last = object()
        game_objects = [
            first,
            dead_ability,
            live_ability,
            dead_asteroid,
            live_asteroid,
            last,
        ]
        authoritative = game_objects

        collisions._remove_dead_collision_objects(game_objects)

        self.assertIs(game_objects, authoritative)
        self.assertEqual(game_objects, [first, live_ability, live_asteroid, last])

    def test_round_reset_preserves_group_and_intra_group_order(self):
        retained_ship = SpaceShip.__new__(SpaceShip)
        retained_ship.currently_alive = True
        retained_ship.current_hp = 1
        replaced_ship = SpaceShip.__new__(SpaceShip)
        replacement_ship = SpaceShip.__new__(SpaceShip)
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
        aftermath = {
            "started_frame": 10,
            "pending_explosions": [{
                "frame": 10,
                "ship": dead_ship,
                "position": [1, 2],
                "scale": 1.0,
                "is_final": True,
            }],
            "death_effects": {1: []},
            "ships_pending_hide": {dead_ship},
            "ditty_started": False,
            "tie_break_ship": None,
        }

        with mock.patch(
            "src.Battle.battle.BattleEffect.ship_explosion",
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
        self.assertEqual(aftermath["pending_explosions"], [])
        self.assertEqual(aftermath["ships_pending_hide"], set())


class CollisionCharacterizationTests(unittest.TestCase):
    @staticmethod
    def make_ship():
        ship = SpaceShip.__new__(SpaceShip)
        ship.name = "Target"
        ship.player = 2
        ship.position = [5, 100]
        ship.previous_position = ship.position.copy()
        ship.size = [20, 20]
        ship.velocity = [0.0, 0.0]
        ship.mass = 1.0
        ship.can_move = True
        ship.inertia = True
        ship.collision_velocity = [0.0, 0.0]
        ship.planet_contacts = set()
        ship.current_hp = 10
        ship.currently_alive = True
        ship.collision_capabilities = CollisionCapabilities(CollisionRole.SHIP)
        ship.laser_target_capabilities = LaserTargetCapabilities()
        ship.area_damage_capabilities = AreaDamageCapabilities(targetable=True)
        return ship

    @staticmethod
    def make_projectile(parent, projectile_class=Ability):
        projectile = projectile_class.__new__(projectile_class)
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
        projectile.collision_capabilities = CollisionCapabilities(
            CollisionRole.PROJECTILE
        )
        projectile.laser_target_capabilities = LaserTargetCapabilities()
        projectile.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        return projectile

    def make_projectile_pair(
        self,
        *,
        first_name="FirstProjectile",
        second_name="SecondProjectile",
        first_player=1,
        second_player=2,
        first_hp=1,
        second_hp=1,
        first_damage=4,
        second_damage=4,
    ):
        first_parent = self.make_ship()
        first_parent.player = first_player
        second_parent = self.make_ship()
        second_parent.player = second_player
        first = self.make_projectile(first_parent)
        second = self.make_projectile(second_parent)

        first.name = first.projectile_name = first_name
        second.name = second.projectile_name = second_name
        first.player = first_player
        second.player = second_player
        first.current_hp = first_hp
        second.current_hp = second_hp
        first.hp_array = [first_hp]
        second.hp_array = [second_hp]
        first.current_damage = first_damage
        second.current_damage = second_damage
        first.position = [100, 100]
        second.position = [108, 100]
        first.previous_position = first.position.copy()
        second.previous_position = second.position.copy()
        return first, second

    @staticmethod
    def make_fighter(
        *,
        collides_with_planets=True,
        collides_with_asteroids=True,
        damages_asteroids=True,
        collides_with_projectiles=True,
        damages_projectiles=True,
        collides_with_enemy_ships=True,
        collides_with_friendly_ships=False,
        collides_with_fighters=True,
        laser_vulnerable=True,
        fighter_class=Ability,
    ):
        fighter = fighter_class.__new__(fighter_class)
        fighter.name = "TestFighter"
        fighter.projectile_name = "TestFighter"
        fighter.type = "fighter"
        fighter.player = 1
        fighter.position = [100, 100]
        fighter.previous_position = fighter.position.copy()
        fighter.size = [20, 20]
        fighter.masks = None
        fighter.heading = 0
        fighter.frames = 1
        fighter.can_move = True
        fighter.can_collide = True
        fighter.currently_alive = True
        fighter.current_hp = 1
        fighter.current_damage = 1
        fighter.death_animation = []
        fighter.velocity = [1.0, 0.0]
        fighter.collision_capabilities = CollisionCapabilities(
            CollisionRole.FIGHTER
        )
        fighter.laser_target_capabilities = LaserTargetCapabilities(
            vulnerable=laser_vulnerable
        )
        fighter.fighter_collision_capabilities = FighterCollisionCapabilities(
            collides_with_planets=collides_with_planets,
            collides_with_asteroids=collides_with_asteroids,
            damages_asteroids=damages_asteroids,
            collides_with_projectiles=collides_with_projectiles,
            damages_projectiles=damages_projectiles,
            collides_with_enemy_ships=collides_with_enemy_ships,
            collides_with_friendly_ships=collides_with_friendly_ships,
            collides_with_fighters=collides_with_fighters,
        )
        fighter.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        return fighter

    @staticmethod
    def make_laser(parent, *, hit_parent=False, hit_self=False, target=None):
        laser = Ability.__new__(Ability)
        laser.name = "TestLaser"
        laser.projectile_name = "TestLaser"
        laser.type = "laser"
        laser.player = parent.player
        laser.parent = parent
        laser.position = parent.position.copy()
        laser.previous_position = laser.position.copy()
        laser.start_position = laser.position.copy()
        laser.end_position = [laser.position[0] + 300, laser.position[1]]
        laser.size = [1, 1]
        laser.masks = None
        laser.heading = 0
        laser.frames = 1
        laser.can_collide = True
        laser.currently_alive = True
        laser.current_hp = 1
        laser.current_damage = 2
        laser.hit_parent = hit_parent
        laser.hit_self = hit_self
        laser.target = target
        laser.collision_capabilities = CollisionCapabilities(CollisionRole.LASER)
        laser.laser_target_capabilities = LaserTargetCapabilities(targetable=False)
        laser.area_damage_capabilities = AreaDamageCapabilities()
        return laser

    @staticmethod
    def make_planet(position):
        planet = Planet.__new__(Planet)
        planet.position = list(position)
        planet.previous_position = planet.position.copy()
        planet.diameter = 20
        planet.size = [20, 20]
        planet.mask = None
        planet.can_move = False
        planet.collision_capabilities = CollisionCapabilities(CollisionRole.PLANET)
        planet.laser_target_capabilities = LaserTargetCapabilities()
        planet.area_damage_capabilities = AreaDamageCapabilities()
        return planet

    @staticmethod
    def make_asteroid(position):
        asteroid = Asteroid.__new__(Asteroid)
        asteroid.position = list(position)
        asteroid.previous_position = asteroid.position.copy()
        asteroid.size = [20, 20]
        asteroid.masks = [None]
        asteroid.current_sprite = 0
        asteroid.currently_alive = True
        asteroid.death_animation = []
        asteroid.can_move = True
        asteroid.can_collide = True
        asteroid.velocity = [0.0, 0.0]
        asteroid.collision_capabilities = CollisionCapabilities(
            CollisionRole.ASTEROID
        )
        asteroid.laser_target_capabilities = LaserTargetCapabilities()
        asteroid.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        return asteroid

    @staticmethod
    def make_area_damage(position, damage_at_distance):
        ability = Ability.__new__(Ability)
        ability.name = "TestAreaDamage"
        ability.type = "area"
        ability.position = list(position)
        ability.previous_position = ability.position.copy()
        ability.currently_alive = True
        ability.area_damage_pending = True
        ability.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=True,
            vulnerable=False,
        )
        ability.current_hp = 100
        ability.damage_at_distance = damage_at_distance
        ability.collision_capabilities = CollisionCapabilities(
            CollisionRole.NONE
        )
        return ability

    def test_area_damage_is_consumed_once_and_uses_wrapped_distance(self):
        area = self.make_area_damage(
            [const.ARENA_SIZE - 5, 100],
            lambda distance: 4 if distance == 10 else 0,
        )
        ship = self.make_ship()
        ship.position = [5, 100]

        collisions._handle_area_damage([area, ship], [])
        collisions._handle_area_damage([area, ship], [])

        self.assertFalse(area.area_damage_pending)
        self.assertEqual(ship.current_hp, 6)

    def test_area_damage_excludes_planets_lasers_dead_and_invulnerable_targets(self):
        area = self.make_area_damage([100, 100], lambda distance: 5)
        planet = self.make_planet([100, 100])
        parent = self.make_ship()
        laser = self.make_laser(parent)
        laser.position = [100, 100]
        dead_projectile = self.make_projectile(parent)
        dead_projectile.position = [100, 100]
        dead_projectile.currently_alive = False
        invulnerable_ship = self.make_ship()
        invulnerable_ship.position = [100, 100]
        invulnerable_ship.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True,
            vulnerable=False,
        )
        unrelated = SimpleNamespace(
            position=[100, 100],
            current_hp=10,
            currently_alive=True,
            area_damage_capabilities=AreaDamageCapabilities(),
            collision_capabilities=CollisionCapabilities(),
        )

        collisions._handle_area_damage(
            [
                area,
                planet,
                laser,
                dead_projectile,
                invulnerable_ship,
                unrelated,
            ],
            [],
        )

        self.assertEqual(laser.current_hp, 1)
        self.assertEqual(dead_projectile.current_hp, 1)
        self.assertEqual(invulnerable_ship.current_hp, 10)
        self.assertEqual(unrelated.current_hp, 10)

    def test_area_damage_clamps_ship_hp_to_zero(self):
        area = self.make_area_damage([100, 100], lambda distance: 12)
        ship = self.make_ship()
        ship.position = [100, 100]

        collisions._handle_area_damage([area, ship], [])

        self.assertEqual(ship.current_hp, 0)

    def test_area_damage_uses_projectile_hp_hook_when_target_survives(self):
        area = self.make_area_damage([100, 100], lambda distance: 2)
        projectile = self.make_projectile(self.make_ship())
        projectile.position = [110, 100]
        projectile.current_hp = 5
        projectile.set_hp = mock.Mock()

        collisions._handle_area_damage([area, projectile], [])

        projectile.set_hp.assert_called_once_with(3)
        self.assertTrue(projectile.currently_alive)

    def test_area_damage_destroys_ability_with_outward_effect_direction(self):
        area = self.make_area_damage([100, 100], lambda distance: 2)
        fighter = self.make_fighter()
        fighter.position = [100, 110]
        blast = object()

        with mock.patch.object(
            collisions.BattleEffect, "from_blast", return_value=blast
        ) as from_blast:
            effects = []
            collisions._handle_area_damage([area, fighter], effects)

        self.assertFalse(fighter.currently_alive)
        self.assertEqual(fighter.current_hp, 0)
        self.assertEqual(effects, [blast])
        from_blast.assert_called_once_with(
            fighter.position, [0.0, 1.0], 2, align_edge=False
        )

    def test_area_damage_destroys_asteroid_without_hp_state(self):
        area = self.make_area_damage([100, 100], lambda distance: 1)
        asteroid = self.make_asteroid([120, 100])

        collisions._handle_area_damage([area, asteroid], [])

        self.assertFalse(asteroid.currently_alive)

    def test_equal_mass_ships_exchange_velocity_and_separate(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [115, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]

        collisions._handle_ship_ship_collisions([first, second])

        self.assertEqual(first.velocity, [-1.0, 0.0])
        self.assertEqual(second.velocity, [1.0, 0.0])
        self.assertEqual(first.position, [97.5, 100.0])
        self.assertEqual(second.position, [117.5, 100.0])
        self.assertEqual((first.current_hp, second.current_hp), (10, 10))

    def test_unequal_mass_ship_bounce_uses_mass_weighted_separation(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [115, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]
        first.mass = 1.0
        second.mass = 3.0

        collisions._handle_ship_ship_collisions([first, second])

        self.assertEqual(first.velocity, [-2.0, 0.0])
        self.assertEqual(second.velocity, [0.0, 0.0])
        self.assertEqual(first.position, [96.25, 100.0])
        self.assertEqual(second.position, [116.25, 100.0])

    def test_ship_ship_bounce_uses_wrapped_boundary_geometry(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [5, 100]
        second.position = [const.ARENA_SIZE - 5, 100]
        first.velocity = [-1.0, 0.0]
        second.velocity = [1.0, 0.0]

        collisions._handle_ship_ship_collisions([first, second])

        self.assertEqual(first.velocity, [1.0, 0.0])
        self.assertEqual(second.velocity, [-1.0, 0.0])
        self.assertEqual(first.position, [10.0, 100.0])
        self.assertEqual(second.position, [const.ARENA_SIZE - 10.0, 100.0])

    def test_non_overlapping_ships_keep_velocity_and_position(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [200, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]

        collisions._handle_ship_ship_collisions([first, second])

        self.assertEqual(first.velocity, [1.0, 0.0])
        self.assertEqual(second.velocity, [-1.0, 0.0])
        self.assertEqual(first.position, [100, 100])
        self.assertEqual(second.position, [200, 100])

    def test_ship_impact_hook_can_add_damage_without_changing_dispatch(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [115, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]
        first.on_ship_impact = mock.Mock(
            return_value=ShipImpactResult(damage_to_other=3)
        )

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_ship_ship_collisions([first, second], [])

        self.assertEqual(first.current_hp, 10)
        self.assertEqual(second.current_hp, 7)
        play_boom.assert_called_once_with(3)
        impact = first.on_ship_impact.call_args.args[1]
        self.assertIsInstance(impact, ShipImpactContext)
        self.assertEqual(impact.normal, (-1.0, 0.0))
        self.assertEqual(impact.distance, 15.0)
        self.assertEqual(impact.overlap, 5.0)
        self.assertEqual(impact.closing_speed, 2.0)

    def test_ship_asteroid_bounce_uses_asteroid_radius_mass_fallback(self):
        ship = self.make_ship()
        asteroid = self.make_asteroid([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        ship.mass = 1.0
        asteroid.velocity = [-1.0, 0.0]

        collisions._handle_ship_asteroid_collisions([ship], [asteroid])

        self.assertAlmostEqual(ship.velocity[0], 1 / 3)
        self.assertAlmostEqual(asteroid.velocity[0], 7 / 3)
        self.assertAlmostEqual(ship.position[0], 100 - 5 / 6)
        self.assertAlmostEqual(asteroid.position[0], 115 + 25 / 6)
        self.assertEqual(ship.current_hp, 10)

    def test_dead_asteroid_does_not_collide_with_ship(self):
        ship = self.make_ship()
        asteroid = self.make_asteroid([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        asteroid.velocity = [-1.0, 0.0]
        asteroid.currently_alive = False

        collisions._handle_ship_asteroid_collisions([ship], [asteroid])

        self.assertEqual(ship.velocity, [1.0, 0.0])
        self.assertEqual(asteroid.velocity, [-1.0, 0.0])
        self.assertEqual(ship.position, [100, 100])
        self.assertEqual(asteroid.position, [115, 100])

    def test_first_approaching_planet_contact_bounces_and_damages_ship(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(ship.position, [95.0, 100.0])
        self.assertEqual(ship.current_hp, 8)
        self.assertEqual(ship.planet_contacts, {id(planet)})
        play_boom.assert_called_once_with(2)

    def test_persistent_planet_contact_stops_ship_without_repeated_damage(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_ship_planet_collisions([ship], [planet])
            ship.position = [100, 100]
            ship.velocity = [1.0, 0.0]
            collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.velocity, [0.0, 0.0])
        self.assertEqual(ship.position, [95.0, 100.0])
        self.assertEqual(ship.current_hp, 8)
        play_boom.assert_called_once_with(2)

    def test_separating_first_planet_contact_does_not_damage_ship(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [-1.0, 0.0]

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(ship.current_hp, 10)
        self.assertEqual(ship.planet_contacts, {id(planet)})
        play_boom.assert_not_called()

    def test_planet_contact_clears_only_beyond_exit_margin(self):
        ship = self.make_ship()
        planet = self.make_planet([100, 100])
        ship.planet_contacts.add(id(planet))
        ship.position = [123, 100]

        collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.planet_contacts, {id(planet)})

        ship.position = [125, 100]
        collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.planet_contacts, set())

    def test_planet_contact_uses_wrapped_geometry(self):
        ship = self.make_ship()
        planet = self.make_planet([const.ARENA_SIZE - 5, 100])
        ship.position = [5, 100]
        ship.velocity = [-1.0, 0.0]

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.velocity, [1.0, 0.0])
        self.assertEqual(ship.position, [15.0, 100.0])
        self.assertEqual(ship.current_hp, 8)

    def test_planet_contact_respects_collision_masks(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        ship_mask = pygame.mask.Mask((20, 20), fill=False)
        planet.mask = pygame.mask.Mask((20, 20), fill=True)
        ship.get_collision_mask = lambda: ship_mask

        collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.velocity, [1.0, 0.0])
        self.assertEqual(ship.current_hp, 10)
        self.assertEqual(ship.planet_contacts, set())

    def test_non_inertial_ship_keeps_planet_bounce_collision_velocity(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        ship.inertia = False

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_ship_planet_collisions([ship], [planet])

        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(ship.collision_velocity, [-1.0, 0.0])

    def test_asteroid_planet_contact_destroys_asteroid_with_animation(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([108, 100])
        asteroid.death_animation = [object()]
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_animation",
                return_value=sentinel_effect,
            ) as from_animation,
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_asteroid_planet_collisions(
                [asteroid], [planet], [], effects
            )

        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        from_animation.assert_called_once_with(
            asteroid.position, asteroid.death_animation
        )
        play_boom.assert_called_once_with(1)

    def test_offscreen_asteroid_planet_contact_is_silent(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([108, 100])

        with (
            mock.patch.object(collisions, "_object_on_screen", return_value=False),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_asteroid_planet_collisions(
                [asteroid], [planet], [self.make_ship(), self.make_ship()], []
            )

        self.assertFalse(asteroid.currently_alive)
        play_boom.assert_not_called()

    def test_asteroid_planet_contact_uses_wrapped_geometry(self):
        asteroid = self.make_asteroid([5, 100])
        planet = self.make_planet([const.ARENA_SIZE - 5, 100])

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_asteroid_planet_collisions(
                [asteroid], [planet], [], []
            )

        self.assertFalse(asteroid.currently_alive)

    def test_dead_asteroid_is_ignored_by_planet_collision(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([108, 100])
        asteroid.currently_alive = False

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_asteroid_planet_collisions(
                [asteroid], [planet], [], []
            )

        play_boom.assert_not_called()

    def test_dead_asteroids_are_replaced_with_incremental_avoidance(self):
        class ReplacementAsteroid:
            created = []

            def __init__(self):
                self.currently_alive = True
                self.can_collide = True
                self.planet = None
                self.spawn_args = None
                self.position = None
                self.created.append(self)

            def set_planet(self, planet):
                self.planet = planet

            def get_respawn_position(self, planet, ships, avoid_bodies):
                self.spawn_args = (planet, tuple(ships), tuple(avoid_bodies))
                return [300 + len(self.created), 400]

        planet = self.make_planet([100, 100])
        ship = self.make_ship()
        alive = self.make_asteroid([200, 200])
        first_dead = self.make_asteroid([300, 300])
        second_dead = self.make_asteroid([400, 400])
        first_dead.currently_alive = False
        second_dead.currently_alive = False
        game_objects = [planet, ship, alive, first_dead, second_dead]

        with mock.patch.object(collisions, "Asteroid", ReplacementAsteroid):
            collisions._spawn_replacement_asteroids(
                game_objects,
                [alive, first_dead, second_dead],
                [ship],
                [planet],
            )

        self.assertEqual(len(ReplacementAsteroid.created), 2)
        first, second = ReplacementAsteroid.created
        self.assertIs(first.planet, planet)
        self.assertIs(second.planet, planet)
        self.assertEqual(first.spawn_args[1], (ship,))
        self.assertEqual(first.spawn_args[2], (ship, alive))
        self.assertEqual(second.spawn_args[2], (ship, alive, first))
        self.assertEqual(game_objects[-2:], [first, second])

    def test_asteroids_are_not_replaced_without_a_planet(self):
        asteroid = self.make_asteroid([100, 100])
        asteroid.currently_alive = False
        replacement_factory = mock.Mock()

        with mock.patch.object(collisions, "Asteroid", replacement_factory):
            collisions._spawn_replacement_asteroids(
                [asteroid], [asteroid], [], []
            )

        replacement_factory.assert_not_called()

    def test_fighter_planet_contact_separates_and_begins_avoidance(self):
        fighter = self.make_fighter()
        planet = self.make_planet([115, 100])
        fighter.begin_planet_avoidance = mock.Mock()

        collisions._handle_fighter_planet_collisions([fighter], [planet])

        self.assertEqual(fighter.position, [94.0, 100.0])
        self.assertTrue(fighter.currently_alive)
        fighter.begin_planet_avoidance.assert_called_once_with(
            planet, [-1.0, 0.0]
        )

    def test_fighter_with_planet_collision_disabled_is_ignored(self):
        fighter = self.make_fighter(collides_with_planets=False)
        planet = self.make_planet([115, 100])
        fighter.begin_planet_avoidance = mock.Mock()

        collisions._handle_fighter_planet_collisions([fighter], [planet])

        self.assertEqual(fighter.position, [100, 100])
        fighter.begin_planet_avoidance.assert_not_called()

    def test_fighter_planet_collision_defaults_to_enabled(self):
        fighter = self.make_fighter()
        planet = self.make_planet([115, 100])

        collisions._handle_fighter_planet_collisions([fighter], [planet])

        self.assertEqual(fighter.position, [94.0, 100.0])

    def test_swept_fighter_planet_contact_begins_avoidance(self):
        fighter = self.make_fighter()
        planet = self.make_planet([150, 100])
        fighter.size = [10, 10]
        fighter.previous_position = [100, 100]
        fighter.position = [200, 100]
        fighter.begin_planet_avoidance = mock.Mock()

        collisions._handle_fighter_planet_collisions([fighter], [planet])

        self.assertEqual(fighter.position, [200, 100])
        fighter.begin_planet_avoidance.assert_called_once_with(
            planet, [1.0, 0.0]
        )

    def test_fighter_planet_contact_uses_wrapped_geometry(self):
        fighter = self.make_fighter()
        planet = self.make_planet([const.ARENA_SIZE - 5, 100])
        fighter.position = [5, 100]
        fighter.previous_position = fighter.position.copy()

        collisions._handle_fighter_planet_collisions([fighter], [planet])

        self.assertEqual(fighter.position, [16.0, 100.0])

    def test_fighter_stops_after_first_planet_contact(self):
        fighter = self.make_fighter()
        first = self.make_planet([115, 100])
        second = self.make_planet([85, 100])
        fighter.begin_planet_avoidance = mock.Mock()

        collisions._handle_fighter_planet_collisions(
            [fighter], [first, second]
        )

        fighter.begin_planet_avoidance.assert_called_once()
        self.assertIs(fighter.begin_planet_avoidance.call_args.args[0], first)

    def test_fighter_asteroid_contact_destroys_both_by_default(self):
        fighter = self.make_fighter()
        asteroid = self.make_asteroid([108, 100])
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_fighter_asteroid_collisions(
                [fighter], [asteroid], effects
            )

        self.assertFalse(fighter.currently_alive)
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(1)

    def test_fighter_can_collide_without_damaging_asteroid(self):
        fighter = self.make_fighter(damages_asteroids=False)
        asteroid = self.make_asteroid([108, 100])

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_asteroid_collisions(
                [fighter], [asteroid], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertTrue(asteroid.currently_alive)

    def test_fighter_with_asteroid_collision_disabled_is_ignored(self):
        fighter = self.make_fighter(collides_with_asteroids=False)
        asteroid = self.make_asteroid([108, 100])

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_fighter_asteroid_collisions(
                [fighter], [asteroid], []
            )

        self.assertTrue(fighter.currently_alive)
        self.assertTrue(asteroid.currently_alive)
        play_boom.assert_not_called()

    def test_fighter_ignores_dead_asteroid_and_hits_next_live_target(self):
        fighter = self.make_fighter()
        dead = self.make_asteroid([108, 100])
        live = self.make_asteroid([108, 100])
        dead.currently_alive = False

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_asteroid_collisions(
                [fighter], [dead, live], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_swept_fighter_asteroid_impact_is_not_tunneled(self):
        fighter = self.make_fighter()
        asteroid = self.make_asteroid([150, 100])
        fighter.size = [10, 10]
        asteroid.size = [10, 10]
        fighter.previous_position = [100, 100]
        fighter.position = [200, 100]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_asteroid_collisions(
                [fighter], [asteroid], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertFalse(asteroid.currently_alive)

    def test_fighter_projectile_contact_destroys_both_by_default(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_fighter_projectile_collisions(
                [fighter], [projectile], effects
            )

        self.assertFalse(fighter.currently_alive)
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(effects, [sentinel_effect, sentinel_effect])
        play_boom.assert_called_once_with(1)

    def test_fighter_can_collide_without_damaging_projectile(self):
        fighter = self.make_fighter(damages_projectiles=False)
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_projectile_collisions(
                [fighter], [projectile], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 1)

    def test_fighter_with_projectile_collision_disabled_is_ignored(self):
        fighter = self.make_fighter(collides_with_projectiles=False)
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_fighter_projectile_collisions(
                [fighter], [projectile], []
            )

        self.assertTrue(fighter.currently_alive)
        self.assertTrue(projectile.currently_alive)
        play_boom.assert_not_called()

    def test_projectile_with_remaining_hp_survives_fighter_impact(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()
        projectile.current_hp = 3
        projectile.hp_array = [3]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_projectile_collisions(
                [fighter], [projectile], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 2)

    def test_fighter_ignores_dead_projectile_and_hits_next_live_target(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        dead = self.make_projectile(parent)
        live = self.make_projectile(parent)
        for projectile in (dead, live):
            projectile.position = [108, 100]
            projectile.previous_position = projectile.position.copy()
        dead.currently_alive = False
        dead.current_hp = 0

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_projectile_collisions(
                [fighter], [dead, live], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_swept_fighter_projectile_impact_is_not_tunneled(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        fighter.size = [10, 10]
        projectile.size = [10, 10]
        fighter.previous_position = [100, 100]
        fighter.position = [200, 100]
        projectile.previous_position = [150, 100]
        projectile.position = [150, 100]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_projectile_collisions(
                [fighter], [projectile], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertFalse(projectile.currently_alive)

    def test_fighter_enemy_ship_contact_damages_ship_and_destroys_fighter(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        target = self.make_ship()
        target.player = 2
        target.position = [108, 100]
        fighter.parent = parent
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_fighter_ship_collisions(
                [fighter], [target], effects
            )

        self.assertEqual(target.current_hp, 9)
        self.assertFalse(fighter.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(1)

    def test_fighter_ignores_friendly_ship_by_default(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        friendly = self.make_ship()
        friendly.player = 1
        friendly.position = [108, 100]
        fighter.parent = parent

        collisions._handle_fighter_ship_collisions(
            [fighter], [friendly], []
        )

        self.assertEqual(friendly.current_hp, 10)
        self.assertTrue(fighter.currently_alive)

    def test_fighter_can_collide_with_friendly_ship(self):
        fighter = self.make_fighter(collides_with_friendly_ships=True)
        parent = self.make_ship()
        parent.player = 1
        friendly = self.make_ship()
        friendly.player = 1
        friendly.position = [108, 100]
        fighter.parent = parent

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_ship_collisions(
                [fighter], [friendly], []
            )

        self.assertEqual(friendly.current_hp, 9)
        self.assertFalse(fighter.currently_alive)

    def test_fighter_with_enemy_ship_collision_disabled_is_ignored(self):
        fighter = self.make_fighter(collides_with_enemy_ships=False)
        parent = self.make_ship()
        parent.player = 1
        enemy = self.make_ship()
        enemy.player = 2
        enemy.position = [108, 100]
        fighter.parent = parent

        collisions._handle_fighter_ship_collisions([fighter], [enemy], [])

        self.assertEqual(enemy.current_hp, 10)
        self.assertTrue(fighter.currently_alive)

    def test_fighter_cannot_recover_with_parent_until_returning(self):
        fighter = self.make_fighter(fighter_class=KzerZaA2)
        parent = self.make_ship()
        parent.player = 1
        parent.position = [108, 100]
        parent.max_hp = 10
        parent.current_hp = 5
        fighter.parent = parent
        fighter.mode = fighter.ATTACKING
        fighter.return_sound = None

        collisions._handle_fighter_ship_collisions([fighter], [parent], [])

        self.assertEqual(parent.current_hp, 5)
        self.assertTrue(fighter.currently_alive)

    def test_returning_fighter_recovers_with_parent(self):
        fighter = self.make_fighter(fighter_class=KzerZaA2)
        parent = self.make_ship()
        parent.player = 1
        parent.position = [108, 100]
        parent.max_hp = 10
        parent.current_hp = 5
        fighter.parent = parent
        fighter.mode = fighter.RETURNING
        fighter.return_sound = None

        collisions._handle_fighter_ship_collisions([fighter], [parent], [])

        self.assertEqual(parent.current_hp, 6)
        self.assertFalse(fighter.currently_alive)
        self.assertEqual(fighter.current_hp, 0)

    def test_fighter_skips_dead_ship_and_hits_next_live_target(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        fighter.parent = parent
        dead = self.make_ship()
        dead.player = 2
        dead.position = [108, 100]
        dead.current_hp = 0
        live = self.make_ship()
        live.player = 2
        live.position = [108, 100]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_ship_collisions(
                [fighter], [dead, live], []
            )

        self.assertEqual(live.current_hp, 9)
        self.assertFalse(fighter.currently_alive)

    def test_swept_fighter_enemy_ship_impact_is_not_tunneled(self):
        fighter = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        fighter.parent = parent
        enemy = self.make_ship()
        enemy.player = 2
        fighter.size = [10, 10]
        enemy.size = [10, 10]
        fighter.previous_position = [100, 100]
        fighter.position = [200, 100]
        enemy.previous_position = [150, 100]
        enemy.position = [150, 100]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_ship_collisions(
                [fighter], [enemy], []
            )

        self.assertEqual(enemy.current_hp, 9)
        self.assertFalse(fighter.currently_alive)

    def test_fighter_fighter_contact_destroys_both_by_default(self):
        first = self.make_fighter()
        second = self.make_fighter()
        second.position = [108, 100]
        second.previous_position = second.position.copy()
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_fighter_fighter_collisions(
                [first, second], effects
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertEqual(effects, [sentinel_effect, sentinel_effect])
        play_boom.assert_called_once_with(1)

    def test_one_sided_fighter_collision_only_damages_disabled_fighter(self):
        first = self.make_fighter()
        second = self.make_fighter(collides_with_fighters=False)
        first.current_hp = 3
        second.current_damage = 5
        second.position = [108, 100]
        second.previous_position = second.position.copy()

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_fighter_fighter_collisions(
                [first, second], []
            )

        self.assertTrue(first.currently_alive)
        self.assertEqual(first.current_hp, 3)
        self.assertFalse(second.currently_alive)
        play_boom.assert_called_once_with(5)

    def test_two_fighters_with_collision_disabled_ignore_each_other(self):
        first = self.make_fighter(collides_with_fighters=False)
        second = self.make_fighter(collides_with_fighters=False)
        second.position = [108, 100]
        second.previous_position = second.position.copy()

        with mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom:
            collisions._handle_fighter_fighter_collisions(
                [first, second], []
            )

        self.assertTrue(first.currently_alive)
        self.assertTrue(second.currently_alive)
        play_boom.assert_not_called()

    def test_fighters_with_remaining_hp_survive_mutual_impact(self):
        first = self.make_fighter()
        second = self.make_fighter()
        first.current_hp = 3
        second.current_hp = 3
        second.position = [108, 100]
        second.previous_position = second.position.copy()

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_fighter_fighter_collisions(
                [first, second], []
            )

        self.assertEqual(first.current_hp, 2)
        self.assertEqual(second.current_hp, 2)
        self.assertTrue(first.currently_alive)
        self.assertTrue(second.currently_alive)

    def test_fighter_ignores_dead_fighter_and_hits_next_live_target(self):
        first = self.make_fighter()
        dead = self.make_fighter()
        live = self.make_fighter()
        for fighter in (dead, live):
            fighter.position = [108, 100]
            fighter.previous_position = fighter.position.copy()
        dead.current_hp = 0
        dead.currently_alive = False

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_fighter_collisions(
                [first, dead, live], []
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_swept_fighter_fighter_impact_is_not_tunneled(self):
        first = self.make_fighter()
        second = self.make_fighter()
        first.size = [10, 10]
        second.size = [10, 10]
        first.previous_position = [100, 100]
        first.position = [200, 100]
        second.previous_position = [150, 100]
        second.position = [150, 100]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_fighter_fighter_collisions(
                [first, second], []
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_destroyed_fighter_finishes_remaining_pairs_in_same_frame(self):
        first = self.make_fighter()
        second = self.make_fighter()
        third = self.make_fighter()
        for fighter in (second, third):
            fighter.position = [108, 100]
            fighter.previous_position = fighter.position.copy()
        effects = []

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_fighter_fighter_collisions(
                [first, second, third], effects
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertFalse(third.currently_alive)
        self.assertEqual(len(effects), 3)
        self.assertEqual(play_boom.call_count, 2)

    def test_laser_targets_apply_role_and_ownership_rules(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)

        enemy_ship = self.make_ship()
        enemy_ship.player = 2
        enemy_ship.position = [180, 100]
        friendly_ship = self.make_ship()
        friendly_ship.player = 1
        friendly_ship.position = [190, 100]

        enemy_projectile = self.make_projectile(parent)
        enemy_projectile.player = 2
        enemy_projectile.position = [200, 100]
        friendly_projectile = self.make_projectile(parent)
        friendly_projectile.player = 1
        friendly_projectile.position = [210, 100]

        enemy_fighter = self.make_fighter()
        enemy_fighter.player = 2
        enemy_fighter.position = [220, 100]
        friendly_fighter = self.make_fighter()
        friendly_fighter.player = 1
        friendly_fighter.position = [230, 100]
        invulnerable_fighter = self.make_fighter(laser_vulnerable=False)
        invulnerable_fighter.player = 2
        invulnerable_fighter.position = [240, 100]

        asteroid = self.make_asteroid([250, 100])
        dead_asteroid = self.make_asteroid([260, 100])
        dead_asteroid.currently_alive = False
        planet = self.make_planet([270, 100])

        targets = collisions._laser_targets(
            laser,
            [parent, enemy_ship, friendly_ship],
            [enemy_projectile, friendly_projectile],
            [enemy_fighter, friendly_fighter, invulnerable_fighter],
            [asteroid, dead_asteroid],
            [planet],
        )

        for target in (
            enemy_ship,
            enemy_projectile,
            enemy_fighter,
            friendly_fighter,
            asteroid,
            planet,
        ):
            self.assertIn(target, targets)
        for target in (
            parent,
            friendly_ship,
            friendly_projectile,
            invulnerable_fighter,
            dead_asteroid,
        ):
            self.assertNotIn(target, targets)

    def test_laser_hit_flags_enable_parent_and_friendly_projectiles_only(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        friendly_ship = self.make_ship()
        friendly_ship.player = 1
        friendly_ship.position = [180, 100]
        friendly_projectile = self.make_projectile(parent)
        friendly_projectile.player = 1
        friendly_projectile.position = [190, 100]
        laser = self.make_laser(parent, hit_parent=True, hit_self=True)

        targets = collisions._laser_targets(
            laser,
            [parent, friendly_ship],
            [friendly_projectile],
            [],
            [],
            [],
        )

        self.assertIn(parent, targets)
        self.assertIn(friendly_projectile, targets)
        self.assertNotIn(friendly_ship, targets)

    def test_explicit_laser_target_overrides_normal_eligibility(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        explicit = self.make_fighter(laser_vulnerable=False)
        explicit.player = 1
        explicit.position = [180, 100]
        interceptor = self.make_fighter()
        interceptor.player = 2
        interceptor.position = [160, 100]
        unrelated_ship = self.make_ship()
        unrelated_ship.player = 2
        unrelated_ship.position = [150, 100]
        laser = self.make_laser(parent, target=explicit)

        targets = collisions._laser_targets(
            laser,
            [unrelated_ship],
            [],
            [explicit, interceptor],
            [],
            [],
        )

        self.assertEqual(targets, [explicit, interceptor])

    def test_laser_selects_nearest_intercept_across_target_roles(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        enemy_ship = self.make_ship()
        enemy_ship.player = 2
        enemy_ship.position = [240, 100]
        asteroid = self.make_asteroid([170, 100])
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_laser_collisions(
                [laser], [enemy_ship], [], [], [asteroid], [], effects
            )

        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(enemy_ship.current_hp, 10)
        self.assertTrue(laser.intercepted)
        self.assertLess(laser.end_position[0], enemy_ship.position[0])

    def test_laser_hit_info_uses_wrapped_segment(self):
        parent = self.make_ship()
        parent.position = [const.ARENA_SIZE - 20, 100]
        laser = self.make_laser(parent)
        laser.start_position = parent.position.copy()
        laser.end_position = [30, 100]
        target = self.make_ship()
        target.position = [5, 100]

        hit_info = collisions._laser_hit_info(laser, target)

        self.assertIsNotNone(hit_info)
        self.assertIs(hit_info["target"], target)

    def test_laser_mask_sampling_rejects_empty_target_mask(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        laser.start_position = [100, 100]
        laser.end_position = [200, 100]
        target = self.make_ship()
        target.position = [150, 100]
        empty_mask = pygame.mask.Mask((20, 20), fill=False)
        target.get_collision_mask = lambda: empty_mask

        self.assertIsNone(collisions._laser_hit_info(laser, target))

        full_mask = pygame.mask.Mask((20, 20), fill=True)
        target.get_collision_mask = lambda: full_mask
        self.assertIsNotNone(collisions._laser_hit_info(laser, target))

    def test_laser_ship_impact_clips_endpoint_and_applies_effect(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        target = self.make_ship()
        target.player = 2
        target.position = [150, 100]
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ) as from_blast,
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_laser_collisions(
                [laser], [target], [], [], [], [], effects
            )

        self.assertEqual(target.current_hp, 8)
        self.assertEqual(laser.end_position, [140.0, 100.0])
        self.assertTrue(laser.intercepted)
        self.assertEqual(effects, [sentinel_effect])
        from_blast.assert_called_once_with(
            [140.0, 100.0], [-1.0, 0.0], 2, align_edge=True
        )
        play_boom.assert_called_once_with(2)

    def test_planet_absorbs_laser_without_damage_state(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        planet = self.make_planet([150, 100])

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_laser_collisions(
                [laser], [], [], [], [], [planet], []
            )

        self.assertFalse(hasattr(planet, "current_hp"))
        self.assertTrue(laser.intercepted)

    def test_laser_destroys_projectile_at_zero_hp(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        projectile = self.make_projectile(parent)
        projectile.player = 2
        projectile.position = [150, 100]
        projectile.previous_position = projectile.position.copy()
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_laser_collisions(
                [laser], [], [projectile], [], [], [], effects
            )

        self.assertFalse(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 0)
        self.assertEqual(effects, [sentinel_effect, sentinel_effect])

    def test_laser_destroys_fighter_at_zero_hp(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        fighter = self.make_fighter()
        fighter.position = [150, 100]
        fighter.previous_position = fighter.position.copy()

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_laser_collisions(
                [laser], [], [], [fighter], [], [], []
            )

        self.assertFalse(fighter.currently_alive)
        self.assertEqual(fighter.current_hp, 0)

    def test_laser_directly_reduces_surviving_projectile_hp(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        projectile = self.make_projectile(parent)
        projectile.player = 2
        projectile.position = [150, 100]
        projectile.previous_position = projectile.position.copy()
        projectile.current_hp = 3
        projectile.set_hp = mock.Mock()

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_laser_collisions(
                [laser], [], [projectile], [], [], [], []
            )

        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 1)
        projectile.set_hp.assert_not_called()

    def test_enemy_projectile_with_greater_remaining_hp_survives(self):
        first, second = self.make_projectile_pair(
            first_hp=10,
            second_hp=3,
            first_damage=4,
            second_damage=2,
        )
        effects = []

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second], effects
            )

        self.assertTrue(first.currently_alive)
        self.assertEqual(first.current_hp, 8)
        self.assertFalse(second.currently_alive)
        self.assertEqual(len(effects), 1)
        play_boom.assert_called_once_with(4)

    def test_second_enemy_projectile_with_greater_remaining_hp_survives(self):
        first, second = self.make_projectile_pair(
            first_hp=3,
            second_hp=10,
            first_damage=2,
            second_damage=4,
        )

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second], []
            )

        self.assertFalse(first.currently_alive)
        self.assertTrue(second.currently_alive)
        self.assertEqual(second.current_hp, 8)

    def test_equal_positive_projectile_hp_after_impact_destroys_both(self):
        first, second = self.make_projectile_pair(
            first_hp=5,
            second_hp=5,
            first_damage=2,
            second_damage=2,
        )

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second], []
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_same_name_enemy_projectiles_destroy_each_other_regardless_of_hp(self):
        first, second = self.make_projectile_pair(
            first_name="MatchingProjectile",
            second_name="MatchingProjectile",
            first_hp=10,
            second_hp=10,
            first_damage=1,
            second_damage=1,
        )

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second], []
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_same_player_projectiles_require_matching_names_and_mutual_self_hit(self):
        cases = (
            ("First", "Second", True, True),
            ("Matching", "Matching", True, False),
            ("Matching", "Matching", False, True),
        )
        for first_name, second_name, first_hit_self, second_hit_self in cases:
            with self.subTest(
                names=(first_name, second_name),
                hit_self=(first_hit_self, second_hit_self),
            ):
                first, second = self.make_projectile_pair(
                    first_name=first_name,
                    second_name=second_name,
                    first_player=1,
                    second_player=1,
                )
                first.hit_self = first_hit_self
                second.hit_self = second_hit_self

                with (
                    mock.patch.object(collisions.BattleEffect, "from_blast"),
                    mock.patch.object(collisions.BattleEffect, "play_boom"),
                ):
                    collisions._handle_projectile_projectile_collisions(
                        [first, second], []
                    )

                self.assertTrue(first.currently_alive)
                self.assertTrue(second.currently_alive)

    def test_matching_same_player_projectiles_with_self_hit_destroy_each_other(self):
        first, second = self.make_projectile_pair(
            first_name="Matching",
            second_name="Matching",
            first_player=1,
            second_player=1,
        )
        first.hit_self = True
        second.hit_self = True

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second], []
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_swept_projectile_projectile_impact_is_not_tunneled(self):
        first, second = self.make_projectile_pair()
        first.size = [10, 10]
        second.size = [10, 10]
        first.previous_position = [100, 100]
        first.position = [200, 100]
        second.previous_position = [150, 100]
        second.position = [150, 100]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second], []
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_destroyed_projectile_finishes_remaining_pairs_in_same_frame(self):
        first, second = self.make_projectile_pair(
            first_name="Matching",
            second_name="Matching",
        )
        third_parent = self.make_ship()
        third_parent.player = 2
        third = self.make_projectile(third_parent)
        third.name = third.projectile_name = "Matching"
        third.player = 2
        third.position = [108, 100]
        third.previous_position = third.position.copy()
        effects = []

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_projectile_projectile_collisions(
                [first, second, third], effects
            )

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertFalse(third.currently_alive)
        self.assertEqual(len(effects), 3)
        self.assertEqual(play_boom.call_count, 2)

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

    def test_generic_projectile_ship_impact_does_not_add_momentum(self):
        parent = self.make_ship()
        parent.player = 1
        target = self.make_ship()
        target.add_impulse = mock.Mock()
        projectile = self.make_projectile(parent)

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_ship_collisions(
                [projectile], [target], []
            )

        target.add_impulse.assert_not_called()

    def test_druuge_projectile_ship_impact_adds_projectile_momentum(self):
        parent = self.make_ship()
        parent.player = 1
        parent.mass = 10
        target = self.make_ship()
        target.mass = 20
        target.add_impulse = mock.Mock()
        projectile = self.make_projectile(parent, DruugeA1)
        projectile.projectile_name = "DruugeA1"
        projectile.RECOIL_INCREMENT = 24
        projectile.velocity = [3, 4]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_ship_collisions(
                [projectile], [target], []
            )

        target.add_impulse.assert_called_once_with(7.2, 9.6)

    def test_projectile_impact_with_planet_destroys_projectile_at_contact(self):
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        planet = self.make_planet([108, 100])
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ) as from_blast,
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_projectile_planet_collisions(
                [projectile], [planet], effects
            )

        self.assertFalse(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 0)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(4)
        from_blast.assert_called_once_with(
            [98.0, 100.0], [-1.0, 0.0], 4, align_edge=True
        )

    def test_swept_projectile_impact_with_planet_is_not_tunneled(self):
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.size = [10, 10]
        projectile.previous_position = [100, 100]
        projectile.position = [200, 100]
        planet = self.make_planet([150, 100])
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_projectile_planet_collisions(
                [projectile], [planet], effects
            )

        self.assertFalse(projectile.currently_alive)
        self.assertEqual(effects, [sentinel_effect])

    def test_projectile_impact_with_asteroid_destroys_both_objects(self):
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        asteroid = self.make_asteroid([108, 100])
        effects = []
        sentinel_effect = object()

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=sentinel_effect,
            ) as from_blast,
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions._handle_projectile_asteroid_collisions(
                [projectile], [asteroid], effects
            )

        self.assertFalse(projectile.currently_alive)
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(4)
        from_blast.assert_called_once_with(
            [98.0, 100.0], [-1.0, 0.0], 4, align_edge=True
        )


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
        self.assertEqual(aftermath["dead_players"], {1})
        self.assertEqual(aftermath.get("started_frame"), 30)
        self.assertIsInstance(aftermath.pending_explosions[0], ScheduledExplosion)
        self.assertEqual(aftermath.pending_explosions[0]["frame"], 30)
        self.assertEqual(aftermath_camera_targets(aftermath, dead, survivor, 30), [survivor, dead])
        self.assertFalse(aftermath_ready_for_selection(aftermath, 30, sound_enabled=False))
        self.assertTrue(
            aftermath_ready_for_selection(
                aftermath, 30 + const.POST_DEATH_CONTROL_FRAMES, sound_enabled=False
            )
        )

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
                "src.Battle.battle.start_or_update_aftermath",
                wraps=start_or_update_aftermath,
            ) as register_deaths,
            mock.patch(
                "src.Battle.battle.BattleEffect.ship_explosion",
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
            "src.Battle.battle.BattleEffect.ship_explosion",
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

        self.assertEqual(
            aftermath_camera_targets(
                aftermath,
                dead,
                survivor,
                20 + const.POST_DEATH_ANIMATION_VIEW_FRAMES - 1,
            ),
            [survivor, dead],
        )
        self.assertIsNone(
            aftermath_camera_targets(
                aftermath,
                dead,
                survivor,
                20 + const.POST_DEATH_ANIMATION_VIEW_FRAMES,
            )
        )

    def test_victory_audio_starts_once_at_view_boundary_and_honors_sound_setting(self):
        dead = self.make_ship(1, hp=0, alive=False)
        survivor = self.make_ship(2)
        start_frame = 50
        aftermath = AftermathState(start_frame, start_frame)

        with mock.patch("src.Battle.battle.play_victory_ditty") as play_ditty:
            update_aftermath(
                aftermath,
                dead,
                survivor,
                [],
                start_frame + const.POST_DEATH_ANIMATION_VIEW_FRAMES - 1,
                sound_enabled=True,
            )
            play_ditty.assert_not_called()
            update_aftermath(
                aftermath,
                dead,
                survivor,
                [],
                start_frame + const.POST_DEATH_ANIMATION_VIEW_FRAMES,
                sound_enabled=True,
            )
            update_aftermath(
                aftermath,
                dead,
                survivor,
                [],
                start_frame + const.POST_DEATH_ANIMATION_VIEW_FRAMES + 1,
                sound_enabled=True,
            )
            play_ditty.assert_called_once_with(survivor)

        muted_aftermath = AftermathState(start_frame, start_frame)
        with mock.patch("src.Battle.battle.play_victory_ditty") as play_ditty:
            update_aftermath(
                muted_aftermath,
                dead,
                survivor,
                [],
                start_frame + const.POST_DEATH_ANIMATION_VIEW_FRAMES,
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
        )
        with mock.patch("src.Battle.battle.play_victory_ditty") as play_ditty:
            update_aftermath(
                tie_aftermath,
                dead,
                other_dead,
                [],
                start_frame + const.POST_DEATH_ANIMATION_VIEW_FRAMES,
                sound_enabled=True,
            )
        play_ditty.assert_called_once_with(dead)

    def test_selection_becomes_ready_on_exact_control_boundary(self):
        aftermath = AftermathState(started_frame=75, latest_death_frame=75)

        self.assertFalse(
            aftermath_ready_for_selection(
                aftermath,
                75 + const.POST_DEATH_CONTROL_FRAMES - 1,
                sound_enabled=True,
            )
        )
        self.assertTrue(
            aftermath_ready_for_selection(
                aftermath,
                75 + const.POST_DEATH_CONTROL_FRAMES,
                sound_enabled=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
