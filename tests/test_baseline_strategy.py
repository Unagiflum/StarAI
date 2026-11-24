"""Baseline strategy engine, trainee phase mixing, opponents and battle mode."""

import dataclasses
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pygame

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Battle.battle_ai import (
    AI_MODE_BASELINE,
    AI_MODE_TRAINED,
    BaselineBattleController,
    BattleAIManager,
    BattleControllerStatus,
    normalize_ai_setting,
)
from src.Objects.Ships.registry import create_ship
from src.audio import NullAudioService
from src.configuration import (
    FLEET_AI_MODE_BASELINE,
    Fleets,
    FleetsCodec,
    PlayerFleet,
)
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.training.baseline import (
    BaselineController,
    BaselineOpponentController,
    BaselinePhaseMixer,
    STRATEGIES,
    strategy_for,
)
from src.training.baseline.engine import (
    ACTION_INDEX_BY_MASK,
    AVOID_RANGE_FRACTION,
    BaselineContext,
    BaselineMemory,
    ControlPlan,
    ORZ_TURRET_STEP,
    TURN_LEFT,
    TURN_NONE,
    TURN_RIGHT,
    PLANET_HORIZON_MARGIN_FRAMES,
    planet_contact_distance,
    planet_horizon_frames,
    simulate_planet_pass,
    top_speed_frames,
    turn_about_frames,
)
from src.training.baseline import behaviours as behaviour_table
from src.training.baseline import strategies as strategy_module
from src.training.baseline.behaviours import (
    BAND_INSIDE_CHASE,
    BAND_INSIDE_FACE,
    BAND_INSIDE_ORBIT,
    DEFAULT_BEHAVIOUR,
    FIRE_AIMED,
    FIRE_DEFEND,
    FIRE_LAUNCH,
    FIRE_SPECIAL,
    SHIP_BEHAVIOURS,
    Band,
    ShipBehaviour,
    WeaponUse,
    ship_behaviour,
)
from src.training.baseline.engine import (
    EVADE_HEADING,
    RECHARGE_REACH_MARGIN,
    run_bearing,
)
from src.training.baseline.flight import ShipFlight
from src.training.baseline.threats import (
    fly_move,
    judge_move,
    threat_tracks,
)
from src.training.baseline.observation_view import (
    ObservationView,
    ObservedObject,
    wrap_degrees,
)
from src.training.baseline.ship_specs import (
    PLANET_HULL_REACH,
    PLANET_RADIUS,
    ship_spec,
    shofixti_blast_damage,
    shofixti_kill_distance,
    weapon_spec,
)
from src.training.baseline.strategies import (
    CHMMR_TRACTOR_BATTERY,
    CHMMR_TRACTOR_FAR,
    EARTHLING_CHMMR_ORBIT_OFFSET,
    EARTHLING_NUKE_MIN_DISTANCE,
    EARTHLING_NUKE_OVERRUN_SPEED,
    may_evade,
    SLYLANDRO_CHMMR_FAR,
    SLYLANDRO_CHMMR_FIRE,
    SLYLANDRO_CHMMR_NEAR,
    SPATHI_CHMMR_FAR,
    SPATHI_CHMMR_NEAR,
    SYREEN_CHMMR_SONG_DISTANCE,
    UMGAH_CHMMR_HOP_DISTANCE,
    UMGAH_CONE_HALF_ANGLE,
    UMGAH_PASS_CLEARANCE,
    UMGAH_RUN_CLEARANCE,
    UTWIG_CHMMR_RUN_BATTERY,
    YEHAT_CHMMR_FIRE_RESERVE,
)
from src.training.baseline.strategies import _umgah_hops_needed as umgah_hops_needed
from src.training.baseline.strategies import _umgah_tail_heading as umgah_tail_heading
from src.training.baseline.strategies import _supox_band as supox_band
from src.training.baseline.strategies import (
    _blazer_intercept_bearing as blazer_intercept_bearing,
)
from src.training.contracts import ACTION_INDEX_TABLE, SHIP_TYPE_CATALOG_ORDER
from src.training.coordinated import (
    CoordinatedActionRequest,
    CoordinatedExplorationSchedule,
)
from src.training.cpu_contracts import (
    TrainingOrchestrationConfig,
    advance_baseline_use,
    initial_baseline_use,
)
from src.training.model_registry import (
    TrainingModelRepository,
    metadata_from_state,
    model_architecture_metadata,
)
from src.training.observation import encode_observation
from src.training.orchestration import (
    TrainingBatchResult,
    ValueNetworkPolicy,
    baseline_mixer_for_config,
)
from src.training.session import (
    TrainingSession,
    baseline_regimen_metadata,
    evaluation_batch_config,
)


def _simulation(ship_a="Earthling", ship_b="Spathi", seed=3):
    audio = NullAudioService()
    return BattleSimulation(
        None,
        create_ship(ship_a, 1, audio_service=audio),
        create_ship(ship_b, 2, audio_service=audio),
        audio_service=audio,
        rng=random.Random(seed),
        include_stars=False,
    )


def _observation(simulation):
    return encode_observation(
        simulation.player1,
        simulation.player2,
        frame_id=simulation.frame_id,
        game_objects=simulation.world,
    )


class ObservationViewTests(unittest.TestCase):
    def test_view_recovers_game_units_from_the_observation(self):
        simulation = _simulation()
        me = simulation.player1
        enemy = simulation.player2
        me.position = [4000.0, 1000.0]
        me.velocity = [0.0, 0.0]
        me.heading = 0
        me.rotation = 0.0
        enemy.position = [4300.0, 1000.0]
        enemy.velocity = [0.0, 0.0]

        view = ObservationView(_observation(simulation))

        self.assertEqual(view.enemy_ship_type, "Spathi")
        self.assertAlmostEqual(view.enemy.distance, 300.0, places=3)
        # The enemy is directly to starboard of a ship facing screen-up.
        self.assertAlmostEqual(view.enemy.bearing, 90.0, places=3)
        self.assertAlmostEqual(view.enemy.forward, 0.0, places=3)
        self.assertAlmostEqual(view.enemy.right, 300.0, places=3)
        self.assertAlmostEqual(view.me.crew, me.current_hp, places=3)
        self.assertAlmostEqual(view.me.battery, me.current_energy, places=3)
        self.assertAlmostEqual(view.me.max_thrust, me.max_thrust, places=3)

    def test_wrap_degrees_uses_half_open_range(self):
        self.assertEqual(wrap_degrees(190.0), -170.0)
        self.assertEqual(wrap_degrees(-180.0), 180.0)
        self.assertEqual(wrap_degrees(540.0), 180.0)


class ControlPlanTests(unittest.TestCase):
    def test_first_rule_to_claim_a_control_wins(self):
        plan = ControlPlan()

        self.assertTrue(plan.claim_a2(True))
        self.assertFalse(plan.claim_a2(False))
        self.assertTrue(plan.claim_turn(TURN_LEFT))
        self.assertFalse(plan.claim_turn(TURN_RIGHT))
        self.assertTrue(plan.claim_thrust(True))

        self.assertEqual(plan.mask(), 1 | 2 | 16)
        self.assertEqual(ACTION_INDEX_TABLE[plan.action_index()].mask, 1 | 2 | 16)

    def test_unclaimed_controls_stay_off(self):
        self.assertEqual(ControlPlan().action_index(), ACTION_INDEX_BY_MASK[0])


class TurningTests(unittest.TestCase):
    def _context(self, *, held_left=False, held_right=False):
        view = mock.Mock()
        view.me.held.side_effect = lambda control: (
            (control == "left" and held_left) or (control == "right" and held_right)
        )
        memory = BaselineMemory()
        memory.tap_degrees = 10.0
        memory.hold_degrees = 40.0
        return BaselineContext(
            ship_name="Earthling",
            view=view,
            memory=memory,
        )

    def test_turns_toward_bearing_outside_half_a_tap(self):
        context = self._context()

        self.assertEqual(context.turn_toward(4.0), TURN_NONE)
        self.assertEqual(context.turn_toward(30.0), TURN_RIGHT)
        self.assertEqual(context.turn_toward(-30.0), TURN_LEFT)

    def test_releases_a_held_turn_near_the_target_for_a_fresh_tap(self):
        context = self._context(held_right=True)

        self.assertEqual(context.turn_toward(20.0), TURN_NONE)
        self.assertEqual(context.turn_toward(90.0), TURN_RIGHT)


class StrategyTests(unittest.TestCase):
    def test_every_ship_has_a_strategy(self):
        self.assertEqual(set(STRATEGIES), set(SHIP_TYPE_CATALOG_ORDER))

    def test_every_strategy_flies_real_battles_without_errors(self):
        for ship_name in SHIP_TYPE_CATALOG_ORDER:
            with self.subTest(ship=ship_name):
                simulation = _simulation(ship_name, "Chenjesu", seed=7)
                controller = BaselineController(ship_name)
                for _frame in range(48):
                    decided = controller.decide(_observation(simulation))
                    self.assertIn(decided, range(len(ACTION_INDEX_TABLE)))
                    simulation.step(
                        actions={
                            1: ACTION_INDEX_TABLE[decided],
                            2: ACTION_INDEX_TABLE[0],
                        }
                    )

    def test_weapon_ranges_use_fixed_range_or_projectile_reach(self):
        self.assertEqual(weapon_spec("ZoqFotPik", 2).range, 140.0)
        self.assertEqual(weapon_spec("Vux", 1).range, 644.0)
        spathi = weapon_spec("Spathi", 1)
        self.assertAlmostEqual(spathi.range, 120 * 10 * 0.9)
        self.assertEqual(weapon_spec("Mmrnmrhm", 1, "ywing").ability_name, "MmrnmrhmYWingA1")


def _seen(bearing, distance, *, closing=0.0, group="enemy_ship", crew_effect=0.0):
    """An object at ``bearing`` that closes straight in at ``closing``."""

    radians = math.radians(bearing)
    forward = distance * math.cos(radians)
    right = distance * math.sin(radians)
    return ObservedObject(
        group=group,
        distance=float(distance),
        bearing=float(bearing),
        forward=forward,
        right=right,
        velocity_forward=-closing * math.cos(radians),
        velocity_right=-closing * math.sin(radians),
        closing_speed=float(closing),
        lateral_speed=0.0,
        crew_effect=float(crew_effect),
        battery_effect=0.0,
        hit_points=0.0,
        expires=False,
        remaining_seconds=0.0,
    )


def _rule_context(
    ship_name,
    *,
    enemy=None,
    enemy_name="Spathi",
    planet=None,
    enemy_objects=(),
    battery=None,
    battery_fraction=1.0,
    held=(),
    ywing=False,
    max_battery=30.0,
    memory=None,
    speed=0.0,
    a1_timer=0.0,
    friendly_shots=(),
):
    view = mock.Mock()
    view.friendly_a1_objects = tuple(friendly_shots)
    view.me.heading = 0.0
    view.me.a1_ready = a1_timer <= 0.0
    view.me.timer_seconds.side_effect = lambda field: a1_timer if field == "a1_timer" else 0.0
    view.me.max_battery = max_battery
    view.enemy = enemy
    view.planet = planet
    view.enemy_ship_type = enemy_name
    view.enemy_objects = tuple(enemy_objects)
    view.objects.side_effect = lambda group: tuple(
        item for item in enemy_objects if item.group == group
    )
    view.me.held.side_effect = lambda control: control in held
    view.me.battery_fraction = battery_fraction
    view.me.battery = 100.0 if battery is None else battery
    view.me.mmrnmrhm_ywing = ywing
    view.me.androsynth_blazer = False
    view.me.velocity_forward = 0.0
    view.me.velocity_right = 0.0
    view.me.speed = speed
    view.enemy_block.androsynth_blazer = False
    view.enemy_block.mmrnmrhm_ywing = False
    view.enemy_block.opponent_bearing = None
    return BaselineContext(
        ship_name=ship_name,
        view=view,
        memory=memory or BaselineMemory(),
    )


class AvoidOpponentTests(unittest.TestCase):
    def test_flees_at_135_degrees_only_from_a_near_closing_opponent(self):
        near = const.ARENA_SIZE * AVOID_RANGE_FRACTION - 100.0
        far = const.ARENA_SIZE * AVOID_RANGE_FRACTION + 100.0

        fleeing = _rule_context("Spathi", enemy=_seen(10.0, near, closing=5.0))
        self.assertTrue(fleeing.avoid_opponent())
        # 10 - 135 = -125 needs less turning than 10 + 135 = 145.
        self.assertEqual(fleeing.plan.turn, TURN_LEFT)
        self.assertTrue(fleeing.plan.thrust)

        self.assertFalse(
            _rule_context("Spathi", enemy=_seen(10.0, far, closing=5.0)).avoid_opponent()
        )
        self.assertFalse(
            _rule_context("Spathi", enemy=_seen(10.0, near, closing=-5.0)).avoid_opponent()
        )

    def test_escape_toward_the_planet_turns_45_degrees_off_it_on_the_quicker_side(self):
        # Escape heading 10 - 135 = -125 points almost at the planet at -110.
        context = _rule_context(
            "Spathi",
            enemy=_seen(10.0, 500.0, closing=5.0),
            planet=_seen(-110.0, 1200.0, group="planet"),
        )
        with mock.patch.object(context, "steer", return_value=True) as steer:
            context.avoid_opponent()
        # -110 + 45 = -65 is nearer the nose than -110 - 45 = -155.
        self.assertAlmostEqual(steer.call_args.args[0], -65.0)

    def test_escape_clear_of_the_planet_keeps_its_heading(self):
        context = _rule_context(
            "Spathi",
            enemy=_seen(10.0, 500.0, closing=5.0),
            planet=_seen(-40.0, 1200.0, group="planet"),
        )
        with mock.patch.object(context, "steer", return_value=True) as steer:
            context.avoid_opponent()
        self.assertAlmostEqual(steer.call_args.args[0], -125.0)


def _planet(bearing, distance, *, own_forward=0.0, own_right=0.0):
    """The planet at ``bearing``, seen by a ship moving at ``own_*``."""

    radians = math.radians(bearing)
    return dataclasses.replace(
        _seen(bearing, distance, group="planet"),
        velocity_forward=-own_forward,
        velocity_right=-own_right,
        closing_speed=own_forward * math.cos(radians) + own_right * math.sin(radians),
    )


class PlanetRuleTests(unittest.TestCase):
    def test_coasting_prediction_matches_the_simulation(self):
        simulation = _simulation("Chenjesu", "Spathi")
        # Asteroids spawn at random and could cross the path.
        simulation.world.objects[:] = [
            obj for obj in simulation.world.objects if type(obj).__name__ != "Asteroid"
        ]
        planet = simulation.world.planets[0]
        me = simulation.player1
        me.position = [planet.position[0], planet.position[1] - 900.0]
        me.velocity = [6.0, 0.0]
        simulation.player2.position = [
            (planet.position[0] + 3500.0) % const.ARENA_SIZE,
            planet.position[1],
        ]
        seen = ObservationView(_observation(simulation)).planet
        predicted = simulate_planet_pass(
            (-seen.forward, -seen.right),
            (-seen.velocity_forward, -seen.velocity_right),
            96,
            ship_spec("Chenjesu"),
        )

        contact_frame = None
        for frame in range(96):
            simulation.step(actions={1: ACTION_INDEX_TABLE[0], 2: ACTION_INDEX_TABLE[0]})
            if me.distance_to(planet)[1] <= planet_contact_distance(ship_spec("Chenjesu")):
                contact_frame = frame + 1
                break
        self.assertIsNotNone(contact_frame)
        self.assertAlmostEqual(predicted.impact_frame, contact_frame, delta=1.0)

    def test_hull_reach_table_matches_the_real_collision_masks(self):
        from src.Battle.collision_geometry import (
            invalidate_geometry,
            objects_overlap_at_positions,
        )
        from src.resources import HeadlessAssetManager

        resources = HeadlessAssetManager()
        audio = NullAudioService()
        planet = _simulation().world.planets[0]
        # 16 headings and 24 approach directions, 2 units apart.
        heading_step = max(1, const.SHIP_DIRECTIONS // 16)
        for ship_name in SHIP_TYPE_CATALOG_ORDER:
            ship = create_ship(ship_name, 1, resources=resources, audio_service=audio)
            reach = 0.0
            for heading in range(0, const.SHIP_DIRECTIONS, heading_step):
                ship.heading = heading
                ship.rotation = heading * const.TURN_ANGLE
                invalidate_geometry(ship)
                for direction in range(0, 360, 15):
                    radians = math.radians(direction)
                    for distance in range(300, int(PLANET_RADIUS), -2):
                        position = [
                            planet.position[0] + math.sin(radians) * distance,
                            planet.position[1] - math.cos(radians) * distance,
                        ]
                        if objects_overlap_at_positions(
                            ship, planet, position, planet.position
                        ):
                            reach = max(reach, distance - PLANET_RADIUS)
                            break
            with self.subTest(ship=ship_name):
                self.assertAlmostEqual(PLANET_HULL_REACH[ship_name], reach, delta=6.0)

    def test_leaves_a_pass_that_misses_the_planet_to_the_strategy(self):
        # Inside the well, but moving across it fast enough to swing past.
        context = _rule_context("Chenjesu", planet=_planet(0.0, 900.0, own_right=30.0))

        self.assertFalse(context.avoid_planet())
        self.assertIsNone(context.plan.turn)
        self.assertIsNone(context.plan.thrust)

    def test_steps_in_when_the_coasting_path_hits_the_surface(self):
        context = _rule_context("Chenjesu", planet=_planet(0.0, 900.0))

        self.assertTrue(context.avoid_planet())
        self.assertIsNotNone(context.plan.turn)

    def test_ships_without_inertia_are_not_pulled_in(self):
        context = _rule_context("Arilou", planet=_planet(0.0, 600.0))

        self.assertFalse(context.avoid_planet())

    def test_escapes_on_the_side_of_the_drift_already_carried(self):
        # The left escape heading (20 - 90 = -70) needs less turning, but
        # the ship already drifts right, toward the other (20 + 90 = 110).
        context = _rule_context(
            "Chenjesu", planet=_planet(20.0, 700.0, own_right=10.0)
        )

        self.assertTrue(context.avoid_planet())
        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertEqual(context.memory.sides["avoid_planet"], 1)

    def test_holds_thrust_while_the_nose_turns_toward_the_escape(self):
        # Drifting left, it escapes left of the planet at -30, more than 90
        # degrees round from the nose: thrust now would push it inward.
        context = _rule_context(
            "Chenjesu", planet=_planet(-30.0, 700.0, own_right=-10.0)
        )

        self.assertTrue(context.avoid_planet())
        self.assertEqual(context.plan.turn, TURN_LEFT)
        self.assertFalse(context.plan.thrust)

    def test_keeps_its_escape_unless_another_is_clearly_better(self):
        # Dead ahead, both sides are equally good: a fresh choice goes right.
        fresh = _rule_context("Chenjesu", planet=_planet(0.0, 700.0))
        self.assertTrue(fresh.avoid_planet())
        self.assertEqual(fresh.plan.turn, TURN_RIGHT)

        memory = BaselineMemory()
        memory.sides["avoid_planet"] = -1
        memory.sides["avoid_planet_offset"] = fresh.memory.sides["avoid_planet_offset"]
        kept = _rule_context("Chenjesu", planet=_planet(0.0, 700.0), memory=memory)

        self.assertTrue(kept.avoid_planet())
        self.assertEqual(kept.plan.turn, TURN_LEFT)

    def test_horizon_of_an_agile_ship_is_its_turn_and_thrust_time(self):
        # The Supox turns one step every 2 frames and pushes every frame:
        # 8 steps take 15 frames, 5 pushes to top speed take 5.
        spec = ship_spec("Supox")
        self.assertEqual(turn_about_frames(spec), 15)
        self.assertEqual(top_speed_frames(spec), 5)
        self.assertEqual(planet_horizon_frames(spec), 15 + 5 + PLANET_HORIZON_MARGIN_FRAMES)

    def test_horizon_of_a_slow_ship_is_capped(self):
        # The Chenjesu needs 50 frames to turn about and 41 to reach top speed.
        spec = ship_spec("Chenjesu")
        self.assertEqual(turn_about_frames(spec), 50)
        self.assertEqual(top_speed_frames(spec), 41)
        self.assertEqual(planet_horizon_frames(spec), 4 * const.FPS)

    def test_ships_at_top_speed_at_once_need_no_acceleration_time(self):
        for name in ("Arilou", "Slylandro"):
            with self.subTest(ship=name):
                spec = ship_spec(name)
                self.assertEqual(top_speed_frames(spec), 0)
                self.assertEqual(turn_about_frames(spec), 8)
                # 8 + 0 + margin is under the floor of one second.
                self.assertEqual(planet_horizon_frames(spec), const.FPS)

    def test_agile_ship_leaves_a_distant_hit_to_the_strategy(self):
        # Falling in from 900 hits the surface in about 29 frames: past the
        # Pkunk's horizon, within the Chenjesu's.
        planet = _planet(0.0, 900.0, own_forward=10.0)
        self.assertFalse(_rule_context("Pkunk", planet=planet).avoid_planet())
        self.assertTrue(_rule_context("Chenjesu", planet=planet).avoid_planet())


def _shot(bearing, distance, *, closing, damage=1.0, hit_points=1.0, group="enemy_a1",
          life_seconds=None, sideways=0.0):
    """An enemy shot at ``bearing`` flying in at ``closing``, drifting
    ``sideways`` clockwise across my view."""

    radians = math.radians(bearing)
    seen = _seen(bearing, distance, closing=closing, group=group, crew_effect=-damage)
    return dataclasses.replace(
        seen,
        velocity_forward=seen.velocity_forward - sideways * math.sin(radians),
        velocity_right=seen.velocity_right + sideways * math.cos(radians),
        lateral_speed=float(sideways),
        hit_points=float(hit_points),
        expires=life_seconds is not None,
        remaining_seconds=0.0 if life_seconds is None else float(life_seconds),
    )


def _duel(ship_a, ship_b, *, gap=1000.0):
    """Ship A at rest, facing screen-up, with ship B ``gap`` below it facing it.

    Real sprites and masks: guns sit off centre and hulls have their own
    shapes, which decide whether a shot lands.
    """

    from src.resources import HeadlessAssetManager

    resources = HeadlessAssetManager()
    audio = NullAudioService()
    simulation = BattleSimulation(
        None,
        create_ship(ship_a, 1, resources=resources, audio_service=audio),
        create_ship(ship_b, 2, resources=resources, audio_service=audio),
        audio_service=audio,
        rng=random.Random(3),
        include_stars=False,
        resources=resources,
    )
    simulation.world.objects[:] = [
        obj for obj in simulation.world.objects if type(obj).__name__ != "Asteroid"
    ]
    me, enemy = simulation.player1, simulation.player2
    me.position = [2000.0, 2000.0]
    enemy.position = [2000.0, 2000.0 + gap]
    for ship in (me, enemy):
        ship.velocity = [0.0, 0.0]
        ship.heading = 0
        ship.rotation = 0.0
    return simulation


def _first_hit_frame(simulation, frames):
    me = simulation.player1
    crew = me.current_hp
    idle = {1: ACTION_INDEX_TABLE[0], 2: ACTION_INDEX_TABLE[0]}
    for frame in range(frames):
        simulation.step(actions=idle)
        if me.current_hp < crew:
            return frame + 1
    return None


class EvadeRuleTests(unittest.TestCase):
    def setUp(self):
        # These check the shot prediction against the simulation, so the
        # shot must be seen the frame it is fired.
        patcher = mock.patch.object(const, "PERCEPTION_DELAY_FRAMES", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _predicted_hit_frame(self, simulation, ship_name, frames):
        view = ObservationView(_observation(simulation))
        velocity = (view.me.velocity_forward, view.me.velocity_right)
        tracks = threat_tracks(
            view.enemy_objects, velocity, enemy_name=view.enemy_ship_type, enemy_form="base"
        )
        self.assertEqual(len(tracks), 1)
        spec = ship_spec(ship_name)
        path, _ = fly_move(spec, velocity, frames, lambda flight: (None, False, False), planet=None)
        outcome = judge_move(tracks, path, spec.shot_reach, planet=None)
        self.assertEqual(outcome.hits, [0])
        from src.training.baseline.threats import first_contact, threat_path

        xs, ys = threat_path(tracks[0], path, planet=None)
        frame, _ = first_contact(xs, ys, path, tracks[0].contact(spec.shot_reach))
        return frame

    def _fire(self, simulation):
        fire = ACTION_INDEX_TABLE[ACTION_INDEX_BY_MASK[8]]
        simulation.step(actions={1: ACTION_INDEX_TABLE[0], 2: fire})

    def test_straight_shot_prediction_matches_the_simulation(self):
        simulation = _duel("Mycon", "KzerZa")
        self._fire(simulation)

        predicted = self._predicted_hit_frame(simulation, "Mycon", 30)
        actual = _first_hit_frame(simulation, 30)

        self.assertIsNotNone(actual)
        self.assertAlmostEqual(predicted, actual, delta=1.0)

    def test_homing_shot_prediction_matches_the_simulation(self):
        simulation = _duel("Mycon", "Earthling", gap=700.0)
        simulation.player1.velocity = [12.0, 0.0]
        self._fire(simulation)

        predicted = self._predicted_hit_frame(simulation, "Mycon", 48)
        actual = _first_hit_frame(simulation, 48)

        self.assertIsNotNone(actual)
        self.assertAlmostEqual(predicted, actual, delta=1.0)

    def test_leaves_a_shot_that_misses_to_the_strategy(self):
        # Ahead and to starboard, crossing my nose well clear of the hull.
        passing = _shot(60.0, 700.0, closing=0.0, sideways=-40.0)
        context = _rule_context("Pkunk", enemy_objects=[passing])

        self.assertFalse(context.evade_threats())
        self.assertIsNone(context.plan.turn)

    def test_ignores_a_shot_that_expires_before_it_arrives(self):
        short_lived = _shot(0.0, 800.0, closing=40.0, life_seconds=0.25)
        context = _rule_context("Chenjesu", enemy_objects=[short_lived])

        self.assertFalse(context.evade_threats())

    def test_waits_while_a_later_dodge_still_clears(self):
        # A quick ship has time to spare against a far, slow shot.
        context = _rule_context("Pkunk", enemy_objects=[_shot(90.0, 1500.0, closing=40.0)])

        self.assertFalse(context.evade_threats())
        self.assertIsNone(context.plan.turn)

    def test_thrusts_out_of_the_path_of_a_shot_from_the_side(self):
        context = _rule_context(
            "Pkunk", enemy_objects=[_shot(90.0, 220.0, closing=40.0, hit_points=2.0)]
        )

        self.assertTrue(context.evade_threats())
        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertTrue(context.plan.thrust)
        self.assertIsNone(context.plan.a1)

    def test_too_slow_to_dodge_shoots_down_the_threat_ahead(self):
        nuke = _shot(0.0, 700.0, closing=40.0, damage=4.0)
        context = _rule_context("Chenjesu", enemy_name="Earthling", enemy_objects=[nuke])

        self.assertTrue(context.evade_threats())
        self.assertTrue(context.plan.a1)
        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertFalse(context.plan.thrust)

    def test_druuge_fires_at_the_threat_ahead(self):
        shell = _shot(0.0, 900.0, closing=60.0, damage=6.0, hit_points=4.0)
        context = _rule_context("Druuge", enemy_name="Druuge", enemy_objects=[shell])

        self.assertTrue(context.evade_threats())
        self.assertTrue(context.plan.a1)

    def test_holds_fire_until_the_weapon_is_ready(self):
        nuke = _shot(0.0, 900.0, closing=40.0, damage=4.0)
        context = _rule_context(
            "Chenjesu", enemy_name="Earthling", enemy_objects=[nuke], a1_timer=0.2
        )

        self.assertTrue(context.evade_threats())
        # A held weapon is kept released, so the strategy can't fire it early.
        self.assertFalse(context.plan.a1)

    def test_leaves_steering_claimed_by_the_planet_rule(self):
        context = _rule_context("Pkunk", enemy_objects=[_shot(90.0, 220.0, closing=40.0)])
        context.plan.claim_turn(TURN_LEFT)

        self.assertFalse(context.evade_threats())

    def test_flight_slides_hops_and_turns_about_as_the_ships_do(self):
        slide = ShipFlight(ship_spec("Supox"), (0.0, 0.0), (0.0, 0.0))
        slide.thrust_angle = 90.0
        for _ in range(3):
            slide.step(None, True)
        self.assertGreater(slide.y, 0.0)
        self.assertAlmostEqual(slide.x, 0.0, places=6)
        self.assertEqual(slide.nose, 0.0)

        hop = ShipFlight(ship_spec("Umgah"), (0.0, 0.0), (5.0, 3.0))
        hop.step(None, False, hop=160.0)
        self.assertEqual((hop.x, hop.y, hop.vx, hop.vy), (-160.0, 0.0, 0.0, 0.0))

        probe = ShipFlight(ship_spec("Slylandro"), (0.0, 0.0), (0.0, 0.0))
        probe.step(None, False)
        self.assertAlmostEqual(probe.vx, ship_spec("Slylandro").max_thrust)
        probe.step(None, True)
        self.assertEqual(abs(probe.nose), 180.0)
        self.assertAlmostEqual(probe.vx, -ship_spec("Slylandro").max_thrust)

    def test_supox_slides_off_a_shot_dead_ahead(self):
        # Too close to turn away: only a slide clears the hull in time.
        shell = _shot(0.0, 300.0, closing=40.0, hit_points=5.0)
        context = _rule_context("Supox", enemy_objects=[shell])

        self.assertTrue(context.evade_threats())
        self.assertTrue(context.plan.a2)
        self.assertIn(context.plan.turn, (TURN_LEFT, TURN_RIGHT))

    def test_umgah_hops_back_from_a_shot_from_the_side(self):
        context = _rule_context("Umgah", enemy_objects=[_shot(90.0, 150.0, closing=40.0)])

        self.assertTrue(context.evade_threats())
        self.assertTrue(context.plan.a2)
        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertFalse(context.plan.thrust)

    def test_umgah_without_battery_has_no_dodge(self):
        context = _rule_context(
            "Umgah", enemy_objects=[_shot(90.0, 150.0, closing=40.0)], battery=0.0
        )

        context.evade_threats()
        self.assertIsNone(context.plan.a2)

    def test_slylandro_turns_off_a_shot_dead_ahead(self):
        shell = _shot(0.0, 600.0, closing=40.0)
        context = _rule_context("Slylandro", enemy_objects=[shell])

        self.assertTrue(context.evade_threats())
        self.assertIn(context.plan.turn, (TURN_LEFT, TURN_RIGHT))
        self.assertFalse(context.plan.thrust)

    def test_utwig_dodges_only_without_the_battery_to_shield(self):
        self.assertFalse(may_evade(_rule_context("Utwig", battery=5.0)))
        self.assertTrue(may_evade(_rule_context("Utwig", battery=0.0)))

    def test_syreen_dodges_except_while_closing_in_to_sing(self):
        def syreen(battery):
            context = _rule_context("Syreen", battery=battery, enemy_name="Spathi")
            context.view.me.crew = 30.0
            context.view.enemy_block.crew = 10.0
            context.view.enemy_block.max_thrust = 10.0
            return context

        self.assertFalse(may_evade(syreen(battery=16.0)))
        self.assertTrue(may_evade(syreen(battery=0.0)))
        self.assertTrue(may_evade(_rule_context("Pkunk")))


def _own_shot(bearing, distance, *, speed, heading=None, damage, hit_points=10.0):
    """One of my weapon 1 shots at ``bearing``, flying at ``speed`` toward
    ``heading`` (both degrees off my nose), with me at rest."""

    radians = math.radians(bearing)
    course = math.radians(bearing if heading is None else heading)
    return ObservedObject(
        group="friendly_a1",
        distance=float(distance),
        bearing=float(bearing),
        forward=distance * math.cos(radians),
        right=distance * math.sin(radians),
        velocity_forward=speed * math.cos(course),
        velocity_right=speed * math.sin(course),
        closing_speed=0.0,
        lateral_speed=0.0,
        crew_effect=-float(damage),
        battery_effect=0.0,
        hit_points=float(hit_points),
        expires=False,
        remaining_seconds=0.0,
    )


class SharedBehaviourTests(unittest.TestCase):
    """The per-ship behaviour table and the shared Phase 6 behaviours."""

    def test_table_keeps_the_gates_each_ship_had(self):
        def ships(field, value):
            return {name for name, entry in SHIP_BEHAVIOURS.items() if getattr(entry, field) == value}

        self.assertEqual(
            ships("shoot_down", False),
            {"Androsynth", "Arilou", "Earthling", "KohrAh", "Mycon", "Umgah"},
        )
        self.assertEqual(ships("evade_headings", False), {"Umgah"})
        self.assertEqual(ships("keep_moving", False), {"Slylandro", "Umgah"})
        self.assertEqual(ships("keep_moving_takes_aim", True), {"Chenjesu", "Mmrnmrhm"})
        self.assertEqual(ships("coast_above_top_speed", True), {"Earthling", "Thraddash"})
        self.assertIs(ship_behaviour("NoSuchShip"), DEFAULT_BEHAVIOUR)
        self.assertIs(ship_behaviour(None), DEFAULT_BEHAVIOUR)
        self.assertTrue(_rule_context("Thraddash").coast_above_top_speed)

    def test_ship_without_evade_skips_rule_1(self):
        context = _rule_context("Spathi")
        context.behaviour = dataclasses.replace(context.behaviour, evade=False)

        self.assertFalse(may_evade(context))

    def test_run_bearing_points_away_from_the_threats_by_their_cost(self):
        def tracks(*shots):
            return threat_tracks(shots, (0.0, 0.0), enemy_name="Spathi", enemy_form="base")

        self.assertAlmostEqual(run_bearing(tracks(_shot(30.0, 500.0, closing=20.0))), -150.0)
        # Equal threats on either beam cancel out.
        self.assertIsNone(
            run_bearing(
                tracks(_shot(90.0, 500.0, closing=20.0), _shot(-90.0, 500.0, closing=20.0))
            )
        )
        # The costlier (and nearer) one ahead outweighs one astern.
        away = run_bearing(
            tracks(
                _shot(0.0, 400.0, closing=20.0, damage=4.0),
                _shot(180.0, 800.0, closing=20.0, damage=1.0),
            )
        )
        self.assertAlmostEqual(abs(away), 180.0)

    def test_run_from_threats_adds_the_heading_away_from_them(self):
        shot = _shot(30.0, 500.0, closing=20.0)
        tracks = threat_tracks([shot], (0.0, 0.0), enemy_name="Spathi", enemy_form="base")

        def headings(run):
            context = _rule_context("Pkunk", enemy_objects=[shot])
            context.behaviour = dataclasses.replace(context.behaviour, run_from_threats=run)
            moves = context._evade_moves(None, None, False, tracks)
            return {plan[1] for plan, *_ in moves if plan[0] == EVADE_HEADING}

        self.assertNotIn(210, headings(False))
        self.assertIn(210, headings(True))
        self.assertEqual(headings(True) - headings(False), {210})

    def _recharging(self, enemy, *, battery, threshold=6.0):
        context = _rule_context("Spathi", enemy=enemy, enemy_name="Earthling", battery=battery)
        context.behaviour = dataclasses.replace(context.behaviour, recharge_battery=threshold)
        return context

    def test_recharge_retreat_runs_from_a_closer_while_below_the_threshold(self):
        reach = _rule_context("Spathi", enemy_name="Earthling").enemy_reach()
        near = reach + 45.0 + RECHARGE_REACH_MARGIN
        closing = _seen(0.0, near + 50.0, closing=30.0)

        running = self._recharging(closing, battery=2.0)
        with mock.patch.object(running, "steer", return_value=True) as steer:
            self.assertTrue(running.recharge_retreat())
        self.assertAlmostEqual(abs(steer.call_args.args[0]), 180.0)

        charged = self._recharging(closing, battery=6.0)
        self.assertFalse(charged.recharge_retreat())
        self.assertIsNone(charged.plan.turn)

    def test_recharge_retreat_never_closes_in(self):
        far = self._recharging(_seen(0.0, 3000.0, closing=-5.0), battery=0.0)

        self.assertFalse(far.recharge_retreat())
        self.assertIsNone(far.plan.turn)
        self.assertIsNone(far.plan.thrust)

    def test_recharge_retreat_is_off_without_a_threshold(self):
        context = _rule_context("Druuge", enemy=_seen(0.0, 300.0, closing=30.0), battery=0.0)

        self.assertFalse(context.recharge_retreat())
        self.assertIsNone(context.plan.turn)

    def test_hold_band_uses_the_given_band_or_the_ship_band(self):
        far = _seen(20.0, 2500.0)
        self.assertFalse(_rule_context("Spathi", enemy=far).hold_band())

        given = _rule_context("Spathi", enemy=far)
        with mock.patch.object(given, "steer", return_value=True) as steer:
            self.assertFalse(given.hold_band((800.0, 1200.0)))
        steer.assert_called_once()

        table = _rule_context("Spathi", enemy=_seen(0.0, 900.0, closing=40.0))
        table.behaviour = dataclasses.replace(table.behaviour, band=(800.0, 1200.0))
        with mock.patch.object(table, "steer", return_value=True) as steer:
            self.assertTrue(table.hold_band())
        self.assertAlmostEqual(abs(steer.call_args.args[0]), 180.0)


class BehaviourSchemaTests(unittest.TestCase):
    """Per-opponent overrides, bands, recharging, weapon use and the shared
    strategy run from the table."""

    def _with(self, ship, entry, *, enemy=None, enemy_name="Mycon", **kwargs):
        patch = mock.patch.dict(behaviour_table.SHIP_BEHAVIOURS, {ship: entry})
        patch.start()
        self.addCleanup(patch.stop)
        return _rule_context(ship, enemy=enemy, enemy_name=enemy_name, **kwargs)

    def test_overrides_apply_against_their_opponent_only(self):
        entry = ShipBehaviour(
            recharge_battery=5.0,
            opponents={"Vux": {"recharge_battery": 2.0, "run_from_threats": True}},
        )
        with mock.patch.dict(behaviour_table.SHIP_BEHAVIOURS, {"Spathi": entry}):
            against = ship_behaviour("Spathi", "Vux")
            self.assertEqual((against.recharge_battery, against.run_from_threats), (2.0, True))
            self.assertIs(ship_behaviour("Spathi", "Mycon"), entry)
            self.assertIs(ship_behaviour("Spathi"), entry)
            # A replaced entry is not served from the cache.
            changed = dataclasses.replace(entry, recharge_battery=7.0)
            behaviour_table.SHIP_BEHAVIOURS["Spathi"] = changed
            self.assertIs(ship_behaviour("Spathi", "Vux").evade, True)
            self.assertEqual(ship_behaviour("Spathi", "Mycon").recharge_battery, 7.0)
        self.assertEqual(_rule_context("Spathi", enemy_name="Vux").behaviour.band.near, 300.0)

    def test_unknown_postures_and_fire_kinds_are_rejected(self):
        with self.assertRaises(ValueError):
            Band(500.0, 900.0, inside="sideways")
        with self.assertRaises(ValueError):
            WeaponUse("lob")

    def test_band_rises_outside_the_opponent_reach(self):
        context = self._with(
            "Spathi", ShipBehaviour(), enemy=_seen(0.0, 2000.0), enemy_name="Earthling"
        )
        reach = context.enemy_reach()
        self.assertEqual(
            context.band_limits(Band(300.0, 500.0, outside_reach_margin=100.0)),
            (reach + 45.0 + 100.0, reach + 45.0 + 100.0 + 150.0),
        )
        self.assertEqual(context.band_limits(Band(5000.0, 6000.0, outside_reach_margin=0.0)), (5000.0, 6000.0))
        self.assertEqual(context.band_limits(Band(500.0, None)), (500.0, math.inf))

    def test_a_reach_cap_keeps_the_band_within_my_own_reach(self):
        # Capped at 0.95 of the Supox gun, the band is the Supox's own rule
        # (``_supox_band``): outside a shorter reach while the gun still
        # reaches, else 600-900.
        capped = Band(600.0, 900.0, outside_reach_margin=150.0, reach_cap=0.95)
        uncapped = Band(600.0, 900.0, outside_reach_margin=150.0)
        moved = 0
        for opponent in ("Earthling", "Shofixti", "Mycon", "Spathi", "Chmmr", "Druuge", "Melnorme"):
            with self.subTest(opponent=opponent):
                context = self._with("Supox", ShipBehaviour(), enemy=_seen(0.0, 2000.0), enemy_name=opponent)
                self.assertEqual(context.band_limits(capped), strategy_module._supox_band(context))
                self.assertLessEqual(context.band_limits(capped)[1], 0.95 * weapon_spec("Supox", 1).range)
                moved += context.band_limits(uncapped) != context.band_limits(capped)
        self.assertGreater(moved, 0)

    def test_band_without_a_far_edge_never_closes_in(self):
        context = self._with("Spathi", ShipBehaviour(band=Band(500.0, None)), enemy=_seen(0.0, 5000.0))

        self.assertFalse(context.hold_band())
        self.assertIsNone(context.plan.turn)
        self.assertIsNone(context.plan.thrust)

    def test_band_postures_inside_the_band(self):
        inside = _seen(40.0, 800.0)
        free = self._with("Spathi", ShipBehaviour(band=Band(600.0, 1000.0)), enemy=inside)
        free.hold_band()
        self.assertIsNone(free.plan.turn)

        face = self._with(
            "Spathi", ShipBehaviour(band=Band(600.0, 1000.0, inside=BAND_INSIDE_FACE)), enemy=inside
        )
        face.hold_band()
        self.assertEqual(face.plan.turn, TURN_RIGHT)
        self.assertFalse(face.plan.thrust)

        orbit = self._with(
            "Spathi", ShipBehaviour(band=Band(600.0, 1000.0, inside=BAND_INSIDE_ORBIT)), enemy=inside
        )
        with mock.patch.object(orbit, "steer", return_value=True) as steer:
            orbit.hold_band()
        self.assertAlmostEqual(abs(wrap_degrees(steer.call_args.args[0] - 40.0)), 90.0)

    def test_band_and_recharge_can_run_off_the_opponent_line(self):
        closer = _seen(0.0, 650.0, closing=40.0)
        veer = self._with(
            "Spathi", ShipBehaviour(band=Band(600.0, 1000.0, run_offset=110.0)), enemy=closer
        )
        with mock.patch.object(veer, "steer", return_value=True) as steer:
            self.assertTrue(veer.hold_band())
        self.assertAlmostEqual(abs(steer.call_args.args[0]), 110.0)

        reach = _rule_context("Spathi", enemy_name="Earthling").enemy_reach()
        entry = ShipBehaviour(recharge_battery=5.0, recharge_run_offset=135.0)
        recharging = self._with(
            "Spathi",
            entry,
            enemy=_seen(0.0, reach + 45.0 + RECHARGE_REACH_MARGIN + 50.0, closing=30.0),
            enemy_name="Earthling",
            battery=1.0,
        )
        with mock.patch.object(recharging, "steer", return_value=True) as steer:
            self.assertTrue(recharging.recharge_retreat())
        self.assertAlmostEqual(abs(steer.call_args.args[0]), 135.0)

    def test_a_back_off_band_backs_off_inside_its_near_edge_closing_or_not(self):
        band = Band(280.0, 400.0, approach_speed=math.inf, run_offset=110.0, back_off=True)
        entry = ShipBehaviour(band=band)
        for closing in (5.0, -5.0):
            inside = self._with("Arilou", entry, enemy=_seen(0.0, 200.0, closing=closing))
            with mock.patch.object(inside, "steer", return_value=True) as steer:
                self.assertFalse(inside.hold_band())
            self.assertAlmostEqual(abs(steer.call_args.args[0]), 110.0)

        between = self._with("Arilou", entry, enemy=_seen(0.0, 350.0, closing=40.0))
        self.assertFalse(between.hold_band())
        self.assertIsNone(between.plan.turn)

        beyond = self._with("Arilou", entry, enemy=_seen(0.0, 900.0, closing=40.0))
        with mock.patch.object(beyond, "steer", return_value=True) as steer:
            beyond.hold_band()
        self.assertTrue(steer.call_args.kwargs["thrust"])

    def test_a_chase_band_flies_straight_at_the_opponent_inside_it(self):
        band = Band(0.0, None, inside=BAND_INSIDE_CHASE, back_off=True)
        context = self._with("Chmmr", ShipBehaviour(band=band), enemy=_seen(30.0, 900.0))
        with mock.patch.object(context, "approach_opponent", return_value=True) as chase:
            context.hold_band()
        chase.assert_called_once()

    def test_spathi_settings_differ_by_opponent(self):
        self.assertEqual(
            ship_behaviour("Spathi", "Mycon").band, Band(756.0, 1080.0, approach_speed=60.0)
        )
        self.assertIsNone(ship_behaviour("Spathi", "Mycon").recharge_battery)
        self.assertIs(ship_behaviour("Spathi", "Chmmr"), ship_behaviour("Spathi"))
        vux = ship_behaviour("Spathi", "Vux")
        self.assertEqual((vux.band.near, vux.band.far, vux.band.run_offset), (300.0, 756.0, 110.0))

    def test_recharging_lasts_until_the_resume_battery(self):
        memory = BaselineMemory()
        entry = ShipBehaviour(recharge_battery=3.0, recharge_resume=8.0)

        def recharging(battery):
            return self._with("Spathi", entry, battery=battery, memory=memory).recharging()

        self.assertFalse(recharging(5.0))
        self.assertTrue(recharging(2.0))
        self.assertTrue(recharging(5.0))
        self.assertFalse(recharging(8.0))
        self.assertFalse(recharging(5.0))

    def test_aimed_weapon_turns_onto_the_opponent_and_fires_on_target(self):
        entry = ShipBehaviour(weapons=(WeaponUse(FIRE_AIMED),))
        off = self._with("Spathi", entry, enemy=_seen(40.0, 800.0))
        self.assertFalse(off.fire_weapon(1))
        self.assertEqual(off.plan.turn, TURN_RIGHT)

        on = self._with("Spathi", entry, enemy=_seen(0.0, 800.0))
        self.assertTrue(on.fire_weapon(1))
        self.assertTrue(on.plan.a1)

        beyond = self._with("Spathi", entry, enemy=_seen(0.0, 800.0))
        self.assertFalse(beyond.fire_weapon(1, WeaponUse(range_fraction=0.5)))

    def test_aimed_weapon_without_aim_leaves_the_nose_alone(self):
        context = self._with("Spathi", ShipBehaviour(), enemy=_seen(40.0, 800.0))

        self.assertFalse(context.fire_weapon(1, WeaponUse(aim=False)))
        self.assertIsNone(context.plan.turn)

    def test_reserve_keeps_battery_for_another_weapon(self):
        use = WeaponUse(FIRE_AIMED, reserve=3.0)
        enemy = _seen(0.0, 800.0)
        # The Spathi gun costs 2.
        self.assertFalse(self._with("Spathi", ShipBehaviour(), enemy=enemy, battery=4.0).fire_weapon(1, use))
        self.assertTrue(self._with("Spathi", ShipBehaviour(), enemy=enemy, battery=5.0).fire_weapon(1, use))

    def test_aimed_weapon_can_aim_at_a_satellite(self):
        enemy = _seen(0.0, 1200.0)
        satellite = _satellite(enemy, 180.0)
        context = self._with("Spathi", ShipBehaviour(), enemy=enemy, enemy_name="Chmmr")
        with mock.patch.object(BaselineContext, "satellite_shot", return_value=(satellite, 30.0)):
            self.assertFalse(context.fire_weapon(1, WeaponUse(satellites=True)))
        self.assertEqual(context.plan.turn, TURN_RIGHT)

    def test_launched_weapon_needs_battery_and_to_outrun_me(self):
        use = WeaponUse(FIRE_LAUNCH)
        behind = _seen(180.0, 600.0)

        launches = self._with("Spathi", ShipBehaviour(), enemy=behind)
        self.assertTrue(launches.fire_weapon(2, use))
        self.assertIsNone(launches.plan.turn)

        self.assertFalse(self._with("Spathi", ShipBehaviour(), enemy=behind, battery=2.0).fire_weapon(2, use))
        self.assertFalse(
            self._with("Spathi", ShipBehaviour(), enemy=_seen(180.0, 2000.0)).fire_weapon(2, use)
        )
        # Backing up faster than the rear missile flies would overrun it.
        overrun = self._with("Spathi", ShipBehaviour(), enemy=behind)
        overrun.view.me.velocity_forward = -(weapon_spec("Spathi", 2).speed + 1.0)
        self.assertFalse(overrun.fire_weapon(2, use))

    def test_defending_weapon_presses_on_an_imminent_hit_or_crew_loss(self):
        use = WeaponUse(FIRE_DEFEND, threat_seconds=0.5, on_crew_loss=True)
        enemy = _seen(0.0, 2000.0)
        close_shot = _shot(30.0, 60.0, closing=40.0)
        far_shot = _shot(30.0, 1500.0, closing=40.0)

        def presses(objects=(), *, dropped=False, battery=None):
            memory = BaselineMemory()
            context = self._with(
                "Yehat", ShipBehaviour(), enemy=enemy, enemy_objects=objects,
                memory=memory, battery=battery,
            )
            context.view.me.crew = 10.0
            memory.previous_crew = 12.0 if dropped else 10.0
            return context.fire_weapon(2, use)

        self.assertTrue(presses([close_shot]))
        self.assertFalse(presses([far_shot]))
        self.assertTrue(presses(dropped=True))
        self.assertFalse(presses([close_shot], battery=0.0))

    def test_special_and_unused_weapons_fire_nothing(self):
        context = self._with("Spathi", ShipBehaviour(weapons=(None, WeaponUse(FIRE_SPECIAL))), enemy=_seen(0.0, 500.0))

        self.assertFalse(context.fire_weapon(1))
        self.assertFalse(context.fire_weapon(2))
        self.assertIsNone(context.plan.a1)
        self.assertIsNone(context.plan.a2)

    def test_shared_strategy_runs_specials_and_form_switch_first(self):
        calls = []
        entry = ShipBehaviour(
            specials=("first", "second"),
            form_switch="switch",
            band=Band(600.0, 1000.0),
            weapons=(WeaponUse(FIRE_AIMED),),
        )
        context = self._with("Spathi", entry, enemy=_seen(0.0, 800.0))
        rules = {
            "first": lambda ctx: calls.append("first"),
            "second": lambda ctx: calls.append("second") or ctx.a2(True),
        }
        with mock.patch.dict(strategy_module.SPECIAL_RULES, rules), mock.patch.dict(
            strategy_module.FORM_SWITCHES, {"switch": lambda ctx: calls.append("switch")}
        ):
            strategy_module.shared(context)

        self.assertEqual(calls, ["first", "second", "switch"])
        self.assertTrue(context.plan.a2)
        self.assertTrue(context.plan.a1)


class HeldWeaponTests(unittest.TestCase):
    """Phase 3A: weapon 1 held by the Chenjesu, Kohr-Ah and Melnorme."""

    ENEMY_AHEAD = _seen(0.0, 1500.0)

    def _context(self, ship, *, held=("a1",), shots=(), enemy=ENEMY_AHEAD, **kwargs):
        context = _rule_context(ship, enemy=enemy, held=held, friendly_shots=shots, **kwargs)
        context.view.me.chenjesu_dogis = 4
        return context

    def test_a_crystal_is_worth_its_damage_only_on_course_for_the_enemy(self):
        def value(heading, held=("a1",)):
            crystal = _own_shot(0.0, 500.0, speed=64.0, heading=heading, damage=6.0)
            return self._context("Chenjesu", held=held, shots=[crystal]).held_shot_value()

        self.assertAlmostEqual(value(0.0), 6.0)
        self.assertEqual(value(90.0), 0.0)
        self.assertEqual(value(180.0), 0.0)
        self.assertIsNone(value(0.0, held=()))

    def test_chenjesu_keeps_holding_a_crystal_closing_on_the_enemy(self):
        crystal = _own_shot(0.0, 500.0, speed=64.0, damage=6.0)
        context = self._context("Chenjesu", shots=[crystal])
        STRATEGIES["Chenjesu"](context)

        self.assertTrue(context.plan.a1)

    def test_a_crystal_closing_but_due_to_miss_is_kept(self):
        # Passing 750 wide of the enemy: the enemy may still turn into it.
        crystal = _own_shot(30.0, 500.0, speed=64.0, damage=6.0)
        context = self._context("Chenjesu", shots=[crystal])
        STRATEGIES["Chenjesu"](context)

        self.assertEqual(context.held_shot_value(), 0.0)
        self.assertTrue(context.plan.a1)

    def test_a_receding_saw_is_let_go_and_a_fresh_press_follows(self):
        saw = _own_shot(0.0, 500.0, speed=64.0, heading=180.0, damage=4.0)
        holding = self._context("KohrAh", shots=[saw])
        STRATEGIES["KohrAh"](holding)
        released = self._context("KohrAh", held=(), shots=[saw])
        STRATEGIES["KohrAh"](released)

        self.assertFalse(holding.plan.a1)
        self.assertTrue(released.plan.a1)

    def test_a_parked_saw_is_not_held_but_still_stops_a_threat(self):
        parked = _own_shot(0.0, 400.0, speed=0.0, damage=4.0)
        shell = _shot(0.0, 900.0, closing=40.0, damage=3.0, hit_points=2.0)
        context = self._context("KohrAh", shots=[parked], enemy_objects=[shell])

        self.assertIsNone(context.held_shot())
        self.assertFalse(context.evade_threats())

    def test_a_threat_my_crystal_will_meet_first_is_left_to_it(self):
        crystal = _own_shot(0.0, 200.0, speed=64.0, damage=6.0)
        nuke = _shot(0.0, 800.0, closing=40.0, damage=4.0, hit_points=4.0)
        context = self._context(
            "Chenjesu", shots=[crystal], enemy=_seen(90.0, 1500.0), enemy_objects=[nuke]
        )
        STRATEGIES["Chenjesu"](context)

        self.assertEqual(context.stopped_threats(threat_tracks(
            [nuke], (0.0, 0.0), enemy_name="Spathi", enemy_form="base"
        )), {0})
        # Kept held for the threat, though the enemy is elsewhere.
        self.assertTrue(context.plan.a1)

    def test_other_ships_do_not_count_their_shots_in_flight(self):
        shot = _own_shot(0.0, 200.0, speed=64.0, damage=6.0)
        nuke = _shot(0.0, 800.0, closing=40.0, damage=4.0, hit_points=4.0)
        context = _rule_context("Druuge", friendly_shots=[shot], enemy_objects=[nuke])
        tracks = threat_tracks([nuke], (0.0, 0.0), enemy_name="Spathi", enemy_form="base")

        self.assertEqual(context.stopped_threats(tracks), set())

    def test_chenjesu_lets_go_of_a_crystal_to_shoot_down_a_threat_and_fires_next(self):
        # The crystal recedes from the enemy behind it; a nuke comes in ahead.
        crystal = _own_shot(90.0, 500.0, speed=64.0, damage=6.0)
        nuke = _shot(0.0, 700.0, closing=40.0, damage=4.0)
        holding = self._context(
            "Chenjesu", shots=[crystal], enemy_objects=[nuke], enemy_name="Earthling"
        )
        released = self._context(
            "Chenjesu", held=(), shots=[crystal], enemy_objects=[nuke], enemy_name="Earthling"
        )

        self.assertTrue(holding.evade_threats())
        self.assertFalse(holding.plan.a1)
        self.assertTrue(released.evade_threats())
        self.assertTrue(released.plan.a1)

    def test_a_crystal_worth_more_than_the_threat_is_kept(self):
        crystal = _own_shot(0.0, 500.0, speed=64.0, heading=180.0, damage=6.0)
        enemy_behind = _seen(180.0, 1500.0)
        nuke = _shot(0.0, 700.0, closing=40.0, damage=1.0)
        context = self._context(
            "Chenjesu",
            shots=[crystal],
            enemy=enemy_behind,
            enemy_objects=[nuke],
            enemy_name="Earthling",
        )
        context.evade_threats()

        self.assertIsNone(context.plan.a1)

    def test_a_shoot_down_with_weapon_1_held_fires_after_the_release_frame(self):
        nuke = _shot(0.0, 700.0, closing=40.0, damage=4.0)
        tracks = threat_tracks([nuke], (0.0, 0.0), enemy_name="Earthling", enemy_form="base")
        spec = ship_spec("Chenjesu")
        coast, _ = fly_move(spec, (0.0, 0.0), 40, lambda flight: (None, False, False), planet=None)

        def fire_frames(held):
            context = self._context("Chenjesu", held=held, enemy_name="Earthling")
            frames = []

            def judge(controller, **kwargs):
                frames.append(kwargs["shot"][0])
                return coast, judge_move(tracks, coast, spec.shot_reach, planet=None)

            context._shoot_down_options(tracks, coast, judge)
            return frames

        self.assertEqual(fire_frames(()), [0])
        self.assertEqual(fire_frames(("a1",)), [1])

    def test_melnorme_keeps_charging_when_nothing_competes(self):
        charge = _own_shot(0.0, 100.0, speed=0.0, damage=8.0, hit_points=8.0)
        context = self._context("Melnorme", shots=[charge], enemy=_seen(60.0, 3000.0))
        STRATEGIES["Melnorme"](context)

        self.assertTrue(context.plan.a1)

    def test_melnorme_shoots_a_threat_down_with_the_charge_it_holds(self):
        charge = _own_shot(0.0, 100.0, speed=0.0, damage=4.0, hit_points=4.0)
        missile = _shot(45.0, 600.0, closing=40.0, damage=3.0, hit_points=2.0)
        tracks = threat_tracks([missile], (0.0, 0.0), enemy_name="Spathi", enemy_form="base")
        spec = ship_spec("Melnorme")
        coast, _ = fly_move(spec, (0.0, 0.0), 30, lambda flight: (None, False, False), planet=None)
        shots = []

        def judge(controller, **kwargs):
            shots.append(kwargs["shot"])
            outcome = judge_move(tracks, coast, spec.shot_reach, planet=None, shot=kwargs["shot"])
            return coast, outcome

        enemy = _seen(180.0, 3000.0)
        charging = self._context("Melnorme", shots=[charge], enemy=enemy)
        options = charging._shoot_down_options(tracks, coast, judge)
        idle = self._context("Melnorme", held=(), enemy=enemy)

        self.assertEqual(len(options), 1)
        # The partial charge flies at its own damage, and stops the missile.
        self.assertEqual(shots[0][1].damage, 4.0)
        self.assertEqual(options[0][0], 0.0)
        self.assertEqual(idle._shoot_down_options(tracks, coast, judge), [])

    def test_melnorme_keeps_charging_while_turning_and_lets_go_once_on(self):
        turning = self._context("Melnorme")
        turning._shoot_down(45.0)
        on_target = self._context("Melnorme")
        on_target._shoot_down(0.0)

        self.assertEqual(turning.plan.turn, TURN_RIGHT)
        self.assertTrue(turning.plan.a1)
        self.assertFalse(on_target.plan.a1)

    def test_melnorme_charge_on_the_nose_stops_a_threat_coming_at_it(self):
        charge = _own_shot(0.0, 100.0, speed=0.0, damage=4.0, hit_points=4.0)
        missile = _shot(0.0, 600.0, closing=40.0, damage=3.0, hit_points=2.0)
        context = self._context("Melnorme", shots=[charge], enemy_objects=[missile])

        self.assertFalse(context.evade_threats())


class KeepMovingRuleTests(unittest.TestCase):
    def test_a_ship_at_rest_turns_across_the_line_to_the_opponent(self):
        context = _rule_context("Chenjesu", enemy=_seen(10.0, 1500.0))

        self.assertTrue(context.keep_moving())
        # 10 - 90 = -80 needs less turning than 10 + 90 = 100.
        self.assertEqual(context.plan.turn, TURN_LEFT)
        self.assertFalse(context.plan.thrust)

    def test_thrusts_once_the_nose_points_across(self):
        context = _rule_context("Chenjesu", enemy=_seen(90.0, 1500.0))

        self.assertTrue(context.keep_moving())
        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertTrue(context.plan.thrust)

    def test_builds_on_the_drift_already_carried(self):
        context = _rule_context("Chenjesu", enemy=_seen(0.0, 1500.0))
        # Drifting to port: keep building that way, not toward the nearer side.
        context.view.me.velocity_right = -5.0

        self.assertTrue(context.keep_moving())
        self.assertEqual(context.plan.turn, TURN_LEFT)

    def test_leaves_a_ship_that_moves_or_already_has_speed_alone(self):
        claimed = _rule_context("Chenjesu", enemy=_seen(10.0, 1500.0))
        claimed.plan.claim_thrust(False)
        self.assertFalse(claimed.keep_moving())

        quick = ship_spec("Chenjesu").max_thrust
        fast = _rule_context("Chenjesu", enemy=_seen(10.0, 1500.0), speed=quick)
        self.assertFalse(fast.keep_moving())

        self.assertFalse(_rule_context("Umgah", enemy=_seen(10.0, 1500.0)).keep_moving())

    def test_takes_the_turn_from_an_aim_only_while_the_weapon_cannot_fire(self):
        in_range = _rule_context("Chenjesu", enemy=_seen(10.0, 400.0))
        in_range.plan.claim_turn(TURN_RIGHT)
        self.assertFalse(in_range.keep_moving())
        self.assertEqual(in_range.plan.turn, TURN_RIGHT)

        reloading = _rule_context("Chenjesu", enemy=_seen(10.0, 400.0), a1_timer=0.5)
        reloading.plan.claim_turn(TURN_RIGHT)
        self.assertTrue(reloading.keep_moving())
        self.assertEqual(reloading.plan.turn, TURN_LEFT)

        # Only some ships leave their aim.
        druuge = _rule_context("Druuge", enemy=_seen(10.0, 400.0), a1_timer=0.5)
        druuge.plan.claim_turn(TURN_RIGHT)
        self.assertFalse(druuge.keep_moving())
        self.assertEqual(druuge.plan.turn, TURN_RIGHT)


class ShipRuleTests(unittest.TestCase):
    def test_pkunk_approaches_from_half_battery_and_flees_below(self):
        enemy = _seen(0.0, 2000.0, closing=5.0)
        charged = _rule_context("Pkunk", enemy=enemy, battery_fraction=0.5)
        strategy_for("Pkunk", "earlier")(charged)
        low = _rule_context("Pkunk", enemy=enemy, battery_fraction=0.49)
        strategy_for("Pkunk", "earlier")(low)

        self.assertTrue(charged.plan.a2)
        self.assertTrue(low.plan.a2)
        # Charged, it heads for a pass beside the opponent; low, it flees at 135.
        self.assertTrue(charged.plan.thrust)
        self.assertTrue(low.plan.thrust)
        self.assertEqual(charged.memory.sides.keys(), {"pkunk_pass"})
        self.assertIn("avoid_opponent", low.memory.sides)
        self.assertNotIn("pkunk_pass", low.memory.sides)

    def test_the_pkunk_entry_never_recharges(self):
        low = _rule_context("Pkunk", enemy=_seen(0.0, 2000.0, closing=5.0), battery=1.0, max_battery=12.0)

        self.assertIs(STRATEGIES["Pkunk"], strategy_module.shared)
        self.assertFalse(low.recharging())

    def test_the_pkunk_entry_fires_the_nearest_gun_within_45_degrees(self):
        reach = weapon_spec("Pkunk", 1).range

        def fires(bearing, distance=250.0):
            context = _rule_context("Pkunk", enemy=_seen(bearing, distance))
            STRATEGIES["Pkunk"](context)
            return bool(context.plan.a1)

        # The front gun at 40 degrees off, a side gun at 50 and 130.
        self.assertTrue(fires(40.0))
        self.assertTrue(fires(50.0))
        self.assertTrue(fires(-130.0))
        # Behind it, every gun is more than 45 degrees off.
        self.assertFalse(fires(170.0))
        self.assertFalse(fires(40.0, distance=reach + 60.0))

    def test_pkunk_fires_a_side_gun_at_an_opponent_abeam(self):
        # Inside firing distance but outside the band where approach backs off.
        context = _rule_context("Pkunk", enemy=_seen(90.0, 250.0))
        strategy_for("Pkunk", "earlier")(context)

        self.assertTrue(context.plan.a1)
        self.assertEqual(context.plan.turn, TURN_NONE)

    def test_pkunk_holds_fire_at_long_reach_unless_the_opponent_closes(self):
        a1_range = weapon_spec("Pkunk", 1).range
        distance = a1_range * 0.85

        def fires(closing):
            context = _rule_context(
                "Pkunk",
                enemy=_seen(90.0, distance, closing=closing),
                battery_fraction=0.2,
            )
            strategy_for("Pkunk", "earlier")(context)
            return bool(context.plan.a1)

        self.assertFalse(fires(0.0))
        self.assertTrue(fires(5.0))

    def test_low_battery_pkunk_keeps_station_just_outside_weapon_range(self):
        far = _rule_context("Pkunk", enemy=_seen(0.0, 2000.0), battery_fraction=0.2)
        strategy_for("Pkunk", "earlier")(far)
        self.assertTrue(far.plan.thrust)
        self.assertIn("station_far", far.memory.sides)

        # Against a short-reach opponent the station lies just beyond its
        # reach (432) plus a hull and margin: 627, with the band from 494.
        holding = _rule_context(
            "Pkunk",
            enemy=_seen(90.0, 560.0),
            enemy_name="Pkunk",
            battery_fraction=0.2,
        )
        strategy_for("Pkunk", "earlier")(holding)
        self.assertFalse(holding.plan.thrust)
        # Points the starboard gun at the opponent while it waits.
        self.assertEqual(holding.plan.turn, TURN_NONE)
        self.assertFalse(holding.plan.a1)

    def test_zoq_fot_pik_saves_battery_for_the_tongue(self):
        a1_range = weapon_spec("ZoqFotPik", 1).range
        enemy = _seen(0.0, a1_range * 0.8)
        a2_cost = SHIP_DEFINITIONS["ZoqFotPik"].a2_cost

        def fires_a1(battery, enemy_name="Spathi"):
            context = _rule_context(
                "ZoqFotPik",
                enemy=enemy,
                enemy_name=enemy_name,
                battery=battery,
            )
            context.view.enemy_block.trackable = True
            context.view.enemy_block.max_thrust = 0.0
            STRATEGIES["ZoqFotPik"](context)
            return bool(context.plan.a1)

        self.assertFalse(fires_a1(a2_cost))
        self.assertTrue(fires_a1(a2_cost + 1))
        # The tongue cannot catch a Slylandro, so weapon 1 fires freely.
        self.assertTrue(fires_a1(a2_cost, enemy_name="Slylandro"))

    def test_zoq_fot_pik_lunges_and_tongues_when_ready(self):
        a2_cost = SHIP_DEFINITIONS["ZoqFotPik"].a2_cost

        def context_at(bearing, distance, battery, variant=None):
            context = _rule_context(
                "ZoqFotPik",
                enemy=_seen(bearing, distance),
                battery=battery,
                battery_fraction=battery / 10.0,
            )
            context.view.enemy_block.trackable = True
            context.view.enemy_block.max_thrust = 0.0
            strategy_for("ZoqFotPik", variant)(context)
            return context

        lunging = context_at(0.0, 600.0, a2_cost)
        self.assertTrue(lunging.plan.thrust)
        self.assertEqual(lunging.plan.turn, TURN_NONE)
        self.assertFalse(lunging.plan.a2)

        # Centre distance beyond the tongue's length still touches the hull,
        # and the narrow tongue reaches a hull several degrees off the nose.
        striking = context_at(6.0, weapon_spec("ZoqFotPik", 2).range + 15.0, a2_cost)
        self.assertTrue(striking.plan.a2)

        # Without tongue battery it holds its band (backing off only inside
        # 100) instead of lunging; the earlier rules backed off inside 0.7
        # of weapon 1's reach.
        waiting = context_at(0.0, 600.0, a2_cost - 1)
        self.assertFalse(waiting.plan.a2)
        self.assertNotIn("band_near", waiting.memory.sides)
        backing_off = context_at(0.0, 80.0, a2_cost - 1)
        self.assertIn("band_near", backing_off.memory.sides)
        earlier = context_at(0.0, 600.0, a2_cost - 1, variant="earlier")
        self.assertIn("a1_range_near", earlier.memory.sides)
        self.assertFalse(earlier.plan.a2)

    def test_intercept_seconds_use_top_speed_and_opponent_motion(self):
        max_thrust = SHIP_DEFINITIONS["ZoqFotPik"].max_thrust
        idle = _rule_context("ZoqFotPik", enemy=_seen(0.0, 100.0 + max_thrust * const.FPS))
        self.assertAlmostEqual(idle.intercept_seconds(100.0), 1.0)
        self.assertEqual(idle.intercept_seconds(5000.0), 0.0)

        fleeing = _rule_context(
            "ZoqFotPik",
            enemy=_seen(0.0, 1000.0, closing=-(max_thrust + 1.0)),
        )
        self.assertEqual(fleeing.intercept_seconds(100.0), math.inf)

    def _umgah(
        self,
        enemy,
        *,
        battery=30.0,
        enemy_objects=(),
        memory=None,
        enemy_name="Spathi",
        facing=None,
    ):
        context = _rule_context(
            "Umgah",
            enemy=enemy,
            enemy_name=enemy_name,
            battery=battery,
            enemy_objects=enemy_objects,
            memory=memory,
        )
        context.view.enemy_block.opponent_bearing = facing
        STRATEGIES["Umgah"](context)
        return context

    def _umgah_tail_aligned_enemy(self, distance):
        """An opponent placed so the hop line passes just beside it."""

        offset = math.degrees(math.asin(UMGAH_PASS_CLEARANCE / distance))
        return _seen(180.0 - offset, distance)

    def test_umgah_hops_only_until_the_opponent_will_sit_in_the_cone(self):
        entry = UMGAH_RUN_CLEARANCE / math.tan(math.radians(UMGAH_CONE_HALF_ANGLE))
        behind = dataclasses.replace(_seen(180.0, 300.0), right=UMGAH_RUN_CLEARANCE)
        # 300 behind plus the entry distance, at 160 a hop: three hops.
        self.assertEqual(umgah_hops_needed(behind), math.ceil((300.0 + entry) / 160.0))
        in_front = dataclasses.replace(_seen(0.0, entry + 1.0), right=UMGAH_RUN_CLEARANCE)
        self.assertEqual(umgah_hops_needed(in_front), 0)

    def test_umgah_blitzes_from_afar_only_with_the_battery_for_the_whole_run(self):
        enemy = self._umgah_tail_aligned_enemy(3000.0)

        self.assertTrue(self._umgah(enemy).plan.a2)
        self.assertFalse(self._umgah(enemy, battery=12.0).plan.a2)

    def test_umgah_will_not_hop_onto_a_mine_but_hops_through_an_incoming_shot(self):
        enemy = self._umgah_tail_aligned_enemy(1000.0)
        mine = _seen(180.0, 400.0, group="enemy_a2")
        shot = _seen(180.0, 400.0, closing=30.0, group="enemy_a1")

        self.assertFalse(self._umgah(enemy, enemy_objects=(mine,)).plan.a2)
        # It hits whether the Umgah hops or waits.
        self.assertTrue(self._umgah(enemy, enemy_objects=(shot,)).plan.a2)

    def test_umgah_passes_on_the_side_away_from_the_opponents_nose(self):
        enemy = _seen(0.0, 1000.0)
        # I am to its right, so its nose points to my right: pass on my left,
        # which keeps the opponent on the right once the tail points at it.
        right_of_nose = _rule_context("Umgah", enemy=enemy)
        right_of_nose.view.enemy_block.opponent_bearing = 20.0
        self.assertGreater(umgah_tail_heading(right_of_nose, enemy), 0.0)
        left_of_nose = _rule_context("Umgah", enemy=enemy)
        left_of_nose.view.enemy_block.opponent_bearing = -20.0
        self.assertLess(umgah_tail_heading(left_of_nose, enemy), 0.0)

    def test_umgah_waits_facing_a_short_reach_opponent(self):
        context = self._umgah(_seen(30.0, 1000.0), enemy_name="Pkunk")

        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertFalse(context.plan.thrust)
        self.assertFalse(context.plan.a2)

    def test_umgah_hops_backward_past_an_opponent_behind_it(self):
        context = self._umgah(self._umgah_tail_aligned_enemy(1000.0))

        self.assertTrue(context.plan.a2)
        self.assertFalse(context.plan.thrust)
        self.assertFalse(context.plan.a1)

    def test_umgah_turns_its_tail_toward_a_distant_opponent_ahead(self):
        context = self._umgah(_seen(0.0, 1000.0))

        self.assertNotEqual(context.plan.turn, TURN_NONE)
        self.assertFalse(context.plan.a2)

    def test_umgah_cones_an_opponent_ahead_within_reach(self):
        context = self._umgah(_seen(30.0, 200.0))

        self.assertTrue(context.plan.a1)
        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertFalse(context.plan.a2)

    def test_umgah_hops_while_recharging_only_with_the_battery_for_a_run(self):
        memory = BaselineMemory()
        enemy = self._umgah_tail_aligned_enemy(1000.0)

        self.assertFalse(self._umgah(enemy, battery=5.0, memory=memory).plan.a2)
        self.assertTrue(memory.values["umgah_recharging"])
        # Seven hops reach the cone from 1000 away.
        self.assertFalse(self._umgah(enemy, battery=7.0, memory=memory).plan.a2)
        self.assertTrue(self._umgah(enemy, battery=20.0, memory=memory).plan.a2)

    def _pkunk(
        self,
        enemy,
        *,
        battery_fraction=1.0,
        enemy_name="Spathi",
        enemy_objects=(),
        facing=None,
        variant="earlier",
    ):
        context = _rule_context(
            "Pkunk",
            enemy=enemy,
            enemy_name=enemy_name,
            battery_fraction=battery_fraction,
            enemy_objects=enemy_objects,
            max_battery=12.0,
        )
        context.view.enemy_block.opponent_bearing = facing
        strategy_for("Pkunk", variant)(context)
        return context

    def test_pkunk_passes_beside_the_opponent_away_from_its_nose(self):
        # The aim point 250 beside it is about 17 degrees off its bearing.
        enemy = _seen(0.0, 800.0)
        # I am on its right, so its nose points to my right: pass to my left.
        context = self._pkunk(enemy, facing=20.0)
        self.assertEqual(context.plan.turn, TURN_LEFT)
        self.assertTrue(context.plan.thrust)
        self.assertEqual(self._pkunk(enemy, facing=-20.0).plan.turn, TURN_RIGHT)

    def test_pkunk_strafes_side_on_within_gun_range(self):
        beside = self._pkunk(_seen(90.0, 300.0))
        self.assertTrue(beside.plan.thrust)
        self.assertTrue(beside.plan.a1)
        self.assertFalse(self._pkunk(_seen(0.0, 300.0)).plan.thrust)

    def test_pkunk_recharges_outside_the_opponents_reach(self):
        enemy = _seen(0.0, 700.0)
        far_reaching = self._pkunk(enemy, battery_fraction=0.3, enemy_name="Druuge")
        self.assertIn("station_near", far_reaching.memory.sides)
        short_reaching = self._pkunk(enemy, battery_fraction=0.3, enemy_name="Pkunk")
        self.assertNotIn("station_near", short_reaching.memory.sides)

    def test_pkunk_dodge_variant_sidesteps_a_shot_about_to_hit(self):
        # A shot from the right flies straight at me: sidestep across its
        # track, which is straight ahead, rather than turn for the pass.
        shot = _seen(90.0, 60.0, closing=40.0, group="enemy_a1")
        enemy = _seen(-60.0, 1500.0)
        context = self._pkunk(enemy, enemy_objects=(shot,), variant="dodge")
        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertTrue(context.plan.thrust)
        self.assertEqual(self._pkunk(enemy, enemy_objects=(shot,)).plan.turn, TURN_LEFT)

    def _earthling(self, enemy, *, battery=18.0, enemy_objects=(), planet=None):
        context = _rule_context(
            "Earthling",
            enemy=enemy,
            battery=battery,
            enemy_objects=enemy_objects,
            planet=planet,
            max_battery=18.0,
        )
        context.view.me.a1_ready = True
        STRATEGIES["Earthling"](context)
        return context

    def test_earthling_point_defense_ignores_shots_flying_away(self):
        enemy = _seen(0.0, 3000.0)
        leaving = _seen(90.0, 300.0, closing=-10.0, group="enemy_a1")
        coming = _seen(90.0, 300.0, closing=10.0, group="enemy_a1")

        self.assertFalse(self._earthling(enemy, enemy_objects=(leaving,)).plan.a2)
        self.assertTrue(self._earthling(enemy, enemy_objects=(coming,)).plan.a2)

    def test_earthling_nukes_only_with_room_for_the_nuke_to_home(self):
        self.assertTrue(self._earthling(_seen(0.0, 1500.0)).plan.a1)
        near = _seen(0.0, EARTHLING_NUKE_MIN_DISTANCE - 50.0)
        self.assertFalse(self._earthling(near).plan.a1)

    def test_earthling_turns_to_nuke_without_thrusting_and_coasts_above_top_speed(self):
        context = self._earthling(_seen(60.0, 1500.0))

        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertFalse(context.plan.thrust)
        self.assertTrue(context.coast_above_top_speed)

    def test_earthling_whips_only_off_a_planet_away_from_the_opponent(self):
        enemy = _seen(10.0, 3000.0)
        behind = _seen(180.0, 2000.0, group="planet")
        # 180 - 45 = 135 and 180 + 45 = -135: -135 leaves heading further
        # from the opponent at 10.
        whip = self._earthling(enemy, battery=0.0, planet=behind)
        self.assertEqual(whip.plan.turn, TURN_LEFT)
        self.assertTrue(whip.plan.thrust)
        toward = _seen(30.0, 2000.0, group="planet")
        no_whip = self._earthling(enemy, battery=0.0, planet=toward)
        # Far off and not closing, it approaches the opponent instead, 45
        # degrees off on the nearer side (10 - 45 = -35), not the planet's
        # sides at -15 or 75.
        self.assertNotIn("gravity_whip", no_whip.memory.sides)
        self.assertEqual(no_whip.plan.turn, TURN_LEFT)
        self.assertTrue(no_whip.plan.thrust)

    def test_umgah_keeps_thrust_off_while_hostile_shots_fly(self):
        enemy = _seen(0.0, 300.0)
        shot = _seen(10.0, 600.0, group="enemy_a1")

        self.assertTrue(self._umgah(enemy).plan.thrust)
        self.assertFalse(self._umgah(enemy, enemy_objects=(shot,)).plan.thrust)

    def _shofixti(
        self,
        enemy,
        *,
        enemy_name="Spathi",
        enemy_crew=30.0,
        crew=6.0,
        battery=4.0,
        enemy_objects=(),
        facing=None,
        shielded=False,
        memory=None,
    ):
        context = _rule_context(
            "Shofixti",
            enemy=enemy,
            enemy_name=enemy_name,
            enemy_objects=enemy_objects,
            battery=battery,
            battery_fraction=battery / 4.0,
            max_battery=4.0,
            memory=memory,
        )
        context.view.me.crew = crew
        context.view.enemy_block.crew = enemy_crew
        context.view.enemy_block.opponent_bearing = facing
        context.view.enemy_block.damage_shield_active = shielded
        context.view.enemy_block.mmrnmrhm_ywing = False
        STRATEGIES["Shofixti"](context)
        return context

    def test_glory_blast_damage_falls_off_with_distance(self):
        self.assertEqual(shofixti_blast_damage(0.0), 19)
        self.assertEqual(shofixti_blast_damage(360.0), 10)
        self.assertEqual(shofixti_blast_damage(720.0), 1)
        self.assertEqual(shofixti_blast_damage(721.0), 0)
        self.assertAlmostEqual(shofixti_kill_distance(6.0), 520.0)
        self.assertEqual(shofixti_blast_damage(520.0), 6)
        self.assertEqual(shofixti_blast_damage(521.0), 5)
        self.assertEqual(shofixti_kill_distance(1.0), 720.0)
        self.assertEqual(shofixti_kill_distance(20.0), -math.inf)

    def test_shofixti_blasts_only_where_the_blast_kills(self):
        # Eight crew die within 720 * (1 - 7/18) = 440, less the margin.
        self.assertTrue(self._shofixti(_seen(0.0, 400.0), enemy_crew=8.0).plan.a2)
        self.assertFalse(self._shofixti(_seen(0.0, 450.0), enemy_crew=8.0).plan.a2)
        # An opponent pulling away may leave the kill distance first.
        self.assertFalse(
            self._shofixti(_seen(0.0, 400.0, closing=-30.0), enemy_crew=8.0).plan.a2
        )
        self.assertFalse(
            self._shofixti(_seen(0.0, 400.0), enemy_crew=8.0, shielded=True).plan.a2
        )
        self.assertFalse(self._shofixti(_seen(0.0, 100.0), enemy_crew=20.0).plan.a2)

    def test_shofixti_blasts_when_a_shofixti_shot_is_about_to_kill_it(self):
        enemy = _seen(0.0, 300.0)
        shot = _seen(0.0, 150.0, closing=96.0, group="enemy_a1", crew_effect=-1.0)

        # Blasting a Shofixti kills both ships, so a kill alone is no reason.
        self.assertFalse(
            self._shofixti(enemy, enemy_name="Shofixti", enemy_crew=1.0).plan.a2
        )
        dying = self._shofixti(
            enemy, enemy_name="Shofixti", crew=1.0, enemy_objects=(shot,)
        )
        self.assertTrue(dying.plan.a2)
        surviving = self._shofixti(
            enemy, enemy_name="Shofixti", crew=2.0, enemy_objects=(shot,)
        )
        self.assertFalse(surviving.plan.a2)

    def test_shofixti_blasts_before_a_laser_it_cannot_outlast(self):
        enemy = _seen(0.0, 500.0)

        def blasts(facing, crew):
            return bool(
                self._shofixti(
                    enemy, enemy_name="Chmmr", enemy_crew=42.0, crew=crew, facing=facing
                ).plan.a2
            )

        # The Chmmr laser removes two crew when its nose points at me.
        self.assertTrue(blasts(0.0, 2.0))
        self.assertFalse(blasts(0.0, 3.0))
        self.assertFalse(blasts(90.0, 2.0))

    def test_shofixti_backs_off_until_its_battery_is_full(self):
        memory = BaselineMemory()
        enemy = _seen(0.0, 2000.0, closing=5.0)

        # Recharging from empty, the band doesn't close in until it is full.
        empty = self._shofixti(enemy, battery=0.0, memory=memory)
        self.assertTrue(empty.recharging())
        self.assertNotIn("band_far", empty.memory.sides)
        refilling = self._shofixti(enemy, battery=3.0, memory=memory)
        self.assertNotIn("band_far", refilling.memory.sides)
        full = self._shofixti(enemy, battery=4.0, memory=memory)
        self.assertIn("band_far", full.memory.sides)

    def test_shield_ships_shield_where_a_shofixti_blast_would_kill(self):
        def shields(distance, crew):
            context = _rule_context("Yehat", enemy=_seen(0.0, distance), enemy_name="Shofixti")
            context.view.me.crew = crew
            STRATEGIES["Yehat"](context)
            return bool(context.plan.a2)

        # Five crew die within 720 * (1 - 4/18) = 560.
        self.assertTrue(shields(550.0, 5.0))
        self.assertFalse(shields(550.0, 20.0))
        self.assertTrue(shields(350.0, 20.0))

    def _ilwrath(
        self,
        enemy,
        *,
        cloaked=False,
        fading=0,
        battery=16.0,
        enemy_thrust=0.0,
        variant=None,
    ):
        context = _rule_context(
            "Ilwrath",
            enemy=enemy,
            battery=battery,
            battery_fraction=battery / 16.0,
            max_battery=16.0,
        )
        context.view.me.trackable = not cloaked
        context.view.me.cloak_transition_direction = fading
        context.view.enemy_block.max_thrust = enemy_thrust
        strategy_for("Ilwrath", variant)(context)
        return context

    def test_cloaked_ilwrath_flames_at_any_bearing_within_reach(self):
        # The flame from the cloak turns the nose onto the opponent.
        context = self._ilwrath(_seen(120.0, 250.0), cloaked=True)

        self.assertTrue(context.plan.a1)
        self.assertFalse(context.plan.a2)

    def test_visible_ilwrath_flames_only_when_aimed(self):
        aimed = self._ilwrath(_seen(5.0, 250.0))
        self.assertTrue(aimed.plan.a1)
        self.assertFalse(aimed.plan.a2)

        wide = self._ilwrath(_seen(40.0, 250.0))
        self.assertFalse(wide.plan.a1)
        self.assertEqual(wide.plan.turn, TURN_RIGHT)

    def test_ilwrath_cloaks_only_beyond_reach_with_battery_to_spare(self):
        far = _seen(0.0, 1000.0)

        self.assertTrue(self._ilwrath(far).plan.a2)
        self.assertFalse(self._ilwrath(far, battery=6.0).plan.a2)
        self.assertFalse(self._ilwrath(far, fading=1).plan.a2)
        self.assertFalse(self._ilwrath(far, cloaked=True).plan.a2)
        self.assertFalse(self._ilwrath(_seen(0.0, 250.0)).plan.a2)

    def test_ilwrath_waits_for_a_faster_opponent_that_keeps_away(self):
        waiting = self._ilwrath(_seen(0.0, 1000.0), enemy_thrust=40.0)
        self.assertFalse(waiting.plan.thrust)

        # An opponent already coming in is met.
        coming = self._ilwrath(_seen(0.0, 1000.0, closing=5.0), enemy_thrust=40.0)
        self.assertTrue(coming.plan.thrust)

        slower = self._ilwrath(_seen(0.0, 1000.0), enemy_thrust=10.0)
        self.assertTrue(slower.plan.thrust)
        self.assertIn("band_far", slower.memory.sides)

        chasing = self._ilwrath(_seen(0.0, 1000.0), enemy_thrust=40.0, variant="chase")
        self.assertTrue(chasing.plan.thrust)

    def _syreen(
        self,
        enemy,
        *,
        enemy_name="Spathi",
        enemy_crew=20.0,
        enemy_thrust=20.0,
        crew=12.0,
        battery=16.0,
        pods=(),
        variant=None,
    ):
        context = _rule_context(
            "Syreen",
            enemy=enemy,
            enemy_name=enemy_name,
            battery=battery,
            battery_fraction=battery / 16.0,
            max_battery=16.0,
        )
        context.view.me.crew = crew
        context.view.crew_pods = tuple(pods)
        context.view.enemy_block.crew = enemy_crew
        context.view.enemy_block.max_thrust = enemy_thrust
        strategy_for("Syreen", variant)(context)
        return context

    def test_syreen_sings_only_up_close_at_crew_it_can_take(self):
        self.assertTrue(self._syreen(_seen(0.0, 250.0)).plan.a2)
        self.assertFalse(self._syreen(_seen(0.0, 600.0)).plan.a2)
        self.assertFalse(self._syreen(_seen(0.0, 250.0), battery=4.0).plan.a2)
        self.assertFalse(self._syreen(_seen(0.0, 250.0), enemy_crew=1.0).plan.a2)
        self.assertFalse(
            self._syreen(_seen(0.0, 250.0), enemy_name="Slylandro").plan.a2
        )

    def test_syreen_closes_to_sing_and_keeps_battery_for_the_song(self):
        approaching = self._syreen(_seen(0.0, 600.0))
        self.assertIn("song_far", approaching.memory.sides)
        self.assertTrue(approaching.plan.a1)

        a2_cost = SHIP_DEFINITIONS["Syreen"].a2_cost
        self.assertFalse(self._syreen(_seen(0.0, 600.0), battery=a2_cost).plan.a1)
        # With no crew to take, weapon 1 fires freely.
        self.assertTrue(
            self._syreen(_seen(0.0, 600.0), battery=a2_cost, enemy_crew=1.0).plan.a1
        )

    def test_syreen_collects_crew_pods_first(self):
        pod = _seen(-60.0, 200.0, group="syreen_crew")
        context = self._syreen(_seen(0.0, 600.0), pods=(pod,))

        self.assertEqual(context.plan.turn, TURN_LEFT)
        self.assertTrue(context.plan.thrust)

    def test_small_syreen_keeps_its_band_from_fast_close_fighters(self):
        enemy = _seen(0.0, 2000.0)

        cautious = self._syreen(enemy, enemy_name="Shofixti", enemy_thrust=40.0)
        self.assertIn("band_far", cautious.memory.sides)
        self.assertNotIn("song_far", cautious.memory.sides)
        grown = self._syreen(enemy, enemy_name="Shofixti", enemy_thrust=40.0, crew=24.0)
        self.assertIn("song_far", grown.memory.sides)
        slower = self._syreen(enemy, enemy_name="Shofixti", enemy_thrust=20.0)
        self.assertIn("song_far", slower.memory.sides)
        earlier = self._syreen(enemy, enemy_name="Shofixti", enemy_thrust=40.0, variant="earlier")
        self.assertIn("gun_far", earlier.memory.sides)

    def test_syreen_keeps_out_of_reach_while_the_song_recharges(self):
        context = self._syreen(_seen(0.0, 600.0, closing=5.0), battery=4.0)

        self.assertTrue(context.memory.values.get("recharge_running"))
        self.assertTrue(context.plan.steering_claimed)
        earlier = self._syreen(_seen(0.0, 600.0, closing=5.0), battery=4.0, variant="earlier")
        self.assertIn("avoid_opponent", earlier.memory.sides)

    def test_no_thrust_above_top_speed_outside_gravity(self):
        top = SHIP_DEFINITIONS["Thraddash"].max_thrust

        coasting = _rule_context("Thraddash", speed=top + 10.0)
        coasting.steer(0.0)
        self.assertFalse(coasting.plan.thrust)

        braking = _rule_context("Thraddash", speed=top + 10.0)
        braking.steer(0.0, brake=True)
        self.assertTrue(braking.plan.thrust)

        cruising = _rule_context("Thraddash", speed=top)
        cruising.steer(0.0)
        self.assertTrue(cruising.plan.thrust)

        # Ships that have not opted in thrust regardless.
        other = _rule_context("Spathi", speed=SHIP_DEFINITIONS["Spathi"].max_thrust + 10.0)
        other.steer(0.0)
        self.assertTrue(other.plan.thrust)

        # With no drag, an overspeed ship must still thrust to turn its motion.
        turning = _rule_context("Thraddash", speed=top + 10.0)
        turning.view.me.velocity_forward = top + 10.0
        turning.steer(120.0)
        self.assertTrue(turning.plan.thrust)

        # Inside a gravity well thrust is not capped, so it still helps.
        whipping = _rule_context(
            "Thraddash",
            speed=top + 10.0,
            planet=_seen(90.0, const.GRAVITY_RANGE - 100.0, group="planet"),
        )
        whipping.steer(0.0)
        self.assertTrue(whipping.plan.thrust)

    def _thraddash(self, enemy, *, enemy_name="Spathi", battery=24.0, enemy_objects=(), speed=0.0):
        context = _rule_context(
            "Thraddash",
            enemy=enemy,
            enemy_name=enemy_name,
            battery=battery,
            battery_fraction=battery / 24.0,
            max_battery=24.0,
            enemy_objects=enemy_objects,
            speed=speed,
        )
        STRATEGIES["Thraddash"](context)
        return context

    def test_thraddash_burns_away_tail_first_from_a_closing_opponent(self):
        behind = self._thraddash(_seen(180.0, 600.0, closing=10.0))
        self.assertTrue(behind.plan.a2)
        self.assertEqual(behind.plan.turn, TURN_NONE)

        ahead = self._thraddash(_seen(0.0, 600.0, closing=10.0))
        self.assertFalse(ahead.plan.a2)
        self.assertNotEqual(ahead.plan.turn, TURN_NONE)

    def test_thraddash_baits_slow_turners_closer(self):
        # The Umgah turns slowly and reaches little: it is let in nearer.
        self.assertFalse(
            self._thraddash(_seen(180.0, 600.0, closing=10.0), enemy_name="Umgah").plan.a2
        )
        self.assertTrue(
            self._thraddash(_seen(180.0, 450.0, closing=10.0), enemy_name="Umgah").plan.a2
        )

    def test_thraddash_leads_chasers_through_its_trail(self):
        enemy = _seen(0.0, 3000.0)
        fighter = _seen(180.0, 300.0, closing=10.0, group="enemy_non_a1")
        self.assertTrue(
            self._thraddash(enemy, enemy_name="KzerZa", enemy_objects=(fighter,)).plan.a2
        )
        distant = _seen(180.0, 600.0, closing=10.0, group="enemy_non_a1")
        waiting = self._thraddash(enemy, enemy_name="KzerZa", enemy_objects=(distant,))
        self.assertFalse(waiting.plan.a2)
        self.assertEqual(waiting.plan.turn, TURN_NONE)

        # Homing missiles are led through the trail; plain shots are not.
        missile = _seen(180.0, 300.0, closing=5.0, group="enemy_a1")
        self.assertTrue(
            self._thraddash(enemy, enemy_name="Earthling", enemy_objects=(missile,)).plan.a2
        )
        self.assertFalse(
            self._thraddash(enemy, enemy_name="Spathi", enemy_objects=(missile,)).plan.a2
        )

    def test_thraddash_fires_from_standoff_keeping_a_burn_reserve(self):
        enemy = _seen(0.0, 1200.0)

        self.assertTrue(self._thraddash(enemy).plan.a1)
        self.assertFalse(self._thraddash(enemy, battery=9.0).plan.a1)

    def test_thraddash_dashes_in_from_afar_and_then_coasts(self):
        # Opponent 45 degrees off the nose: the approach heading is the nose.
        dashing = self._thraddash(_seen(45.0, 3000.0))
        self.assertTrue(dashing.plan.a2)
        self.assertTrue(dashing.plan.thrust)

        coasting = self._thraddash(_seen(45.0, 3000.0), speed=70.0)
        self.assertFalse(coasting.plan.a2)
        self.assertFalse(coasting.plan.thrust)

    def test_orz_keeps_its_marines_from_a_thraddash(self):
        def launches(enemy_name):
            context = _rule_context("Orz", enemy=_seen(0.0, 500.0), enemy_name=enemy_name)
            context.view.me.orz_marines = 0
            context.view.me.crew = 16.0
            context.view.me.orz_turret_bearing = 0.0
            STRATEGIES["Orz"](context)
            return bool(context.plan.a2)

        self.assertTrue(launches("Spathi"))
        self.assertFalse(launches("Thraddash"))

    def _supox(self, enemy, *, enemy_name="Spathi", facing=None, enemy_objects=(), memory=None):
        context = _rule_context(
            "Supox",
            enemy=enemy,
            enemy_name=enemy_name,
            enemy_objects=enemy_objects,
            memory=memory,
        )
        context.view.enemy_block.opponent_bearing = facing
        STRATEGIES["Supox"](context)
        return context

    def test_supox_turns_to_aim_with_a2_released(self):
        context = self._supox(_seen(30.0, 700.0))

        self.assertFalse(context.plan.a2)
        self.assertEqual(context.plan.turn, TURN_RIGHT)

    def test_supox_backs_off_without_turning_when_too_close(self):
        # Inside the band's 400 near edge.
        context = self._supox(_seen(0.0, 300.0))

        self.assertTrue(context.plan.a2)
        self.assertTrue(context.plan.thrust)
        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertTrue(context.plan.a1)

    def test_supox_slides_off_the_nose_line_of_an_opponent_aiming_at_it(self):
        # On the opponent's right means sliding to my left, and back again
        # once its aim swings past.
        right_of_nose = self._supox(_seen(0.0, 700.0), facing=5.0)
        self.assertTrue(right_of_nose.plan.a2)
        self.assertEqual(right_of_nose.plan.turn, TURN_LEFT)
        self.assertFalse(right_of_nose.plan.thrust)
        self.assertTrue(right_of_nose.plan.a1)

        left_of_nose = self._supox(_seen(0.0, 700.0), facing=-5.0)
        self.assertEqual(left_of_nose.plan.turn, TURN_RIGHT)

        unwatched = self._supox(_seen(0.0, 700.0), facing=90.0)
        self.assertFalse(unwatched.plan.a2)
        self.assertFalse(unwatched.plan.thrust)

    def test_supox_slides_out_of_the_path_of_a_hit(self):
        # A shot crossing from the right passes on my right: slide left.
        shot = ObservedObject(
            group="enemy_a1",
            distance=100.0,
            bearing=60.0,
            forward=50.0,
            right=86.6,
            velocity_forward=-40.0,
            velocity_right=-20.0,
            closing_speed=40.0,
            lateral_speed=0.0,
            crew_effect=-1.0,
            battery_effect=0.0,
            hit_points=1.0,
            expires=True,
            remaining_seconds=1.0,
        )
        context = self._supox(_seen(0.0, 700.0), enemy_objects=(shot,))

        self.assertTrue(context.plan.a2)
        self.assertEqual(context.plan.turn, TURN_LEFT)

    def test_supox_keeps_outside_shorter_reaching_weapons(self):
        chmmr = _rule_context("Supox", enemy=_seen(0.0, 700.0), enemy_name="Chmmr")
        near, far = supox_band(chmmr)
        self.assertGreater(near, weapon_spec("Chmmr", 1).range)
        self.assertGreater(far, near)

        # The Chmmr laser reaches 700: back off.
        context = self._supox(_seen(0.0, 700.0), enemy_name="Chmmr")
        self.assertTrue(context.plan.a2)
        self.assertTrue(context.plan.thrust)

        self.assertEqual(
            supox_band(_rule_context("Supox", enemy=_seen(0.0, 700.0))), (600.0, 900.0)
        )

    def test_supox_leaves_a2_off_while_the_planet_rule_steers(self):
        context = _rule_context("Supox", enemy=_seen(0.0, 400.0))
        context.plan.claim_turn(TURN_LEFT)
        context.plan.claim_thrust(True)
        STRATEGIES["Supox"](context)

        self.assertFalse(context.plan.a2)

    def _vux(
        self,
        enemy,
        *,
        enemy_thrust=64.0,
        enemy_turn_wait=0.0,
        battery=40.0,
        facing=None,
        velocity_forward=0.0,
        memory=None,
    ):
        context = _rule_context(
            "Vux",
            enemy=enemy,
            enemy_name="Pkunk",
            battery=battery,
            battery_fraction=battery / 40.0,
            max_battery=40.0,
            memory=memory,
        )
        context.view.me.velocity_forward = velocity_forward
        context.view.enemy_block.max_thrust = enemy_thrust
        context.view.enemy_block.turn_wait = enemy_turn_wait
        context.view.enemy_block.opponent_bearing = facing
        STRATEGIES["Vux"](context)
        return context

    def test_vux_holds_its_beam_on_a_target_within_reach(self):
        # As at the close start: in reach and on the beam, keep burning.
        holding = self._vux(_seen(2.0, 600.0))
        self.assertTrue(holding.plan.a1)
        self.assertEqual(holding.plan.turn, TURN_NONE)
        self.assertFalse(holding.plan.thrust)

        # Out of battery, it turns away instead.
        drained = self._vux(_seen(2.0, 600.0), battery=2.0)
        self.assertNotEqual(drained.plan.turn, TURN_NONE)

    def test_vux_turns_away_without_thrusting_toward_the_opponent(self):
        facing_it = self._vux(_seen(0.0, 800.0))
        self.assertNotEqual(facing_it.plan.turn, TURN_NONE)
        self.assertFalse(facing_it.plan.thrust)

        tail_on = self._vux(_seen(180.0, 800.0))
        self.assertEqual(tail_on.plan.turn, TURN_NONE)
        self.assertTrue(tail_on.plan.thrust)

    def test_vux_faces_the_opponent_once_drifting_away(self):
        top = SHIP_DEFINITIONS["Vux"].max_thrust
        # Opponent dead ahead while I move straight back from it.
        drifting = self._vux(_seen(0.0, 800.0), velocity_forward=-top)

        self.assertEqual(drifting.plan.turn, TURN_NONE)
        self.assertFalse(drifting.plan.thrust)

    def test_vux_limpets_until_the_opponent_is_crippled(self):
        enemy = _seen(0.0, 1000.0)

        self.assertTrue(self._vux(enemy).plan.a2)
        self.assertFalse(
            self._vux(enemy, enemy_thrust=15.0, enemy_turn_wait=5.0).plan.a2
        )
        # Near laser reach, limpets leave the laser its reserve.
        self.assertFalse(self._vux(_seen(0.0, 600.0), battery=20.0).plan.a2)
        self.assertTrue(self._vux(_seen(0.0, 600.0), battery=30.0).plan.a2)

    def test_vux_lasers_whenever_the_hull_crosses_the_beam(self):
        self.assertTrue(self._vux(_seen(6.0, 300.0)).plan.a1)
        self.assertTrue(self._vux(_seen(2.0, 650.0)).plan.a1)
        self.assertFalse(self._vux(_seen(15.0, 300.0)).plan.a1)
        self.assertFalse(self._vux(_seen(0.0, 800.0)).plan.a1)

    def test_vux_closes_on_a_crippled_opponent_from_its_blind_side(self):
        crippled = {"enemy_thrust": 15.0, "enemy_turn_wait": 5.0}

        watched = self._vux(_seen(0.0, 1000.0), facing=5.0, **crippled)
        # On its right, so come in to my left, off its nose line.
        self.assertEqual(watched.plan.turn, TURN_LEFT)

        unwatched = self._vux(_seen(0.0, 1000.0), facing=120.0, **crippled)
        self.assertEqual(unwatched.plan.turn, TURN_NONE)
        self.assertTrue(unwatched.plan.thrust)

    def test_variants_are_named_per_ship(self):
        self.assertIs(strategy_for("ZoqFotPik"), STRATEGIES["ZoqFotPik"])
        self.assertIs(strategy_for("ZoqFotPik", "default"), STRATEGIES["ZoqFotPik"])
        self.assertIsNot(strategy_for("ZoqFotPik", "a1_first"), STRATEGIES["ZoqFotPik"])
        self.assertIsNot(strategy_for("Pkunk", "full_range"), STRATEGIES["Pkunk"])
        with self.assertRaises(KeyError):
            strategy_for("Spathi", "a1_first")

    def test_druuge_recoils_only_away_from_closing_dangers_ahead(self):
        ahead = _seen(20.0, 400.0, closing=10.0, group="enemy_non_a1")
        behind = _seen(160.0, 400.0, closing=10.0, group="enemy_non_a1")
        opening = _seen(20.0, 400.0, closing=-10.0, group="enemy_non_a1")
        enemy = _seen(90.0, 3000.0)

        def fires(enemy_name, *dangers):
            context = _rule_context(
                "Druuge",
                enemy=enemy,
                enemy_name=enemy_name,
                enemy_objects=dangers,
            )
            STRATEGIES["Druuge"](context)
            return bool(context.plan.a1)

        self.assertTrue(fires("KzerZa", ahead))
        self.assertTrue(fires("Vux", ahead))
        self.assertFalse(fires("KzerZa", ahead, behind))
        self.assertFalse(fires("KzerZa", opening))
        self.assertFalse(fires("Spathi", ahead))

    def test_druuge_points_at_the_nearest_target_and_burns_crew_when_drained(self):
        dogi = _seen(-60.0, 900.0, group="enemy_non_a1")
        context = _rule_context(
            "Druuge",
            enemy=_seen(60.0, 1500.0),
            enemy_name="Chenjesu",
            enemy_objects=(dogi,),
            battery=1.0,
        )
        STRATEGIES["Druuge"](context)

        self.assertEqual(context.plan.turn, TURN_LEFT)
        self.assertFalse(context.plan.a1)
        self.assertTrue(context.plan.a2)

    def test_mmrnmrhm_switches_forms_around_xform_range(self):
        x_range = weapon_spec("Mmrnmrhm", 1, "xform").range
        near = _seen(30.0, x_range - 50.0, closing=5.0)
        far = _seen(30.0, x_range + 500.0, closing=5.0)

        x_near = _rule_context("Mmrnmrhm", enemy=near)
        STRATEGIES["Mmrnmrhm"](x_near)
        self.assertTrue(x_near.plan.a1)
        self.assertTrue(x_near.plan.thrust)
        self.assertEqual(x_near.plan.turn, TURN_RIGHT)
        self.assertFalse(x_near.plan.a2)

        x_far = _rule_context("Mmrnmrhm", enemy=far)
        STRATEGIES["Mmrnmrhm"](x_far)
        self.assertTrue(x_far.plan.a2)

        y_near = _rule_context("Mmrnmrhm", enemy=near, ywing=True)
        STRATEGIES["Mmrnmrhm"](y_near)
        self.assertTrue(y_near.plan.a2)

        y_far = _rule_context("Mmrnmrhm", enemy=far, ywing=True)
        STRATEGIES["Mmrnmrhm"](y_far)
        self.assertFalse(y_far.plan.a2)
        self.assertTrue(y_far.plan.a1)
        # The Y-wing no longer flees a closing opponent; it faces it. The
        # earlier rules fled.
        self.assertEqual(y_far.plan.turn, TURN_RIGHT)
        self.assertNotIn("avoid_opponent", y_far.memory.sides)
        fled = _rule_context("Mmrnmrhm", enemy=far, ywing=True)
        strategy_for("Mmrnmrhm", "earlier")(fled)
        self.assertEqual(fled.memory.sides.keys(), {"avoid_opponent"})


def _satellite(enemy, orbit_angle, *, hp=10.0, spin=1, orbit=256.0):
    """A satellite on the enemy's orbit at ``orbit_angle`` (degrees from my
    forward axis around the enemy), circling clockwise for ``spin`` 1."""

    radians = math.radians(orbit_angle)
    forward = enemy.forward + orbit * math.cos(radians)
    right = enemy.right + orbit * math.sin(radians)
    # Tangential speed of one turn every satellite period.
    period = SHIP_DEFINITIONS["Chmmr"].satellite_period
    speed = 2.0 * math.pi * orbit / period
    velocity_forward = enemy.velocity_forward - spin * speed * math.sin(radians)
    velocity_right = enemy.velocity_right + spin * speed * math.cos(radians)
    return ObservedObject(
        group="enemy_non_a1",
        distance=math.hypot(forward, right),
        bearing=math.degrees(math.atan2(right, forward)),
        forward=forward,
        right=right,
        velocity_forward=velocity_forward,
        velocity_right=velocity_right,
        closing_speed=0.0,
        lateral_speed=0.0,
        crew_effect=0.0,
        battery_effect=0.0,
        hit_points=float(hp),
        expires=False,
        remaining_seconds=0.0,
    )


class ChmmrSatelliteHelperTests(unittest.TestCase):
    def test_finds_only_the_chmmr_satellites(self):
        enemy = _seen(0.0, 1200.0)
        satellite = _satellite(enemy, 180.0)
        laser = dataclasses.replace(_satellite(enemy, 90.0), expires=True)
        stray = _seen(30.0, 900.0, group="enemy_non_a1")
        stray = dataclasses.replace(stray, hit_points=1.0)
        objects = (satellite, laser, stray)

        chmmr = _rule_context("Thraddash", enemy=enemy, enemy_name="Chmmr", enemy_objects=objects)
        self.assertEqual(chmmr.chmmr_satellites(), (satellite,))
        other = _rule_context("Thraddash", enemy=enemy, enemy_name="Spathi", enemy_objects=objects)
        self.assertEqual(other.chmmr_satellites(), ())

    def test_finds_a_satellite_seen_across_the_arena_wrap(self):
        enemy = _seen(0.0, 3900.0)
        beyond = _satellite(enemy, 0.0)
        # Its nearest copy lies one arena behind me (screen down).
        across = dataclasses.replace(
            beyond, forward=beyond.forward - const.ARENA_SIZE, bearing=180.0
        )
        context = _rule_context(
            "Thraddash", enemy=enemy, enemy_name="Chmmr", enemy_objects=(across,)
        )

        self.assertEqual(context.chmmr_satellites(), (across,))
        forward, right = context.satellite_offset(across, 0)
        self.assertAlmostEqual(forward, enemy.forward + 256.0, places=6)
        self.assertAlmostEqual(right, 0.0, places=6)

    def test_predicts_a_satellite_along_its_orbit(self):
        enemy = _seen(0.0, 1200.0)
        quarter = SHIP_DEFINITIONS["Chmmr"].satellite_period // 4
        for spin, expected_right in ((1, -256.0), (-1, 256.0)):
            with self.subTest(spin=spin):
                # Behind the Chmmr (toward me), a quarter turn on.
                satellite = _satellite(enemy, 180.0, spin=spin)
                context = _rule_context(
                    "Thraddash", enemy=enemy, enemy_name="Chmmr", enemy_objects=(satellite,)
                )
                forward, right = context.satellite_offset(satellite, quarter)
                self.assertAlmostEqual(forward, enemy.forward, places=6)
                self.assertAlmostEqual(right, expected_right, places=6)

    def test_shoots_the_weakest_satellite_in_reach(self):
        enemy = _seen(0.0, 1200.0)
        strong = _satellite(enemy, 180.0, hp=10.0)
        weak = _satellite(enemy, 60.0, hp=3.0)
        context = _rule_context(
            "Thraddash", enemy=enemy, enemy_name="Chmmr", enemy_objects=(strong, weak)
        )

        target, _ = context.satellite_shot(1)

        self.assertIs(target, weak)
        far_enemy = _seen(0.0, 3000.0)
        far = _rule_context(
            "Thraddash",
            enemy=far_enemy,
            enemy_name="Chmmr",
            enemy_objects=(_satellite(far_enemy, 180.0),),
        )
        self.assertIsNone(far.satellite_shot(1))

    def test_intercept_meets_a_target_where_it_will_be(self):
        context = _rule_context("Thraddash", enemy=_seen(0.0, 1200.0), enemy_name="Chmmr")
        speed = context.weapon(1).speed

        bearing, frames = context.intercept(1, lambda frames: (600.0, 0.0))
        self.assertAlmostEqual(bearing, 0.0)
        self.assertEqual(frames, math.ceil(600.0 / speed))
        # A target moving right is led to the right.
        bearing, _ = context.intercept(1, lambda frames: (600.0, 20.0 * frames))
        self.assertGreater(bearing, 0.0)
        self.assertIsNone(context.intercept(1, lambda frames: (5000.0, 0.0)))

    def test_satellite_ring_lies_beyond_both_lasers_within_reach(self):
        context = _rule_context("Thraddash", enemy=_seen(0.0, 1200.0), enemy_name="Chmmr")

        self.assertEqual(context.satellite_ring(1620.0), (600.0, 1876.0))
        self.assertIsNone(context.satellite_ring(300.0))


class ChmmrMatchupTests(unittest.TestCase):
    def test_thraddash_fires_at_a_lined_up_satellite_from_the_band(self):
        enemy = _seen(0.0, 1200.0)
        satellite = _satellite(enemy, 180.0)
        context = _rule_context(
            "Thraddash", enemy=enemy, enemy_name="Chmmr", enemy_objects=(satellite,)
        )
        with mock.patch.object(
            BaselineContext, "satellite_shot", return_value=(satellite, 0.0)
        ):
            strategy_for("Thraddash")(context)

        self.assertTrue(context.plan.a1)
        self.assertFalse(context.plan.a2)

    def test_thraddash_turns_tail_on_a_chmmr_closing_fast(self):
        enemy = _seen(180.0, 1100.0, closing=40.0)
        context = _rule_context("Thraddash", enemy=enemy, enemy_name="Chmmr")

        strategy_for("Thraddash")(context)

        # The tail already faces it: burn away, laying the trail between.
        self.assertTrue(context.plan.a2)
        self.assertFalse(context.plan.a1)

    def test_thraddash_never_dashes_in_at_a_chmmr(self):
        context = _rule_context(
            "Thraddash", enemy=_seen(0.0, 3000.0, closing=20.0), enemy_name="Chmmr"
        )

        strategy_for("Thraddash")(context)

        self.assertFalse(context.plan.a2)
        self.assertFalse(context.plan.thrust)

    def test_arilou_fires_only_while_a_satellite_blocks_its_beam(self):
        enemy = _seen(0.0, 640.0)
        # One frame's turn short of the beam: the laser fires a frame later.
        step = 360.0 / SHIP_DEFINITIONS["Chmmr"].satellite_period
        on_beam = _satellite(enemy, 180.0 - step)
        off_beam = _satellite(enemy, 90.0)
        for satellite, fires in ((on_beam, True), (off_beam, False)):
            with self.subTest(fires=fires):
                context = _rule_context(
                    "Arilou", enemy=enemy, enemy_name="Chmmr", enemy_objects=(satellite,)
                )
                strategy_for("Arilou")(context)
                self.assertIs(bool(context.plan.a1), fires)

    def test_arilou_backs_off_nose_away_inside_the_hold_distance(self):
        enemy = _seen(180.0, 600.0, closing=30.0)
        context = _rule_context(
            "Arilou",
            enemy=enemy,
            enemy_name="Chmmr",
            enemy_objects=(_satellite(enemy, 90.0),),
        )

        strategy_for("Arilou")(context)

        self.assertEqual(context.plan.turn, TURN_NONE)
        self.assertTrue(context.plan.thrust)


class ChmmrMatchupMoreTests(unittest.TestCase):
    def _orz(self, satellites):
        enemy = _seen(0.0, 1000.0)
        objects = tuple(_satellite(enemy, angle) for angle in (90.0, 210.0, 330.0)[:satellites])
        context = _rule_context("Orz", enemy=enemy, enemy_name="Chmmr", enemy_objects=objects)
        context.view.me.orz_marines = 0
        context.view.me.crew = 16.0
        context.view.me.orz_turret_bearing = 0.0
        STRATEGIES["Orz"](context)
        return context

    def test_orz_boards_a_chmmr_only_with_one_satellite_or_none_left(self):
        self.assertFalse(self._orz(3).plan.a2)
        self.assertFalse(self._orz(2).plan.a2)
        self.assertTrue(self._orz(1).plan.a2)
        self.assertTrue(self._orz(0).plan.a2)

    def _vux(self, enemy, objects):
        context = _rule_context("Vux", enemy=enemy, enemy_name="Chmmr", enemy_objects=objects)
        # A healthy Chmmr: not slowed by limpets.
        context.view.enemy_block.max_thrust = 35.0
        context.view.enemy_block.turn_wait = 3
        STRATEGIES["Vux"](context)
        return context

    def test_vux_lays_limpets_only_while_satellites_shoot_at_it(self):
        far_enemy = _seen(90.0, 1500.0)
        guarded = self._vux(far_enemy, (_satellite(far_enemy, 0.0),))
        self.assertFalse(guarded.plan.a2)

        near_enemy = _seen(90.0, 420.0)
        busy = self._vux(near_enemy, (_satellite(near_enemy, 270.0),))
        self.assertTrue(busy.plan.a2)

    def test_vux_lasers_a_satellite_crossing_its_beam(self):
        enemy = _seen(20.0, 600.0)
        step = 360.0 / SHIP_DEFINITIONS["Chmmr"].satellite_period
        # Where its orbit crosses the nose line on my side, within reach;
        # it gets there next frame.
        crossing = enemy.forward - math.sqrt(256.0**2 - enemy.right**2)
        orbit_angle = math.degrees(math.atan2(-enemy.right, crossing - enemy.forward))
        satellite = _satellite(enemy, orbit_angle - step)
        context = self._vux(enemy, (satellite,))

        self.assertTrue(context.satellite_on_beam(0.0, 664.0))
        self.assertTrue(context.plan.a1)

    def test_supox_turns_to_a_satellite_while_settled_in_its_band(self):
        enemy = _seen(0.0, 900.0)
        # Its shot bearing is well off the nose, past the tap tolerance.
        satellite = _satellite(enemy, 60.0)
        context = _rule_context(
            "Supox", enemy=enemy, enemy_name="Chmmr", enemy_objects=(satellite,)
        )
        context.view.enemy_block.opponent_bearing = 90.0
        _, bearing = context.satellite_shot(1)
        STRATEGIES["Supox"](context)

        self.assertFalse(context.plan.a2)
        expected = TURN_RIGHT if bearing > 0.0 else TURN_LEFT
        self.assertEqual(context.plan.turn, expected)

    def test_standoff_runs_early_from_a_fast_closer(self):
        context = _rule_context("Orz", enemy=_seen(0.0, 1500.0, closing=60.0), enemy_name="Chmmr")

        self.assertTrue(context.keep_standoff("test", 900.0, 1250.0))
        # Running: the nose turns away from it.
        self.assertNotEqual(context.plan.turn, TURN_NONE)

    def test_standoff_closes_in_slowly_and_leaves_the_band_free(self):
        slow = _rule_context("Orz", enemy=_seen(0.0, 2000.0, closing=5.0), enemy_name="Chmmr")
        self.assertFalse(slow.keep_standoff("test", 900.0, 1250.0))
        self.assertTrue(slow.plan.thrust)

        fast = _rule_context("Orz", enemy=_seen(0.0, 4000.0, closing=30.0), enemy_name="Chmmr")
        self.assertFalse(fast.keep_standoff("test", 900.0, 1250.0))
        self.assertFalse(fast.plan.thrust)

        settled = _rule_context("Orz", enemy=_seen(0.0, 1100.0), enemy_name="Chmmr")
        self.assertFalse(settled.keep_standoff("test", 900.0, 1250.0))
        self.assertIsNone(settled.plan.turn)
        self.assertIsNone(settled.plan.thrust)


class AndrosynthChmmrTests(unittest.TestCase):
    def _context(self, enemy, *, blazer=False, battery=24.0):
        context = _rule_context(
            "Androsynth",
            enemy=enemy,
            enemy_name="Chmmr",
            enemy_objects=(_satellite(enemy, 90.0),),
            battery=battery,
            max_battery=24.0,
        )
        context.view.me.androsynth_blazer = blazer
        return context

    def test_saves_the_battery_and_blazes_in_from_close_on_a_full_battery(self):
        far = self._context(_seen(30.0, 1500.0))
        STRATEGIES["Androsynth"](far)
        self.assertFalse(far.plan.a1)
        self.assertFalse(far.plan.a2)
        # The nose comes round to the Chmmr meanwhile.
        self.assertEqual(far.plan.turn, TURN_RIGHT)

        close = self._context(_seen(0.0, 700.0))
        STRATEGIES["Androsynth"](close)
        self.assertTrue(close.plan.a2)
        self.assertFalse(close.plan.a1)

        drained = self._context(_seen(0.0, 700.0), battery=12.0)
        STRATEGIES["Androsynth"](drained)
        self.assertFalse(drained.plan.a2)

    def test_blazer_rams_where_the_chmmr_will_be(self):
        # Off to the right, crossing ahead of me.
        crossing = dataclasses.replace(_seen(90.0, 600.0), velocity_forward=30.0)
        context = self._context(crossing, blazer=True)

        bearing = blazer_intercept_bearing(context)
        STRATEGIES["Androsynth"](context)

        self.assertGreater(bearing, 0.0)
        self.assertLess(bearing, 90.0)
        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertTrue(context.plan.thrust)


class UtwigChmmrTests(unittest.TestCase):
    def _context(self, enemy, *, enemy_name="Chmmr", enemy_battery=42.0):
        # The satellite on the far side of the Chmmr.
        satellites = (_satellite(enemy, 0.0),) if enemy_name == "Chmmr" else ()
        context = _rule_context(
            "Utwig",
            enemy=enemy,
            enemy_name=enemy_name,
            enemy_objects=satellites,
            battery=10.0,
            max_battery=20.0,
        )
        context.view.me.crew = 20.0
        context.view.enemy_block.battery = enemy_battery
        context.view.enemy_block.battery_fraction = enemy_battery / 42.0
        return context

    def test_runs_shielded_from_a_charged_chmmr_within_laser_reach(self):
        context = self._context(_seen(0.0, 550.0, closing=40.0))

        STRATEGIES["Utwig"](context)

        self.assertTrue(context.memory.values.get("utwig_chmmr_running"))
        self.assertTrue(context.plan.a2)
        self.assertFalse(context.plan.a1)

    def test_runs_unshielded_while_out_of_every_laser_reach(self):
        # Closing fast enough to run from, but beyond the Chmmr laser.
        context = self._context(_seen(0.0, 900.0, closing=60.0))

        STRATEGIES["Utwig"](context)

        self.assertTrue(context.memory.values.get("utwig_chmmr_running"))
        self.assertFalse(context.plan.a2)

    def test_fires_on_the_chmmr_from_the_band(self):
        context = self._context(_seen(0.0, 950.0))

        STRATEGIES["Utwig"](context)

        self.assertFalse(context.memory.values.get("utwig_chmmr_running", False))
        self.assertTrue(context.plan.a1)
        self.assertFalse(context.plan.a2)

    def test_stands_shielded_against_a_low_chmmr_that_can_still_fire(self):
        low = UTWIG_CHMMR_RUN_BATTERY * 42.0 - 1.0
        context = self._context(_seen(0.0, 550.0, closing=40.0), enemy_battery=low)

        STRATEGIES["Utwig"](context)

        self.assertFalse(context.memory.values.get("utwig_chmmr_running"))
        self.assertTrue(context.plan.a2)
        self.assertFalse(context.plan.a1)

    def test_fires_at_a_chmmr_without_battery_for_a_shot(self):
        context = self._context(_seen(0.0, 550.0, closing=40.0), enemy_battery=1.0)

        STRATEGIES["Utwig"](context)

        self.assertFalse(context.memory.values.get("utwig_chmmr_running"))
        self.assertFalse(context.plan.a2)
        self.assertTrue(context.plan.a1)

    def test_other_opponents_keep_the_shield_rules(self):
        context = self._context(_seen(0.0, 550.0, closing=40.0), enemy_name="Spathi")

        STRATEGIES["Utwig"](context)

        self.assertNotIn("utwig_chmmr_running", context.memory.values)
        self.assertFalse(context.plan.a2)


class UmgahChmmrTests(unittest.TestCase):
    def _umgah(self, enemy):
        context = _rule_context("Umgah", enemy=enemy, enemy_name="Chmmr", battery=30.0)
        STRATEGIES["Umgah"](context)
        return context

    def test_faces_the_chmmr_without_thrust_with_the_cone_on(self):
        context = self._umgah(_seen(30.0, 1200.0, closing=40.0))

        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertFalse(context.plan.thrust)
        self.assertTrue(context.plan.a1)
        self.assertFalse(context.plan.a2)

    def test_hops_back_off_the_chmmr_hull(self):
        context = self._umgah(_seen(0.0, UMGAH_CHMMR_HOP_DISTANCE - 10.0))

        self.assertTrue(context.plan.a2)
        self.assertTrue(context.plan.a1)
        self.assertFalse(context.plan.thrust)


def _first_steer(context, strategy):
    """The bearing ``strategy`` first steers toward, or None."""

    with mock.patch.object(BaselineContext, "steer", wraps=context.steer) as steer:
        strategy(context)
    return steer.call_args_list[0].args[0] if steer.call_args_list else None


class SlylandroChmmrTests(unittest.TestCase):
    def _context(self, enemy, enemy_name="Chmmr"):
        context = _rule_context(
            "Slylandro", enemy=enemy, enemy_name=enemy_name, battery=20.0, max_battery=20.0
        )
        context.view.asteroids = ()
        return context

    def test_circles_the_chmmr_in_its_band_firing(self):
        context = self._context(_seen(0.0, (SLYLANDRO_CHMMR_NEAR + SLYLANDRO_CHMMR_FAR) / 2))

        heading = _first_steer(context, STRATEGIES["Slylandro"])

        self.assertAlmostEqual(abs(heading), 90.0)
        self.assertTrue(context.plan.a1)
        self.assertFalse(context.plan.a2)

    def test_runs_from_a_closing_chmmr(self):
        context = self._context(_seen(0.0, SLYLANDRO_CHMMR_NEAR + 20.0, closing=30.0))

        heading = _first_steer(context, STRATEGIES["Slylandro"])

        self.assertAlmostEqual(abs(heading), 180.0)

    def test_holds_fire_beyond_the_fire_distance(self):
        context = self._context(_seen(0.0, SLYLANDRO_CHMMR_FIRE + 100.0))

        STRATEGIES["Slylandro"](context)

        self.assertFalse(context.plan.a1)

    def test_other_opponents_keep_the_harvester_on(self):
        context = self._context(_seen(0.0, 900.0), enemy_name="Spathi")

        STRATEGIES["Slylandro"](context)

        self.assertTrue(context.plan.a2)


class AndrosynthSharedTests(unittest.TestCase):
    def _context(self, enemy, *, battery, enemy_name="Mycon"):
        return _rule_context(
            "Androsynth",
            enemy=enemy,
            enemy_name=enemy_name,
            battery=battery,
            battery_fraction=battery / 24.0,
            max_battery=24.0,
        )

    def test_runs_from_its_table_entry(self):
        self.assertIs(STRATEGIES["Androsynth"], strategy_module.shared)

    def test_blazes_in_close_on_half_a_battery(self):
        near = self._context(_seen(0.0, 800.0), battery=14.0)
        STRATEGIES["Androsynth"](near)
        self.assertTrue(near.plan.a2)

        low = self._context(_seen(0.0, 800.0), battery=10.0)
        STRATEGIES["Androsynth"](low)
        self.assertFalse(low.plan.a2)

    def test_launches_bubbles_keeping_a_reserve(self):
        far = _seen(0.0, 3000.0)
        charged = self._context(far, battery=14.0)
        STRATEGIES["Androsynth"](charged)
        self.assertTrue(charged.plan.a1)

        reserve = self._context(far, battery=11.0)
        STRATEGIES["Androsynth"](reserve)
        self.assertFalse(reserve.plan.a1)


class SpathiChmmrTests(unittest.TestCase):
    """The earlier Chmmr rule, kept in the ``approach`` variant."""

    rules = staticmethod(strategy_for("Spathi", "approach"))

    def _context(self, enemy, *, enemy_name="Chmmr", satellites=()):
        return _rule_context(
            "Spathi",
            enemy=enemy,
            enemy_name=enemy_name,
            enemy_objects=satellites,
            battery=10.0,
            max_battery=10.0,
        )

    def test_shoots_a_satellite_from_the_band(self):
        enemy = _seen(0.0, (SPATHI_CHMMR_NEAR + SPATHI_CHMMR_FAR) / 2)
        satellite = _satellite(enemy, 180.0)
        self.assertIsNotNone(self._context(enemy, satellites=(satellite,)).satellite_shot(1))

        for lead, fires, turn in ((0.0, True, TURN_NONE), (40.0, False, TURN_RIGHT)):
            context = self._context(enemy, satellites=(satellite,))
            with mock.patch.object(
                BaselineContext, "satellite_shot", return_value=(satellite, lead)
            ):
                self.rules(context)
            self.assertEqual(bool(context.plan.a1), fires)
            self.assertEqual(context.plan.turn or TURN_NONE, turn)

    def test_runs_from_a_closing_chmmr(self):
        context = self._context(_seen(0.0, SPATHI_CHMMR_NEAR + 20.0, closing=30.0))

        heading = _first_steer(context, self.rules)

        self.assertAlmostEqual(abs(heading), 180.0)

    def test_no_rear_missiles_at_the_chmmr(self):
        behind = _seen(180.0, 800.0)
        chmmr = self._context(behind)
        self.rules(chmmr)
        self.assertFalse(chmmr.plan.a2)

        other = self._context(behind, enemy_name="Earthling")
        self.rules(other)
        self.assertTrue(other.plan.a2)


class SpathiBandTests(unittest.TestCase):
    def _context(self, enemy, *, enemy_name="Mycon", battery=10.0):
        return _rule_context(
            "Spathi",
            enemy=enemy,
            enemy_name=enemy_name,
            battery=battery,
            max_battery=10.0,
            battery_fraction=battery / 10.0,
        )

    def test_runs_straight_from_a_closer_entering_the_band(self):
        context = self._context(_seen(0.0, 800.0, closing=40.0))

        heading = _first_steer(context, STRATEGIES["Spathi"])

        self.assertAlmostEqual(abs(heading), 180.0)

    def test_a_low_battery_does_not_stop_it_closing_in(self):
        # Pass 1 dropped recharging: it dealt less harm.
        far = _seen(20.0, 2500.0, closing=-5.0)

        for battery in (10.0, 1.0):
            context = self._context(far, battery=battery)
            self.assertIsNotNone(_first_steer(context, STRATEGIES["Spathi"]))

    def test_fires_the_gun_ahead_and_the_rear_missile_behind(self):
        ahead = self._context(_seen(0.0, 800.0))
        STRATEGIES["Spathi"](ahead)
        self.assertTrue(ahead.plan.a1)

        behind = self._context(_seen(180.0, 800.0))
        STRATEGIES["Spathi"](behind)
        self.assertTrue(behind.plan.a2)


class SyreenChmmrTests(unittest.TestCase):
    def _syreen(self, enemy):
        context = _rule_context(
            "Syreen", enemy=enemy, enemy_name="Chmmr", battery=16.0, max_battery=16.0
        )
        context.view.crew_pods = ()
        STRATEGIES["Syreen"](context)
        return context

    def test_waits_nose_first_without_thrust(self):
        context = self._syreen(_seen(30.0, 900.0, closing=40.0))

        self.assertEqual(context.plan.turn, TURN_RIGHT)
        self.assertFalse(context.plan.thrust)
        self.assertFalse(context.plan.a2)

    def test_sings_only_once_the_chmmr_is_close(self):
        self.assertTrue(self._syreen(_seen(0.0, SYREEN_CHMMR_SONG_DISTANCE - 20.0)).plan.a2)
        self.assertFalse(self._syreen(_seen(0.0, SYREEN_CHMMR_SONG_DISTANCE + 50.0)).plan.a2)


class YehatChmmrTests(unittest.TestCase):
    def _yehat(self, distance, *, shield_up=False, battery=10.0):
        context = _rule_context(
            "Yehat",
            enemy=_seen(0.0, distance),
            enemy_name="Chmmr",
            battery=battery,
            max_battery=10.0,
        )
        context.view.me.damage_shield_active = shield_up
        # The table flies its defaults against a Chmmr since pass 3; the
        # Chmmr rule remains in the earlier rules.
        strategy_for("Yehat", "earlier")(context)
        return context

    def test_raises_the_shield_within_the_chmmr_laser(self):
        context = self._yehat(500.0)

        self.assertTrue(context.plan.a2)
        self.assertFalse(context.plan.a1)

    def test_fires_under_a_raised_shield_keeping_the_reserve(self):
        context = self._yehat(500.0, shield_up=True, battery=5.0)
        self.assertFalse(context.plan.a2)
        self.assertTrue(context.plan.a1)

        low = self._yehat(500.0, shield_up=True, battery=1.0 + YEHAT_CHMMR_FIRE_RESERVE - 0.5)
        self.assertFalse(low.plan.a1)

    def test_no_shield_beyond_the_chmmr_laser(self):
        context = self._yehat(700.0, battery=1.0)

        self.assertFalse(context.plan.a2)
        self.assertTrue(context.plan.a1)


class ChmmrTractorTests(unittest.TestCase):
    def _tractor(self, enemy, *, enemy_name="Spathi", battery_fraction=1.0, cloaked=False, blazer=False):
        context = _rule_context(
            "Chmmr", enemy=enemy, enemy_name=enemy_name, battery_fraction=battery_fraction
        )
        context.view.enemy_block.trackable = not cloaked
        context.view.enemy_block.androsynth_blazer = blazer
        STRATEGIES["Chmmr"](context)
        return bool(context.plan.a2)

    def test_pulls_a_mid_range_opponent_in_but_not_a_far_one(self):
        self.assertTrue(self._tractor(_seen(0.0, 1000.0)))
        self.assertTrue(self._tractor(_seen(0.0, CHMMR_TRACTOR_FAR - 50.0)))
        self.assertFalse(self._tractor(_seen(0.0, CHMMR_TRACTOR_FAR + 50.0)))

    def test_holds_a_close_opponent_getting_away_but_not_one_closing(self):
        self.assertTrue(self._tractor(_seen(0.0, 400.0, closing=-10.0)))
        self.assertFalse(self._tractor(_seen(0.0, 400.0, closing=10.0)))

    def test_no_pull_on_ships_it_cannot_hold_or_see_or_without_battery(self):
        mid = _seen(0.0, 1000.0)
        self.assertFalse(self._tractor(mid, enemy_name="Arilou"))
        self.assertFalse(self._tractor(mid, enemy_name="Slylandro"))
        self.assertFalse(self._tractor(mid, enemy_name="Androsynth", blazer=True))
        self.assertTrue(self._tractor(mid, enemy_name="Androsynth"))
        self.assertFalse(self._tractor(mid, enemy_name="Ilwrath", cloaked=True))
        self.assertFalse(
            self._tractor(mid, battery_fraction=CHMMR_TRACTOR_BATTERY - 0.05)
        )


class EarthlingNukeAndChmmrTests(unittest.TestCase):
    def _context(self, enemy, *, enemy_name="Spathi", battery=18.0, forward_speed=0.0):
        context = _rule_context(
            "Earthling",
            enemy=enemy,
            enemy_name=enemy_name,
            battery=battery,
            max_battery=18.0,
        )
        context.view.me.a1_ready = True
        context.view.me.velocity_forward = forward_speed
        context.view.me.speed = abs(forward_speed)
        return context

    def test_holds_nukes_while_flying_fast_enough_to_overrun_them(self):
        fast = self._context(_seen(0.0, 1500.0), forward_speed=EARTHLING_NUKE_OVERRUN_SPEED + 5.0)
        STRATEGIES["Earthling"](fast)
        self.assertFalse(fast.plan.a1)

        slow = self._context(_seen(0.0, 1500.0), forward_speed=EARTHLING_NUKE_OVERRUN_SPEED - 5.0)
        STRATEGIES["Earthling"](slow)
        self.assertTrue(slow.plan.a1)

    def test_flies_across_a_chmmrs_bearing_between_nukes(self):
        # No battery for a nuke: it steers 90 degrees off the Chmmr.
        context = self._context(_seen(20.0, 1500.0), enemy_name="Chmmr", battery=0.0)

        with mock.patch.object(BaselineContext, "steer", wraps=context.steer) as steer:
            STRATEGIES["Earthling"](context)

        heading = steer.call_args_list[0].args[0]
        self.assertAlmostEqual(abs(wrap_degrees(heading - 20.0)), EARTHLING_CHMMR_ORBIT_OFFSET)

    def test_still_nukes_a_chmmr_when_it_can(self):
        context = self._context(_seen(0.0, 1500.0), enemy_name="Chmmr")

        STRATEGIES["Earthling"](context)

        self.assertTrue(context.plan.a1)


class AndrosynthBlazerSteeringTests(unittest.TestCase):
    def _blazer(self, enemy_name):
        # Off to the right, crossing ahead of me.
        crossing = dataclasses.replace(_seen(90.0, 600.0), velocity_forward=30.0)
        context = _rule_context(
            "Androsynth", enemy=crossing, enemy_name=enemy_name, battery=24.0, max_battery=24.0
        )
        context.view.me.androsynth_blazer = True
        context.view.me.battery_fraction = 1.0
        return context

    def _steered_bearing(self, context):
        with mock.patch.object(BaselineContext, "steer", wraps=context.steer) as steer:
            STRATEGIES["Androsynth"](context)
        return steer.call_args_list[0].args[0]

    def test_blazer_rams_at_the_intercept_point(self):
        context = self._blazer("Spathi")

        bearing = self._steered_bearing(context)

        self.assertAlmostEqual(bearing, blazer_intercept_bearing(context))
        # It leads the crossing opponent, ahead of its bearing.
        self.assertGreater(bearing, 0.0)
        self.assertLess(bearing, 80.0)

    def test_blazer_rams_straight_at_a_supox_or_mmrnmrhm(self):
        for name in ("Supox", "Mmrnmrhm"):
            with self.subTest(name=name):
                self.assertAlmostEqual(self._steered_bearing(self._blazer(name)), 90.0)


class OrzTurretTests(unittest.TestCase):
    def _context(self, enemy=None, *, turret=0.0, enemy_name="Spathi", **kwargs):
        context = _rule_context(
            "Orz", enemy=enemy or _seen(0.0, 2000.0), enemy_name=enemy_name, **kwargs
        )
        context.view.me.orz_turret_bearing = turret
        # No marine to launch.
        context.view.me.orz_marines = 2
        context.view.me.crew = 16.0
        return context

    def test_observes_the_turret_bearing_from_the_nose(self):
        simulation = _duel("Orz", "Spathi")
        me = simulation.player1
        me.heading = const.SHIP_DIRECTIONS // 8
        me.rotation = me.heading * const.TURN_ANGLE
        # Three facings clockwise of the nose.
        me.turret.relative_heading = 3

        view = ObservationView(_observation(simulation))

        self.assertAlmostEqual(view.me.orz_turret_bearing, 3 * ORZ_TURRET_STEP)

    def test_turns_the_turret_while_the_hull_holds_its_heading(self):
        context = self._context()
        context.plan.claim_turn(TURN_NONE)
        context.plan.claim_thrust(True)

        self.assertFalse(context.turn_turret(90.0))

        plan = context.plan
        self.assertEqual((plan.turn, plan.a2, plan.a1, plan.thrust), (TURN_RIGHT, True, False, True))

    def test_leaves_the_turret_while_the_hull_turns_or_a_shot_fires(self):
        turning = self._context()
        turning.plan.claim_turn(TURN_LEFT)
        self.assertFalse(turning.turn_turret(90.0))
        self.assertEqual(turning.plan.turn, TURN_LEFT)
        self.assertIsNone(turning.plan.a2)

        firing = self._context()
        firing.plan.claim_a1(True)
        self.assertFalse(firing.turn_turret(-90.0))
        self.assertIsNone(firing.plan.turn)
        self.assertIsNone(firing.plan.a2)

    def test_turret_points_within_half_a_facing(self):
        context = self._context(turret=45.0)

        self.assertTrue(context.turn_turret(45.0 + 0.4 * ORZ_TURRET_STEP))
        self.assertIsNone(context.plan.turn)
        self.assertIsNone(context.plan.a2)

    def test_runs_from_a_kited_opponent_and_fires_back_with_the_turret(self):
        chaser = _seen(180.0, 800.0, closing=30.0)
        turning = self._context(chaser, enemy_name="KzerZa")
        STRATEGIES["Orz"](turning)
        # Hull straight away and thrusting; the turret swings round.
        self.assertTrue(turning.plan.thrust)
        self.assertIn(turning.plan.turn, (TURN_LEFT, TURN_RIGHT))
        self.assertTrue(turning.plan.a2)
        self.assertFalse(turning.plan.a1)

        facing_back = self._context(chaser, enemy_name="KzerZa", turret=180.0)
        STRATEGIES["Orz"](facing_back)
        self.assertTrue(facing_back.plan.thrust)
        self.assertEqual(facing_back.plan.turn, TURN_NONE)
        self.assertTrue(facing_back.plan.a1)
        self.assertFalse(facing_back.plan.a2)

    def test_turret_aims_on_the_approach_except_at_shots_it_must_dodge(self):
        ahead = _seen(45.0, 3000.0)
        approach = self._context(ahead)
        STRATEGIES["Orz"](approach)
        self.assertEqual(approach.plan.turn, TURN_RIGHT)
        self.assertTrue(approach.plan.a2)

        dodger = self._context(ahead, enemy_name="KohrAh")
        STRATEGIES["Orz"](dodger)
        self.assertIsNone(dodger.plan.a2)

    def test_shoots_a_threat_down_with_the_turret_keeping_its_course(self):
        shell = _shot(ORZ_TURRET_STEP, 300.0, closing=60.0, damage=3.0, hit_points=2.0)
        turning = self._context(enemy_objects=[shell])

        self.assertTrue(turning.evade_threats())
        plan = turning.plan
        self.assertEqual((plan.turn, plan.a2, plan.a1, plan.thrust), (TURN_RIGHT, True, False, False))

        on_it = self._context(enemy_objects=[shell], turret=ORZ_TURRET_STEP)
        self.assertTrue(on_it.evade_threats())
        self.assertTrue(on_it.plan.a1)
        self.assertFalse(on_it.plan.a2)


class BaselinePhaseMixerTests(unittest.TestCase):
    def test_phases_last_span_times_multiplier_frames(self):
        rng = mock.Mock()
        rng.random.side_effect = [0.1, 0.9, 0.1]
        mixer = BaselinePhaseMixer(
            "Earthling",
            baseline_use=0.5,
            frames_per_phase=3,
            rng=rng,
        )

        phases = [mixer.begin_decision() for _ in range(9)]

        self.assertEqual(phases, [True] * 3 + [False] * 3 + [True] * 3)
        self.assertEqual(rng.random.call_count, 3)
        self.assertEqual(mixer.decisions, 9)

    def test_reset_starts_a_new_phase_roll(self):
        rng = mock.Mock()
        rng.random.side_effect = [0.1, 0.9]
        mixer = BaselinePhaseMixer(
            "Earthling",
            baseline_use=0.5,
            frames_per_phase=10,
            rng=rng,
        )

        self.assertTrue(mixer.begin_decision())
        mixer.reset()
        self.assertFalse(mixer.begin_decision())

    def test_turn_memory_skips_frames_flown_by_exploration(self):
        mixer = BaselinePhaseMixer("Earthling", baseline_use=1.0, frames_per_phase=10)

        with (
            mock.patch.object(mixer.controller, "decide", return_value=0),
            mock.patch.object(mixer.controller, "interrupt") as interrupt,
        ):
            mixer.begin_decision()
            mixer.baseline_action([0.0])
            mixer.begin_decision()
            mixer.baseline_action([0.0])
            interrupt.assert_not_called()
            mixer.begin_decision()  # An exploratory frame.
            mixer.begin_decision()
            mixer.baseline_action([0.0])

        interrupt.assert_called_once()
        self.assertEqual((mixer.baseline_decisions, mixer.decisions), (3, 4))

    def test_mixer_is_only_built_when_baseline_use_is_positive(self):
        config = TrainingOrchestrationConfig(trainee_ship="Earthling")
        self.assertIsNone(baseline_mixer_for_config(config))
        mixer = baseline_mixer_for_config(
            TrainingOrchestrationConfig(
                trainee_ship="Earthling",
                baseline_use=0.25,
                action_frame_span=4,
                frame_span_multiplier=7,
            )
        )
        self.assertEqual(mixer.frames_per_phase, 28)
        self.assertEqual(mixer.baseline_use, 0.25)


class PolicyMixingTests(unittest.TestCase):
    def _mixer(self, phase):
        mixer = mock.Mock()
        mixer.begin_decision.return_value = phase
        mixer.baseline_action.return_value = 5
        mixer.baseline_decisions = 1
        mixer.decisions = 2
        return mixer

    def test_baseline_phase_decides_every_frame(self):
        rng = mock.Mock()
        rng.random.return_value = 0.9
        mixer = self._mixer(True)
        policy = ValueNetworkPolicy(
            object(),
            epsilon=0.1,
            action_frame_span=3,
            rng=rng,
            baseline_mixer=mixer,
        )

        selections = [policy.prepare_action_selection([0.0]) for _ in range(4)]

        for selection in selections:
            self.assertEqual(selection.action_index, 5)
            self.assertTrue(selection.guided)
            self.assertFalse(selection.exploratory)
            self.assertFalse(selection.held)
        self.assertEqual(mixer.baseline_action.call_count, 4)
        self.assertEqual(mixer.begin_decision.call_count, 4)
        # Exploration is rolled once per span window, not every frame.
        self.assertEqual(rng.random.call_count, 2)

    def test_exploratory_action_is_held_for_the_span_while_phases_advance(self):
        rng = mock.Mock()
        rng.random.return_value = 0.0
        rng.randrange.return_value = 2
        mixer = self._mixer(True)
        policy = ValueNetworkPolicy(
            object(),
            epsilon=0.5,
            action_frame_span=3,
            rng=rng,
            baseline_mixer=mixer,
        )

        selections = [policy.prepare_action_selection([0.0]) for _ in range(3)]

        self.assertEqual([s.action_index for s in selections], [2, 2, 2])
        self.assertEqual([s.held for s in selections], [False, True, True])
        self.assertTrue(all(s.exploratory for s in selections))
        self.assertEqual(mixer.begin_decision.call_count, 3)
        mixer.baseline_action.assert_not_called()

    def test_exploration_applies_during_baseline_phases(self):
        rng = mock.Mock()
        rng.random.return_value = 0.0
        rng.randrange.return_value = 2
        mixer = self._mixer(True)
        policy = ValueNetworkPolicy(
            object(),
            epsilon=0.5,
            rng=rng,
            baseline_mixer=mixer,
        )

        selection = policy.prepare_action_selection([0.0])

        self.assertTrue(selection.exploratory)
        self.assertEqual(selection.action_index, 2)
        mixer.baseline_action.assert_not_called()

    def test_network_phase_defers_to_inference_and_reset_resets_mixer(self):
        rng = mock.Mock()
        rng.random.return_value = 0.9
        mixer = self._mixer(False)
        policy = ValueNetworkPolicy(object(), epsilon=0.1, rng=rng, baseline_mixer=mixer)

        self.assertIsNone(policy.prepare_action_selection([0.0]))
        policy.reset_exploration_span()
        mixer.reset.assert_called_once()
        self.assertEqual(policy.baseline_counts, (1, 2))


class CoordinatedScheduleMixingTests(unittest.TestCase):
    def test_baseline_phase_records_skip_inference_and_evaluation_records_fly_alone(self):
        rng = mock.Mock()
        rng.random.return_value = 0.9
        training_mixer = mock.Mock()
        training_mixer.begin_decision.return_value = True
        training_mixer.baseline_action.return_value = 4
        evaluation_mixer = mock.Mock()
        schedule = CoordinatedExplorationSchedule(
            epsilon=0.1,
            frame_span=2,
            rng=rng,
            greedy_record_ids=(2,),
            baseline_mixers={1: training_mixer, 2: evaluation_mixer},
        )
        requests = (
            CoordinatedActionRequest(record_id=1, policy=None, observation=[0.0]),
            CoordinatedActionRequest(record_id=2, policy=None, observation=[0.0]),
        )

        selections, deciders = schedule.prepare_selections(requests)

        self.assertEqual([request.record_id for request in deciders], [2])
        self.assertEqual(selections[1].action_index, 4)
        self.assertTrue(selections[1].guided)
        evaluation_mixer.begin_decision.assert_not_called()
        again, _ = schedule.prepare_selections(requests[:1])
        self.assertFalse(again[1].held)
        self.assertTrue(again[1].guided)
        self.assertEqual(training_mixer.baseline_action.call_count, 2)


class BaselineUseDecayTests(unittest.TestCase):
    def test_decay_respects_floor(self):
        config = TrainingOrchestrationConfig(
            trainee_ship="Earthling",
            baseline_use=0.5,
            baseline_floor=0.2,
            baseline_decay=0.5,
        )
        self.assertEqual(initial_baseline_use(config), 0.5)
        self.assertEqual(advance_baseline_use(config, 0.5), 0.25)
        self.assertEqual(advance_baseline_use(config, 0.25), 0.2)

    def test_evaluation_config_turns_baseline_control_off(self):
        config = evaluation_batch_config(
            TrainingOrchestrationConfig(trainee_ship="Earthling", baseline_use=0.6)
        )
        self.assertEqual(config.baseline_use, 0.0)
        self.assertEqual(config.epsilon, 0.0)
        self.assertEqual(config.opponent_mode, "simple")

    def test_session_decays_after_training_batches_only_and_saves_regimen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = TrainingModelRepository(root / "bundled", root / "user")
            metadata = metadata_from_state(
                ship="Earthling",
                slot=1,
                description="Baseline",
                architecture=model_architecture_metadata(8, 1),
                training={"regimen": {}},
            )
            slot = repository.create_or_update_user_model(metadata)
            seen = []

            def batch_runner(**kwargs):
                seen.append(kwargs["config"])
                return TrainingBatchResult(
                    completed_rounds=0,
                    replay_size=0,
                    optimization_losses=(),
                    round_results=(),
                    baseline_decisions=3,
                    total_decisions=4,
                )

            session = TrainingSession(
                repository=repository,
                slot=slot,
                metadata=metadata,
                config=TrainingOrchestrationConfig(
                    trainee_ship="Earthling",
                    opponent_mode="simple",
                    hidden_layer_width=8,
                    hidden_layer_count=1,
                    starting_baseline_use=0.5,
                    baseline_use=0.5,
                    baseline_floor=0.1,
                    baseline_decay=0.8,
                    frame_span_multiplier=5,
                ),
                batch_grouping=2,
                batch_runner=batch_runner,
            )
            session.run_synchronously(max_batches=2)
            saved = json.loads((root / "user" / "Earthling-01.json").read_text())

        self.assertEqual([config.baseline_use for config in seen], [0.5, 0.0])
        self.assertAlmostEqual(session.status.current_baseline_use, 0.4)
        self.assertAlmostEqual(session.status.last_baseline_share, 0.75)
        regimen = saved["training"]["regimen"]
        self.assertAlmostEqual(regimen["current_baseline_use"], 0.4)
        self.assertEqual(regimen["frame_span_multiplier"], 5)
        self.assertEqual(
            set(baseline_regimen_metadata(session.config, current_baseline_use=0.4)),
            {
                "starting_baseline_use",
                "current_baseline_use",
                "baseline_floor",
                "baseline_decay",
                "frame_span_multiplier",
            },
        )


class BaselineOpponentControllerTests(unittest.TestCase):
    def test_decides_every_frame(self):
        simulation = _simulation("Earthling", "Spathi")
        controller = BaselineOpponentController()

        with mock.patch.object(
            BaselineController,
            "decide",
            side_effect=[ACTION_INDEX_BY_MASK[1], ACTION_INDEX_BY_MASK[8]],
        ) as decide:
            actions = [controller.direct_controls_for_frame(simulation) for _ in range(2)]

        self.assertEqual(decide.call_count, 2)
        self.assertEqual([action.mask for action in actions], [1, 8])


class BattleModeBaselineTests(unittest.TestCase):
    def test_ai_settings_normalize_legacy_flags_and_modes(self):
        self.assertIsNone(normalize_ai_setting(False))
        self.assertEqual(normalize_ai_setting(True), AI_MODE_TRAINED)
        self.assertEqual(normalize_ai_setting("baseline"), AI_MODE_BASELINE)
        self.assertIsNone(normalize_ai_setting("unknown"))

    def test_baseline_mode_skips_model_resolution(self):
        simulation = _simulation("Earthling", "Spathi")
        manager = BattleAIManager({1: "baseline"}, model_inference_available=False)

        with mock.patch.object(manager, "_resolve_model") as resolve:
            manager.bind_round(simulation)
            actions = manager.actions_for_frame(simulation)

        resolve.assert_not_called()
        self.assertEqual(manager.status_for_player(1), BattleControllerStatus("baseline"))
        self.assertEqual(manager.label_for_player(1), "Baseline")
        self.assertIsInstance(manager._controllers[1], BaselineBattleController)
        self.assertEqual(set(actions[1]), {"forward", "left", "right", "action1", "action2"})

    def test_fleet_ai_mode_round_trips_and_defaults_to_trained(self):
        codec = FleetsCodec(SHIP_DEFINITIONS)
        fleets = Fleets(
            PlayerFleet(("Earthling",), True, FLEET_AI_MODE_BASELINE),
            PlayerFleet(("Spathi",), True),
        )

        decoded = codec.decode(codec.encode(fleets))

        self.assertEqual(decoded.player1.ai_setting, FLEET_AI_MODE_BASELINE)
        self.assertIs(decoded.player2.ai_setting, True)
        legacy = codec.decode({"Player1": {"ships": [], "ai": True}})
        self.assertIs(legacy.player1.ai_setting, True)

    def test_fleet_ai_toggle_cycles_human_trained_baseline(self):
        from src.Menus.pick_fleet import AIModeToggleButton

        pygame.font.init()
        toggle = AIModeToggleButton(0, 0, 120, 30, "AI")
        click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(5, 5), button=1)

        values = []
        for _ in range(3):
            toggle.handle_event(click)
            values.append((toggle.value, toggle.text))

        self.assertEqual(
            values,
            [(True, "AI"), ("baseline", "Base AI"), (False, "AI")],
        )


class TrainAIBaselineSettingsTests(unittest.TestCase):
    def test_saved_baseline_settings_load_and_old_models_restart_at_starting(self):
        from src.Menus.train_ai import TrainingUIState, apply_model_conditions_to_state

        state = TrainingUIState()
        skipped = apply_model_conditions_to_state(
            state,
            {
                "training": {
                    "opponent": {"mode": "all", "ai_opponent_chance": 65.0},
                    "regimen": {
                        "starting_baseline_use": 0.7,
                        "current_baseline_use": 0.42,
                        "baseline_floor": 0.05,
                        "baseline_decay": 0.99,
                        "frame_span_multiplier": 12,
                    },
                }
            },
        )

        self.assertFalse(skipped)
        self.assertEqual(state.baseline_opponent_chance, 35.0)
        self.assertEqual(state.starting_baseline_use, 0.7)
        self.assertEqual(state.current_baseline_use, 0.42)
        self.assertEqual(state.baseline_floor, 0.05)
        # 0.99 halves in 69 batches; the nearest setting is 100.
        self.assertEqual(state.baseline_decay_half_life, 100)
        self.assertEqual(state.frame_span_multiplier, 12)

        legacy = TrainingUIState(starting_baseline_use=0.3, current_baseline_use=0.1)
        apply_model_conditions_to_state(
            legacy,
            {"training": {"regimen": {"epsilon_decay": 0.99}}},
        )
        self.assertEqual(legacy.current_baseline_use, 0.3)

    def test_settings_comparison_ignores_progress_and_retired_opponent_keys(self):
        from src.Menus.train_ai import _training_settings_match

        first = {
            "opponent": {"mode": "all", "ai_opponent_chance": 50.0, "a1_activity": 25.0},
            "regimen": {"starting_baseline_use": 0.5, "current_baseline_use": 0.5},
        }
        second = {
            "opponent": {"mode": "all", "ai_opponent_chance": 50.0},
            "regimen": {"starting_baseline_use": 0.5, "current_baseline_use": 0.31},
        }

        self.assertTrue(_training_settings_match(first, second))


if __name__ == "__main__":
    unittest.main()
