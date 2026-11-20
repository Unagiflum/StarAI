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
    TURN_LEFT,
    TURN_NONE,
    TURN_RIGHT,
    planet_contact_distance,
    simulate_planet_pass,
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
    EARTHLING_NUKE_MIN_DISTANCE,
    may_evade,
    UMGAH_CONE_HALF_ANGLE,
    UMGAH_PASS_CLEARANCE,
    UMGAH_RUN_CLEARANCE,
)
from src.training.baseline.strategies import _umgah_hops_needed as umgah_hops_needed
from src.training.baseline.strategies import _umgah_tail_heading as umgah_tail_heading
from src.training.baseline.strategies import _supox_band as supox_band
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
):
    view = mock.Mock()
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
        self.assertIsNone(context.plan.a1)

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
        STRATEGIES["Pkunk"](charged)
        low = _rule_context("Pkunk", enemy=enemy, battery_fraction=0.49)
        STRATEGIES["Pkunk"](low)

        self.assertTrue(charged.plan.a2)
        self.assertTrue(low.plan.a2)
        # Charged, it heads for a pass beside the opponent; low, it flees at 135.
        self.assertTrue(charged.plan.thrust)
        self.assertTrue(low.plan.thrust)
        self.assertEqual(charged.memory.sides.keys(), {"pkunk_pass"})
        self.assertIn("avoid_opponent", low.memory.sides)
        self.assertNotIn("pkunk_pass", low.memory.sides)

    def test_pkunk_fires_a_side_gun_at_an_opponent_abeam(self):
        # Inside firing distance but outside the band where approach backs off.
        context = _rule_context("Pkunk", enemy=_seen(90.0, 250.0))
        STRATEGIES["Pkunk"](context)

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
            STRATEGIES["Pkunk"](context)
            return bool(context.plan.a1)

        self.assertFalse(fires(0.0))
        self.assertTrue(fires(5.0))

    def test_low_battery_pkunk_keeps_station_just_outside_weapon_range(self):
        far = _rule_context("Pkunk", enemy=_seen(0.0, 2000.0), battery_fraction=0.2)
        STRATEGIES["Pkunk"](far)
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
        STRATEGIES["Pkunk"](holding)
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

        def context_at(bearing, distance, battery):
            context = _rule_context(
                "ZoqFotPik",
                enemy=_seen(bearing, distance),
                battery=battery,
                battery_fraction=battery / 10.0,
            )
            context.view.enemy_block.trackable = True
            context.view.enemy_block.max_thrust = 0.0
            STRATEGIES["ZoqFotPik"](context)
            return context

        lunging = context_at(0.0, 600.0, a2_cost)
        self.assertTrue(lunging.plan.thrust)
        self.assertEqual(lunging.plan.turn, TURN_NONE)
        self.assertFalse(lunging.plan.a2)

        # Centre distance beyond the tongue's length still touches the hull,
        # and the narrow tongue reaches a hull several degrees off the nose.
        striking = context_at(6.0, weapon_spec("ZoqFotPik", 2).range + 15.0, a2_cost)
        self.assertTrue(striking.plan.a2)

        # Without tongue battery it keeps weapon 1 distance instead.
        recharging = context_at(0.0, 600.0, a2_cost - 1)
        self.assertIn("a1_range_near", recharging.memory.sides)
        self.assertFalse(recharging.plan.a2)

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
        variant=None,
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

        empty = self._shofixti(enemy, battery=0.0, memory=memory)
        self.assertIn("avoid_opponent", empty.memory.sides)
        refilling = self._shofixti(enemy, battery=3.0, memory=memory)
        self.assertNotIn("a1_range_far", refilling.memory.sides)
        full =self._shofixti(enemy, battery=4.0, memory=memory)
        self.assertIn("a1_range_far", full.memory.sides)

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
        self.assertIn("flame_far", slower.memory.sides)

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
        STRATEGIES["Syreen"](context)
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

    def test_small_syreen_keeps_gun_range_from_fast_close_fighters(self):
        enemy = _seen(0.0, 2000.0)

        cautious = self._syreen(enemy, enemy_name="ZoqFotPik", enemy_thrust=40.0)
        self.assertIn("gun_far", cautious.memory.sides)
        grown = self._syreen(enemy, enemy_name="ZoqFotPik", enemy_thrust=40.0, crew=24.0)
        self.assertIn("song_far", grown.memory.sides)
        slower = self._syreen(enemy, enemy_name="ZoqFotPik", enemy_thrust=20.0)
        self.assertIn("song_far", slower.memory.sides)

    def test_syreen_backs_off_while_the_song_recharges(self):
        context = self._syreen(_seen(0.0, 600.0, closing=5.0), battery=4.0)

        self.assertIn("avoid_opponent", context.memory.sides)

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
        context = self._supox(_seen(0.0, 400.0))

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
        self.assertEqual(y_far.memory.sides.keys(), {"avoid_opponent"})


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
        self.assertEqual(state.baseline_decay, 0.99)
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
