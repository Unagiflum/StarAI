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
)
from src.training.baseline.observation_view import (
    ObservationView,
    ObservedObject,
    wrap_degrees,
)
from src.training.baseline.ship_specs import (
    shofixti_blast_damage,
    shofixti_kill_distance,
    weapon_spec,
)
from src.training.baseline.strategies import (
    EARTHLING_NUKE_MIN_DISTANCE,
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
):
    view = mock.Mock()
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


class ShipRuleTests(unittest.TestCase):
    def test_pkunk_approaches_from_half_battery_and_flees_below(self):
        enemy = _seen(0.0, 2000.0, closing=5.0)
        charged = _rule_context("Pkunk", enemy=enemy, battery_fraction=0.5)
        STRATEGIES["Pkunk"](charged)
        low = _rule_context("Pkunk", enemy=enemy, battery_fraction=0.49)
        STRATEGIES["Pkunk"](low)

        self.assertTrue(charged.plan.a2)
        self.assertTrue(low.plan.a2)
        # Approaching heads 45 degrees off the opponent; fleeing 135.
        self.assertTrue(charged.plan.thrust)
        self.assertTrue(low.plan.thrust)
        self.assertEqual(charged.memory.sides.keys(), {"a1_range_far"})
        self.assertIn("avoid_opponent", low.memory.sides)
        self.assertNotIn("a1_range_far", low.memory.sides)

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
        a1_range = weapon_spec("Pkunk", 1).range
        far = _rule_context("Pkunk", enemy=_seen(0.0, 2000.0), battery_fraction=0.2)
        STRATEGIES["Pkunk"](far)
        self.assertTrue(far.plan.thrust)
        self.assertIn("station_far", far.memory.sides)

        holding = _rule_context(
            "Pkunk",
            enemy=_seen(90.0, a1_range * 1.1),
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
