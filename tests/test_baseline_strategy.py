"""Baseline strategy engine, trainee phase mixing, opponents and battle mode."""

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
from src.training.baseline.ship_specs import weapon_spec
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


def _seen(bearing, distance, *, closing=0.0, group="enemy_ship"):
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
        crew_effect=0.0,
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
):
    view = mock.Mock()
    view.enemy = enemy
    view.planet = planet
    view.enemy_ship_type = enemy_name
    view.enemy_objects = tuple(enemy_objects)
    view.objects.side_effect = lambda group: (
        tuple(enemy_objects) if group == "enemy_non_a1" else ()
    )
    view.me.held.side_effect = lambda control: control in held
    view.me.battery_fraction = battery_fraction
    view.me.battery = 100.0 if battery is None else battery
    view.me.mmrnmrhm_ywing = ywing
    view.me.androsynth_blazer = False
    view.me.velocity_forward = 0.0
    view.me.velocity_right = 0.0
    view.enemy_block.androsynth_blazer = False
    return BaselineContext(ship_name=ship_name, view=view, memory=BaselineMemory())


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
        self.assertEqual(low.memory.sides.keys(), {"avoid_opponent"})

    def test_pkunk_fires_a_side_gun_at_an_opponent_abeam(self):
        # Inside weapon range but outside the band where approach backs off.
        context = _rule_context("Pkunk", enemy=_seen(90.0, 380.0))
        STRATEGIES["Pkunk"](context)

        self.assertTrue(context.plan.a1)
        self.assertEqual(context.plan.turn, TURN_NONE)

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
