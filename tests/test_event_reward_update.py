import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle import collision_responses
from src.Objects.Ships.Chenjesu.A1.ChenjesuA1 import ChenjesuA1Shard
from src.Objects.Ships.Chmmr.A3.ChmmrSatellite import ChmmrSatellite
from src.Objects.Ships.Chmmr.A3.ChmmrSatelliteLaser import ChmmrSatelliteLaser
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.registry import create_ability
from src.Objects.Ships.catalog import ABILITIES_DATA
from src.audio import NullAudioService
from src.training import event_ledger
from src.training.causal_credit import (
    AbilityRewardCredit,
    ORIGIN_KIND_AUTONOMOUS_FIRE,
    ORIGIN_KIND_LAUNCH,
    ORIGIN_KIND_PRESS,
    ORIGIN_KIND_RELEASE,
    RewardOrigin,
    full_weight_credit,
    reward_credit_for,
)
from src.training.rewards import (
    REWARD_DIE,
    REWARD_DEBUFF_ENEMY,
    REWARD_ENEMY_LOSES_CREW,
    REWARD_KILL_ENEMY,
    REWARD_KILL_ENEMY_OBJECT,
    REWARD_LOSE_CREW,
    REWARD_POINT_A1,
    REWARD_SPAWN_A1,
    RewardDecisionFrame,
    RewardFrameOutcome,
    RollingReturnPipeline,
    StagedTrajectoryPipeline,
)
from src.training.cpu_contracts import OpponentSpec
from src.training.cpu_contracts import TrainingOrchestrationConfig
from src.training.contracts import OBSERVATION_INPUT_SIZE
from src.training.episode_metrics import PendingCombatEpisode, finalize_pending_episodes


class CausalCreditContractTests(unittest.TestCase):
    def test_training_orchestration_defaults_to_causal_without_metadata_flag(self):
        config = TrainingOrchestrationConfig(trainee_ship="Earthling")
        self.assertEqual(config.reward_mode, "causal")

    def test_weighted_origins_validate_identity_and_sum(self):
        credit = AbilityRewardCredit(
            "trajectory",
            (
                RewardOrigin("trajectory", 5, 0.5, "press"),
                RewardOrigin("trajectory", 8, 0.5, "release"),
            ),
        )
        self.assertEqual([origin.frame_index for origin in credit.origins], [5, 8])
        with self.assertRaises(ValueError):
            AbilityRewardCredit(
                "trajectory",
                (RewardOrigin("trajectory", 5, 0.75, "press"),),
            )
        with self.assertRaises(ValueError):
            RewardOrigin("trajectory", 5, float("nan"), "press")

    def test_committed_trainee_action_binds_origin_but_opponent_action_does_not(self):
        trainee = SimpleNamespace()
        opponent = SimpleNamespace()
        trainee_obj = SimpleNamespace()
        opponent_obj = SimpleNamespace()
        ledger = event_ledger.BattleEventLedger()
        ledger.start_reward_trajectory(trainee, trajectory_id="trajectory")
        ledger.begin_decision(trainee, 17, 9)

        ledger.bind_committed_action(trainee, 1, (trainee_obj,))
        ledger.bind_committed_action(opponent, 1, (opponent_obj,))

        credit = reward_credit_for(trainee_obj)
        self.assertEqual(credit.trajectory_id, "trajectory")
        self.assertEqual(credit.origins[0].frame_index, 17)
        self.assertEqual(credit.origins[0].kind, ORIGIN_KIND_PRESS)
        self.assertIsNone(reward_credit_for(opponent_obj))

    def test_detached_derived_object_retains_inherited_credit(self):
        source = SimpleNamespace()
        child = SimpleNamespace(parent=source)
        credit = full_weight_credit("trajectory", 4, kind=ORIGIN_KIND_LAUNCH)
        event_ledger.bind_reward_credit(source, credit)
        event_ledger.inherit_credit(child, source)
        child.parent = None
        self.assertIs(reward_credit_for(child), credit)

    def test_credited_derived_object_gets_its_own_later_spawn_stamp(self):
        trainee = SimpleNamespace()
        fighter = SimpleNamespace(parent=trainee)
        child = SimpleNamespace(
            name="KzerZaA2Laser",
            type="laser",
            parent=fighter,
        )
        ledger = event_ledger.BattleEventLedger()
        ledger.start_reward_trajectory(trainee, trajectory_id="trajectory")
        ledger.begin_decision(trainee, 1, 0)
        ledger.bind_committed_action(trainee, 2, (fighter,))
        event_ledger.inherit_credit(child, fighter)

        ledger.current_frame = 2
        ledger.record_object_spawned(child)

        self.assertEqual(event_ledger.spawn_stamp_for(fighter), (1, 1))
        self.assertEqual(event_ledger.spawn_stamp_for(child), (2, 2))

    def test_closed_origin_is_rejected_and_diagnosed(self):
        trainee = SimpleNamespace()
        source = SimpleNamespace(name="TestAbility")
        ledger = event_ledger.BattleEventLedger()
        ledger.start_reward_trajectory(trainee, trajectory_id="trajectory")
        event_ledger.bind_reward_credit(
            source,
            full_weight_credit("trajectory", 1),
        )
        event = ledger.record_debuff_applied(
            SimpleNamespace(),
            event_ledger.DEBUFF_DOGI_DRAIN,
            source=source,
        )
        ledger.close_reward_trajectory()
        self.assertIsNone(
            ledger.resolve_event_credit(event, component="Debuff enemy", expected=True)
        )
        self.assertEqual(ledger.diagnostics.closed_trajectory_rejections["TestAbility"], 1)


class StagedTrajectoryParityTests(unittest.TestCase):
    def test_packed_staging_matches_rolling_pipeline_exact_contract(self):
        weights = {REWARD_POINT_A1: 2.5}
        rolling = RollingReturnPipeline(gamma=0.9, reward_weights=weights)
        staged = StagedTrajectoryPipeline(gamma=0.9, reward_weights=weights)
        expected = []
        frame_count = rolling.discount_cutoff_frames + 7

        for frame_id in range(1, frame_count + 1):
            decision = RewardDecisionFrame(
                frame_id=frame_id,
                observation=(float(frame_id), float(frame_id % 3)),
                action_index=frame_id % 16,
                a1_pointing=frame_id % 2 == 0,
            )
            outcome = RewardFrameOutcome(
                frame_id=frame_id,
                terminal=frame_id == frame_count,
            )
            staged.stage_decision(decision, trajectory_id="trajectory")
            expected.extend(rolling.add_frame(decision, outcome))
            actual = staged.add_frame(decision, outcome)

        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertEqual(left.observation, right.observation)
            self.assertEqual(left.action_index, right.action_index)
            self.assertEqual(left.start_frame_id, right.start_frame_id)
            self.assertEqual(left.end_frame_id, right.end_frame_id)
            self.assertEqual(left.actual_frame_count, right.actual_frame_count)
            self.assertEqual(left.terminal_truncated, right.terminal_truncated)
            self.assertAlmostEqual(left.return_value, right.return_value, places=12)
            for component in left.component_values:
                self.assertAlmostEqual(
                    left.component_values[component],
                    right.component_values[component],
                    places=12,
                )
        self.assertEqual(staged.pending_count, 0)
        self.assertFalse(staged.is_open)

    def test_staging_storage_is_packed_and_reports_peak_bytes(self):
        pipeline = StagedTrajectoryPipeline(gamma=0.99, mode="causal")
        observation = tuple(float(index) for index in range(OBSERVATION_INPUT_SIZE))
        for frame_id in range(1, 11):
            pipeline.stage_decision(
                RewardDecisionFrame(frame_id, observation, frame_id % 16),
                trajectory_id="trajectory",
            )
        expected_bytes = 10 * (OBSERVATION_INPUT_SIZE * 4 + 4 + 8 + 19 * 8)
        self.assertEqual(pipeline.staged_storage_bytes, expected_bytes)
        self.assertEqual(pipeline.peak_staged_frames, 10)
        self.assertEqual(pipeline.peak_staged_bytes, expected_bytes)
        with self.assertRaises(RuntimeError):
            pipeline.shadow_immediate_components_for_frame(1)

        shadow = StagedTrajectoryPipeline(gamma=0.99, mode="shadow")
        for frame_id in range(1, 11):
            shadow.stage_decision(
                RewardDecisionFrame(frame_id, observation, frame_id % 16),
                trajectory_id="trajectory",
            )
        shadow_bytes = 10 * (OBSERVATION_INPUT_SIZE * 4 + 4 + 8 + 38 * 8)
        self.assertEqual(shadow.staged_storage_bytes, shadow_bytes)
        self.assertEqual(shadow.peak_staged_bytes, shadow_bytes)

    def test_shadow_finalization_reports_aligned_target_distributions(self):
        trainee = SimpleNamespace(name="Earthling")
        enemy = SimpleNamespace(name="Enemy", current_hp=10)
        source = SimpleNamespace(
            name="EarthlingA1",
            type="projectile",
            parent=trainee,
        )
        ledger = event_ledger.BattleEventLedger()
        ledger.start_reward_trajectory(trainee, trajectory_id="trajectory")
        pipeline = StagedTrajectoryPipeline(gamma=0.5, mode="shadow")

        first = RewardDecisionFrame(
            1, (1.0,), 3, self_ship=trainee, enemy_ship=enemy
        )
        ledger.begin_decision(trainee, 1, 3, reward_mode="shadow")
        event_ledger.bind_reward_credit(
            source,
            full_weight_credit("trajectory", 1, kind=ORIGIN_KIND_PRESS),
        )
        pipeline.stage_decision(first, trajectory_id="trajectory")
        pipeline.add_frame(first, RewardFrameOutcome(1), ledger=ledger)

        second = RewardDecisionFrame(
            2, (2.0,), 4, self_ship=trainee, enemy_ship=enemy
        )
        ledger.begin_decision(trainee, 2, 4, reward_mode="shadow")
        event = ledger.record_crew_changed(enemy, -1, source=source)
        pipeline.stage_decision(second, trajectory_id="trajectory")
        baseline = pipeline.add_frame(
            second,
            RewardFrameOutcome(2, events=(event,), terminal=True),
            ledger=ledger,
        )
        proposed = pipeline.last_shadow_samples
        comparison = pipeline.last_shadow_comparison

        self.assertEqual(
            [sample.component_values[REWARD_ENEMY_LOSES_CREW] for sample in baseline],
            [0.5, 1.0],
        )
        self.assertEqual(
            [sample.component_values[REWARD_ENEMY_LOSES_CREW] for sample in proposed],
            [1.0, 0.0],
        )
        component = comparison.by_component[REWARD_ENEMY_LOSES_CREW]
        self.assertEqual(component.baseline.mean, 0.75)
        self.assertEqual(component.proposed.mean, 0.5)
        self.assertEqual(component.delta.mean, -0.25)
        self.assertEqual(
            comparison.by_component_and_action[
                (REWARD_ENEMY_LOSES_CREW, 3)
            ].proposed.maximum,
            1.0,
        )
        self.assertEqual(ledger.diagnostics.peak_staged_frames, 2)
        self.assertGreater(ledger.diagnostics.peak_staged_bytes, 0)
        self.assertEqual(ledger.diagnostics.finalized_trajectory_lengths, [2])
        self.assertEqual(ledger.diagnostics.shadow_comparison_count, 1)
        self.assertEqual(ledger.diagnostics.shadow_comparisons, [comparison])
        self.assertIs(ledger.diagnostics.last_shadow_comparison, comparison)


class RewardTrajectoryLifecycleTests(unittest.TestCase):
    def test_enemy_death_does_not_close_causal_trajectory_and_metrics_wait(self):
        pipeline = StagedTrajectoryPipeline(
            gamma=0.9,
            reward_weights={REWARD_POINT_A1: 1.0},
        )
        first = RewardDecisionFrame(1, (1.0,), 0, a1_pointing=False)
        second = RewardDecisionFrame(2, (2.0,), 1, a1_pointing=True)
        pipeline.stage_decision(first, trajectory_id="trajectory")
        self.assertEqual(
            pipeline.add_frame(first, RewardFrameOutcome(1, terminal=False)),
            [],
        )
        self.assertTrue(pipeline.is_open)
        pipeline.stage_decision(second, trajectory_id="trajectory")
        finalized = pipeline.add_frame(second, RewardFrameOutcome(2, terminal=True))

        boundaries = (
            PendingCombatEpisode(
                OpponentSpec("Earthling"), 0, 1, "resolved", True, False, False, 1, 0
            ),
            PendingCombatEpisode(
                OpponentSpec("Earthling"), 1, 2, "timeout", False, False, True, 0, 0
            ),
        )
        episodes = finalize_pending_episodes(boundaries, finalized)
        self.assertEqual([episode.mature_samples for episode in episodes], [1, 1])
        self.assertGreater(episodes[0].total_return, 0.0)
        self.assertEqual(sum(episode.mature_samples for episode in episodes), 2)

    def test_trainee_death_closes_once_and_new_life_gets_unique_trajectory(self):
        trainee = SimpleNamespace()
        replacement = SimpleNamespace()
        ledger = event_ledger.BattleEventLedger()
        first_id = ledger.start_reward_trajectory(trainee)
        ledger.begin_decision(trainee, 1, 0)
        ledger.close_reward_trajectory()
        ledger.begin_decision(replacement, 2, 0)
        self.assertNotEqual(first_id, ledger.active_trajectory_id)

        pipeline = StagedTrajectoryPipeline(gamma=0.0)
        decision = RewardDecisionFrame(1, (1.0,), 0)
        self.assertEqual(
            len(pipeline.add_frame(decision, RewardFrameOutcome(1, terminal=True))),
            1,
        )
        with self.assertRaises(RuntimeError):
            pipeline.add_frame(decision, RewardFrameOutcome(1, terminal=True))

    def test_simultaneous_death_frame_components_are_processed_before_close(self):
        trainee = SimpleNamespace()
        enemy = SimpleNamespace()
        events = (
            event_ledger.TrainingBattleEvent(
                1, event_ledger.EVENT_SHIP_DIED, target=enemy
            ),
            event_ledger.TrainingBattleEvent(
                1, event_ledger.EVENT_SHIP_DIED, target=trainee
            ),
        )
        decision = RewardDecisionFrame(
            1,
            (1.0,),
            0,
            self_ship=trainee,
            enemy_ship=enemy,
        )
        pipeline = StagedTrajectoryPipeline(
            gamma=0.0,
            reward_weights={REWARD_KILL_ENEMY: 1.0, REWARD_DIE: -1.0},
        )
        samples = pipeline.add_frame(
            decision,
            RewardFrameOutcome(1, events=events, terminal=True),
        )
        self.assertEqual(samples[0].component_values[REWARD_KILL_ENEMY], 1.0)
        self.assertEqual(samples[0].component_values[REWARD_DIE], 1.0)


class LongLivedAbilityRoutingTests(unittest.TestCase):
    def setUp(self):
        self.trainee = SimpleNamespace(name="Earthling")
        self.enemy = SimpleNamespace(name="Enemy", current_hp=10)
        self.ledger = event_ledger.BattleEventLedger()
        self.ledger.start_reward_trajectory(
            self.trainee,
            trajectory_id="trajectory",
        )

    def decision(self, frame_id):
        return RewardDecisionFrame(
            frame_id,
            (float(frame_id),),
            frame_id % 16,
            self_ship=self.trainee,
            enemy_ship=self.enemy,
        )

    def stage_origin(self, pipeline, source, *, frame_id=2):
        before = self.decision(frame_id - 1)
        self.ledger.begin_decision(self.trainee, before.frame_id, before.action_index)
        pipeline.stage_decision(before, trajectory_id="trajectory")
        pipeline.add_frame(before, RewardFrameOutcome(before.frame_id), ledger=self.ledger)
        origin = self.decision(frame_id)
        self.ledger.begin_decision(self.trainee, origin.frame_id, origin.action_index)
        pipeline.stage_decision(origin, trajectory_id="trajectory")
        event_ledger.bind_reward_credit(
            source,
            full_weight_credit("trajectory", frame_id),
        )
        source._training_origin_enemy_death_count = 0
        pipeline.add_frame(origin, RewardFrameOutcome(origin.frame_id), ledger=self.ledger)

    def test_repeated_dogi_debuffs_accumulate_only_at_launch_origin(self):
        source = SimpleNamespace(name="ChenjesuA2", parent=self.trainee)
        pipeline = StagedTrajectoryPipeline(gamma=0.5, mode="causal")
        self.stage_origin(pipeline, source)

        for frame_id in (3, 4):
            self.ledger.current_frame = frame_id
            self.ledger.begin_decision(self.trainee, frame_id, 0)
            event = self.ledger.record_debuff_applied(
                self.enemy,
                event_ledger.DEBUFF_DOGI_DRAIN,
                actor=self.trainee,
                source=source,
            )
            decision = self.decision(frame_id)
            pipeline.stage_decision(decision, trajectory_id="trajectory")
            pipeline.add_frame(
                decision,
                RewardFrameOutcome(frame_id, events=(event,)),
                ledger=self.ledger,
            )

        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_DEBUFF_ENEMY],
            2.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(3)[REWARD_DEBUFF_ENEMY],
            0.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(4)[REWARD_DEBUFF_ENEMY],
            0.0,
        )
        finalized = pipeline.flush_pending(end_frame_id=4)
        self.assertAlmostEqual(finalized[0].component_values[REWARD_DEBUFF_ENEMY], 1.0)
        self.assertEqual(finalized[1].component_values[REWARD_DEBUFF_ENEMY], 2.0)

    def test_marine_boarding_crew_loss_and_kill_share_launch_origin(self):
        source = SimpleNamespace(name="OrzA3", parent=self.trainee)
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.stage_origin(pipeline, source, frame_id=1)
        self.enemy.current_hp = 0
        self.ledger.current_frame = 2
        self.ledger.begin_decision(self.trainee, 2, 0)
        debuff = self.ledger.record_debuff_applied(
            self.enemy,
            event_ledger.DEBUFF_BOARDING_MARINE,
            source=source,
        )
        self.ledger.record_crew_changed(self.enemy, -1, source=source)
        events = tuple(event for event in self.ledger.events if event.frame_id == 2)
        decision = self.decision(2)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(2, events=(debuff,) + events[1:]),
            ledger=self.ledger,
        )

        origin = pipeline.immediate_components_for_frame(1)
        effect = pipeline.immediate_components_for_frame(2)
        self.assertEqual(origin[REWARD_DEBUFF_ENEMY], 1.0)
        self.assertEqual(origin[REWARD_ENEMY_LOSES_CREW], 1.0)
        self.assertEqual(origin[REWARD_KILL_ENEMY], 1.0)
        self.assertEqual(effect[REWARD_DEBUFF_ENEMY], 0.0)
        self.assertEqual(effect[REWARD_ENEMY_LOSES_CREW], 0.0)
        self.assertEqual(effect[REWARD_KILL_ENEMY], 0.0)

    def test_shadow_mode_keeps_legacy_effect_sample_and_computes_origin(self):
        source = SimpleNamespace(name="ChenjesuA2", parent=self.trainee)
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="shadow")
        self.stage_origin(pipeline, source, frame_id=1)
        self.ledger.current_frame = 2
        self.ledger.begin_decision(self.trainee, 2, 0)
        event = self.ledger.record_debuff_applied(
            self.enemy,
            event_ledger.DEBUFF_DOGI_DRAIN,
            source=source,
        )
        decision = self.decision(2)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(2, events=(event,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_DEBUFF_ENEMY],
            1.0,
        )
        self.assertEqual(
            pipeline.shadow_immediate_components_for_frame(1)[REWARD_DEBUFF_ENEMY],
            1.0,
        )
        self.assertEqual(
            pipeline.shadow_immediate_components_for_frame(2)[REWARD_DEBUFF_ENEMY],
            0.0,
        )

    def test_crew_bearing_fighter_destruction_routes_both_distinct_components(self):
        source = SimpleNamespace(
            name="KzerZaA2Laser",
            type="laser",
            parent=self.trainee,
        )
        destroyed = SimpleNamespace(
            name="EnemyFighter",
            type="projectile",
            parent=self.enemy,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.stage_origin(pipeline, source, frame_id=1)
        self.ledger.current_frame = 2
        self.ledger.begin_decision(self.trainee, 2, 0)
        removal = self.ledger.record_object_removed(
            destroyed,
            destroyed=True,
            reason="destruction",
            source=source,
        )
        crew = self.ledger.record_crew_changed(self.enemy, -1, source=source)
        decision = self.decision(2)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(2, events=(removal, crew)),
            ledger=self.ledger,
        )

        origin = pipeline.immediate_components_for_frame(1)
        effect = pipeline.immediate_components_for_frame(2)
        self.assertEqual(origin[REWARD_KILL_ENEMY_OBJECT], 1.0)
        self.assertEqual(origin[REWARD_ENEMY_LOSES_CREW], 1.0)
        self.assertEqual(effect[REWARD_KILL_ENEMY_OBJECT], 0.0)
        self.assertEqual(effect[REWARD_ENEMY_LOSES_CREW], 0.0)

    def test_natural_enemy_fighter_loss_keeps_fallback_without_fake_origin(self):
        enemy_fighter = SimpleNamespace(
            name="KzerZaA2",
            type="projectile",
            parent=self.enemy,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        decision = self.decision(1)
        self.ledger.begin_decision(self.trainee, 1, 0)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        event = self.ledger.record_crew_changed(
            self.enemy,
            -1,
            source=enemy_fighter,
        )
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(1, events=(event,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_ENEMY_LOSES_CREW],
            1.0,
        )
        self.assertEqual(self.ledger.diagnostics.missing_provenance, {})

    def test_closed_long_lived_origin_falls_back_and_cannot_update_old_trajectory(self):
        source = SimpleNamespace(name="ChenjesuA2", parent=self.trainee)
        old_pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.stage_origin(old_pipeline, source, frame_id=1)
        old_pipeline.flush_pending(end_frame_id=1)
        self.ledger.close_reward_trajectory()

        replacement = SimpleNamespace(name="EarthlingReplacement")
        replacement_enemy = SimpleNamespace(name="EnemyReplacement", current_hp=10)
        self.ledger.start_reward_trajectory(replacement, trajectory_id="replacement")
        self.ledger.current_frame = 2
        self.ledger.begin_decision(replacement, 2, 0)
        event = self.ledger.record_debuff_applied(
            replacement_enemy,
            event_ledger.DEBUFF_DOGI_DRAIN,
            source=source,
        )
        decision = RewardDecisionFrame(
            2,
            (2.0,),
            0,
            self_ship=replacement,
            enemy_ship=replacement_enemy,
        )
        new_pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        new_pipeline.stage_decision(decision, trajectory_id="replacement")
        new_pipeline.add_frame(
            decision,
            RewardFrameOutcome(2, events=(event,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            new_pipeline.immediate_components_for_frame(2)[REWARD_DEBUFF_ENEMY],
            1.0,
        )
        self.assertEqual(
            self.ledger.diagnostics.closed_trajectory_rejections["ChenjesuA2"],
            1,
        )

    def test_dogi_effect_after_enemy_replacement_keeps_original_launch(self):
        source = SimpleNamespace(name="ChenjesuA2", parent=self.trainee)
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.stage_origin(pipeline, source, frame_id=1)
        self.ledger.enemy_death_count = 1
        self.enemy = SimpleNamespace(name="ReplacementEnemy", current_hp=10)
        self.ledger.current_frame = 2
        self.ledger.begin_decision(self.trainee, 2, 0)
        event = self.ledger.record_debuff_applied(
            self.enemy,
            event_ledger.DEBUFF_DOGI_DRAIN,
            source=source,
        )
        decision = self.decision(2)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(2, events=(event,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_DEBUFF_ENEMY],
            1.0,
        )
        self.assertEqual(
            self.ledger.diagnostics.cross_enemy_death_effects["ChenjesuA2"],
            1,
        )


class OwnLaunchedCrewLossRoutingTests(unittest.TestCase):
    def setUp(self):
        self.trainee = SimpleNamespace(name="KzerZa", current_hp=10)
        self.enemy = SimpleNamespace(name="Earthling", current_hp=10)
        self.ledger = event_ledger.BattleEventLedger()
        event_ledger.bind_ledger(self.trainee, self.ledger)
        self.ledger.start_reward_trajectory(
            self.trainee,
            trajectory_id="trajectory",
        )

    def decision(self, frame_id):
        return RewardDecisionFrame(
            frame_id,
            (float(frame_id),),
            frame_id % 16,
            self_ship=self.trainee,
            enemy_ship=self.enemy,
        )

    def launch(self, pipeline, obj, frame_id, action_number):
        decision = self.decision(frame_id)
        self.ledger.current_frame = frame_id
        self.ledger.begin_decision(
            self.trainee,
            frame_id,
            decision.action_index,
            reward_mode=pipeline.mode,
        )
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        self.ledger.bind_committed_action(
            self.trainee,
            action_number,
            (obj,),
        )
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(frame_id),
            ledger=self.ledger,
        )

    def add_loss(self, pipeline, unit, frame_id, *, source=None):
        decision = self.decision(frame_id)
        self.ledger.current_frame = frame_id
        self.ledger.begin_decision(
            self.trainee,
            frame_id,
            decision.action_index,
            reward_mode=pipeline.mode,
        )
        event_ledger.record_launched_crew_lost(
            unit,
            source=unit if source is None else source,
        )
        event = self.ledger.events[-1]
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(frame_id, events=(event,)),
            ledger=self.ledger,
        )
        return event

    def test_natural_loss_moves_to_fighter_launch_in_live_causal_mode(self):
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.launch(pipeline, fighter, 1, 2)
        self.add_loss(pipeline, fighter, 2)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            1.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            self.ledger.diagnostics.launched_crew_loss_routes["natural"],
            1,
        )

    def test_external_loss_moves_to_fighter_launch(self):
        fighter = SimpleNamespace(
            name="OrzA3",
            type="special_object",
            parent=self.trainee,
        )
        enemy_shot = SimpleNamespace(
            name="EarthlingA2",
            type="laser",
            parent=self.enemy,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.launch(pipeline, fighter, 1, 3)
        self.add_loss(pipeline, fighter, 2, source=enemy_shot)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            1.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            self.ledger.diagnostics.launched_crew_loss_routes["external"],
            1,
        )

    def test_later_friendly_projectile_launch_receives_loss(self):
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
        )
        projectile = SimpleNamespace(
            name="KzerZaA1",
            type="projectile",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.launch(pipeline, fighter, 1, 2)
        self.launch(pipeline, projectile, 2, 1)
        self.add_loss(pipeline, fighter, 3, source=projectile)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            1.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(3)[REWARD_LOSE_CREW],
            0.0,
        )

    def test_later_fighter_launch_receives_friendly_fire_loss(self):
        projectile = SimpleNamespace(
            name="KzerZaA1",
            type="projectile",
            parent=self.trainee,
        )
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.launch(pipeline, projectile, 1, 1)
        self.launch(pipeline, fighter, 2, 2)
        self.add_loss(pipeline, fighter, 3, source=projectile)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            1.0,
        )

    def test_same_spawn_stamp_splits_friendly_fire_loss(self):
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
            _training_spawn_stamp=(2, 1),
        )
        projectile = SimpleNamespace(
            name="KzerZaA1",
            type="projectile",
            parent=self.trainee,
            _training_spawn_stamp=(2, 1),
        )
        event_ledger.bind_reward_credit(
            fighter,
            full_weight_credit("trajectory", 1),
        )
        event_ledger.bind_reward_credit(
            projectile,
            full_weight_credit("trajectory", 2),
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        for frame_id in (1, 2):
            decision = self.decision(frame_id)
            self.ledger.begin_decision(self.trainee, frame_id, 0)
            pipeline.stage_decision(decision, trajectory_id="trajectory")
            pipeline.add_frame(
                decision,
                RewardFrameOutcome(frame_id),
                ledger=self.ledger,
            )
        self.add_loss(pipeline, fighter, 3, source=projectile)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            0.5,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            0.5,
        )

    def test_closed_launch_provenance_omits_loss(self):
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
            _training_spawn_stamp=(1, 1),
        )
        event_ledger.bind_reward_credit(
            fighter,
            full_weight_credit("closed-trajectory", 1),
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.add_loss(pipeline, fighter, 1)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            self.ledger.diagnostics.launched_crew_loss_routes["closed"],
            1,
        )

    def test_missing_launch_provenance_keeps_loss_on_death_frame(self):
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.add_loss(pipeline, fighter, 1)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            1.0,
        )
        self.assertEqual(
            self.ledger.diagnostics.launched_crew_loss_routes["missing_unit"],
            1,
        )

    def test_shadow_mode_keeps_loss_on_death_frame(self):
        fighter = SimpleNamespace(
            name="KzerZaA2",
            type="special_object",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="shadow")
        self.launch(pipeline, fighter, 1, 2)
        self.add_loss(pipeline, fighter, 2)

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            1.0,
        )
        self.assertEqual(
            pipeline.shadow_immediate_components_for_frame(1)[REWARD_LOSE_CREW],
            0.0,
        )
        self.assertEqual(
            pipeline.shadow_immediate_components_for_frame(2)[REWARD_LOSE_CREW],
            1.0,
        )


class GeneralAbilityRoutingTests(LongLivedAbilityRoutingTests):
    def test_ordinary_projectile_delayed_hit_matches_normative_timeline(self):
        source = SimpleNamespace(
            name="EarthlingA1",
            type="projectile",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.5, mode="causal")
        self.stage_origin(pipeline, source, frame_id=2)
        for frame_id in range(3, 6):
            self.ledger.begin_decision(self.trainee, frame_id, 0)
            decision = self.decision(frame_id)
            pipeline.stage_decision(decision, trajectory_id="trajectory")
            pipeline.add_frame(
                decision,
                RewardFrameOutcome(frame_id),
                ledger=self.ledger,
            )

        self.enemy.current_hp = 0
        self.ledger.current_frame = 6
        self.ledger.begin_decision(self.trainee, 6, 0)
        self.ledger.record_crew_changed(self.enemy, -1, source=source)
        events = tuple(event for event in self.ledger.events if event.frame_id == 6)
        effect = self.decision(6)
        pipeline.stage_decision(effect, trajectory_id="trajectory")
        pipeline.add_frame(
            effect,
            RewardFrameOutcome(6, events=events),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_ENEMY_LOSES_CREW],
            1.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(2)[REWARD_KILL_ENEMY],
            1.0,
        )
        for frame_id in range(3, 7):
            components = pipeline.immediate_components_for_frame(frame_id)
            self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 0.0)
            self.assertEqual(components[REWARD_KILL_ENEMY], 0.0)
        finalized = pipeline.flush_pending(end_frame_id=6)
        self.assertAlmostEqual(
            finalized[0].component_values[REWARD_ENEMY_LOSES_CREW],
            0.5,
        )
        self.assertEqual(
            finalized[1].component_values[REWARD_ENEMY_LOSES_CREW],
            1.0,
        )

    def test_effect_later_than_cutoff_still_credits_live_origin(self):
        source = SimpleNamespace(
            name="EarthlingA1",
            type="projectile",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.5, mode="causal")
        self.stage_origin(pipeline, source, frame_id=1)
        effect_frame = pipeline.discount_cutoff_frames + 5
        for frame_id in range(2, effect_frame):
            self.ledger.begin_decision(self.trainee, frame_id, 0)
            decision = self.decision(frame_id)
            pipeline.stage_decision(decision, trajectory_id="trajectory")
            pipeline.add_frame(
                decision,
                RewardFrameOutcome(frame_id),
                ledger=self.ledger,
            )
        self.ledger.current_frame = effect_frame
        self.ledger.begin_decision(self.trainee, effect_frame, 0)
        event = self.ledger.record_crew_changed(self.enemy, -1, source=source)
        decision = self.decision(effect_frame)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(effect_frame, events=(event,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_ENEMY_LOSES_CREW],
            1.0,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(effect_frame)[REWARD_ENEMY_LOSES_CREW],
            0.0,
        )

    def test_fighter_child_weapon_inherits_fighter_launch_credit(self):
        from src.Objects.Ships.KzerZa.A2.KzerZaA2Laser import KzerZaA2Laser

        fighter = SimpleNamespace(
            name="KzerZaA2",
            parent=self.trainee,
            player=1,
            position=[0.0, 0.0],
            previous_position=[0.0, 0.0],
            resources=getattr(self.trainee, "resources", None),
        )
        # The real constructor needs an initialized ability parent, so exercise
        # the generic inheritance helper with the real derived class contract
        # represented by its source/child identity.
        child = SimpleNamespace(name=KzerZaA2Laser.__name__, parent=fighter)
        credit = full_weight_credit("trajectory", 4)
        event_ledger.bind_reward_credit(fighter, credit)
        event_ledger.inherit_credit(child, fighter)
        self.assertIs(reward_credit_for(child), credit)

    def test_chmmr_laser_fractional_and_lethal_effects_use_firing_origin(self):
        satellite = SimpleNamespace(
            name="ChmmrSatellite",
            type="special_object",
            parent=self.trainee,
        )
        laser = SimpleNamespace(
            name="ChmmrSatelliteLaser",
            type="laser",
            parent=satellite,
        )
        enemy_satellite = SimpleNamespace(
            name="ChmmrSatellite",
            type="special_object",
            parent=self.enemy,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        decision = self.decision(1)
        self.ledger.begin_decision(self.trainee, 1, 7)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        self.ledger.bind_autonomous_fire(laser, self.trainee)
        self.ledger.current_frame = 1
        hp_event = self.ledger.record_object_hp_changed(
            enemy_satellite,
            1,
            source=laser,
        )
        removal = self.ledger.record_object_removed(
            enemy_satellite,
            destroyed=True,
            reason="destruction",
            source=laser,
        )
        self.enemy.current_hp = 0
        self.ledger.record_crew_changed(self.enemy, -1, source=laser)
        events = tuple(event for event in self.ledger.events if event.frame_id == 1)
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(1, events=events),
            ledger=self.ledger,
        )

        components = pipeline.immediate_components_for_frame(1)
        self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 1.5)
        self.assertEqual(components[REWARD_KILL_ENEMY], 1.5)
        self.assertEqual(reward_credit_for(laser).origins[0].frame_index, 1)

    def test_satellite_body_contact_is_nonrouteable_fallback(self):
        body = SimpleNamespace(
            name="ChmmrSatellite",
            type="special_object",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        decision = self.decision(1)
        self.ledger.begin_decision(self.trainee, 1, 0)
        pipeline.stage_decision(decision, trajectory_id="trajectory")
        event = self.ledger.record_crew_changed(self.enemy, -1, source=body)
        pipeline.add_frame(
            decision,
            RewardFrameOutcome(1, events=(event,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_ENEMY_LOSES_CREW],
            1.0,
        )
        self.assertEqual(self.ledger.diagnostics.missing_provenance, {})


class PressReleaseAttributionTests(LongLivedAbilityRoutingTests):
    def test_post_release_effect_uses_press_release_split_without_rewriting_prior(self):
        source = SimpleNamespace(
            name="ChenjesuA1",
            type="projectile",
            parent=self.trainee,
        )
        pipeline = StagedTrajectoryPipeline(gamma=0.9, mode="causal")
        self.ledger.reward_mode = "causal"
        self.stage_origin(pipeline, source, frame_id=1)
        press_credit = reward_credit_for(source)

        self.ledger.current_frame = 2
        self.ledger.begin_decision(self.trainee, 2, 4, reward_mode="causal")
        pre_release = self.ledger.record_crew_changed(
            self.enemy,
            -1,
            source=source,
        )
        decision2 = self.decision(2)
        pipeline.stage_decision(decision2, trajectory_id="trajectory")
        pipeline.add_frame(
            decision2,
            RewardFrameOutcome(2, events=(pre_release,)),
            ledger=self.ledger,
        )

        self.ledger.begin_decision(self.trainee, 3, 13, reward_mode="causal")
        decision3 = RewardDecisionFrame(
            3,
            (3.0,),
            13,
            self_ship=self.trainee,
            enemy_ship=self.enemy,
        )
        pipeline.stage_decision(decision3, trajectory_id="trajectory")
        release = self.ledger.record_action_released(self.trainee, (source,))
        pipeline.add_frame(
            decision3,
            RewardFrameOutcome(3, events=(release,)),
            ledger=self.ledger,
        )
        split_credit = reward_credit_for(source)
        self.assertIsNot(split_credit, press_credit)
        self.assertEqual(
            [(origin.frame_index, origin.weight, origin.kind) for origin in split_credit.origins],
            [(1, 0.5, ORIGIN_KIND_PRESS), (3, 0.5, ORIGIN_KIND_RELEASE)],
        )
        self.assertEqual(release.metadata["action_index"], 13)
        self.assertEqual(
            pipeline.immediate_components_for_frame(3)[REWARD_SPAWN_A1],
            0.0,
        )

        self.ledger.current_frame = 4
        self.ledger.begin_decision(self.trainee, 4, 0, reward_mode="causal")
        post_release = self.ledger.record_crew_changed(
            self.enemy,
            -1,
            source=source,
        )
        decision4 = self.decision(4)
        pipeline.stage_decision(decision4, trajectory_id="trajectory")
        pipeline.add_frame(
            decision4,
            RewardFrameOutcome(4, events=(post_release,)),
            ledger=self.ledger,
        )

        self.assertEqual(
            pipeline.immediate_components_for_frame(1)[REWARD_ENEMY_LOSES_CREW],
            1.5,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(3)[REWARD_ENEMY_LOSES_CREW],
            0.5,
        )
        self.assertEqual(
            pipeline.immediate_components_for_frame(4)[REWARD_ENEMY_LOSES_CREW],
            0.0,
        )
        self.assertEqual(pre_release.metadata["reward_credit"], press_credit)

    def test_split_origin_discount_example_is_exact(self):
        source = SimpleNamespace(
            name="MelnormeA1",
            type="projectile",
            parent=self.trainee,
        )
        gamma = 0.8
        pipeline = StagedTrajectoryPipeline(gamma=gamma, mode="causal")
        self.ledger.reward_mode = "causal"
        self.stage_origin(pipeline, source, frame_id=1)
        for frame_id in (2, 3):
            self.ledger.begin_decision(
                self.trainee, frame_id, frame_id, reward_mode="causal"
            )
            decision = self.decision(frame_id)
            pipeline.stage_decision(decision, trajectory_id="trajectory")
            pipeline.add_frame(
                decision,
                RewardFrameOutcome(frame_id),
                ledger=self.ledger,
            )
        self.ledger.begin_decision(self.trainee, 4, 11, reward_mode="causal")
        release_decision = self.decision(4)
        pipeline.stage_decision(release_decision, trajectory_id="trajectory")
        self.ledger.record_action_released(self.trainee, (source,))
        pipeline.add_frame(
            release_decision,
            RewardFrameOutcome(4),
            ledger=self.ledger,
        )
        self.ledger.current_frame = 5
        self.ledger.begin_decision(self.trainee, 5, 0, reward_mode="causal")
        event = self.ledger.record_crew_changed(self.enemy, -1, source=source)
        effect = self.decision(5)
        pipeline.stage_decision(effect, trajectory_id="trajectory")
        samples = pipeline.add_frame(
            effect,
            RewardFrameOutcome(5, events=(event,), terminal=True),
            ledger=self.ledger,
        )

        samples_by_frame = {sample.start_frame_id: sample for sample in samples}
        self.assertAlmostEqual(
            samples_by_frame[1].component_values[REWARD_ENEMY_LOSES_CREW],
            0.5 + gamma**3 * 0.5,
        )
        self.assertAlmostEqual(
            samples_by_frame[2].component_values[REWARD_ENEMY_LOSES_CREW],
            gamma**2 * 0.5,
        )
        self.assertAlmostEqual(
            samples_by_frame[3].component_values[REWARD_ENEMY_LOSES_CREW],
            gamma * 0.5,
        )
        self.assertEqual(
            samples_by_frame[5].component_values[REWARD_ENEMY_LOSES_CREW],
            0.0,
        )

    def test_release_without_affected_object_creates_no_event_or_origin(self):
        self.ledger.reward_mode = "causal"
        self.ledger.begin_decision(self.trainee, 2, 9, reward_mode="causal")
        self.assertIsNone(self.ledger.record_action_released(self.trainee, ()))
        self.assertEqual(self.ledger.events, [])

    def test_repeated_release_changes_only_future_credit_snapshot(self):
        saw = SimpleNamespace(
            name="KohrAhA1",
            type="projectile",
            parent=self.trainee,
        )
        event_ledger.bind_reward_credit(
            saw,
            full_weight_credit("trajectory", 1, kind=ORIGIN_KIND_PRESS),
        )
        saw._training_origin_enemy_death_count = 0
        self.ledger.begin_decision(self.trainee, 3, 0, reward_mode="causal")
        self.ledger.record_action_released(self.trainee, (saw,))
        self.ledger.current_frame = 4
        prior_event = self.ledger.record_crew_changed(self.enemy, -1, source=saw)
        prior_credit = prior_event.metadata["reward_credit"]
        self.ledger.begin_decision(self.trainee, 5, 0, reward_mode="causal")
        self.ledger.record_action_released(self.trainee, (saw,))
        future_credit = reward_credit_for(saw)

        self.assertEqual(prior_credit.origins[1].frame_index, 3)
        self.assertEqual(future_credit.origins[0].frame_index, 1)
        self.assertEqual(future_credit.origins[1].frame_index, 5)

    def test_legacy_release_event_keeps_press_only_credit(self):
        projectile = SimpleNamespace(name="ChenjesuA1")
        credit = full_weight_credit("trajectory", 1, kind=ORIGIN_KIND_PRESS)
        event_ledger.bind_reward_credit(projectile, credit)
        self.ledger.begin_decision(self.trainee, 2, 0, reward_mode="legacy")
        self.ledger.record_action_released(self.trainee, (projectile,))
        self.assertIs(reward_credit_for(projectile), credit)


class BuiltInProvenanceTests(unittest.TestCase):
    def make_ship(self, name, player=1):
        ship = create_ship(name, player, audio_service=NullAudioService())
        ship.position = [1000.0, 1000.0]
        ship.previous_position = ship.position.copy()
        ship.velocity = [0.0, 0.0]
        ship.rotation = 0.0
        ship.heading = 0
        return ship

    def test_chenjesu_shards_inherit_crystal_press_origin(self):
        ship = self.make_ship("Chenjesu")
        ledger = event_ledger.BattleEventLedger()
        event_ledger.bind_ledger(ship, ledger)
        ledger.begin_decision(ship, 5, 8)
        crystal = ship.perform_action1()
        crystal.fragment()
        shards = crystal.drain_spawned_objects()

        self.assertTrue(shards)
        self.assertTrue(all(isinstance(shard, ChenjesuA1Shard) for shard in shards))
        self.assertTrue(all(reward_credit_for(shard) == reward_credit_for(crystal) for shard in shards))
        self.assertEqual(reward_credit_for(shards[0]).origins[0].kind, ORIGIN_KIND_PRESS)

    def test_chenjesu_and_melnorme_release_hooks_report_only_live_abilities(self):
        for ship_name in ("Chenjesu", "Melnorme"):
            with self.subTest(ship=ship_name):
                ship = self.make_ship(ship_name)
                ledger = event_ledger.BattleEventLedger()
                event_ledger.bind_ledger(ship, ledger)
                ledger.begin_decision(ship, 5, 1, reward_mode="causal")
                projectile = ship.perform_action1()

                ledger.begin_decision(ship, 8, 13, reward_mode="causal")
                affected = ship.perform_action1_release()
                release = ledger.record_action_released(ship, affected)
                credit = reward_credit_for(projectile)

                self.assertEqual(affected, (projectile,))
                self.assertEqual(release.metadata["action_index"], 13)
                self.assertEqual(
                    [
                        (origin.frame_index, origin.weight, origin.kind)
                        for origin in credit.origins
                    ],
                    [
                        (5, 0.5, ORIGIN_KIND_PRESS),
                        (8, 0.5, ORIGIN_KIND_RELEASE),
                    ],
                )
                self.assertEqual(ship.perform_action1_release(), ())

    def test_kohr_ah_release_updates_every_live_saw_independently(self):
        ship = self.make_ship("KohrAh")
        ledger = event_ledger.BattleEventLedger()
        event_ledger.bind_ledger(ship, ledger)
        first = create_ability("KohrAhA1", ship)
        second = create_ability("KohrAhA1", ship)
        event_ledger.bind_reward_credit(
            first,
            full_weight_credit("trajectory", 2, kind=ORIGIN_KIND_PRESS),
        )
        event_ledger.bind_reward_credit(
            second,
            full_weight_credit("trajectory", 4, kind=ORIGIN_KIND_PRESS),
        )
        ledger.start_reward_trajectory(ship, trajectory_id="trajectory")
        ledger.begin_decision(ship, 7, 6, reward_mode="causal")
        ship.friendly_objects = [first, second]

        affected = ship.perform_action1_release()
        ledger.record_action_released(ship, affected)

        self.assertEqual(affected, (first, second))
        self.assertEqual(
            [origin.frame_index for origin in reward_credit_for(first).origins],
            [2, 7],
        )
        self.assertEqual(
            [origin.frame_index for origin in reward_credit_for(second).origins],
            [4, 7],
        )

    def test_dogi_and_marine_credit_survives_enemy_replacement(self):
        for ship_name, action_number in (("Chenjesu", 2), ("Orz", 3)):
            with self.subTest(ship=ship_name):
                ship = self.make_ship(ship_name)
                ship.opponent = self.make_ship("Earthling", 2)
                ledger = event_ledger.BattleEventLedger()
                event_ledger.bind_ledger(ship, ledger)
                ledger.begin_decision(ship, 11, 3)
                result = ship.commit_action(ship._select_action_plan(action_number))
                self.assertTrue(result.valid)
                self.assertTrue(result.spawned_objects)
                ability = result.spawned_objects[0]
                original_credit = reward_credit_for(ability)
                ship.opponent = self.make_ship("Earthling", 2)
                ability.opponent = ship.opponent
                self.assertIs(reward_credit_for(ability), original_credit)

    def test_chmmr_laser_uses_fresh_parent_firing_frame_origin(self):
        ship = self.make_ship("Chmmr")
        ledger = event_ledger.BattleEventLedger()
        event_ledger.bind_ledger(ship, ledger)
        satellite = ChmmrSatellite(ship)
        event_ledger.bind_ledger(satellite, ledger)

        ledger.begin_decision(ship, 23, 14)
        first = ChmmrSatelliteLaser(satellite)
        ledger.begin_decision(ship, 24, 2)
        second = ChmmrSatelliteLaser(satellite)

        first_origin = reward_credit_for(first).origins[0]
        second_origin = reward_credit_for(second).origins[0]
        self.assertEqual((first_origin.frame_index, first_origin.kind), (23, ORIGIN_KIND_AUTONOMOUS_FIRE))
        self.assertEqual(second_origin.frame_index, 24)
        self.assertNotEqual(first_origin, second_origin)

    def test_fighter_removal_and_permanent_crew_loss_share_destroying_source(self):
        trainee = self.make_ship("Earthling", 1)
        enemy = self.make_ship("KzerZa", 2)
        source = SimpleNamespace(
            name="EarthlingA1",
            projectile_name="EarthlingA1",
            type="projectile",
            parent=trainee,
        )
        fighter = KzerZaA2(enemy)
        ledger = event_ledger.BattleEventLedger()
        for obj in (trainee, enemy, source, fighter):
            event_ledger.bind_ledger(obj, ledger)
        ledger.start_reward_trajectory(trainee, trajectory_id="trajectory")
        event_ledger.bind_reward_credit(source, full_weight_credit("trajectory", 7))

        collision_responses.set_projectile_hp(fighter, 0, source=source)
        ledger.record_object_removed(
            fighter,
            destroyed=True,
            reason="destruction",
            source=source,
        )
        fighter.on_destroyed()
        removal = next(event for event in ledger.events if event.event_type == event_ledger.EVENT_OBJECT_REMOVED)
        crew = next(event for event in ledger.events if event.event_type == event_ledger.EVENT_CREW_CHANGED)
        self.assertIs(removal.metadata["source"], source)
        self.assertIs(crew.obj, source)
        self.assertEqual(removal.metadata["reward_credit"], crew.metadata["reward_credit"])

    def test_every_cataloged_controlled_ability_accepts_committed_origin_credit(self):
        excluded_autonomous_or_derived = {
            "ChmmrSatellite",
            "ChmmrSatelliteLaser",
            "ChenjesuA1Shard",
            "KzerZaA2Laser",
            "SyreenCrew",
        }
        for ability_name, ability_data in ABILITIES_DATA.items():
            if ability_name in excluded_autonomous_or_derived:
                continue
            ship_name = ability_data.get("ship_name")
            if not ship_name:
                continue
            with self.subTest(ability=ability_name):
                ship = self.make_ship(ship_name)
                ledger = event_ledger.BattleEventLedger()
                event_ledger.bind_ledger(ship, ledger)
                ledger.begin_decision(ship, 3, 0)
                ability = create_ability(ability_name, ship)
                action_number = 1 if "A1" in ability_name else 2 if "A2" in ability_name else 3
                ledger.bind_committed_action(ship, action_number, (ability,))
                credit = reward_credit_for(ability)
                self.assertIsNotNone(credit)
                self.assertEqual(credit.origins[0].frame_index, 3)


if __name__ == "__main__":
    unittest.main()
