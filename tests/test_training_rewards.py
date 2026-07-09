import unittest
from types import SimpleNamespace

from src.training.event_ledger import (
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_REMOVED,
    EVENT_OBJECT_SPAWNED,
    EVENT_SHIP_DIED,
    TrainingBattleEvent,
)
from src.training.rewards import (
    REWARD_A1_RANGE,
    REWARD_BATTERY_AT_ZERO,
    REWARD_DEBUFF_ENEMY,
    REWARD_DIE,
    REWARD_ENEMY_LOSES_CREW,
    REWARD_GAIN_BATTERY,
    REWARD_GAIN_CREW,
    REWARD_GET_DEBUFFED,
    REWARD_HIGH_SPEED,
    REWARD_KILL_ENEMY,
    REWARD_KILL_ENEMY_OBJECT,
    REWARD_LOSE_BATTERY,
    REWARD_LOSE_CREW,
    REWARD_POINT_A1,
    REWARD_POINT_A2,
    REWARD_SPAWN_A1,
    REWARD_SPAWN_A2,
    RewardDecisionFrame,
    RewardFrameOutcome,
    RollingReturnPipeline,
    calculate_reward_components,
)


def decision(
    frame_id,
    self_ship,
    enemy_ship,
    *,
    battery=10,
    speed=0,
    max_thrust=12,
    a1_pointing=False,
    a1_in_range=False,
    a2_pointing=False,
    a2_in_range=False,
    action_index=3,
):
    return RewardDecisionFrame(
        frame_id=frame_id,
        observation=(float(frame_id), 1.0),
        action_index=action_index,
        self_ship=self_ship,
        enemy_ship=enemy_ship,
        self_battery=battery,
        self_speed=speed,
        self_max_thrust=max_thrust,
        a1_pointing=a1_pointing,
        a1_in_range=a1_in_range,
        a2_pointing=a2_pointing,
        a2_in_range=a2_in_range,
    )


def outcome(frame_id, *, battery=10, speed=0, max_thrust=12, events=(), terminal=False):
    return RewardFrameOutcome(
        frame_id=frame_id,
        self_battery=battery,
        self_speed=speed,
        self_max_thrust=max_thrust,
        events=tuple(events),
        terminal=terminal,
    )


class TrainingRewardComponentTests(unittest.TestCase):
    def setUp(self):
        self.trainee = SimpleNamespace(name="Earthling")
        self.enemy = SimpleNamespace(name="Chenjesu")

    def test_pointing_and_range_are_normalized_by_actual_window_length(self):
        decisions = [
            decision(10, self.trainee, self.enemy, a1_pointing=True, a1_in_range=True),
            decision(11, self.trainee, self.enemy, a1_pointing=False, a1_in_range=True),
            decision(12, self.trainee, self.enemy, a2_pointing=True),
            decision(13, self.trainee, self.enemy, a2_pointing=True),
        ]
        outcomes = [outcome(decision.frame_id) for decision in decisions]

        components = calculate_reward_components(decisions[0], decisions, outcomes)

        self.assertEqual(components[REWARD_POINT_A1], 0.25)
        self.assertEqual(components[REWARD_A1_RANGE], 0.5)
        self.assertEqual(components[REWARD_POINT_A2], 0.5)

    def test_events_and_endpoint_rewards_remain_distinct(self):
        events = (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_SPAWNED,
                owner=self.trainee,
                action="A1",
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_SPAWNED,
                owner=self.trainee,
                action="A3",
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_CREW_CHANGED,
                target=self.enemy,
                magnitude=-2,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_CREW_CHANGED,
                target=self.trainee,
                magnitude=-3,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_CREW_CHANGED,
                target=self.trainee,
                magnitude=1,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_DEBUFF_APPLIED,
                target=self.enemy,
                magnitude=2,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_DEBUFF_APPLIED,
                target=self.trainee,
                magnitude=1,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.enemy,
                destroyed=True,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_SHIP_DIED,
                target=self.enemy,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_SHIP_DIED,
                target=self.trainee,
            ),
        )
        start = decision(1, self.trainee, self.enemy, battery=5, speed=8, max_thrust=10)
        end = outcome(1, battery=7, speed=11, max_thrust=10, events=events)

        components = calculate_reward_components(start, [start], [end])

        self.assertEqual(components[REWARD_SPAWN_A1], 1.0)
        self.assertEqual(components[REWARD_SPAWN_A2], 1.0)
        self.assertEqual(components[REWARD_HIGH_SPEED], 1.0)
        self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 2.0)
        self.assertEqual(components[REWARD_LOSE_CREW], 3.0)
        self.assertEqual(components[REWARD_GAIN_CREW], 1.0)
        self.assertEqual(components[REWARD_DEBUFF_ENEMY], 2.0)
        self.assertEqual(components[REWARD_GET_DEBUFFED], 1.0)
        self.assertEqual(components[REWARD_KILL_ENEMY_OBJECT], 1.0)
        self.assertEqual(components[REWARD_KILL_ENEMY], 1.0)
        self.assertEqual(components[REWARD_DIE], 1.0)
        self.assertEqual(components[REWARD_GAIN_BATTERY], 2.0)
        self.assertEqual(components[REWARD_LOSE_BATTERY], 0.0)

    def test_battery_loss_and_zero_are_endpoint_rewards(self):
        start = decision(1, self.trainee, self.enemy, battery=4)
        end = outcome(1, battery=0)

        components = calculate_reward_components(start, [start], [end])

        self.assertEqual(components[REWARD_LOSE_BATTERY], 4.0)
        self.assertEqual(components[REWARD_BATTERY_AT_ZERO], 1.0)

    def test_natural_expiration_does_not_count_as_kill_enemy_object(self):
        event = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_OBJECT_REMOVED,
            owner=self.enemy,
            destroyed=False,
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(event,))],
        )

        self.assertEqual(components[REWARD_KILL_ENEMY_OBJECT], 0.0)


class RollingReturnPipelineTests(unittest.TestCase):
    def setUp(self):
        self.trainee = SimpleNamespace(name="Earthling")
        self.enemy = SimpleNamespace(name="Chenjesu")

    def test_sample_matures_when_prediction_window_is_full(self):
        pipeline = RollingReturnPipeline(
            prediction_window=3,
            reward_weights={REWARD_POINT_A1: 6.0, REWARD_GAIN_BATTERY: 2.0},
        )

        self.assertEqual(
            pipeline.add_frame(
                decision(1, self.trainee, self.enemy, battery=1, a1_pointing=True),
                outcome(1, battery=2),
            ),
            [],
        )
        self.assertEqual(
            pipeline.add_frame(
                decision(2, self.trainee, self.enemy, battery=2),
                outcome(2, battery=3),
            ),
            [],
        )
        matured = pipeline.add_frame(
            decision(3, self.trainee, self.enemy, battery=3),
            outcome(3, battery=4),
        )

        self.assertEqual(len(matured), 1)
        sample = matured[0]
        self.assertEqual(sample.observation, (1.0, 1.0))
        self.assertEqual(sample.action_index, 3)
        self.assertEqual(sample.actual_frame_count, 3)
        self.assertFalse(sample.terminal_truncated)
        self.assertAlmostEqual(sample.component_values[REWARD_POINT_A1], 1 / 3)
        self.assertEqual(sample.component_values[REWARD_GAIN_BATTERY], 3.0)
        self.assertAlmostEqual(sample.return_value, 8.0)

    def test_terminal_frame_flushes_every_pending_window_and_truncates_return(self):
        pipeline = RollingReturnPipeline(
            prediction_window=5,
            reward_weights={REWARD_POINT_A1: 10.0},
        )

        pipeline.add_frame(
            decision(1, self.trainee, self.enemy, a1_pointing=True),
            outcome(1),
        )
        pipeline.add_frame(
            decision(2, self.trainee, self.enemy, a1_pointing=False),
            outcome(2),
        )
        matured = pipeline.add_frame(
            decision(3, self.trainee, self.enemy, a1_pointing=True),
            outcome(3, terminal=True),
        )

        self.assertEqual([sample.start_frame_id for sample in matured], [1, 2, 3])
        self.assertEqual([sample.actual_frame_count for sample in matured], [3, 2, 1])
        self.assertTrue(all(sample.terminal_truncated for sample in matured))
        self.assertAlmostEqual(matured[0].return_value, 10.0 * (2 / 3))
        self.assertAlmostEqual(matured[1].return_value, 10.0 * (1 / 2))
        self.assertAlmostEqual(matured[2].return_value, 10.0)

        with self.assertRaises(RuntimeError):
            pipeline.add_frame(
                decision(4, self.trainee, self.enemy),
                outcome(4),
            )


if __name__ == "__main__":
    unittest.main()
