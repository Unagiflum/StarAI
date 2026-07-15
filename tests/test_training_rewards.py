import unittest
from types import SimpleNamespace

from src.collision_capabilities import CollisionCapabilities, CollisionRole
from src.training.event_ledger import BattleEventLedger
from src.training.event_ledger import (
    EVENT_ACTION_USED,
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_REMOVED,
    EVENT_OBJECT_HP_CHANGED,
    EVENT_OBJECT_SPAWNED,
    EVENT_SHIP_DIED,
    TrainingBattleEvent,
)
from src.training.rewards import (
    REWARD_A1_RANGE,
    REWARD_A2_RANGE,
    REWARD_BATTERY_AT_ZERO,
    REWARD_DEBUFF_ENEMY,
    REWARD_DIE,
    REWARD_DESTROY_OWN_OBJECT,
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
    discount_cutoff_frames,
    frame_outcome_from_battle_state,
    normalize_reward_weights,
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


def outcome(
    frame_id,
    *,
    battery=10,
    speed=0,
    max_thrust=12,
    sustained_a2_active=False,
    events=(),
    terminal=False,
):
    return RewardFrameOutcome(
        frame_id=frame_id,
        self_battery=battery,
        self_speed=speed,
        self_max_thrust=max_thrust,
        self_sustained_a2_active=sustained_a2_active,
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

    def test_range_reward_labels_and_legacy_weights(self):
        self.assertEqual(REWARD_A1_RANGE, "In A1 range")
        self.assertEqual(REWARD_A2_RANGE, "In A2 range")
        self.assertEqual(REWARD_SPAWN_A1, "Use A1")
        self.assertEqual(REWARD_SPAWN_A2, "Use A2")
        self.assertEqual(REWARD_DESTROY_OWN_OBJECT, "Destroy own object")

        weights = normalize_reward_weights(
            {
                "Get in A1 weapon range": 1.5,
                "Get in A2 weapon range": -2.0,
                "Spawn A1 object": 3.0,
                "Spawn A2 object": 4.0,
            }
        )
        self.assertEqual(weights[REWARD_A1_RANGE], 1.5)
        self.assertEqual(weights[REWARD_A2_RANGE], -2.0)
        self.assertEqual(weights[REWARD_SPAWN_A1], 3.0)
        self.assertEqual(weights[REWARD_SPAWN_A2], 4.0)

    def test_satellite_damage_and_destruction_replace_generic_object_rewards(self):
        trainee = SimpleNamespace(name="Chmmr")
        enemy = SimpleNamespace(name="Chmmr")
        own_satellite = SimpleNamespace(name="ChmmrSatellite")
        enemy_satellite = SimpleNamespace(name="ChmmrSatellite")
        events = (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_HP_CHANGED,
                actor=enemy,
                owner=trainee,
                target=own_satellite,
                magnitude=-3,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_HP_CHANGED,
                actor=trainee,
                owner=enemy,
                target=enemy_satellite,
                magnitude=-4,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=enemy,
                obj=enemy_satellite,
                destroyed=True,
                metadata={"source_owner": trainee, "source_type": "projectile"},
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=trainee,
                obj=own_satellite,
                destroyed=True,
                metadata={"source_owner": enemy, "source_type": "projectile"},
            ),
        )

        components = calculate_reward_components(
            decision(1, trainee, enemy),
            [decision(1, trainee, enemy)],
            [outcome(1, events=events)],
        )

        self.assertEqual(components[REWARD_LOSE_CREW], 1.5)
        self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 2.0)
        self.assertEqual(components[REWARD_KILL_ENEMY], 0.5)
        self.assertAlmostEqual(components[REWARD_DIE], 1.0 / 3.0)
        self.assertEqual(components[REWARD_KILL_ENEMY_OBJECT], 0.0)
        self.assertEqual(components[REWARD_DESTROY_OWN_OBJECT], 0.0)

    def test_satellite_parent_cleanup_has_no_reward(self):
        trainee = SimpleNamespace(name="Chmmr")
        enemy = SimpleNamespace(name="Chmmr")
        satellite = SimpleNamespace(name="ChmmrSatellite")
        cleanup = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_OBJECT_REMOVED,
            owner=trainee,
            obj=satellite,
            destroyed=False,
            removal_reason="parent_cleanup",
        )

        components = calculate_reward_components(
            decision(1, trainee, enemy),
            [decision(1, trainee, enemy)],
            [outcome(1, events=(cleanup,))],
        )

        self.assertEqual(components[REWARD_DIE], 0.0)
        self.assertEqual(components[REWARD_DESTROY_OWN_OBJECT], 0.0)

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
                action="A2",
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
                obj=SimpleNamespace(type="projectile"),
                destroyed=True,
                metadata={
                    "source_owner": self.trainee,
                    "source_type": "projectile",
                },
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
        self.assertAlmostEqual(components[REWARD_HIGH_SPEED], 0.1)
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
        self.assertEqual(components[REWARD_DESTROY_OWN_OBJECT], 0.0)

    def test_use_rewards_count_successful_action_commits(self):
        events = (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_ACTION_USED,
                owner=self.trainee,
                action="A1",
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_ACTION_USED,
                owner=self.trainee,
                action="A2",
            ),
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=events)],
        )

        self.assertEqual(components[REWARD_SPAWN_A1], 1.0)
        self.assertEqual(components[REWARD_SPAWN_A2], 1.0)

    def test_use_rewards_count_laser_area_and_non_object_abilities(self):
        for event in (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_ACTION_USED,
                owner=self.trainee,
                action="A1",
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_SPAWNED,
                owner=self.trainee,
                action="A1",
                ability_name="ChmmrA1",
                obj=SimpleNamespace(name="ChmmrA1", type="laser"),
            ),
        ):
            with self.subTest(action="A1", event=event.event_type):
                start = decision(1, self.trainee, self.enemy)
                components = calculate_reward_components(
                    start,
                    [start],
                    [outcome(1, events=(event,))],
                )
                self.assertEqual(components[REWARD_SPAWN_A1], 1.0)
                self.assertEqual(components[REWARD_SPAWN_A2], 0.0)

        for event in (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_ACTION_USED,
                owner=self.trainee,
                action="A2",
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_SPAWNED,
                owner=self.trainee,
                action="A2",
                ability_name="ZoqFotPikA2",
                obj=SimpleNamespace(name="ZoqFotPikA2", type="area"),
            ),
        ):
            with self.subTest(action="A2", event=event.event_type):
                start = decision(1, self.trainee, self.enemy)
                components = calculate_reward_components(
                    start,
                    [start],
                    [outcome(1, events=(event,))],
                )
                self.assertEqual(components[REWARD_SPAWN_A1], 0.0)
                self.assertEqual(components[REWARD_SPAWN_A2], 1.0)

    def test_orz_turret_turn_does_not_count_but_marine_counts_as_use_a2(self):
        orz = SimpleNamespace(name="Orz")
        start = decision(1, orz, self.enemy)
        turret_turn = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_ACTION_USED,
            owner=orz,
            action="A2",
        )
        marine_launch = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_ACTION_USED,
            owner=orz,
            action="A3",
            ability_name="OrzA3",
        )

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(turret_turn,))],
        )
        self.assertEqual(components[REWARD_SPAWN_A2], 0.0)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(marine_launch,))],
        )
        self.assertEqual(components[REWARD_SPAWN_A2], 1.0)

    def test_ilwrath_and_androsynth_use_a2_counts_active_state_per_frame(self):
        for ship in (
            SimpleNamespace(name="Ilwrath"),
            SimpleNamespace(name="Androsynth"),
        ):
            decisions = [
                decision(1, ship, self.enemy),
                decision(2, ship, self.enemy),
                decision(3, ship, self.enemy),
            ]
            outcomes = [
                outcome(1, sustained_a2_active=True),
                outcome(2, sustained_a2_active=True),
                outcome(3, sustained_a2_active=False),
            ]

            with self.subTest(ship=ship.name):
                components = calculate_reward_components(
                    decisions[0], decisions, outcomes
                )
                self.assertEqual(components[REWARD_SPAWN_A2], 2 / 3)

    def test_ilwrath_and_androsynth_toggle_events_do_not_count_as_use_a2(self):
        for ship in (
            SimpleNamespace(name="Ilwrath"),
            SimpleNamespace(name="Androsynth"),
        ):
            event = TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_ACTION_USED,
                owner=ship,
                action="A2",
            )
            start = decision(1, ship, self.enemy)

            with self.subTest(ship=ship.name):
                components = calculate_reward_components(
                    start,
                    [start],
                    [outcome(1, events=(event,))],
                )
                self.assertEqual(components[REWARD_SPAWN_A2], 0.0)

    def test_sustained_a2_outcome_snapshots_ship_state(self):
        states = (
            (SimpleNamespace(name="Ilwrath", cloaked=True), True),
            (SimpleNamespace(name="Ilwrath", cloaked=False), False),
            (SimpleNamespace(name="Androsynth", form="A2"), True),
            (SimpleNamespace(name="Androsynth", form="Base"), False),
        )

        for ship, expected in states:
            with self.subTest(ship=ship.name, state=ship.__dict__):
                result = frame_outcome_from_battle_state(frame_id=1, self_ship=ship)
                self.assertEqual(result.self_sustained_a2_active, expected)

    def test_battery_loss_and_zero_are_endpoint_rewards(self):
        start = decision(1, self.trainee, self.enemy, battery=4)
        end = outcome(1, battery=0)

        components = calculate_reward_components(start, [start], [end])

        self.assertEqual(components[REWARD_LOSE_BATTERY], 4.0)
        self.assertEqual(components[REWARD_BATTERY_AT_ZERO], 1.0)

    def test_high_speed_reward_is_per_frame_excess_speed_ratio(self):
        decisions = [
            decision(1, self.trainee, self.enemy, speed=15, max_thrust=10),
            decision(2, self.trainee, self.enemy, speed=12, max_thrust=10),
            decision(3, self.trainee, self.enemy, speed=9, max_thrust=10),
        ]
        outcomes = [
            outcome(1, speed=15, max_thrust=10),
            outcome(2, speed=12, max_thrust=10),
            outcome(3, speed=9, max_thrust=10),
        ]

        components = calculate_reward_components(decisions[0], decisions, outcomes)

        self.assertAlmostEqual(components[REWARD_HIGH_SPEED], (0.5 + 0.2) / 3)

    def test_natural_expiration_does_not_count_as_kill_enemy_object(self):
        event = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_OBJECT_REMOVED,
            owner=self.enemy,
            obj=SimpleNamespace(type="projectile"),
            destroyed=False,
            metadata={
                "source_owner": self.trainee,
                "source_type": "projectile",
            },
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(event,))],
        )

        self.assertEqual(components[REWARD_KILL_ENEMY_OBJECT], 0.0)

    def test_enemy_object_kill_requires_trainee_owned_source(self):
        enemy_projectile = SimpleNamespace(type="projectile")
        trainee_projectile = SimpleNamespace(type="projectile")
        events = (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.enemy,
                obj=enemy_projectile,
                destroyed=True,
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.enemy,
                obj=enemy_projectile,
                destroyed=True,
                metadata={
                    "source_owner": self.enemy,
                    "source_type": "projectile",
                },
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.enemy,
                obj=enemy_projectile,
                destroyed=True,
                metadata={
                    "source_owner": self.trainee,
                    "source_type": "area",
                },
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.enemy,
                obj=trainee_projectile,
                destroyed=True,
                metadata={
                    "source_owner": self.trainee,
                    "source_type": "laser",
                },
            ),
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=events)],
        )

        self.assertEqual(components[REWARD_KILL_ENEMY_OBJECT], 1.0)

    def test_destroy_own_object_counts_friendly_weapon_killing_friendly_object(self):
        friendly_projectile = SimpleNamespace(type="projectile")
        friendly_special = SimpleNamespace(type="special_object")
        events = (
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.trainee,
                obj=friendly_projectile,
                destroyed=True,
                metadata={
                    "source_owner": self.trainee,
                    "source_type": "laser",
                },
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.trainee,
                obj=friendly_special,
                destroyed=True,
                metadata={
                    "source_owner": self.trainee,
                    "source_type": "special_object",
                },
            ),
            TrainingBattleEvent(
                frame_id=1,
                event_type=EVENT_OBJECT_REMOVED,
                owner=self.trainee,
                obj=friendly_projectile,
                destroyed=True,
                metadata={
                    "source_owner": self.enemy,
                    "source_type": "projectile",
                },
            ),
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=events)],
        )

        self.assertEqual(components[REWARD_DESTROY_OWN_OBJECT], 2.0)
        self.assertEqual(components[REWARD_KILL_ENEMY_OBJECT], 0.0)

    def test_enemy_planet_crew_loss_receives_no_credit(self):
        planet = SimpleNamespace(
            collision_capabilities=CollisionCapabilities(CollisionRole.PLANET)
        )
        event = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_CREW_CHANGED,
            target=self.enemy,
            obj=planet,
            magnitude=-4,
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(event,))],
        )

        self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 0.0)

    def test_enemy_druuge_a2_crew_loss_receives_no_credit(self):
        event = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_CREW_CHANGED,
            target=self.enemy,
            obj=SimpleNamespace(name="DruugeA2"),
            ability_name="DruugeA2",
            magnitude=-4,
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(event,))],
        )

        self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 0.0)

    def test_enemy_shofixti_a2_crew_loss_receives_no_credit(self):
        event = TrainingBattleEvent(
            frame_id=1,
            event_type=EVENT_CREW_CHANGED,
            target=self.enemy,
            obj=SimpleNamespace(name="ShofixtiA2"),
            ability_name="ShofixtiA2",
            magnitude=-4,
        )
        start = decision(1, self.trainee, self.enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(event,))],
        )

        self.assertEqual(components[REWARD_ENEMY_LOSES_CREW], 0.0)

    def test_self_crew_loss_and_death_rewards_keep_full_credit(self):
        planet = SimpleNamespace(
            collision_capabilities=CollisionCapabilities(CollisionRole.PLANET)
        )
        sources = (
            ("planet", planet, None),
            ("Druuge A2", SimpleNamespace(name="DruugeA2"), "DruugeA2"),
            (
                "Shofixti A2",
                SimpleNamespace(name="ShofixtiA2"),
                "ShofixtiA2",
            ),
        )
        start = decision(1, self.trainee, self.enemy)

        for label, source, ability_name in sources:
            with self.subTest(source=label):
                crew_loss = TrainingBattleEvent(
                    frame_id=1,
                    event_type=EVENT_CREW_CHANGED,
                    target=self.trainee,
                    obj=source,
                    ability_name=ability_name,
                    magnitude=-4,
                )
                self_death = TrainingBattleEvent(
                    frame_id=1,
                    event_type=EVENT_SHIP_DIED,
                    target=self.trainee,
                    obj=source,
                    ability_name=ability_name,
                    metadata={"enemy_death_reward_credit": 0.0},
                )

                components = calculate_reward_components(
                    start,
                    [start],
                    [outcome(1, events=(crew_loss, self_death))],
                )

                self.assertEqual(components[REWARD_LOSE_CREW], 4.0)
                self.assertEqual(components[REWARD_DIE], 1.0)

    def test_enemy_death_uses_configured_source_weighted_crew_loss_credit(self):
        ledger = BattleEventLedger()
        planet = SimpleNamespace(
            collision_capabilities=CollisionCapabilities(CollisionRole.PLANET)
        )
        druuge_a2 = SimpleNamespace(name="DruugeA2")
        shofixti_a2 = SimpleNamespace(name="ShofixtiA2")
        weapon = SimpleNamespace(name="EarthlingA1")
        enemy = SimpleNamespace(name="Druuge", current_hp=10)

        enemy.current_hp = 8
        ledger.record_crew_changed(enemy, -2, source=planet)
        enemy.current_hp = 6
        ledger.record_crew_changed(enemy, -2, source=druuge_a2)
        enemy.current_hp = 4
        ledger.record_crew_changed(enemy, -2, source=shofixti_a2)
        enemy.current_hp = 0
        ledger.record_crew_changed(enemy, -4, source=weapon)
        death_event = ledger.events[-1]
        start = decision(1, self.trainee, enemy)

        components = calculate_reward_components(
            start,
            [start],
            [outcome(1, events=(death_event,))],
        )

        self.assertEqual(death_event.event_type, EVENT_SHIP_DIED)
        self.assertAlmostEqual(
            components[REWARD_KILL_ENEMY],
            (2 * 0.0 + 2 * 0.0 + 2 * 0.0 + 4 * 1.0) / 10,
        )


class RollingReturnPipelineTests(unittest.TestCase):
    def setUp(self):
        self.trainee = SimpleNamespace(name="Earthling")
        self.enemy = SimpleNamespace(name="Chenjesu")

    def test_sample_matures_when_discount_cutoff_is_reached(self):
        gamma = 0.2
        self.assertEqual(discount_cutoff_frames(gamma), 3)
        pipeline = RollingReturnPipeline(
            gamma=gamma,
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
        discount_sum = 1.0 + gamma + gamma**2
        self.assertAlmostEqual(
            sample.component_values[REWARD_POINT_A1],
            1.0 / discount_sum,
        )
        self.assertEqual(sample.component_values[REWARD_GAIN_BATTERY], 1.0)
        self.assertAlmostEqual(sample.return_value, 6.0 / discount_sum + 2.0)

    def test_terminal_frame_flushes_every_pending_window_and_truncates_return(self):
        gamma = 0.5
        pipeline = RollingReturnPipeline(
            gamma=gamma,
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
        self.assertAlmostEqual(
            matured[0].return_value,
            10.0 * ((1.0 + gamma**2) / (1.0 + gamma + gamma**2)),
        )
        self.assertAlmostEqual(
            matured[1].return_value,
            10.0 * (gamma / (1.0 + gamma)),
        )
        self.assertAlmostEqual(matured[2].return_value, 10.0)

        with self.assertRaises(RuntimeError):
            pipeline.add_frame(
                decision(4, self.trainee, self.enemy),
                outcome(4),
            )

    def test_explicit_flush_matures_pending_windows_without_new_frame(self):
        gamma = 0.5
        pipeline = RollingReturnPipeline(
            gamma=gamma,
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

        matured = pipeline.flush_pending(end_frame_id=99)

        self.assertEqual([sample.start_frame_id for sample in matured], [1, 2])
        self.assertEqual([sample.end_frame_id for sample in matured], [99, 99])
        self.assertEqual([sample.actual_frame_count for sample in matured], [2, 1])
        self.assertTrue(all(sample.terminal_truncated for sample in matured))
        self.assertAlmostEqual(matured[0].return_value, 10.0 / (1.0 + gamma))
        self.assertAlmostEqual(matured[1].return_value, 0.0)
        self.assertEqual(pipeline.pending_count, 0)
        with self.assertRaises(RuntimeError):
            pipeline.add_frame(
                decision(3, self.trainee, self.enemy),
                outcome(3),
            )

    def test_ship_death_rewards_are_discounted_sums_not_window_averages(self):
        gamma = 0.5
        pipeline = RollingReturnPipeline(
            gamma=gamma,
            reward_weights={REWARD_KILL_ENEMY: 1.0, REWARD_DIE: -1.0},
        )
        death_events = (
            TrainingBattleEvent(
                frame_id=2,
                event_type=EVENT_SHIP_DIED,
                target=self.enemy,
            ),
            TrainingBattleEvent(
                frame_id=2,
                event_type=EVENT_SHIP_DIED,
                target=self.trainee,
            ),
        )

        pipeline.add_frame(
            decision(1, self.trainee, self.enemy),
            outcome(1),
        )
        matured = pipeline.add_frame(
            decision(2, self.trainee, self.enemy),
            outcome(2, events=death_events, terminal=True),
        )

        self.assertEqual([sample.start_frame_id for sample in matured], [1, 2])
        self.assertAlmostEqual(
            matured[0].component_values[REWARD_KILL_ENEMY],
            gamma,
        )
        self.assertAlmostEqual(matured[1].component_values[REWARD_KILL_ENEMY], 1.0)
        self.assertAlmostEqual(matured[0].component_values[REWARD_DIE], gamma)
        self.assertAlmostEqual(matured[1].component_values[REWARD_DIE], 1.0)
        self.assertAlmostEqual(matured[0].return_value, 0.0)
        self.assertAlmostEqual(matured[1].return_value, 0.0)

    def test_reincarnating_pkunk_death_counts_as_normal_kill_event(self):
        gamma = 0.5
        pipeline = RollingReturnPipeline(
            gamma=gamma,
            reward_weights={REWARD_KILL_ENEMY: 1.0},
        )
        pkunk = SimpleNamespace(name="Pkunk")
        death_event = TrainingBattleEvent(
            frame_id=2,
            event_type=EVENT_SHIP_DIED,
            target=pkunk,
        )

        matured = []
        for frame_id in range(1, discount_cutoff_frames(gamma) + 1):
            events = (death_event,) if frame_id == 2 else ()
            matured.extend(
                pipeline.add_frame(
                    decision(frame_id, self.trainee, pkunk),
                    outcome(frame_id, events=events, terminal=False),
                )
            )

        self.assertEqual(len(matured), 1)
        self.assertEqual(matured[0].start_frame_id, 1)
        self.assertAlmostEqual(
            matured[0].component_values[REWARD_KILL_ENEMY],
            gamma,
        )
        self.assertFalse(matured[0].terminal_truncated)

    def test_constant_per_frame_rewards_are_match_length_invariant(self):
        gamma = 0.2
        full_pipeline = RollingReturnPipeline(
            gamma=gamma,
            reward_weights={REWARD_POINT_A1: 10.0},
        )
        truncated_pipeline = RollingReturnPipeline(
            gamma=gamma,
            reward_weights={REWARD_POINT_A1: 10.0},
        )

        for frame_id in range(1, 4):
            full_matured = full_pipeline.add_frame(
                decision(frame_id, self.trainee, self.enemy, a1_pointing=True),
                outcome(frame_id),
            )
        truncated_matured = truncated_pipeline.add_frame(
            decision(1, self.trainee, self.enemy, a1_pointing=True),
            outcome(1, terminal=True),
        )

        self.assertEqual(len(full_matured), 1)
        self.assertEqual(len(truncated_matured), 1)
        self.assertAlmostEqual(full_matured[0].return_value, 10.0)
        self.assertAlmostEqual(truncated_matured[0].return_value, 10.0)


if __name__ == "__main__":
    unittest.main()
