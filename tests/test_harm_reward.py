import math
import unittest
from types import SimpleNamespace

from src.training.causal_credit import full_weight_credit
from src.training.event_ledger import (
    DEBUFF_LIMPET,
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_HP_CHANGED,
    EVENT_OBJECT_REMOVED,
    EVENT_REBIRTH_COMPLETED,
    EVENT_SHIP_DIED,
    BattleEventLedger,
    TrainingBattleEvent,
)
from src.training.harm import (
    HARM_DAMAGE_ENEMY,
    HARM_DAMAGE_TAKEN,
    HARM_KILL_ENEMY,
    HARM_OWN_DEATH,
    HarmTracker,
    harm_profile_for_name,
)
from src.training.rewards import (
    REWARD_COMPONENTS,
    RewardDecisionFrame,
    RewardFrameOutcome,
    StagedTrajectoryPipeline,
)

OBSERVATION = (0.0,)


def ship(name, crew, *, limpets=0):
    return SimpleNamespace(name=name, current_hp=crew, limpets_attached=limpets)


def weapon(owner, name="Weapon"):
    return SimpleNamespace(name=name, parent=owner, type="projectile")


def crew_event(target, delta, source=None, **metadata):
    return TrainingBattleEvent(
        frame_id=0,
        event_type=EVENT_CREW_CHANGED,
        owner=target,
        target=target,
        obj=source,
        magnitude=float(delta),
        metadata={"source": source, **metadata} if source is not None else metadata,
    )


def died(target, source=None):
    return TrainingBattleEvent(
        frame_id=0,
        event_type=EVENT_SHIP_DIED,
        owner=target,
        target=target,
        obj=source,
        metadata={"source": source} if source is not None else {},
    )


def limpet(target, source):
    return TrainingBattleEvent(
        frame_id=0,
        event_type=EVENT_DEBUFF_APPLIED,
        owner=target,
        target=target,
        obj=source,
        metadata={"source": source, "debuff_type": DEBUFF_LIMPET},
    )


def satellite_damage(owner, damage, source):
    satellite = SimpleNamespace(name="ChmmrSatellite", parent=owner)
    return TrainingBattleEvent(
        frame_id=0,
        event_type=EVENT_OBJECT_HP_CHANGED,
        owner=owner,
        target=satellite,
        obj=source,
        magnitude=-float(damage),
        metadata={"source": source},
    )


def satellite_destroyed(owner, source):
    satellite = SimpleNamespace(name="ChmmrSatellite", parent=owner)
    return TrainingBattleEvent(
        frame_id=0,
        event_type=EVENT_OBJECT_REMOVED,
        owner=owner,
        obj=satellite,
        destroyed=True,
        metadata={"source": source},
    )


class Fight:
    """Feeds frames of events to a tracker for one trainee and opponent."""

    def __init__(self, trainee, enemy):
        self.trainee = trainee
        self.enemy = enemy
        self.tracker = HarmTracker()
        self.frame = 0

    def frame_of(self, *events):
        self.frame += 1
        decision = RewardDecisionFrame(
            self.frame,
            OBSERVATION,
            0,
            self_ship=self.trainee,
            enemy_ship=self.enemy,
            self_crew=self.trainee.current_hp,
            enemy_crew=self.enemy.current_hp,
            self_limpets=self.trainee.limpets_attached,
            enemy_limpets=self.enemy.limpets_attached,
        )
        for event in events:
            if event.event_type == EVENT_CREW_CHANGED:
                event.target.current_hp += event.magnitude
        return self.tracker.contributions(
            decision,
            RewardFrameOutcome(self.frame, events=tuple(events)),
        )

    def total(self, component):
        return self.tracker.totals[component]


class HarmFormulaTests(unittest.TestCase):
    def test_full_crew_damage_is_half_and_a_kill_is_whole_at_the_ship_scale(self):
        for name, crew in (("Arilou", 6), ("KohrAh", 42)):
            with self.subTest(name=name):
                profile = harm_profile_for_name(name)
                trainee = ship("Earthling", 18)
                enemy = ship(name, crew)
                fight = Fight(trainee, enemy)
                shot = weapon(trainee)
                fight.frame_of(crew_event(enemy, -crew, shot), died(enemy, shot))
                self.assertAlmostEqual(
                    fight.total(HARM_DAMAGE_ENEMY), 0.5 * profile.scale
                )
                self.assertAlmostEqual(fight.total(HARM_KILL_ENEMY), profile.scale)
        self.assertAlmostEqual(harm_profile_for_name("KohrAh").scale, 1.0)
        self.assertAlmostEqual(
            harm_profile_for_name("Arilou").scale, math.sqrt(16 / 30)
        )

    def test_chmmr_with_satellites(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Chmmr", 42)
        fight = Fight(trainee, enemy)
        shot = weapon(trainee)
        fight.frame_of(
            satellite_damage(enemy, 30, shot),
            *(satellite_destroyed(enemy, shot) for _ in range(3)),
        )
        fight.frame_of(crew_event(enemy, -42, shot), died(enemy, shot))

        # (0.5 crew + 0.5 satellites) / 1.5, and (1 ship + 3/3 satellites) / 2.
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 1.0 / 1.5)
        self.assertAlmostEqual(fight.total(HARM_KILL_ENEMY), 1.0)

    def test_vux_limpets_on_a_chmmr_match_the_specification_example(self):
        vux = ship("Vux", 20)
        enemy = ship("Chmmr", 42)
        fight = Fight(vux, enemy)
        shot = weapon(vux)
        fight.frame_of(
            satellite_damage(enemy, 30, shot),
            *(satellite_destroyed(enemy, shot) for _ in range(3)),
            *(limpet(enemy, shot) for _ in range(3)),
        )
        fight.frame_of(crew_event(enemy, -42, shot), died(enemy, shot))

        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 1.4375 / 1.5)
        self.assertAlmostEqual(fight.total(HARM_KILL_ENEMY), 1.0)

    def test_limpet_series_continues_from_limpets_already_attached(self):
        vux = ship("Vux", 20)
        enemy = ship("KohrAh", 42, limpets=2)
        fight = Fight(vux, enemy)
        fight.frame_of(limpet(enemy, weapon(vux)))

        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 0.25 * 0.5**2)

    def test_pkunk_damage_and_deaths_are_halved_and_each_life_counts_afresh(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Pkunk", 8)
        scale = harm_profile_for_name("Pkunk").scale
        fight = Fight(trainee, enemy)
        shot = weapon(trainee)
        fight.frame_of(crew_event(enemy, -8, shot), died(enemy, shot))
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 0.25 * scale)
        self.assertAlmostEqual(fight.total(HARM_KILL_ENEMY), 0.5 * scale)

        enemy.current_hp = 8
        rebirth = TrainingBattleEvent(
            frame_id=0, event_type=EVENT_REBIRTH_COMPLETED, owner=enemy, target=enemy
        )
        fight.frame_of(rebirth)
        fight.frame_of(crew_event(enemy, -8, shot), died(enemy, shot))
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 0.5 * scale)
        self.assertAlmostEqual(fight.total(HARM_KILL_ENEMY), 1.0 * scale)

    def test_trainee_pkunk_is_tracked_per_life(self):
        trainee = ship("Pkunk", 8)
        enemy = ship("Earthling", 18)
        scale = harm_profile_for_name("Pkunk").scale
        fight = Fight(trainee, enemy)
        hit = weapon(enemy)
        fight.frame_of(crew_event(trainee, -8, hit), died(trainee, hit))
        trainee.current_hp = 8
        fight.frame_of(
            TrainingBattleEvent(
                frame_id=0,
                event_type=EVENT_REBIRTH_COMPLETED,
                owner=trainee,
                target=trainee,
            )
        )
        fight.frame_of(crew_event(trainee, -4, hit))

        # The refill is not a gain: 8 + 4 crew lost, halved.
        self.assertAlmostEqual(
            fight.total(HARM_DAMAGE_TAKEN), 0.5 * 0.5 * 12 / 8 * scale
        )
        self.assertAlmostEqual(fight.total(HARM_OWN_DEATH), 0.5 * scale)


class OpponentNewLowTests(unittest.TestCase):
    def test_mycon_example_credits_only_new_lows_the_trainee_causes(self):
        trainee = ship("Earthling", 18)
        mycon = ship("Mycon", 20)
        scale = harm_profile_for_name("Mycon").scale
        fight = Fight(trainee, mycon)
        shot = weapon(trainee)
        planet = SimpleNamespace(name="Planet")
        per_crew = 0.5 * scale / 20

        steps = (
            (crew_event(mycon, -8, shot), 8),
            (crew_event(mycon, 6, weapon(mycon, "MyconHeal")), 0),
            (crew_event(mycon, -10, shot), 4),
            (crew_event(mycon, -3, planet), 0),
            (crew_event(mycon, -5, shot), 5),
        )
        for event, credited in steps:
            found = fight.frame_of(event)
            self.assertAlmostEqual(
                sum(item.amount for item in found), credited * per_crew
            )
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 17 * per_crew)

    def test_ship_that_cannot_gain_crew_is_credited_its_plain_damage(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Spathi", 30)
        fight = Fight(trainee, enemy)
        shot = weapon(trainee)
        for damage in (3, 7, 5):
            fight.frame_of(crew_event(enemy, -damage, shot))
        scale = harm_profile_for_name("Spathi").scale
        self.assertAlmostEqual(
            fight.total(HARM_DAMAGE_ENEMY), 0.5 * scale * 15 / 30
        )

    def test_losses_the_trainee_did_not_cause_earn_nothing(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Druuge", 14)
        fight = Fight(trainee, enemy)
        planet = SimpleNamespace(name="Planet")
        burn = weapon(enemy, "DruugeA2")
        fight.frame_of(crew_event(enemy, -2, planet), crew_event(enemy, -1, burn))
        fight.frame_of(crew_event(enemy, -11, planet), died(enemy, planet))

        self.assertEqual(fight.total(HARM_DAMAGE_ENEMY), 0.0)
        self.assertEqual(fight.total(HARM_KILL_ENEMY), 0.0)

    def test_no_kill_credit_for_druuge_or_shofixti_self_harm(self):
        trainee = ship("Earthling", 18)
        for name, ability in (("Druuge", "DruugeA2"), ("Shofixti", "ShofixtiA2")):
            with self.subTest(name=name):
                enemy = ship(name, 6)
                fight = Fight(trainee, enemy)
                own = weapon(enemy, ability)
                fight.frame_of(crew_event(enemy, -6, own), died(enemy, own))
                self.assertEqual(fight.total(HARM_KILL_ENEMY), 0.0)

    def test_trainee_destroyed_enemy_fighters_count_as_crew_damage(self):
        trainee = ship("Earthling", 18)
        kzer = ship("KzerZa", 42)
        fight = Fight(trainee, kzer)
        fighter = weapon(kzer, "KzerZaA2")
        # Launching moves crew aboard into the fighter without an event.
        kzer.current_hp -= 2
        fight.frame_of(
            crew_event(kzer, -1, weapon(trainee), launched_crew_loss=True,
                       launched_unit=fighter),
        )
        scale = harm_profile_for_name("KzerZa").scale
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_ENEMY), 0.5 * scale / 42)


class TraineeCrewLevelTests(unittest.TestCase):
    def test_syreen_example_nets_to_the_real_crew_change(self):
        syreen = ship("Syreen", 12)
        enemy = ship("Earthling", 18)
        fight = Fight(syreen, enemy)
        pod = weapon(enemy, "SyreenCrew")
        hit = weapon(enemy)
        for delta in (8, -5, 5, 2, -9):
            fight.frame_of(crew_event(syreen, delta, pod if delta > 0 else hit))
        scale = harm_profile_for_name("Syreen").scale

        # Net gain of 1 crew: a credit of 1 crew.
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_TAKEN), -0.5 * scale / 12)

    def test_launched_crew_costs_nothing_until_lost_for_good(self):
        kzer = ship("KzerZa", 42)
        enemy = ship("Earthling", 18)
        fight = Fight(kzer, enemy)
        fighter = weapon(kzer, "KzerZaA2")
        asteroid = SimpleNamespace(name="Asteroid")
        fight.frame_of(
            crew_event(kzer, -1, fighter, launched_crew_escrow_debit=True)
        )
        fight.frame_of(
            crew_event(kzer, 1, fighter, launched_crew_escrow_refund=True)
        )
        self.assertEqual(fight.total(HARM_DAMAGE_TAKEN), 0.0)

        found = fight.frame_of(
            crew_event(kzer, -1, asteroid, launched_crew_loss=True,
                       launched_unit=fighter)
        )
        scale = harm_profile_for_name("KzerZa").scale
        self.assertAlmostEqual(fight.total(HARM_DAMAGE_TAKEN), 0.5 * scale / 42)
        self.assertFalse(found[0].relocate)

    def test_self_inflicted_losses_are_penalised_at_their_origin(self):
        druuge = ship("Druuge", 14)
        enemy = ship("Earthling", 18)
        fight = Fight(druuge, enemy)
        found = fight.frame_of(crew_event(druuge, -1, weapon(druuge, "DruugeA2")))

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].component, HARM_DAMAGE_TAKEN)
        self.assertGreater(found[0].amount, 0.0)
        self.assertTrue(found[0].relocate)

    def test_own_satellites_and_death_count_toward_the_penalties(self):
        chmmr = ship("Chmmr", 42)
        enemy = ship("Earthling", 18)
        fight = Fight(chmmr, enemy)
        hit = weapon(enemy)
        fight.frame_of(
            satellite_damage(chmmr, 30, hit),
            *(satellite_destroyed(chmmr, hit) for _ in range(3)),
        )
        fight.frame_of(crew_event(chmmr, -42, hit), died(chmmr, hit))

        self.assertAlmostEqual(fight.total(HARM_DAMAGE_TAKEN), 1.0 / 1.5)
        self.assertAlmostEqual(fight.total(HARM_OWN_DEATH), 1.0)


class HarmPipelineTests(unittest.TestCase):
    WEIGHTS = {
        HARM_DAMAGE_ENEMY: 1.024,
        HARM_KILL_ENEMY: 1.024,
        HARM_DAMAGE_TAKEN: -0.512,
        HARM_OWN_DEATH: -0.512,
    }

    def _decision(self, frame, trainee, enemy):
        return RewardDecisionFrame(
            frame,
            OBSERVATION,
            0,
            self_ship=trainee,
            enemy_ship=enemy,
            self_crew=trainee.current_hp,
            enemy_crew=enemy.current_hp,
            self_limpets=0,
            enemy_limpets=0,
        )

    def test_per_frame_rewards_sum_to_the_formula_totals(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Spathi", 30)
        shot = weapon(trainee)
        hit = weapon(enemy)
        pipeline = StagedTrajectoryPipeline(gamma=0.99, reward_weights=self.WEIGHTS)
        frames = (
            (crew_event(enemy, -10, shot),),
            (crew_event(trainee, -4, hit),),
            (crew_event(enemy, -20, shot), died(enemy, shot)),
        )
        totals = {component: 0.0 for component in REWARD_COMPONENTS}
        for frame, events in enumerate(frames, start=1):
            decision = self._decision(frame, trainee, enemy)
            for event in events:
                if event.event_type == EVENT_CREW_CHANGED:
                    event.target.current_hp += event.magnitude
            pipeline.add_frame(
                decision, RewardFrameOutcome(frame, events=events)
            )
            for component, value in pipeline.immediate_components_for_frame(
                frame
            ).items():
                totals[component] += value
        spathi = harm_profile_for_name("Spathi").scale
        earthling = harm_profile_for_name("Earthling").scale
        self.assertAlmostEqual(totals[HARM_DAMAGE_ENEMY], 0.5 * spathi)
        self.assertAlmostEqual(totals[HARM_KILL_ENEMY], spathi)
        self.assertAlmostEqual(totals[HARM_DAMAGE_TAKEN], 0.5 * earthling * 4 / 18)
        self.assertEqual(totals["Kill enemy"], 0.0)

    def test_causal_placement_moves_damage_to_its_origin_and_keeps_penalties(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Spathi", 30)
        ledger = BattleEventLedger()
        ledger.start_reward_trajectory(trainee, trajectory_id="trajectory")
        pipeline = StagedTrajectoryPipeline(
            gamma=0.99, reward_weights=self.WEIGHTS, mode="causal"
        )
        shot = weapon(trainee)
        credit = full_weight_credit("trajectory", 1)
        hit = weapon(enemy)

        ledger.begin_decision(trainee, 1, 0, reward_mode="causal")
        pipeline.add_frame(
            self._decision(1, trainee, enemy),
            RewardFrameOutcome(1),
            ledger=ledger,
        )
        ledger.begin_decision(trainee, 2, 0, reward_mode="causal")
        damage = crew_event(enemy, -6, shot, reward_credit=credit)
        taken = crew_event(trainee, -3, hit)
        enemy.current_hp -= 6
        trainee.current_hp -= 3
        pipeline.add_frame(
            self._decision(2, trainee, enemy),
            RewardFrameOutcome(2, events=(damage, taken)),
            ledger=ledger,
        )

        origin = pipeline.immediate_components_for_frame(1)
        effect = pipeline.immediate_components_for_frame(2)
        spathi = harm_profile_for_name("Spathi").scale
        self.assertAlmostEqual(origin[HARM_DAMAGE_ENEMY], 0.5 * spathi * 6 / 30)
        self.assertAlmostEqual(effect[HARM_DAMAGE_ENEMY], 0.0)
        self.assertGreater(effect[HARM_DAMAGE_TAKEN], 0.0)
        self.assertEqual(origin[HARM_DAMAGE_TAKEN], 0.0)

    def test_tracking_outlasts_each_trainee_life_trajectory(self):
        trainee = ship("Earthling", 18)
        enemy = ship("Mycon", 20)
        ledger = BattleEventLedger()
        shot = weapon(trainee)
        first = StagedTrajectoryPipeline(gamma=0.99, reward_weights=self.WEIGHTS)
        decision = self._decision(1, trainee, enemy)
        enemy.current_hp -= 8
        first.add_frame(
            decision,
            RewardFrameOutcome(1, events=(crew_event(enemy, -8, shot),)),
            ledger=ledger,
        )
        # Regeneration, then a new trajectory: the low-water mark carries over.
        second = StagedTrajectoryPipeline(gamma=0.99, reward_weights=self.WEIGHTS)
        decision = self._decision(2, trainee, enemy)
        enemy.current_hp += 6 - 10
        second.add_frame(
            decision,
            RewardFrameOutcome(
                2,
                events=(
                    crew_event(enemy, 6, weapon(enemy, "MyconHeal")),
                    crew_event(enemy, -10, shot),
                ),
            ),
            ledger=ledger,
        )
        scale = harm_profile_for_name("Mycon").scale
        self.assertAlmostEqual(
            second.immediate_components_for_frame(2)[HARM_DAMAGE_ENEMY],
            0.5 * scale * 4 / 20,
        )

    def test_classic_weights_leave_the_harm_tracker_off(self):
        pipeline = StagedTrajectoryPipeline(
            gamma=0.99, reward_weights={"Kill enemy": 1.024}
        )
        self.assertFalse(pipeline._harm_enabled)


if __name__ == "__main__":
    unittest.main()
