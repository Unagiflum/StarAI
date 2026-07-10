import unittest
from types import SimpleNamespace

import src.const as const
from src.Battle.world import World
from src.Objects.Ships.action_transaction import ActionPlan
from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Pkunk.Pkunk import Pkunk
from src.training.combat_adapters import (
    is_enemy_in_effective_range,
    is_pointing_at_enemy,
)
from src.training.event_ledger import (
    BattleEventLedger,
    DEBUFF_CONFUSION,
    EVENT_BATTERY_CHANGED,
    EVENT_CREW_CHANGED,
    EVENT_DEBUFF_APPLIED,
    EVENT_OBJECT_REMOVED,
    EVENT_OBJECT_SPAWNED,
    EVENT_REBIRTH_ATTEMPT,
    EVENT_REBIRTH_COMPLETED,
    bind_ledger,
)


def ship(name, *, position=(1000, 1000), rotation=0, velocity=(0, 0), trackable=True):
    return SimpleNamespace(
        name=name,
        position=list(position),
        previous_position=list(position),
        rotation=rotation,
        velocity=list(velocity),
        size=[40, 40],
        currently_alive=True,
        current_hp=10,
        trackable=trackable,
    )


class TrainingCombatAdapterTests(unittest.TestCase):
    def test_conventional_projectile_pointing_uses_quantized_toroidal_bearing(self):
        trainee = ship("Chenjesu", position=(const.ARENA_SIZE - 20, 100), rotation=90)
        enemy = ship("Chenjesu", position=(20, 100))

        self.assertTrue(is_pointing_at_enemy(trainee, enemy, action_number=1))

        trainee.rotation = 180
        self.assertFalse(is_pointing_at_enemy(trainee, enemy, action_number=1))

    def test_projectile_range_accounts_for_lifetime_and_enemy_motion(self):
        trainee = ship("Earthling", position=(100, 100), rotation=0)
        enemy = ship("Chenjesu", position=(100, 700))

        self.assertTrue(is_enemy_in_effective_range(trainee, enemy, action_number=1))

        enemy.position = [100, 4000]
        self.assertFalse(is_enemy_in_effective_range(trainee, enemy, action_number=1))

        enemy.position = [100, 700]
        enemy.velocity = [0, 80]
        self.assertFalse(is_enemy_in_effective_range(trainee, enemy, action_number=1))

    def test_projectile_range_disregards_facing_but_keeps_parent_velocity(self):
        trainee = ship("Druuge", position=(1000, 1000), rotation=0)
        enemy = ship("Chenjesu", position=(1000, 2500))

        self.assertTrue(is_enemy_in_effective_range(trainee, enemy, action_number=1))

        trainee = ship("Orz", position=(1000, 1000), rotation=0, velocity=(0, 100))
        self.assertTrue(is_enemy_in_effective_range(trainee, enemy, action_number=1))

        trainee.velocity = [0, -100]
        self.assertFalse(is_enemy_in_effective_range(trainee, enemy, action_number=1))

    def test_laser_area_and_linked_orz_marine_ranges_are_directionless(self):
        slylandro = ship("Slylandro", position=(100, 100), rotation=180)
        enemy = ship("Chenjesu", position=(100, 1100))

        self.assertTrue(is_enemy_in_effective_range(slylandro, enemy, action_number=1))

        enemy.position = [100, 1400]
        self.assertFalse(is_enemy_in_effective_range(slylandro, enemy, action_number=1))

        zoq = ship("ZoqFotPik", position=(100, 100), rotation=180)
        enemy.position = [100, 230]
        self.assertTrue(is_enemy_in_effective_range(zoq, enemy, action_number=2))

        enemy.position = [100, 260]
        self.assertFalse(is_enemy_in_effective_range(zoq, enemy, action_number=2))

        orz = ship("Orz", position=(100, 100), rotation=0)
        enemy.position = [3000, 3000]
        self.assertTrue(is_enemy_in_effective_range(orz, enemy, action_number=2))

    def test_laser_range_and_cloaked_tracking_fallback_are_characterized(self):
        chmmr = ship("Chmmr", position=(100, 100), rotation=180)
        enemy = ship("Chenjesu", position=(100, 650))

        self.assertTrue(is_pointing_at_enemy(chmmr, enemy, action_number=1))
        self.assertTrue(is_enemy_in_effective_range(chmmr, enemy, action_number=1))

        enemy.position = [100, 900]
        self.assertFalse(is_enemy_in_effective_range(chmmr, enemy, action_number=1))

        arilou = ship("Arilou", position=(100, 100), rotation=180)
        cloaked = ship("Ilwrath", position=(100, 200), trackable=False)
        self.assertFalse(is_pointing_at_enemy(arilou, cloaked, action_number=1))

    def test_unsupported_action_returns_false(self):
        trainee = ship("Supox")
        enemy = ship("Chenjesu")

        self.assertFalse(is_pointing_at_enemy(trainee, enemy, action_number=2))
        self.assertFalse(is_enemy_in_effective_range(trainee, enemy, action_number=2))


class TrainingEventLedgerTests(unittest.TestCase):
    def make_ship(self):
        test_ship = SpaceShip.__new__(SpaceShip)
        test_ship.name = "Test"
        test_ship.player = 1
        test_ship.current_hp = 5
        test_ship.max_hp = 10
        test_ship.current_energy = 4
        test_ship.max_energy = 10
        test_ship.energy_timer = 0
        test_ship._active_damage_shield = None
        return test_ship

    def test_loss_followed_by_healing_records_both_events_in_order(self):
        ledger = BattleEventLedger()
        ledger.current_frame = 7
        test_ship = self.make_ship()
        bind_ledger(test_ship, ledger)

        test_ship.take_damage(2)
        test_ship.commit_action(
            ActionPlan(
                action_number=2,
                valid=True,
                crew_change=1,
                energy_change=0,
            )
        )

        crew_events = [event for event in ledger.events if event.event_type == EVENT_CREW_CHANGED]
        self.assertEqual([event.magnitude for event in crew_events], [-2.0, 1.0])
        self.assertEqual([event.frame_id for event in crew_events], [7, 7])

    def test_launched_marine_or_fighter_transfer_is_not_crew_loss(self):
        ledger = BattleEventLedger()
        test_ship = self.make_ship()
        bind_ledger(test_ship, ledger)

        test_ship.commit_action(
            ActionPlan(
                action_number=3,
                valid=True,
                spawned_objects=(SimpleNamespace(name="OrzA3"),),
                crew_change=-1,
                energy_change=0,
            )
        )

        self.assertEqual(test_ship.current_hp, 4)
        self.assertFalse(
            any(event.event_type == EVENT_CREW_CHANGED for event in ledger.events)
        )

    def test_debuff_refreshes_and_battery_changes_are_typed(self):
        ledger = BattleEventLedger()
        test_ship = self.make_ship()
        bind_ledger(test_ship, ledger)

        test_ship.apply_confused(("frame",), 10)
        test_ship.apply_confused(("frame",), 10)
        test_ship.change_energy(-3)

        debuffs = [event for event in ledger.events if event.event_type == EVENT_DEBUFF_APPLIED]
        battery = [event for event in ledger.events if event.event_type == EVENT_BATTERY_CHANGED]
        self.assertEqual(len(debuffs), 2)
        self.assertEqual([event.metadata["debuff_type"] for event in debuffs], [DEBUFF_CONFUSION] * 2)
        self.assertEqual(battery[0].magnitude, -3.0)

    def test_world_records_spawn_destruction_and_natural_expiration(self):
        ledger = BattleEventLedger()
        world = World([])
        world.set_training_event_ledger(ledger)
        spawned = SimpleNamespace(
            name="EarthlingA1",
            type="projectile",
            parent=None,
            currently_alive=True,
            current_hp=1,
            position=[0, 0],
            previous_position=[0, 0],
            update=lambda: False,
            drain_spawned_objects=lambda: (),
        )

        world.add(spawned)
        world.update_objects()

        self.assertEqual(ledger.events[0].event_type, EVENT_OBJECT_SPAWNED)
        self.assertEqual(ledger.events[1].event_type, EVENT_OBJECT_REMOVED)
        self.assertFalse(ledger.events[1].destroyed)
        self.assertEqual(ledger.events[1].removal_reason, "natural_expiration")

    def test_pkunk_rebirth_events_do_not_record_crew_gain(self):
        ledger = BattleEventLedger()
        pkunk = Pkunk.__new__(Pkunk)
        pkunk.name = "Pkunk"
        pkunk.player = 1
        pkunk.current_hp = 0
        pkunk.max_hp = 4
        pkunk.current_energy = 0
        pkunk.max_energy = 6
        pkunk.energy_timer = 0
        pkunk.currently_alive = False
        pkunk.cloaked = True
        pkunk.trackable = False
        pkunk.can_move = False
        pkunk.can_collide = False
        pkunk.rebirth_count = 0
        pkunk.initial_rebirth_chance = 1.0
        pkunk.rebirth_chance_decay = 1.0
        pkunk.rng = SimpleNamespace(random=lambda: 0.0)
        pkunk.audio_service = None
        pkunk.limpets_attached = 0
        pkunk.sprites = ()
        pkunk.base_sprites = ()
        pkunk.input_pressed_frames = {}
        pkunk.newly_pressed_controls = set()
        pkunk.released_controls = set()
        bind_ledger(pkunk, ledger)

        self.assertTrue(pkunk.attempt_rebirth())
        pkunk.complete_rebirth()

        event_types = [event.event_type for event in ledger.events]
        self.assertIn(EVENT_REBIRTH_ATTEMPT, event_types)
        self.assertIn(EVENT_REBIRTH_COMPLETED, event_types)
        self.assertNotIn(EVENT_CREW_CHANGED, event_types)


if __name__ == "__main__":
    unittest.main()
