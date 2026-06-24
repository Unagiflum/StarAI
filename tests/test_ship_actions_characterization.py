import os
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle.battle import BattleSimulation
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.action_transaction import ActionOutput, ActionPlan, ActionResult
from src.Objects.Ships.catalog import ABILITIES_DATA, SHIPS_DATA
from src.Objects.Ships.registry import create_ability, create_ship


ORDINARY_SINGLE_ACTIONS = (
    ("Arilou", 1, "src.Objects.Ships.Arilou.Arilou.ArilouA1"),
    ("Earthling", 1, "src.Objects.Ships.Earthling.Earthling.EarthlingA1"),
    ("KzerZa", 1, "src.Objects.Ships.KzerZa.KzerZa.KzerZaA1"),
    ("Mycon", 1, "src.Objects.Ships.Mycon.Mycon.MyconA1"),
    ("Shofixti", 1, "src.Objects.Ships.Shofixti.Shofixti.ShofixtiA1"),
    ("Spathi", 1, "src.Objects.Ships.Spathi.Spathi.SpathiA1"),
    ("Spathi", 2, "src.Objects.Ships.Spathi.Spathi.SpathiA2"),
    ("Supox", 1, "src.Objects.Ships.Supox.Supox.SupoxA1"),
    ("Thraddash", 1, "src.Objects.Ships.Thraddash.Thraddash.ThraddashA1"),
    ("ZoqFot", 1, "src.Objects.Ships.ZoqFot.ZoqFot.ZoqFotA1"),
    ("ZoqFot", 2, "src.Objects.Ships.ZoqFot.ZoqFot.ZoqFotA2"),
)


class ShipActionCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    @staticmethod
    def action_values(ship, action_number):
        return (
            getattr(ship, f"a{action_number}_cost"),
            int(getattr(ship, f"a{action_number}_wait") * const.ACTION_WAIT_SCALE),
        )

    def test_action_plan_validation_does_not_commit_until_requested(self):
        ship = create_ship("Earthling", 1)
        initial_energy = ship.current_energy
        launch_sound = mock.Mock()
        ability = SimpleNamespace(launch_sound=launch_sound)
        ship.action_factories = {1: mock.Mock(return_value=ability)}

        plan = ship.plan_action1()

        self.assertIsInstance(plan, ActionPlan)
        self.assertTrue(plan.valid)
        self.assertEqual(plan.spawned_objects, (ability,))
        self.assertEqual(plan.energy_change, -ship.a1_cost)
        self.assertTrue(plan.cooldown_committed)
        self.assertEqual(ship.current_energy, initial_energy)
        self.assertEqual(ship.action1_timer, 0)
        launch_sound.play.assert_not_called()

        result = ship.commit_action(plan)

        self.assertIsInstance(result, ActionResult)
        self.assertTrue(result.valid)
        self.assertEqual(result.spawned_objects, (ability,))
        self.assertEqual(result.cooldown_action, 1)
        self.assertTrue(result.launch_sound_played)
        self.assertEqual(ship.current_energy, initial_energy - ship.a1_cost)
        launch_sound.play.assert_called_once_with()

    def test_invalid_target_plan_rolls_back_every_commit_field(self):
        ship = create_ship("Earthling", 1)
        initial_energy = ship.current_energy
        point_defense = SimpleNamespace(
            launch_sound=mock.Mock(),
            get_shots=mock.Mock(return_value=None),
        )
        with mock.patch(
            "src.Objects.Ships.Earthling.Earthling.EarthlingA2",
            return_value=point_defense,
        ):
            plan = ship.plan_action2()
        result = ship.commit_action(plan)

        self.assertFalse(plan.valid)
        self.assertFalse(result.valid)
        self.assertEqual(result.spawned_objects, ())
        self.assertEqual(result.energy_change, 0)
        self.assertIsNone(result.cooldown_action)
        self.assertEqual(ship.current_energy, initial_energy)
        self.assertEqual(ship.action2_timer, 0)
        point_defense.launch_sound.play.assert_not_called()

    def test_action_result_preserves_multi_object_order_and_one_sound(self):
        ship = create_ship("Pkunk", 1)
        abilities = [SimpleNamespace(launch_sound=mock.Mock()) for _ in range(3)]
        with mock.patch(
            "src.Objects.Ships.Pkunk.Pkunk.PkunkA1",
            side_effect=abilities,
        ):
            plan = ship.plan_action1()
        result = ship.commit_action(plan)

        self.assertEqual(plan.spawned_objects, tuple(abilities))
        self.assertEqual(result.spawned_objects, tuple(abilities))
        self.assertIs(result.output, ActionOutput.MANY)
        self.assertEqual(result.compatibility_value(), abilities)
        self.assertEqual(
            [ability.launch_sound.play.call_count for ability in abilities],
            [1, 0, 0],
        )

    def test_ordinary_single_actions_apply_cost_cooldown_sound_and_result(self):
        for ship_name, action_number, constructor_path in ORDINARY_SINGLE_ACTIONS:
            with self.subTest(ship=ship_name, action=action_number):
                ship = create_ship(ship_name, 1)
                initial_energy = ship.current_energy
                cost, cooldown = self.action_values(ship, action_number)
                launch_sound = mock.Mock()
                ability = SimpleNamespace(launch_sound=launch_sound)
                constructor = mock.Mock(return_value=ability)
                ship.action_factories = {action_number: constructor}
                result = getattr(ship, f"perform_action{action_number}")()

                self.assertIs(result, ability)
                self.assertEqual(ship.current_energy, initial_energy - cost)
                self.assertEqual(getattr(ship, f"action{action_number}_timer"), cooldown)
                constructor.assert_called_once_with(ship)
                launch_sound.play.assert_called_once_with()

    def test_ordinary_single_actions_do_nothing_when_not_ready(self):
        for ship_name, action_number, constructor_path in ORDINARY_SINGLE_ACTIONS:
            with self.subTest(ship=ship_name, action=action_number, reason="cooldown"):
                ship = create_ship(ship_name, 1)
                timer_name = f"action{action_number}_timer"
                setattr(ship, timer_name, 1)
                initial_energy = ship.current_energy
                constructor = mock.Mock()
                ship.action_factories = {action_number: constructor}
                result = getattr(ship, f"perform_action{action_number}")()
                self.assertIsNone(result)
                self.assertEqual(ship.current_energy, initial_energy)
                self.assertEqual(getattr(ship, timer_name), 1)
                constructor.assert_not_called()

            with self.subTest(ship=ship_name, action=action_number, reason="energy"):
                ship = create_ship(ship_name, 1)
                cost, _ = self.action_values(ship, action_number)
                if cost == 0:
                    continue
                ship.current_energy = cost - 1
                constructor = mock.Mock()
                ship.action_factories = {action_number: constructor}
                result = getattr(ship, f"perform_action{action_number}")()
                self.assertIsNone(result)
                self.assertEqual(ship.current_energy, cost - 1)
                constructor.assert_not_called()

    def test_fixed_multi_actions_spawn_in_order_and_play_one_sound(self):
        cases = (
            ("Pkunk", 1, "src.Objects.Ships.Pkunk.Pkunk.PkunkA1", (-90, 0, 90)),
            ("Yehat", 1, "src.Objects.Ships.Yehat.Yehat.YehatA1", "side_offsets"),
            ("KohrAh", 2, "src.Objects.Ships.KohrAh.KohrAh.KohrAhA2", "gas_ring"),
        )
        for ship_name, action_number, constructor_path, expected in cases:
            with self.subTest(ship=ship_name, action=action_number):
                ship = create_ship(ship_name, 1)
                initial_energy = ship.current_energy
                cost, cooldown = self.action_values(ship, action_number)
                count = 2 if expected == "side_offsets" else ship.GAS_COUNT if expected == "gas_ring" else 3
                abilities = [SimpleNamespace(launch_sound=mock.Mock()) for _ in range(count)]

                with mock.patch(constructor_path, side_effect=abilities) as constructor:
                    result = getattr(ship, f"perform_action{action_number}")()

                self.assertEqual(result, abilities)
                self.assertEqual(ship.current_energy, initial_energy - cost)
                self.assertEqual(getattr(ship, f"action{action_number}_timer"), cooldown)
                self.assertEqual(sum(a.launch_sound.play.call_count for a in abilities), 1)
                if expected == "side_offsets":
                    offsets = (-ship.size[0] / 2, ship.size[0] / 2)
                elif expected == "gas_ring":
                    offsets = tuple(index * ship.angle_increment for index in range(ship.GAS_COUNT))
                else:
                    offsets = expected
                self.assertEqual(
                    constructor.call_args_list,
                    [mock.call(ship, offset) for offset in offsets],
                )

    def test_ordinary_action_energy_and_cooldown_are_stable_frame_by_frame(self):
        ship = create_ship("Earthling", 1)
        ability = SimpleNamespace(launch_sound=None)
        ship.action_factories = {1: mock.Mock(return_value=ability)}
        ship.set_control_state("action1", True, frame_id=10)

        observed = []
        for frame_id in range(10, 14):
            spawned = ship.process_controls(frame_id)
            observed.append((ship.current_energy, ship.action1_timer, spawned))

        self.assertEqual(
            observed,
            [
                (9, 20, [ability]),
                (9, 19, []),
                (9, 18, []),
                (9, 17, []),
            ],
        )

    def test_earthling_point_defense_consumes_energy_per_spawned_shot(self):
        ship = create_ship("Earthling", 1)
        shots = [object(), object(), object()]
        launch_sound = mock.Mock()
        point_defense = SimpleNamespace(
            launch_sound=launch_sound,
            get_shots=mock.Mock(return_value=shots),
        )
        ship.current_energy = ship.a2_cost * 3 + 1

        with mock.patch(
            "src.Objects.Ships.Earthling.Earthling.EarthlingA2",
            return_value=point_defense,
        ):
            result = ship.perform_action2()

        self.assertEqual(result, shots)
        self.assertEqual(ship.current_energy, 1)
        self.assertEqual(ship.action2_timer, int(ship.a2_wait * const.ACTION_WAIT_SCALE))
        point_defense.get_shots.assert_called_once_with(3)
        launch_sound.play.assert_called_once_with()

    def test_earthling_point_defense_does_not_commit_when_there_are_no_targets(self):
        ship = create_ship("Earthling", 1)
        initial_energy = ship.current_energy
        point_defense = SimpleNamespace(
            launch_sound=mock.Mock(),
            get_shots=mock.Mock(return_value=None),
        )
        with mock.patch(
            "src.Objects.Ships.Earthling.Earthling.EarthlingA2",
            return_value=point_defense,
        ):
            result = ship.perform_action2()
        self.assertIsNone(result)
        self.assertEqual(ship.current_energy, initial_energy)
        self.assertEqual(ship.action2_timer, 0)
        point_defense.launch_sound.play.assert_not_called()

    def test_arilou_teleport_commits_action_and_moves_after_constructing_effect(self):
        ship = create_ship("Arilou", 1)
        effect = SimpleNamespace(launch_sound=mock.Mock())
        initial_energy = ship.current_energy
        with (
            mock.patch(
                "src.Objects.Ships.Arilou.Arilou.ArilouA2",
                return_value=effect,
            ),
            mock.patch(
                "src.Objects.Ships.Arilou.Arilou.random.randint",
                side_effect=[123, 456],
            ),
        ):
            result = ship.perform_action2()
        self.assertIs(result, effect)
        self.assertEqual(ship.position, [123, 456])
        self.assertEqual(ship.current_energy, initial_energy - ship.a2_cost)
        self.assertEqual(ship.action2_timer, int(ship.a2_wait * const.ACTION_WAIT_SCALE))
        effect.launch_sound.play.assert_called_once_with()

    def test_arilou_laser_fires_forward_without_a_target(self):
        ship = create_ship("Arilou", 1)
        ship.position = [1000, 2000]
        ship.heading = 4

        laser = create_ability("ArilouA1", ship)
        laser.position = ship.position.copy()
        laser.calculate_end_position()

        self.assertAlmostEqual(laser.end_position[0], 1000 + laser.LASER_RANGE)
        self.assertAlmostEqual(laser.end_position[1], 2000)

    def test_arilou_laser_fires_forward_when_target_is_cloaked(self):
        ship = create_ship("Arilou", 1)
        target = create_ship("Ilwrath", 2)
        ship.position = [1000, 2000]
        ship.heading = 8
        ship.opponent = target
        target.position = [1500, 2000]
        target.trackable = False

        laser = create_ability("ArilouA1", ship)
        laser.position = ship.position.copy()
        laser.calculate_end_position()

        self.assertAlmostEqual(laser.end_position[0], 1000)
        self.assertAlmostEqual(laser.end_position[1], 2000 + laser.LASER_RANGE)

    def test_kohr_ah_primary_is_press_only_and_release_stops_live_saws(self):
        ship = create_ship("KohrAh", 1)
        ship.action1_active = True
        saw = ship.perform_action1()
        self.assertIsNotNone(saw)
        saw.stop_and_track = mock.Mock()
        ship.action1_timer = 0
        self.assertIsNone(ship.perform_action1())

        ship.friendly_objects = [saw]
        ship.perform_action1_release()
        saw.stop_and_track.assert_called_once_with()
        self.assertFalse(ship.last_action1_state)

    def test_kohr_ah_primary_replaces_oldest_saw_at_the_limit(self):
        ship = create_ship("KohrAh", 1)
        ship.SAW_COUNT = 1
        old_saw = create_ability("KohrAhA1", ship)
        ship.friendly_objects = [old_saw]
        ship.action1_active = True
        new_saw = ship.perform_action1()
        self.assertFalse(old_saw.currently_alive)
        self.assertEqual(ship.active_projectiles, [new_saw])

    def test_kzer_za_fighters_preserve_formation_crew_cost_and_launch_index(self):
        ship = create_ship("KzerZa", 1)
        ship.current_hp = 3
        fighters = [SimpleNamespace(launch_sound=mock.Mock()) for _ in range(2)]
        with mock.patch(
            "src.Objects.Ships.KzerZa.KzerZa.KzerZaA2", side_effect=fighters
        ) as constructor:
            result = ship.perform_action2()
        self.assertEqual(result, fighters)
        self.assertEqual(constructor.call_args_list, [mock.call(ship, 135, 0), mock.call(ship, 225, 1)])
        self.assertEqual(ship.current_hp, 1)
        self.assertEqual(ship.fighter_launch_count, 2)
        fighters[0].launch_sound.play.assert_called_once_with()
        fighters[1].launch_sound.play.assert_not_called()

    def test_pkunk_secondary_gains_energy_without_paying_action_cost(self):
        ship = create_ship("Pkunk", 1)
        ship.current_energy = 5
        insult = SimpleNamespace(ENERGY_GAIN=2, play_insult=mock.Mock())
        with mock.patch("src.Objects.Ships.Pkunk.Pkunk.PkunkA2", return_value=insult):
            result = ship.perform_action2()
        self.assertIsNone(result)
        self.assertEqual(ship.current_energy, 7)
        self.assertEqual(ship.action2_timer, int(ship.a2_wait * const.ACTION_WAIT_SCALE))
        insult.play_insult.assert_called_once_with()

    def test_ilwrath_cloak_costs_energy_while_uncloak_is_free(self):
        ship = create_ship("Ilwrath", 1)
        initial_energy = ship.current_energy
        cloak_effect = SimpleNamespace(launch_sound=mock.Mock())
        with mock.patch(
            "src.Objects.Ships.Ilwrath.Ilwrath.IlwrathA2", return_value=cloak_effect
        ):
            ship.perform_action2()
        self.assertTrue(ship.cloaked)
        self.assertFalse(ship.trackable)
        self.assertEqual(ship.current_energy, initial_energy - ship.a2_cost)
        cloak_effect.launch_sound.play.assert_called_once_with()

        ship.action2_timer = 0
        uncloak_sound = mock.Mock()
        ship._uncloak_sound = uncloak_sound
        with mock.patch.object(Ability, "sound_enabled", True):
            ship.perform_action2()
        self.assertFalse(ship.cloaked)
        self.assertTrue(ship.trackable)
        self.assertEqual(ship.current_energy, initial_energy - ship.a2_cost)
        uncloak_sound.play.assert_called_once_with()

    def test_ilwrath_primary_faces_opponent_and_uncloaks_before_firing(self):
        ship = create_ship("Ilwrath", 1)
        opponent = create_ship("Earthling", 2)
        ship.position = [0, 0]
        opponent.position = [100, 0]
        ship.opponent = opponent
        ship.cloak()
        ship.fade_timer = ship.FADE_DURATION
        flame = SimpleNamespace(launch_sound=mock.Mock())
        with mock.patch(
            "src.Objects.Ships.Ilwrath.Ilwrath.IlwrathA1", return_value=flame
        ):
            result = ship.perform_action1()
        self.assertIs(result, flame)
        self.assertEqual(ship.heading, 4)
        self.assertFalse(ship.cloaked)
        self.assertTrue(ship.trackable)
        flame.launch_sound.play.assert_called_once_with()

    def test_ilwrath_cloak_fade_renders_a_blended_sprite(self):
        ship = create_ship("Ilwrath", 1)
        ship.cloak()

        sprite = ship.set_sprite()

        self.assertIsInstance(sprite, pygame.Surface)
        self.assertEqual(sprite.get_size(), ship.sprites[ship.heading].get_size())
        self.assertEqual(ship.fade_timer, 1)

    def test_supox_secondary_changes_movement_controls_without_action_cost(self):
        ship = create_ship("Supox", 1)
        ship.action2_active = True
        ship.thrust_active = True
        ship.turn_left_active = True
        ship.input_pressed_frames = {"thrust": 1, "turn_left": 2}
        self.assertFalse(ship.turn_input_enabled())
        self.assertEqual(ship.get_active_thrust_angles(True, True, True), [-90])
        initial_energy = ship.current_energy
        self.assertIsNone(ship.perform_action2())
        self.assertEqual(ship.current_energy, initial_energy)

    def test_supox_lateral_thrust_marker_uses_opaque_ship_width(self):
        ship = create_ship("Supox", 1)
        ship.position = [100, 100]
        ship.rotation = 0

        marker = ship.get_thrust_marker_position(90)
        expected_offset = ((ship.size[0] / 2) + 6) / 2

        self.assertAlmostEqual(marker[0], 100 - expected_offset)
        self.assertAlmostEqual(marker[1], 100)

    def test_shofixti_secondary_marks_self_destruct_only_in_battle(self):
        setup_ship = create_ship("Shofixti", 1)
        setup_ship.shofixti_self_destruct = False
        setup_ship.perform_action2()
        self.assertFalse(setup_ship.shofixti_self_destruct)
        self.assertEqual(setup_ship.current_hp, setup_ship.start_hp)

        battle_ship = create_ship("Shofixti", 1)
        battle_ship.initialize_in_battle([100, 100], 0)
        result = battle_ship.perform_action2()
        self.assertTrue(battle_ship.shofixti_self_destruct)
        self.assertEqual(battle_ship.current_hp, 0)
        self.assertIsNotNone(result)

    def test_druuge_primary_recoils_and_secondary_converts_crew_to_energy(self):
        ship = create_ship("Druuge", 1)
        cannon = SimpleNamespace(
            MAX_RECOIL=96,
            RECOIL_INCREMENT=24,
            launch_sound=mock.Mock(),
        )
        with (
            mock.patch("src.Objects.Ships.Druuge.Druuge.DruugeA1", return_value=cannon),
            mock.patch.object(ship, "apply_thrust") as apply_thrust,
        ):
            self.assertIs(ship.perform_action1(), cannon)
        apply_thrust.assert_called_once_with(96, 24, 180, False)

        ship.action2_timer = 0
        ship.current_energy = 1
        initial_hp = ship.current_hp
        furnace = SimpleNamespace(ENERGY_GAIN=16, launch_sound=mock.Mock())
        with mock.patch("src.Objects.Ships.Druuge.Druuge.DruugeA2", return_value=furnace):
            self.assertIsNone(ship.perform_action2())
        self.assertEqual(ship.current_energy, 17)
        self.assertEqual(ship.current_hp, initial_hp - 1)
        furnace.launch_sound.play.assert_called_once_with()

    def test_special_energy_actions_do_not_commit_at_their_state_limits(self):
        cases = (
            ("Druuge", "current_energy", "max_energy", "src.Objects.Ships.Druuge.Druuge.DruugeA2"),
            ("Pkunk", "current_energy", "max_energy", "src.Objects.Ships.Pkunk.Pkunk.PkunkA2"),
            ("KzerZa", "current_hp", 1, "src.Objects.Ships.KzerZa.KzerZa.KzerZaA2"),
        )
        for ship_name, state_name, limit, constructor_path in cases:
            with self.subTest(ship=ship_name):
                ship = create_ship(ship_name, 1)
                setattr(ship, state_name, getattr(ship, limit) if isinstance(limit, str) else limit)
                initial_energy = ship.current_energy
                with mock.patch(constructor_path) as constructor:
                    result = ship.perform_action2()
                self.assertIsNone(result)
                self.assertEqual(ship.current_energy, initial_energy)
                self.assertEqual(ship.action2_timer, 0)
                constructor.assert_not_called()

    def test_mycon_secondary_heals_to_cap_and_is_blocked_at_full_crew(self):
        ship = create_ship("Mycon", 1)
        ship.current_hp = ship.max_hp - 2
        heal = SimpleNamespace(HP_GAIN=4, launch_sound=mock.Mock())
        with mock.patch("src.Objects.Ships.Mycon.Mycon.MyconA2", return_value=heal) as constructor:
            self.assertIsNone(ship.perform_action2())
            self.assertEqual(ship.current_hp, ship.max_hp)
            self.assertEqual(ship.current_energy, ship.start_energy - ship.a2_cost)
            heal.launch_sound.play.assert_called_once_with()

            ship.action2_timer = 0
            energy_at_cap = ship.current_energy
            self.assertIsNone(ship.perform_action2())
            self.assertEqual(ship.current_energy, energy_at_cap)
        constructor.assert_called_once_with(ship)

    def test_simultaneous_invalid_combination_runs_both_ship_actions(self):
        ship = create_ship("Spathi", 1)
        first = SimpleNamespace(launch_sound=None)
        second = SimpleNamespace(launch_sound=None)
        ship.action_factories = {
            1: mock.Mock(return_value=first),
            2: mock.Mock(return_value=second),
        }
        ship.set_control_state("action1", True, 10)
        ship.set_control_state("action2", True, 10)
        spawned = ship.process_controls(10)
        self.assertEqual(spawned, [first, second])

    def test_valid_combined_action_suppresses_individual_actions(self):
        ship = create_ship("Orz", 1)
        initial_energy = ship.current_energy
        ship.set_control_state("action1", True, 10)
        ship.set_control_state("action2", True, 10)

        spawned = ship.process_controls(10)

        self.assertEqual(spawned, [])
        self.assertEqual(ship.current_energy, initial_energy - ship.a3_cost)
        self.assertEqual(
            ship.action3_timer,
            int(ship.a3_wait * const.ACTION_WAIT_SCALE),
        )
        self.assertEqual(ship.action1_timer, 0)
        self.assertEqual(ship.action2_timer, 0)

    def test_sustained_thraddash_afterburner_repeats_on_input_cadence(self):
        ship = create_ship("Thraddash", 1)
        afterburner = SimpleNamespace(
            REUNK_THRUST=72,
            REUNK_INCREMENT=12,
            launch_sound=mock.Mock(),
        )
        ship.set_control_state("action2", True, 10)

        spawned = []
        with (
            mock.patch(
                "src.Objects.Ships.Thraddash.Thraddash.ThraddashA2",
                return_value=afterburner,
            ) as constructor,
            mock.patch.object(ship, "apply_thrust") as apply_thrust,
        ):
            for frame_id in range(10, 14):
                spawned.append(ship.process_controls(frame_id))

        self.assertEqual(spawned, [[afterburner], [], [], [afterburner]])
        self.assertEqual(ship.current_energy, ship.start_energy - 2 * ship.a2_cost)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(apply_thrust.call_count, 2)
        self.assertEqual(afterburner.launch_sound.play.call_count, 2)

    def test_every_ship_and_cataloged_ability_constructs_headlessly(self):
        ships = {name: create_ship(name, 1) for name in SHIPS_DATA}
        for ability_name, ability_data in ABILITIES_DATA.items():
            with self.subTest(ability=ability_name):
                ability = create_ability(ability_name, ships[ability_data["ship_name"]])
                self.assertEqual(ability.name, ability_name)

    def test_every_ship_type_runs_in_a_headless_battle(self):
        for ship_name in SHIPS_DATA:
            with self.subTest(ship=ship_name):
                simulation = BattleSimulation(
                    None,
                    create_ship(ship_name, 1),
                    create_ship("Earthling", 2),
                    sound_enabled=False,
                )
                state = simulation.step(actions={1: {}, 2: {}})
                self.assertEqual(simulation.frame_id, 1)
                self.assertIn(simulation.player1, state["game_objects"])


if __name__ == "__main__":
    unittest.main()
