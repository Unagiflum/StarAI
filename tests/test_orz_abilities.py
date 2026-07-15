import math
import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle.battle import initialize_new_round_ships
from src.Battle.battle_aftermath import hide_dead_ship
from src.Battle import collision_responses, collisions
from src.Battle.collision_contract import CollisionContext, CollisionEnvironment
from src.Battle.status_bar import draw_boarded_marine_icons
from src.audio import RecordingAudioService
from src.Objects.object import ThrustMarker
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ability, create_ship
from src.Objects.Ships.Orz.A3.OrzA3 import OrzA3
from src.resources import AssetManager
from src.training.event_ledger import (
    BattleEventLedger,
    EVENT_CREW_CHANGED,
    bind_ledger,
)


class OrzAbilityTests(unittest.TestCase):
    def resolve_collision(self, first, second, effects, ships=None):
        context = CollisionContext(
            effects,
            CollisionEnvironment(ships=tuple(ships or ())),
        )
        return collisions.COLLISION_PAIR_REGISTRY.dispatch(first, second, context)

    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.ship = create_ship("Orz", 1)
        self.ship.initialize_in_battle([500, 500], 4)

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_a1_uses_configured_projectile_characteristics(self):
        projectile = create_ability("OrzA1", self.ship)

        self.assertEqual(projectile.current_hp, 2)
        self.assertEqual(projectile.current_damage, 3)
        self.assertEqual(projectile.speed, 120)
        self.assertEqual(
            projectile.expiration_timer,
            12,
        )
        self.assertEqual(
            len(projectile.death_animation),
            6 * const.VIDEO_FPS_MULTIPLIER,
        )
        self.assertTrue(ABILITY_DEFINITIONS["OrzA1"].has_sound)

    def test_marine_survives_parent_explosion_until_cleanup(self):
        opponent = create_ship("Earthling", 2)
        opponent.initialize_in_battle([700, 500], 0)
        self.ship.opponent = opponent
        marine, _ = self.ship.perform_action3()
        self.ship.current_hp = 0
        self.ship.currently_alive = False

        self.assertTrue(marine.update())

        game_objects = [self.ship, marine]
        hide_dead_ship(self.ship, game_objects)
        self.assertFalse(marine.currently_alive)
        self.assertNotIn(marine, game_objects)

    def test_a2_turns_turret_relative_to_hull_and_wraps(self):
        self.assertEqual(self.ship.turret.relative_heading, 0)
        self.assertEqual(self.ship.turret_heading, self.ship.heading)

        self.ship.set_control_state("turn_right", True)
        for _ in range(const.SHIP_DIRECTIONS + 1):
            self.ship.action2_timer = 0
            self.ship.perform_action2()

        self.assertEqual(self.ship.turret.relative_heading, 1)
        self.assertEqual(
            self.ship.turret_heading,
            (self.ship.heading + 1) % const.SHIP_DIRECTIONS,
        )

        self.ship.heading = 7
        self.ship.rotation = self.ship.heading * const.TURN_ANGLE
        self.assertEqual(self.ship.turret_heading, 8)

    def test_a2_requires_one_direction_and_turns_each_way(self):
        self.assertIsNone(self.ship.perform_action2())
        self.assertEqual(self.ship.turret.relative_heading, 0)
        self.assertEqual(self.ship.action2_timer, 0)

        self.ship.set_control_state("turn_left", True)
        self.ship.perform_action2()
        self.assertEqual(
            self.ship.turret.relative_heading,
            const.SHIP_DIRECTIONS - 1,
        )

        self.ship.action2_timer = 0
        self.ship.set_control_state("turn_right", True)
        self.assertIsNone(self.ship.perform_action2())
        self.assertEqual(
            self.ship.turret.relative_heading,
            const.SHIP_DIRECTIONS - 1,
        )

    def test_a2_direction_controls_turret_without_turning_hull(self):
        initial_heading = self.ship.heading
        self.ship.set_control_state("action2", True, 10)
        self.ship.set_control_state("turn_right", True, 10)

        self.ship.process_controls(10)

        self.assertEqual(self.ship.heading, initial_heading)
        self.assertEqual(self.ship.turret.relative_heading, 1)

    def test_a1_fires_along_current_turret_heading_with_parent_velocity(self):
        self.ship.turret.relative_heading = 3
        self.ship.velocity = [5, -7]

        projectile = self.ship.perform_action1()
        expected_heading = (self.ship.heading + 3) % const.SHIP_DIRECTIONS
        angle = math.radians(expected_heading * const.TURN_ANGLE)

        self.assertEqual(projectile.heading, expected_heading)
        self.assertEqual(projectile.rotation, expected_heading * const.TURN_ANGLE)
        self.assertAlmostEqual(
            projectile.velocity[0],
            math.sin(angle) * projectile.speed + 5,
        )
        self.assertAlmostEqual(
            projectile.velocity[1],
            -math.cos(angle) * projectile.speed - 7,
        )

    def test_turret_sprite_is_centered_over_ship_and_not_a_collision_mask(self):
        base = self.ship.sprites[self.ship.heading]
        composite = self.ship.set_sprite()

        self.assertNotEqual(composite.get_size(), base.get_size())
        # self.assertIs(self.ship.get_collision_mask(), self.ship.masks[self.ship.heading])
        self.assertFalse(ABILITY_DEFINITIONS["OrzA2"].has_sound)

    def test_attaching_limpet_invalidates_cached_turret_composites(self):
        cached_composite = self.ship.set_sprite()

        self.ship.attach_limpet()

        self.assertEqual(self.ship._turret_composites, {})
        self.assertIsNot(self.ship.set_sprite(), cached_composite)

    def test_new_battle_initialization_resets_turret_forward(self):
        self.ship.turret.relative_heading = 9
        self.ship.initialize_in_battle([100, 200], 12)

        self.assertEqual(self.ship.turret.relative_heading, 0)
        self.assertEqual(self.ship.turret_heading, 12)

    def test_round_transition_retains_preserved_winners_turret_orientation(self):
        self.ship.turret.relative_heading = 9
        challenger = create_ship("Earthling", 2)

        initialized = initialize_new_round_ships(
            [self.ship, challenger],
            [self.ship],
            None,
        )

        self.assertEqual(initialized, [challenger])
        self.assertEqual(self.ship.turret.relative_heading, 9)

    def test_holding_a2_suppresses_a1_when_a1_is_then_pressed(self):
        self.ship.set_control_state("action2", True, 10)
        self.assertIsNone(self.ship.perform_action1())
        self.assertEqual(self.ship.process_controls(10), [])

        self.ship.set_control_state("action1", True, 11)
        self.assertEqual(self.ship.process_controls(11), [])

        self.assertEqual(self.ship.action1_timer, 0)
        self.assertGreater(self.ship.action3_timer, 0)

    def test_menu_icon_includes_forward_turret(self):
        resources = AssetManager()
        icon = resources.menu_ship_sprite("Orz")
        ship = resources.ship("Orz").sprites[0]

        self.assertEqual(icon.get_size(), ship.get_size())
        self.assertNotEqual(
            pygame.image.tobytes(icon, "RGBA"),
            pygame.image.tobytes(ship, "RGBA"),
        )

    def test_a3_launches_a_three_hp_marine_and_spends_one_crew(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy

        marine, handled = self.ship.perform_action3()

        self.assertTrue(handled)
        self.assertIsInstance(marine, OrzA3)
        self.assertEqual(marine.current_hp, 3)
        self.assertEqual(self.ship.current_hp, self.ship.max_hp - 1)
        self.assertIn(marine, self.ship.active_marines)

    def test_a3_starts_stationary_and_tracks_on_its_first_update(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy

        marine, _ = self.ship.perform_action3()

        self.assertEqual(marine.mode, OrzA3.OUTBOUND)
        self.assertEqual(marine.velocity, [0.0, 0.0])
        marine.update()
        self.assertAlmostEqual(
            math.hypot(*marine.velocity),
            marine.thrust_increment,
        )

    def test_a3_boards_after_immediate_crew_kill_and_registers_hud_icon(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.position = enemy.position.copy()
        marine.previous_position = marine.position.copy()

        with mock.patch.object(collision_responses.BattleEffect, "play_boom"):
            handled = self.resolve_collision(marine, enemy, [], ships=[marine, enemy])

        self.assertTrue(handled)
        self.assertEqual(enemy.current_hp, enemy.max_hp - 1)
        self.assertTrue(marine.is_boarded)
        self.assertFalse(marine.can_collide)
        self.assertEqual(enemy.boarded_marines, [marine])

    def test_real_a3_destroys_each_fragile_special_object_without_damage(self):
        for ship_name, ability_name in (
            ("KzerZa", "KzerZaA2"),
            ("Vux", "VuxA2"),
            ("Syreen", "SyreenCrew"),
        ):
            with self.subTest(target=ability_name):
                enemy = create_ship(ship_name, 2)
                enemy.initialize_in_battle([900, 900], 0)
                self.ship.opponent = enemy
                enemy.opponent = self.ship
                marine = create_ability("OrzA3", self.ship)
                target = create_ability(ability_name, enemy)
                marine.mode = OrzA3.OUTBOUND
                marine.position = [700.0, 500.0]
                target.position = marine.position.copy()
                marine.previous_position = marine.position.copy()
                target.previous_position = target.position.copy()
                game_objects = [marine, target]

                with mock.patch.object(
                    collision_responses.BattleEffect, "play_boom"
                ):
                    collisions.handle_collisions(game_objects)

                self.assertTrue(marine.currently_alive)
                self.assertEqual(marine.current_hp, 3)
                self.assertFalse(target.currently_alive)
                self.assertNotIn(target, game_objects)

    def test_real_a3_objects_ignore_one_another(self):
        first = create_ability("OrzA3", self.ship)
        second = create_ability("OrzA3", self.ship)
        first.mode = second.mode = OrzA3.OUTBOUND
        first.position = [700.0, 500.0]
        second.position = [708.0, 500.0]
        first.previous_position = first.position.copy()
        second.previous_position = second.position.copy()
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]

        collisions.handle_collisions([first, second])

        self.assertTrue(first.currently_alive)
        self.assertTrue(second.currently_alive)
        self.assertEqual(first.velocity, [1.0, 0.0])
        self.assertEqual(second.velocity, [-1.0, 0.0])

    def test_a3_only_collides_with_its_destination_ship(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        bystander = create_ship("Vux", 2)
        bystander.initialize_in_battle([600, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()

        marine.mode = OrzA3.OUTBOUND
        self.assertTrue(marine.should_collide_with_ship(enemy))
        self.assertFalse(marine.should_collide_with_ship(bystander))
        self.assertFalse(marine.should_collide_with_ship(self.ship))

        marine.mode = OrzA3.RETURNING
        self.assertTrue(marine.should_collide_with_ship(self.ship))
        self.assertFalse(marine.should_collide_with_ship(enemy))
        self.assertFalse(marine.should_collide_with_ship(bystander))

    def test_boarded_a3_uses_original_death_kill_and_no_result_ranges(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)

        marine.rng = mock.Mock()
        marine.rng.randrange.return_value = 144
        marine.boarding_timer = 1
        marine.update()
        self.assertEqual(enemy.current_hp, enemy.max_hp - 1)

        marine.rng.randrange.return_value = 16
        marine.boarding_timer = 1
        marine.update()
        self.assertEqual(enemy.current_hp, enemy.max_hp - 2)

        marine.rng.randrange.return_value = 0
        marine.boarding_timer = 1
        self.assertFalse(marine.update())
        self.assertNotIn(marine, enemy.boarded_marines)

    def test_boarded_a3_action_pauses_during_arilou_teleport(self):
        enemy = create_ship("Arilou", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)
        boarded_hp = enemy.current_hp
        marine.rng = mock.Mock()
        marine.boarding_timer = 1

        with mock.patch.object(enemy.rng, "randint", side_effect=[123, 456]):
            enemy.perform_action2()
        marine.update()

        self.assertEqual(marine.boarding_timer, 1)
        marine.rng.randrange.assert_not_called()
        self.assertEqual(enemy.current_hp, boarded_hp)

    def test_surviving_a3_returns_and_restores_parent_crew(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)
        enemy.current_hp = 0

        marine.update()
        self.assertEqual(marine.mode, OrzA3.RETURNING)
        self.assertTrue(marine.can_collide)
        marine.recover_with_parent()

        self.assertEqual(self.ship.current_hp, self.ship.max_hp)
        self.assertFalse(marine.currently_alive)

    def test_a3_boarding_death_records_parent_crew_loss_once(self):
        ledger = BattleEventLedger()
        bind_ledger(self.ship, ledger)
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)
        marine.rng = mock.Mock()
        marine.rng.randrange.return_value = 0
        marine.boarding_timer = 1

        self.assertFalse(marine.update())
        marine.on_destroyed()

        crew_events = [
            event for event in ledger.events if event.event_type == EVENT_CREW_CHANGED
        ]
        self.assertEqual([event.target for event in crew_events], [self.ship])
        self.assertEqual([event.magnitude for event in crew_events], [-1.0])

    def test_a3_uses_named_launch_alarm_and_death_sounds(self):
        audio = RecordingAudioService()
        ship = create_ship("Orz", 1, audio_service=audio)
        enemy = create_ship("Earthling", 2, audio_service=audio)
        ship.initialize_in_battle([500, 500], 4)
        enemy.initialize_in_battle([700, 500], 0)
        ship.opponent = enemy

        marine, _ = ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)
        marine.rng = mock.Mock()
        marine.rng.randrange.return_value = 0
        marine.boarding_timer = 1
        marine.update()

        played_names = [operation[1].name for operation in audio.operations]
        self.assertEqual(
            played_names,
            ["OrzA3Launch.wav", "OrzA3Alarm.wav", "OrzA3Die.wav"],
        )

    def test_a3_limits_parent_to_eight_active_marines(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy

        for _ in range(OrzA3.MAX_MARINES):
            marine, _ = self.ship.perform_action3()
            self.assertIsInstance(marine, OrzA3)
            self.ship.action3_timer = 0

        ninth, _ = self.ship.perform_action3()
        self.assertIsNone(ninth)
        self.assertEqual(len(self.ship.active_marines), OrzA3.MAX_MARINES)

    def test_status_draws_one_icon_for_each_boarded_marine(self):
        screen = mock.Mock()
        icon = pygame.Surface((12, 12), pygame.SRCALPHA)
        ship = mock.Mock()
        ship.boarded_marines = [
            mock.Mock(currently_alive=True, is_boarded=True, hud_sprite=icon)
            for _ in range(6)
        ]
        ship.returning_marines = []

        draw_boarded_marine_icons(screen, ship, 10, 100, 65)

        self.assertEqual(screen.blit.call_count, 6)
        self.assertTrue(
            all(
                call.args[1][1] == 100 + (20 - call.args[0].get_height()) // 2
                for call in screen.blit.call_args_list
            )
        )

    def test_a3_uses_red_and_green_flight_sprites_and_original_hud_sprite(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND

        marine.handle_ship_contact(enemy)
        self.assertIs(marine.get_sprite(), marine.red_flight_sprite)
        self.assertEqual(marine.get_sprite().get_size(), (16, 16))
        self.assertEqual(marine.get_collision_mask().get_size(), (16, 16))
        self.assertIs(marine.hud_sprite, marine.red_flight_sprite)
        self.assertEqual(marine.hud_sprite.get_size(), (16, 16))

        enemy.current_hp = 0
        marine.update()
        self.assertIs(marine.get_sprite(), marine.green_flight_sprite)
        self.assertIn(marine, self.ship.active_marines)
        self.assertNotIn(marine, enemy.boarded_marines)

    def test_accelerating_a3_emits_ship_thrust_markers(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND

        marine.update()
        markers = marine.drain_spawned_objects()

        self.assertEqual(len(markers), 1)
        self.assertIsInstance(markers[0], ThrustMarker)

    def test_a3_bounces_without_boarding_an_active_shield(self):
        enemy = create_ship("Yehat", 2)
        enemy.initialize_in_battle([700, 500], 0)
        enemy.perform_action2()
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.previous_position = [680, 500]
        marine.position = enemy.position.copy()
        marine.velocity = [10, 0]
        starting_hp = enemy.current_hp

        with mock.patch.object(collision_responses.BattleEffect, "play_boom"):
            handled = self.resolve_collision(marine, enemy, [], ships=[marine, enemy])

        self.assertTrue(handled)
        self.assertEqual(enemy.current_hp, starting_hp)
        self.assertFalse(marine.is_boarded)
        self.assertEqual(enemy.boarded_marines, [])
        self.assertEqual(marine.velocity, [-10, 0])
        self.assertGreater(marine.shield_bounce_timer, 0)

    def test_boarded_a3_kill_plays_zap_sound(self):
        audio = RecordingAudioService()
        ship = create_ship("Orz", 1, audio_service=audio)
        enemy = create_ship("Earthling", 2, audio_service=audio)
        ship.initialize_in_battle([500, 500], 4)
        enemy.initialize_in_battle([700, 500], 0)
        ship.opponent = enemy
        marine, _ = ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)
        marine.rng = mock.Mock()
        marine.rng.randrange.return_value = 16
        marine.boarding_timer = 1

        marine.update()

        played_names = [operation[1].name for operation in audio.operations]
        self.assertEqual(
            played_names,
            ["OrzA3Launch.wav", "OrzA3Alarm.wav", "OrzA3Zap.wav"],
        )

    def test_boarded_a3_is_recorded_as_the_lethal_source(self):
        enemy = create_ship("Earthling", 2)
        enemy.initialize_in_battle([700, 500], 0)
        enemy.current_hp = 2
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)
        marine.rng = mock.Mock()
        marine.rng.randrange.return_value = 16
        marine.boarding_timer = 1

        marine.update()

        self.assertEqual(enemy.current_hp, 0)
        self.assertIs(enemy.last_lethal_damage_source, marine)

    def test_shofixti_self_destruct_kills_boarded_marines(self):
        enemy = create_ship("Shofixti", 2)
        enemy.initialize_in_battle([700, 500], 0)
        self.ship.opponent = enemy
        marine, _ = self.ship.perform_action3()
        marine.mode = OrzA3.OUTBOUND
        marine.handle_ship_contact(enemy)

        enemy.perform_action2()
        enemy.perform_action2()
        enemy.perform_action2()

        self.assertFalse(marine.currently_alive)
        self.assertEqual(enemy.boarded_marines, [])


if __name__ == "__main__":
    unittest.main()
