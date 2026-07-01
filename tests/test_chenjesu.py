import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle import collision_responses, collisions
from src.Objects.Ships.Chenjesu.A1.ChenjesuA1 import ChenjesuA1Shard
from src.Objects.Ships.Chenjesu.A2.ChenjesuA2 import ChenjesuA2
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import (
    create_ability,
    create_ship,
    get_ability_class,
    get_ship_class,
)
from src.audio import NullAudioService, RecordingAudioService


class ChenjesuTests(unittest.TestCase):
    def make_ship(self, player=1, audio=None):
        ship = create_ship(
            "Chenjesu",
            player,
            audio_service=audio or NullAudioService(),
        )
        ship.position = [1000.0, 1000.0]
        ship.previous_position = ship.position.copy()
        ship.velocity = [0.0, 0.0]
        ship.rotation = 0.0
        ship.heading = 0
        return ship

    def test_catalog_and_registry_expose_broodhome(self):
        self.assertEqual(SHIP_DEFINITIONS["Chenjesu"].ship_type, "Broodhome")
        self.assertEqual(ABILITY_DEFINITIONS["ChenjesuA1"].fragment_count, 8)
        self.assertEqual(ABILITY_DEFINITIONS["ChenjesuA2"].mass, 4)
        self.assertEqual(ABILITY_DEFINITIONS["ChenjesuA2"].avoid_angle, 22.5)
        self.assertEqual(get_ship_class("Chenjesu").__name__, "Chenjesu")
        self.assertEqual(get_ability_class("ChenjesuA2").__name__, "ChenjesuA2")

    def test_primary_lives_while_held_and_does_not_repeat(self):
        ship = self.make_ship()
        projectile = ship.perform_action1()
        energy_after_launch = ship.current_energy

        for _ in range(ABILITY_DEFINITIONS["ChenjesuA1"].life_time + 1):
            self.assertTrue(projectile.update())

        self.assertFalse(ship.plan_action1().valid)
        self.assertEqual(ship.current_energy, energy_after_launch)

    def test_primary_release_spawns_evenly_distributed_shards_without_end_animation(self):
        ship = self.make_ship()
        projectile = ship.perform_action1()
        ship.perform_action1_release()
        shards = projectile.drain_spawned_objects()

        self.assertFalse(projectile.currently_alive)
        self.assertEqual(len(shards), 8)
        self.assertTrue(all(isinstance(shard, ChenjesuA1Shard) for shard in shards))
        self.assertTrue(all(not shard.death_animation for shard in shards))
        directions = [
            round(math.degrees(math.atan2(s.velocity[0], -s.velocity[1])) % 360)
            for s in shards
        ]
        self.assertEqual(directions, list(range(0, 360, 45)))

        for _ in range(9):
            self.assertTrue(shards[0].update())
        self.assertFalse(shards[0].update())

    def test_friendly_crystals_ignore_each_other_but_hit_a2(self):
        ship = self.make_ship()
        first = ship.perform_action1()
        first.fragment()
        shard = first.drain_spawned_objects()[0]
        second = type(first)(ship)
        cloud = ChenjesuA2(ship)

        self.assertFalse(shard.should_collide_with_projectile_like(second))
        self.assertFalse(second.should_collide_with_projectile_like(shard))
        self.assertTrue(shard.should_collide_with_projectile_like(cloud))

    def test_a2_obeys_team_count_limit(self):
        ship = self.make_ship()
        ship.current_energy = ship.max_energy
        definition = ABILITY_DEFINITIONS["ChenjesuA2"]
        ship.friendly_objects = [ChenjesuA2(ship) for _ in range(definition.max_count)]

        self.assertFalse(ship.plan_action2().valid)

        ship.friendly_objects[-1].currently_alive = False
        self.assertTrue(ship.plan_action2().valid)

    def test_a2_launch_recoils_parent_according_to_momentum(self):
        ship = self.make_ship()
        ship.rotation = 90.0
        expected_recoil_speed = (
            ABILITY_DEFINITIONS["ChenjesuA2"].mass
            * ABILITY_DEFINITIONS["ChenjesuA2"].speed
            / ship.mass
        )

        plan = ship.plan_action2()
        self.assertEqual(ship.accumulated_impulses, [0.0, 0.0])

        result = ship.commit_action(plan)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(ship.accumulated_impulses[0], expected_recoil_speed)
        self.assertAlmostEqual(ship.accumulated_impulses[1], 0.0)

        ship.update()
        self.assertAlmostEqual(math.hypot(*ship.velocity), expected_recoil_speed)

    def test_a2_pursues_target_and_uses_only_jitter_when_target_is_untrackable(self):
        ship = self.make_ship()
        target = self.make_ship(player=2)
        target.position = [1000.0, 500.0]
        ship.opponent = target
        cloud = ChenjesuA2(ship)
        cloud.position = [1000.0, 1000.0]
        cloud.rng = SimpleNamespace(uniform=lambda start, end: 0.0)

        cloud.update()
        self.assertEqual(cloud.velocity, [0.0, -(cloud.speed + cloud.jitter)])

        target.trackable = False
        cloud.update()
        self.assertEqual(cloud.velocity, [0.0, -cloud.jitter])

    def test_a2_leaves_enemy_front_arc_perpendicularly(self):
        ship = self.make_ship()
        target = self.make_ship(player=2)
        target.position = [1000.0, 1000.0]
        target.rotation = 0.0
        ship.opponent = target
        cloud = ChenjesuA2(ship)
        # Twenty degrees is inside the configured 22.5-degree half-angle.
        cloud.position = [
            target.position[0] + math.sin(math.radians(20)) * 500,
            target.position[1] - math.cos(math.radians(20)) * 500,
        ]
        cloud.rng = SimpleNamespace(uniform=lambda start, end: 0.0)

        base_velocity = cloud._target_velocity(target)
        move_angle = math.degrees(
            math.atan2(base_velocity[0], -base_velocity[1])
        ) % 360

        self.assertAlmostEqual(move_angle, 110.0)

    def test_a2_drains_enemy_energy_and_bounces_without_ship_damage(self):
        ship = self.make_ship()
        target = self.make_ship(player=2)
        target.current_energy = 15
        cloud = ChenjesuA2(ship)
        cloud.position = [100.0, 100.0]
        cloud.previous_position = cloud.position.copy()
        cloud.velocity = [10.0, 0.0]
        target.position = [110.0, 100.0]
        target.previous_position = target.position.copy()
        target.velocity = [0.0, 0.0]
        starting_hp = target.current_hp

        cloud.handle_ship_contact(target)

        self.assertEqual(target.current_energy, 5)
        self.assertEqual(target.current_hp, starting_hp)
        self.assertLess(cloud.velocity[0], 10.0)
        self.assertGreater(target.velocity[0], 0.0)

    def test_a2_takes_projectile_damage_and_kills_projectile(self):
        ship = self.make_ship()
        cloud = ChenjesuA2(ship)
        projectile = ship.perform_action1()
        projectile.current_damage = 2

        self.assertTrue(cloud.handle_projectile_contact(projectile))
        self.assertEqual(cloud.current_hp, 1)
        self.assertFalse(projectile.currently_alive)

    def test_real_a2_destroys_each_fragile_special_object_without_damage(self):
        ship = self.make_ship()
        for ship_name, ability_name in (
            ("KzerZa", "KzerZaA2"),
            ("Vux", "VuxA2"),
            ("Syreen", "SyreenCrew"),
        ):
            with self.subTest(target=ability_name):
                enemy = create_ship(ship_name, 2)
                enemy.initialize_in_battle([900, 900], 0)
                ship.opponent = enemy
                enemy.opponent = ship
                cloud = create_ability("ChenjesuA2", ship)
                target = create_ability(ability_name, enemy)
                cloud.position = [700.0, 500.0]
                target.position = cloud.position.copy()
                cloud.previous_position = cloud.position.copy()
                target.previous_position = target.position.copy()
                game_objects = [cloud, target]

                with mock.patch.object(
                    collision_responses.BattleEffect, "play_boom"
                ):
                    collisions.handle_collisions(game_objects)

                self.assertTrue(cloud.currently_alive)
                self.assertEqual(cloud.current_hp, 3)
                self.assertFalse(target.currently_alive)
                self.assertNotIn(target, game_objects)

    def test_real_a2_bounces_from_same_type_regardless_of_team(self):
        for other_player in (1, 2):
            with self.subTest(other_player=other_player):
                first_parent = self.make_ship(player=1)
                second_parent = self.make_ship(player=other_player)
                first = ChenjesuA2(first_parent)
                second = ChenjesuA2(second_parent)
                first.position = [700.0, 500.0]
                second.position = [710.0, 500.0]
                first.previous_position = first.position.copy()
                second.previous_position = second.position.copy()
                first.velocity = [1.0, 0.0]
                second.velocity = [-1.0, 0.0]

                collisions.handle_collisions([first, second])

                self.assertTrue(first.currently_alive)
                self.assertTrue(second.currently_alive)
                self.assertLess(first.velocity[0], 0.0)
                self.assertGreater(second.velocity[0], 0.0)

    def test_a2_ignores_only_the_two_specified_area_effects(self):
        cloud = ChenjesuA2(self.make_ship())
        sources = {
            name: SimpleNamespace(
                name=name,
                special_object_collision_capabilities=(
                    cloud.special_object_collision_capabilities
                ),
            )
            for name in ("SyreenA2", "SlylandroA2", "OtherArea")
        }

        self.assertFalse(
            collision_responses.special_object_is_area_target(
                sources["SyreenA2"], cloud
            )
        )
        self.assertFalse(
            collision_responses.special_object_is_area_target(
                sources["SlylandroA2"], cloud
            )
        )
        self.assertTrue(
            collision_responses.special_object_is_area_target(
                sources["OtherArea"], cloud
            )
        )

    def test_parent_death_removes_a2_silently(self):
        audio = RecordingAudioService()
        ship = self.make_ship(audio=audio)
        cloud = ChenjesuA2(ship)
        audio.operations.clear()
        ship.current_hp = 0

        self.assertFalse(cloud.update())
        self.assertFalse(cloud.currently_alive)
        self.assertEqual(audio.operations, [])


if __name__ == "__main__":
    unittest.main()
