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
import src.resources as resources_module
from collision_test_support import CollisionTestCase
from src.Battle import collisions
from src.Battle.collision_geometry import collision_info, objects_overlap
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.Umgah.A1.UmgahA1 import UmgahA1
from src.Objects.Ships.Umgah.A2.UmgahA2 import UmgahA2
from src.resources import AssetManager


class UmgahTests(CollisionTestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.resources = AssetManager()
        self.ship = create_ship("Umgah", 1, resources=self.resources)
        self.ship.initialize_in_battle([500.0, 500.0], 0)

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_catalog_and_directional_animation_assets(self):
        ship = SHIP_DEFINITIONS["Umgah"]
        ability = ABILITY_DEFINITIONS["UmgahA1"]
        assets = self.resources.ability("UmgahA1")

        self.assertEqual((ship.ship_type, ship.cost, ship.max_energy), ("Drone", 7, 30))
        self.assertEqual(ability.excluded_radius, 29)
        self.assertEqual(len(assets.sprites), 3)
        self.assertEqual(len(assets.sprites[0]), const.TOTAL_SPRITE_DIRECTIONS)
        center = assets.sprites[0][0].get_rect().center
        self.assertEqual(assets.masks[0][0].get_at(center), 0)
        self.assertFalse(self.resources._asset_errors)

    def test_a1_exclusion_uses_parent_scale_not_ability_scale(self):
        ship_definition = SHIP_DEFINITIONS["Umgah"]
        ability_definition = ABILITY_DEFINITIONS["UmgahA1"]
        expected_radius = round(
            ability_definition.excluded_radius * ship_definition.sprite_scale
        )
        ability_scaled_radius = round(
            ability_definition.excluded_radius * ability_definition.sprite_scale
        )

        with mock.patch(
            "src.resources._exclude_center_circle",
            wraps=resources_module._exclude_center_circle,
        ) as exclude_circle:
            AssetManager().ability("UmgahA1")

        self.assertNotEqual(expected_radius, ability_scaled_radius)
        self.assertTrue(exclude_circle.call_args_list)
        self.assertTrue(
            all(call.args[1] == expected_radius for call in exclude_circle.call_args_list)
        )

    def test_a1_cycles_frames_resets_regeneration_and_lives_for_collision_frame(self):
        self.ship.current_energy = 0
        self.ship.energy_timer = self.ship.energy_wait
        self.ship.update_timers()
        self.assertEqual(self.ship.current_energy, self.ship.max_energy)

        areas = []
        for _ in range(3):
            self.ship.action1_timer = 0
            areas.append(self.ship.perform_action1())

        self.assertEqual([area.current_frame for area in areas], [0, 1, 2])
        self.assertEqual(self.ship.current_energy, 0)
        self.assertEqual(self.ship.energy_timer, 0)
        self.assertTrue(areas[0].update())
        self.assertFalse(areas[0].update())

    def test_a1_hits_both_teams_but_not_its_parent_or_a_shielded_ship(self):
        area = self.ship.perform_action1()
        enemy = self.make_ship()
        enemy.player = 2
        enemy.position = [500.0, 400.0]
        friendly = self.make_ship()
        friendly.player = 1
        friendly.position = [520.0, 400.0]
        shielded = self.make_ship()
        shielded.player = 2
        shielded.position = [480.0, 400.0]
        shielded.damage_shield_is_active = lambda: True
        starting_hp = shielded.current_hp

        collisions._handle_area_damage(
            [area, self.ship, enemy, friendly, shielded],
            [],
        )

        self.assertEqual(self.ship.current_hp, self.ship.start_hp)
        self.assertEqual(enemy.current_hp, 9)
        self.assertEqual(friendly.current_hp, 9)
        self.assertEqual(shielded.current_hp, starting_hp)

    def test_a1_never_targets_parent_when_rotated_masks_overlap(self):
        starting_hp = self.ship.current_hp

        for frame, heading in ((0, 2), (1, 5), (2, 8)):
            with self.subTest(frame=frame, heading=heading):
                self.ship._a1_animation_frame = frame
                self.ship.action1_timer = 0
                self.ship.heading = heading
                self.ship.rotation = heading * const.TURN_ANGLE
                area = self.ship.perform_action1()
                _, _, overlap = collision_info(area, self.ship)
                self.assertTrue(objects_overlap(area, self.ship, overlap))

                collisions._handle_area_damage([area, self.ship], [])

                self.assertEqual(self.ship.current_hp, starting_hp)

    def test_a1_damages_projectiles_special_objects_and_asteroids(self):
        area = self.ship.perform_action1()
        projectile = self.make_projectile(self.ship)
        projectile.position = [500.0, 400.0]
        special_object = self.make_special_object()
        special_object.position = [520.0, 400.0]
        asteroid = self.make_asteroid([480.0, 400.0])

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions._handle_area_damage(
                [area, projectile, special_object, asteroid],
                [],
            )

        self.assertFalse(projectile.currently_alive)
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(asteroid.currently_alive)

    def test_a1_creates_planet_effect_without_planet_damage(self):
        area = self.ship.perform_action1()
        planet = self.make_planet([500.0, 400.0])
        planet.current_hp = 100
        planet.currently_alive = True
        sentinel = object()
        effects = []

        with mock.patch.object(
            collisions.BattleEffect,
            "from_blast",
            return_value=sentinel,
        ):
            collisions._handle_area_damage([area, planet], effects)

        self.assertEqual(planet.current_hp, 100)
        self.assertEqual(effects, [sentinel])

    def test_a1_blocks_lasers_but_is_not_an_area_damage_target(self):
        area = self.ship.perform_action1()
        laser_parent = self.make_ship()
        laser_parent.player = 2
        laser_parent.position = [300.0, 400.0]
        laser = self.make_laser(laser_parent)
        laser.end_position = [700.0, 400.0]
        other_area = self.make_area_damage([500.0, 400.0], lambda distance: 10)

        self.assertFalse(collisions.AREA_TARGET_REGISTRY.is_eligible(other_area, area))
        with (
            mock.patch.object(collisions.BattleEffect, "from_blast", return_value=object()),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions._handle_laser_collisions(
                [laser],
                [],
                [],
                [],
                [],
                [],
                [],
                area_abilities=[area],
            )

        self.assertTrue(laser.intercepted)
        self.assertLess(area.current_hp, area.start_hp)

    def test_a2_moves_backward_once_and_stops_after_collision_processing(self):
        initial_energy = self.ship.current_energy
        burst = self.ship.perform_action2()

        self.assertIsInstance(burst, UmgahA2)
        self.assertEqual(self.ship.current_energy, initial_energy - 1)
        self.assertEqual(self.ship.energy_timer, 0)
        self.assertAlmostEqual(math.hypot(*self.ship.velocity), const.SPEED_LIMIT)

        self.ship.update()
        self.assertEqual(self.ship.position, [500.0, 650.0])
        collisions.handle_collisions([self.ship])
        self.assertEqual(self.ship.velocity, [0.0, 0.0])

    def test_a2_elastic_collision_moves_other_body_then_stops_at_contact(self):
        other = self.make_ship()
        other.position = [500.0, 600.0]
        other.previous_position = other.position.copy()
        self.ship.perform_action2()
        self.ship.update()

        collisions.handle_collisions([self.ship, other])

        self.assertEqual(self.ship.velocity, [0.0, 0.0])
        self.assertGreater(other.velocity[1], 0)
        self.assertLess(self.ship.position[1], 650.0)

    def test_a2_stops_at_swept_projectile_contact_after_thrust_impulse(self):
        projectile = self.make_projectile(self.make_ship())
        projectile.player = 2
        projectile.position = [500.0, 550.0]
        projectile.previous_position = [500.0, 700.0]
        projectile.velocity = [0.0, -150.0]
        projectile.current_damage = 1
        projectile.on_ship_impact = lambda ship: ship.apply_thrust(96, 24, 180, False)
        self.ship.perform_action2()
        self.ship.update()

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([self.ship, projectile])

        self.assertEqual(self.ship.velocity, [0.0, 0.0])
        self.assertLess(self.ship.position[1], 650.0)


if __name__ == "__main__":
    unittest.main()
