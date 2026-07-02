import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
if pygame.display.get_surface() is None:
    pygame.display.set_mode((1, 1))

import src.const as const
from src.Objects.Ships.Chmmr.A1.ChmmrA1 import ChmmrA1Spark
from src.Objects.Ships.Chmmr.A2.ChmmrA2 import ChmmrA2
from src.Objects.Ships.Chmmr.A3.ChmmrSatellite import ChmmrSatellite
from src.Objects.Ships.Chmmr.A3.ChmmrSatelliteLaser import ChmmrSatelliteLaser
from src.Objects.Ships.ability import SPECIAL_OBJECT_AREA_IMMUNITIES
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ability, create_ship, get_ship_class
from src.Battle.collision_responses import resolve_projectile_projectile_collision
from src.audio import NullAudioService
from src.toroidal import wrapped_delta, wrapped_distance


class ChmmrTests(unittest.TestCase):
    def setUp(self):
        self.chmmr = create_ship(
            "Chmmr",
            1,
            audio_service=NullAudioService(),
        )
        self.target = create_ship(
            "Earthling",
            2,
            audio_service=NullAudioService(),
        )
        self.chmmr.initialize_in_battle([1000.0, 1000.0], 0)
        self.target.initialize_in_battle([1000.0, 700.0], 0)
        self.chmmr.opponent = self.target
        self.target.opponent = self.chmmr

    def test_catalog_and_registry_expose_avatar(self):
        ship = SHIP_DEFINITIONS["Chmmr"]
        tractor = ABILITY_DEFINITIONS["ChmmrA2"]

        self.assertEqual(ship.ship_type, "Avatar")
        self.assertEqual(ship.satellite_count, 3)
        self.assertEqual(ship.satellite_period, 64)
        self.assertEqual(ship.satellite_laser_color, (0, 0, 255))
        self.assertEqual(tractor.base_speed, 12)
        self.assertEqual(tractor.silhouette_count, 5)
        self.assertEqual(
            tractor.silhouette_colors,
            (
                (0, 0, 200, 200),
                (0, 0, 200, 200),
                (0, 0, 200, 150),
                (0, 0, 200, 100),
                (0, 0, 200, 50),
            ),
        )
        self.assertEqual(get_ship_class("Chmmr").__name__, "Chmmr")

    def test_tractor_silhouette_uses_configured_alpha(self):
        sprite = pygame.Surface((1, 1), pygame.SRCALPHA)
        sprite.fill((255, 255, 255, 255))

        silhouette = ChmmrA2._silhouette(
            sprite,
            (0, 0, 200, 73),
            SimpleNamespace(),
        )

        self.assertEqual(silhouette.get_at((0, 0)), (0, 0, 200, 73))

    def test_first_update_spawns_three_evenly_spaced_satellites(self):
        self.chmmr.update()
        satellites = self.chmmr.drain_spawned_objects()

        self.assertEqual(len(satellites), 3)
        self.assertTrue(all(isinstance(satellite, ChmmrSatellite) for satellite in satellites))
        self.assertTrue(
            all(
                math.isclose(
                    wrapped_distance(self.chmmr.position, satellite.position),
                    256,
                )
                for satellite in satellites
            )
        )
        angles = sorted(
            round(
                math.degrees(
                    math.atan2(
                        wrapped_delta(self.chmmr.position, satellite.position)[0],
                        -wrapped_delta(self.chmmr.position, satellite.position)[1],
                    )
                )
                % 360
            )
            for satellite in satellites
        )
        self.assertEqual(angles, [0, 120, 240])
        self.chmmr.update()
        self.assertEqual(self.chmmr.drain_spawned_objects(), [])

    def test_primary_creates_one_world_fixed_spark_and_ignores_own_satellites(self):
        with mock.patch.object(self.chmmr.rng, "uniform", return_value=300):
            plan = self.chmmr.plan_action1()
        beam = plan.spawned_objects[0]
        spark = beam.drain_spawned_objects()[0]
        satellite = ChmmrSatellite(self.chmmr)

        self.assertIsInstance(spark, ChmmrA1Spark)
        self.assertEqual(spark.render_layer, "after_lasers")
        self.assertAlmostEqual(wrapped_distance(beam.start_position, spark.position), 300)
        original_position = spark.position.copy()
        self.chmmr.position = [1500.0, 1500.0]
        self.chmmr.rotation = 90
        self.assertEqual(spark.position, original_position)
        self.assertFalse(beam.should_consider_laser_target(satellite))
        enemy_satellite = ChmmrSatellite(self.chmmr)
        enemy_satellite.parent = self.target
        self.assertTrue(beam.should_consider_laser_target(enemy_satellite))

    def test_primary_clamps_spark_to_an_intercepted_laser_endpoint(self):
        with mock.patch.object(self.chmmr.rng, "uniform", return_value=300):
            beam = self.chmmr.plan_action1().spawned_objects[0]
        spark = beam.drain_spawned_objects()[0]
        full_delta = wrapped_delta(beam.start_position, beam.end_position)
        full_distance = math.hypot(*full_delta)
        beam.end_position = [
            (beam.start_position[0] + full_delta[0] / full_distance * 100)
            % const.ARENA_SIZE,
            (beam.start_position[1] + full_delta[1] / full_distance * 100)
            % const.ARENA_SIZE,
        ]
        beam.intercepted = True

        beam.on_laser_hit(self.target, beam.end_position)

        self.assertAlmostEqual(wrapped_distance(beam.start_position, spark.position), 100)

    def test_primary_cycles_through_configured_laser_colors(self):
        expected = (
            (189, 0, 0),
            (255, 24, 0),
            (255, 140, 0),
            (255, 24, 0),
        )

        beams = [self.chmmr.plan_action1().spawned_objects[0] for _ in expected]

        self.assertEqual(ABILITY_DEFINITIONS["ChmmrA1"].laser_color, expected)
        self.assertEqual(tuple(beam.LASER_COLOR for beam in beams), expected)

    def test_primary_overlay_uses_ship_direction_only_while_beam_exists(self):
        self.chmmr.heading = 4
        self.chmmr.previous_heading = 4
        self.chmmr.rotation = 90
        first_beam = self.chmmr.plan_action1().spawned_objects[0]
        second_beam = self.chmmr.plan_action1().spawned_objects[0]
        expected_index = 4 * const.VIDEO_FPS_MULTIPLIER
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))

        with mock.patch.object(
            pygame.transform,
            "smoothscale_by",
            wraps=pygame.transform.smoothscale_by,
        ) as smoothscale:
            first_beam.draw_foreground(
                screen,
                1.0,
                [-520.0, -520.0],
                interp_t=1.0,
            )

        self.assertIs(smoothscale.call_args.args[0], first_beam.sprites[expected_index])
        self.assertIs(
            first_beam.sprites[expected_index],
            second_beam.sprites[expected_index],
        )

    def test_tractor_adds_mass_scaled_impulse_only_to_visible_inertial_targets(self):
        self.chmmr.rotation = 90
        self.chmmr.heading = 4
        tractor = self.chmmr.plan_action2().spawned_objects[0]
        tractor.update()
        delta = wrapped_delta(self.target.position, tractor.position)
        distance = math.hypot(*delta)
        expected = 12 / self.target.mass
        self.assertAlmostEqual(self.target.accumulated_impulses[0], delta[0] / distance * expected)
        self.assertAlmostEqual(self.target.accumulated_impulses[1], delta[1] / distance * expected)
        first_impulse = self.target.accumulated_impulses.copy()
        tractor.update()
        self.assertEqual(self.target.accumulated_impulses, first_impulse)

        for inertia, cloaked in ((False, False), (True, True)):
            with self.subTest(inertia=inertia, cloaked=cloaked):
                self.target.accumulated_impulses = [0.0, 0.0]
                self.target.inertia = inertia
                self.target.cloaked = cloaked
                self.target.trackable = not cloaked
                tractor = self.chmmr.plan_action2().spawned_objects[0]
                tractor.update()
                self.assertEqual(self.target.accumulated_impulses, [0.0, 0.0])

    def test_satellite_chases_orbit_target_at_capped_speed(self):
        satellite = ChmmrSatellite(self.chmmr)
        satellite.enemy_objects = []
        satellite.opponent = None
        satellite.position = [2000.0, 2000.0]
        start = satellite.position.copy()

        satellite.update()

        self.assertAlmostEqual(wrapped_distance(start, satellite.position), 80)

    def test_satellite_targets_lowest_hp_and_waits_two_frames_between_shots(self):
        satellite = ChmmrSatellite(self.chmmr)
        weak = SimpleNamespace(
            name="WeakProjectile",
            player=2,
            currently_alive=True,
            current_hp=1,
            position=satellite.position.copy(),
            trackable=True,
            physical_collision_capabilities=None,
            laser_target_capabilities=None,
        )
        crew = SimpleNamespace(
            name="SyreenCrew",
            player=2,
            currently_alive=True,
            current_hp=0.5,
            position=satellite.position.copy(),
            trackable=True,
            physical_collision_capabilities=None,
            laser_target_capabilities=None,
        )
        self.target.position = satellite.position.copy()
        self.target.current_hp = 5
        satellite.enemy_objects = [crew, weak]

        satellite.update()
        first = satellite.drain_spawned_objects()
        self.assertEqual(len(first), 1)
        self.assertIsInstance(first[0], ChmmrSatelliteLaser)
        self.assertIs(first[0].target, weak)

        satellite.update()
        self.assertEqual(satellite.drain_spawned_objects(), [])
        satellite.update()
        self.assertEqual(satellite.drain_spawned_objects(), [])
        satellite.update()
        self.assertEqual(len(satellite.drain_spawned_objects()), 1)

    def test_satellite_breaks_equal_hp_ties_by_nearest_target(self):
        satellite = ChmmrSatellite(self.chmmr)
        satellite.opponent = None
        far = SimpleNamespace(
            name="FarProjectile",
            player=2,
            currently_alive=True,
            current_hp=1,
            position=[satellite.position[0] + 200, satellite.position[1]],
            trackable=True,
            physical_collision_capabilities=None,
            laser_target_capabilities=None,
        )
        near = SimpleNamespace(
            name="NearProjectile",
            player=2,
            currently_alive=True,
            current_hp=1,
            position=[satellite.position[0] + 20, satellite.position[1]],
            trackable=True,
            physical_collision_capabilities=None,
            laser_target_capabilities=None,
        )
        satellite.enemy_objects = [far, near]

        satellite.update()

        laser = satellite.drain_spawned_objects()[0]
        self.assertIs(laser.target, near)

    def test_satellite_collision_and_area_immunity_contracts(self):
        satellite = ChmmrSatellite(self.chmmr)
        enemy_projectile = SimpleNamespace(type="projectile", player=2)
        friendly_projectile = SimpleNamespace(type="projectile", player=1)
        enemy_fighter = SimpleNamespace(type="special_object", player=2)

        self.assertTrue(satellite.should_collide_with_projectile_like(enemy_projectile))
        self.assertFalse(satellite.should_collide_with_projectile_like(friendly_projectile))
        self.assertFalse(satellite.should_collide_with_projectile_like(enemy_fighter))
        self.assertEqual(
            satellite.area_damage_capabilities.immune_to_sources,
            SPECIAL_OBJECT_AREA_IMMUNITIES,
        )

    def test_enemy_projectile_damages_satellite_and_is_destroyed(self):
        satellite = ChmmrSatellite(self.chmmr)
        projectile = create_ability("EarthlingA1", self.target)
        satellite.position = [1200.0, 1200.0]
        satellite.previous_position = satellite.position.copy()
        projectile.position = satellite.position.copy()
        projectile.previous_position = projectile.position.copy()
        projectile.velocity = [0.0, 0.0]

        resolve_projectile_projectile_collision(satellite, projectile, [])

        self.assertEqual(satellite.current_hp, 6)
        self.assertFalse(projectile.currently_alive)


if __name__ == "__main__":
    unittest.main()
