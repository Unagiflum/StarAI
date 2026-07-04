import os
import random
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from collision_test_support import CollisionTestCase
from src.Battle import collisions
from src.Battle.world import World
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.registry import create_ability, create_ship
from src import const


class CollisionSpatialPipelineTests(CollisionTestCase):
    def run_without_effect_assets(self, objects, **kwargs):
        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                side_effect=lambda *args, **kw: SimpleNamespace(effect=True),
            ) as blast,
            mock.patch.object(collisions.BattleEffect, "play_boom") as boom,
        ):
            collisions.handle_collisions(objects, **kwargs)
        return blast, boom

    def test_wide_laser_uses_spatial_ribbon_query_and_pixel_mask(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100.0, 100.0]
        target = self.make_ship()
        target.player = 2
        target.position = [200.0, 108.0]
        target.previous_position = target.position.copy()
        target.size = [10, 10]
        target.heading = 0
        filled_mask = pygame.mask.Mask((10, 10), fill=True)
        target.masks = [filled_mask] * const.SHIP_DIRECTIONS
        laser = self.make_laser(parent)
        laser.end_position = [300.0, 100.0]
        laser.LASER_WIDTH = 8
        laser.calculate_end_position = mock.Mock()

        metrics = collisions.CollisionMetrics()
        self.run_without_effect_assets(
            [parent, target, laser],
            metrics=metrics,
        )

        self.assertEqual(target.current_hp, 8)
        self.assertGreater(metrics.laser_candidates, 0)

    def test_multisegment_laser_hits_target_on_later_segment(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100.0, 100.0]
        target = self.make_ship()
        target.player = 2
        target.position = [250.0, 300.0]
        target.previous_position = target.position.copy()
        laser = self.make_laser(parent)
        laser.calculate_end_position = mock.Mock()
        laser.collision_segments = lambda: (
            ([100.0, 100.0], [100.0, 300.0]),
            ([100.0, 300.0], [300.0, 300.0]),
        )

        self.run_without_effect_assets([parent, target, laser])

        self.assertEqual(target.current_hp, 8)
        self.assertTrue(laser.intercepted)

    def test_wrapped_laser_segment_hits_across_arena_seam(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [const.ARENA_SIZE - 40.0, 500.0]
        target = self.make_ship()
        target.player = 2
        target.position = [20.0, 500.0]
        target.previous_position = target.position.copy()
        laser = self.make_laser(parent)
        laser.start_position = parent.position.copy()
        laser.end_position = [80.0, 500.0]
        laser.calculate_end_position = mock.Mock()

        self.run_without_effect_assets([parent, target, laser])

        self.assertEqual(target.current_hp, 8)

    def test_spatial_laser_preserves_pass_through_then_blocker_order(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100.0, 100.0]
        pass_through = self.make_special_object()
        pass_through.position = [170.0, 100.0]
        pass_through.laser_target_capabilities = (
            pass_through.laser_target_capabilities.__class__(blocks_lasers=False)
        )
        blocker = self.make_ship()
        blocker.player = 2
        blocker.position = [240.0, 100.0]
        blocker.previous_position = blocker.position.copy()
        far_targets = []
        for index in range(20):
            target = self.make_ship()
            target.player = 2
            target.position = [3000.0 + index * 30, 3000.0]
            target.previous_position = target.position.copy()
            far_targets.append(target)
        laser = self.make_laser(parent)
        laser.calculate_end_position = mock.Mock()

        metrics = collisions.CollisionMetrics()
        self.run_without_effect_assets(
            [parent, blocker, *far_targets, laser, pass_through],
            metrics=metrics,
        )

        self.assertFalse(pass_through.currently_alive)
        self.assertEqual(blocker.current_hp, 8)
        self.assertLess(metrics.laser_candidates, metrics.possible_laser_targets)

    def test_explicit_target_remains_authoritative_with_spatial_blockers(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100.0, 100.0]
        intended = self.make_ship()
        intended.player = 1
        intended.position = [250.0, 100.0]
        intended.previous_position = intended.position.copy()
        blocker = self.make_special_object()
        blocker.player = 2
        blocker.position = [170.0, 100.0]
        laser = self.make_laser(parent, target=intended)
        laser.calculate_end_position = mock.Mock()

        self.run_without_effect_assets([parent, intended, laser, blocker])

        self.assertFalse(blocker.currently_alive)
        self.assertEqual(intended.current_hp, 10)

    def test_bounded_area_query_and_unknown_radius_fallback(self):
        near = self.make_ship()
        near.player = 2
        near.position = [150.0, 100.0]
        near.previous_position = near.position.copy()
        far = self.make_ship()
        far.player = 2
        far.position = [2000.0, 2000.0]
        far.previous_position = far.position.copy()

        bounded = self.make_area_damage(
            [100.0, 100.0],
            lambda distance: 3 if distance <= 100 else 0,
        )
        bounded.player = 1
        bounded.maximum_area_damage_radius = lambda: 100
        bounded_metrics = collisions.CollisionMetrics()
        self.run_without_effect_assets(
            [bounded, near, far],
            metrics=bounded_metrics,
        )
        self.assertEqual(near.current_hp, 7)
        self.assertEqual(far.current_hp, 10)
        self.assertLess(
            bounded_metrics.area_candidates,
            bounded_metrics.possible_area_targets,
        )
        self.assertEqual(bounded_metrics.area_full_scan_fallbacks, 0)

        fallback_near = self.make_ship()
        fallback_near.player = 2
        fallback_near.position = [150.0, 100.0]
        fallback_near.previous_position = fallback_near.position.copy()
        fallback_far = self.make_ship()
        fallback_far.position = [2000.0, 2000.0]
        fallback_far.previous_position = fallback_far.position.copy()
        unknown = self.make_area_damage(
            [100.0, 100.0],
            lambda distance: 2 if distance <= 100 else 0,
        )
        unknown.player = 1
        fallback_metrics = collisions.CollisionMetrics()
        self.run_without_effect_assets(
            [unknown, fallback_near, fallback_far],
            metrics=fallback_metrics,
        )
        self.assertEqual(fallback_near.current_hp, 8)
        self.assertEqual(fallback_metrics.area_full_scan_fallbacks, 1)
        self.assertEqual(
            fallback_metrics.area_candidates,
            fallback_metrics.possible_area_targets,
        )

    def test_bounded_area_requeries_unvisited_targets_after_source_moves(self):
        first = self.make_ship()
        first.player = 2
        first.position = [150.0, 100.0]
        first.previous_position = first.position.copy()
        second = self.make_ship()
        second.player = 2
        second.position = [1050.0, 100.0]
        second.previous_position = second.position.copy()
        source = self.make_area_damage(
            [100.0, 100.0],
            lambda distance: 2 if distance <= 100 else 0,
        )
        source.player = 1
        source.maximum_area_damage_radius = lambda: 100

        def move_after_first_hit(target, damage):
            if target is first:
                source.position = [1000.0, 100.0]

        source.on_area_damage_hit = move_after_first_hit

        self.run_without_effect_assets([source, first, second])

        self.assertEqual(first.current_hp, 8)
        self.assertEqual(second.current_hp, 8)

    def test_spawned_object_is_indexed_and_destroyed_object_is_absent(self):
        target = self.make_ship()
        target.player = 2
        target.position = [500.0, 500.0]
        target.previous_position = target.position.copy()
        parent = self.make_ship()
        parent.player = 1
        projectile = self.make_projectile(parent)
        projectile.position = target.position.copy()
        projectile.previous_position = projectile.position.copy()

        class Spawner:
            def __init__(self, spawned):
                self.spawned = [spawned]

            def update(self):
                return True

            def drain_spawned_objects(self):
                spawned, self.spawned = self.spawned, []
                return spawned

        world = World([Spawner(projectile), target])
        world.update_objects(excluded_objects=(target,))
        self.assertIn(projectile, world.objects)
        self.run_without_effect_assets(world)
        self.assertEqual(target.current_hp, 6)

        untouched = self.make_ship()
        untouched.player = 2
        untouched.position = [700.0, 700.0]
        untouched.previous_position = untouched.position.copy()
        removed = self.make_projectile(parent)
        removed.position = untouched.position.copy()
        removed.previous_position = removed.position.copy()
        removed.update = lambda: False
        second_world = World([removed, untouched])
        second_world.update_objects(excluded_objects=(untouched,))
        self.assertNotIn(removed, second_world.objects)
        self.run_without_effect_assets(second_world)
        self.assertEqual(untouched.current_hp, 10)

    def test_ilwrath_and_mycon_evolution_precedes_spatial_index_build(self):
        original_sound = Ability.sound_enabled
        Ability.sound_enabled = False
        try:
            for ability_name, ship_name in (
                ("IlwrathA1", "Ilwrath"),
                ("MyconA1", "Mycon"),
            ):
                with self.subTest(ability=ability_name):
                    parent = create_ship(ship_name, 1)
                    parent.position = [1000.0, 1000.0]
                    parent.previous_position = parent.position.copy()
                    parent.opponent = None
                    parent.planet = None
                    projectile = create_ability(ability_name, parent)
                    if ability_name == "MyconA1":
                        # Mycon plasma derives its frame directly from its
                        # remaining lifetime, matching UQM plasma_preprocess.
                        projectile.expiration_timer = 130
                    else:
                        projectile.frame_timer = 1
                    self.assertTrue(projectile.update())
                    self.assertEqual(projectile.current_frame, 1)
                    self.assertIs(
                        projectile.get_collision_mask(),
                        projectile.masks[1],
                    )

                    target = self.make_ship()
                    target.player = 2
                    target.current_hp = 20
                    target.position = projectile.position.copy()
                    target.previous_position = target.position.copy()
                    expected_hp = 20 - projectile.current_damage
                    objects = [projectile, target]

                    self.run_without_effect_assets(objects)

                    self.assertEqual(target.current_hp, expected_hp)
                    self.assertNotIn(projectile, objects)
        finally:
            Ability.sound_enabled = original_sound

    def test_randomized_spatial_and_brute_force_results_match(self):
        rng = random.Random(91237)

        def build_scene(specs):
            parents = [self.make_ship(), self.make_ship()]
            parents[0].player = 1
            parents[1].player = 2
            objects = []
            for label, player, position, previous in specs:
                projectile = self.make_projectile(parents[player - 1])
                projectile.label = label
                projectile.player = player
                projectile.position = list(position)
                projectile.previous_position = list(previous)
                objects.append(projectile)
            return objects

        for scene_index in range(20):
            specs = []
            for label in range(14):
                position = [rng.uniform(0, 900), rng.uniform(0, 900)]
                previous = [
                    (position[0] + rng.uniform(-180, 180)) % const.ARENA_SIZE,
                    (position[1] + rng.uniform(-180, 180)) % const.ARENA_SIZE,
                ]
                specs.append((label, 1 + label % 2, position, previous))

            outcomes = []
            for mode in (
                collisions.BROAD_PHASE_BRUTE_FORCE,
                collisions.BROAD_PHASE_SPATIAL,
            ):
                objects = build_scene(specs)
                originals = list(objects)
                blast, boom = self.run_without_effect_assets(
                    objects,
                    broad_phase=mode,
                    shadow_validate=(mode == collisions.BROAD_PHASE_SPATIAL),
                )
                survivors = [
                    obj.label for obj in originals if obj in objects
                ]
                states = [
                    (
                        obj.label,
                        obj.currently_alive,
                        obj.current_hp,
                        tuple(round(value, 8) for value in obj.position),
                    )
                    for obj in originals
                ]
                outcomes.append(
                    (survivors, states, blast.call_count, boom.call_args_list)
                )

            self.assertEqual(
                outcomes[0],
                outcomes[1],
                msg=f"scene {scene_index} diverged",
            )

    def test_shadow_validation_detects_an_omitted_real_collision(self):
        first, second = self.make_projectile_pair()
        with mock.patch.object(
            collisions.ToroidalSpatialIndex,
            "candidates_for",
            return_value=[],
        ):
            with self.assertRaisesRegex(AssertionError, "omitted"):
                self.run_without_effect_assets(
                    [first, second],
                    shadow_validate=True,
                )


if __name__ == "__main__":
    unittest.main()
