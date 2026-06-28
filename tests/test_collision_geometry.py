import unittest
from types import SimpleNamespace

import pygame

import src.const as const
from collision_test_support import CollisionTestCase
from src.Battle.collision_geometry import (
    collision_info,
    contact_point,
    estimated_impact,
    laser_hit_info,
    objects_overlap,
    segment_circle_intercept,
    ship_rotation_blocked,
    solid_sweep_overlap,
    swept_impact,
    wrapped_segment,
)
from src.collision_capabilities import SpecialObjectCollisionCapabilities


def body(position, size=(10, 10), *, previous_position=None, mask=None):
    value = SimpleNamespace(
        position=list(position),
        previous_position=list(
            position if previous_position is None else previous_position
        ),
        size=list(size),
        can_move=True,
    )
    value.get_collision_mask = lambda: mask
    return value


class CollisionGeometryTests(CollisionTestCase):
    def test_ship_rotation_is_blocked_by_live_overlapping_candidates(self):
        ship = body([100, 100], size=(20, 20))
        live_opponent = body([115, 100], size=(20, 20))
        live_opponent.current_hp = 1
        ship.opponent = live_opponent
        ship.asteroids = []
        ship.planet = None

        self.assertTrue(ship_rotation_blocked(ship))

    def test_ship_rotation_ignores_dead_and_non_overlapping_candidates(self):
        ship = body([100, 100], size=(20, 20))
        dead_opponent = body([100, 100], size=(20, 20))
        dead_opponent.current_hp = 0
        dead_asteroid = body([100, 100], size=(20, 20))
        dead_asteroid.currently_alive = False
        ship.opponent = dead_opponent
        ship.asteroids = [dead_asteroid]
        ship.planet = body([200, 100], size=(20, 20))

        self.assertFalse(ship_rotation_blocked(ship))

    def test_ship_rotation_is_blocked_by_elastic_special_objects(self):
        ship = body([100, 100], size=(20, 20))
        special_object = body([110, 100], size=(20, 20))
        special_object.current_hp = 1
        special_object.currently_alive = True
        special_object.special_object_collision_capabilities = (
            SpecialObjectCollisionCapabilities(
                bounces_off_ships_without_damage=True
            )
        )
        ship.opponent = None
        ship.asteroids = []
        ship.planet = None
        ship.friendly_objects = [special_object]
        ship.enemy_objects = []

        self.assertTrue(ship_rotation_blocked(ship))

    def test_contact_info_uses_shortest_toroidal_displacement(self):
        left = body([5, 100], size=(8, 8))
        right = body([const.ARENA_SIZE - 5, 100], size=(8, 8))

        normal, distance, overlap = collision_info(
            left, right
        )

        self.assertEqual([1.0, 0.0], normal)
        self.assertEqual(10, distance)
        self.assertEqual(-2, overlap)

    def test_empty_masks_reject_a_circular_broadphase_overlap(self):
        empty_mask = pygame.mask.Mask((10, 10), fill=False)
        first = body([100, 100], mask=empty_mask)
        second = body([100, 100], mask=empty_mask)
        _, _, overlap = collision_info(first, second)

        self.assertFalse(
            objects_overlap(first, second, overlap)
        )

    def test_collision_radius_uses_logical_size_not_padded_mask_canvas(self):
        padded_mask = pygame.mask.Mask((100, 100), fill=False)
        padded_mask.draw(pygame.mask.Mask((20, 10), fill=True), (40, 45))
        first = body([100, 100], size=(20, 10), mask=padded_mask)
        second = body([130, 100], size=(20, 10), mask=padded_mask)

        _, _, overlap = collision_info(first, second)

        self.assertEqual(overlap, -10)

    def test_exact_mask_overlap_keeps_full_canvas_alignment(self):
        large_canvas = pygame.mask.Mask((100, 100), fill=False)
        large_canvas.draw(pygame.mask.Mask((20, 20), fill=True), (40, 40))
        small_canvas = pygame.mask.Mask((40, 40), fill=False)
        small_canvas.draw(pygame.mask.Mask((20, 20), fill=True), (10, 10))
        first = body([100, 100], size=(20, 20), mask=large_canvas)
        second = body([100, 100], size=(20, 20), mask=small_canvas)
        _, _, overlap = collision_info(first, second)

        self.assertTrue(objects_overlap(first, second, overlap))

    def test_contact_point_uses_opaque_boundary_inside_padded_canvas(self):
        padded_mask = pygame.mask.Mask((100, 100), fill=False)
        padded_mask.draw(pygame.mask.Mask((20, 10), fill=True), (40, 45))
        target = body([100, 100], size=(20, 10), mask=padded_mask)

        self.assertEqual(contact_point(target, [0, -1]), [100.0, 95.0])
        self.assertEqual(contact_point(target, [-1, 0]), [90.0, 100.0])

    def test_estimated_impact_uses_actual_mask_overlap(self):
        projectile_mask = pygame.mask.Mask((10, 10), fill=True)
        target_mask = pygame.mask.Mask((100, 100), fill=False)
        target_mask.draw(pygame.mask.Mask((10, 10), fill=True), (20, 20))
        projectile = body([75, 70], size=(10, 10), mask=projectile_mask)
        target = body([100, 100], size=(10, 10), mask=target_mask)

        contact, _ = estimated_impact(projectile, target)
        projectile_local = (
            int(contact[0] - projectile.position[0] + 5),
            int(contact[1] - projectile.position[1] + 5),
        )
        target_local = (
            int(contact[0] - target.position[0] + 50),
            int(contact[1] - target.position[1] + 50),
        )

        self.assertTrue(projectile_mask.get_at(projectile_local))
        self.assertTrue(target_mask.get_at(target_local))

    def test_swept_impact_finds_contact_between_frame_endpoints(self):
        projectile = body([100, 200], previous_position=[0, 200])
        target = body([50, 200])

        contact, normal = swept_impact(
            projectile, target
        )

        self.assertIsNotNone(contact)
        self.assertEqual([-1.0, 0.0], normal)
        self.assertAlmostEqual(45.0, contact[0])
        self.assertAlmostEqual(200.0, contact[1])

    def test_solid_sweep_uses_twelve_pixel_samples(self):
        full_mask = pygame.mask.Mask((10, 10), fill=True)
        moving = body(
            [100, 200],
            previous_position=[0, 200],
            mask=full_mask,
        )
        target = body([37, 200], mask=full_mask)

        self.assertTrue(solid_sweep_overlap(moving, target))
        self.assertLess(moving.position[0], 50)

    def test_segment_circle_intercept_crosses_wrapped_boundary(self):
        start = [const.ARENA_SIZE - 10, 300]
        _, end = wrapped_segment(start, [10, 300])

        contact = segment_circle_intercept(
            start, end, [0, 300], 5
        )

        self.assertEqual([const.ARENA_SIZE - 5, 300], contact)


    def test_collision_normal_uses_nearest_wrapped_image(self):
        left = body([const.ARENA_SIZE - 5, 100], size=(20, 20))
        right = body([5, 100], size=(20, 20))

        normal, distance, overlap = collision_info(left, right)

        self.assertEqual(normal, [-1.0, 0.0])
        self.assertEqual(distance, 10.0)
        self.assertEqual(overlap, 10.0)
        self.assertTrue(objects_overlap(left, right, overlap))

    def test_laser_hit_info_uses_wrapped_segment(self):
        parent = self.make_ship()
        parent.position = [const.ARENA_SIZE - 20, 100]
        laser = self.make_laser(parent)
        laser.start_position = parent.position.copy()
        laser.end_position = [30, 100]
        target = self.make_ship()
        target.position = [5, 100]

        hit_info = laser_hit_info(laser, target)

        self.assertIsNotNone(hit_info)
        self.assertIs(hit_info["target"], target)

    def test_laser_mask_sampling_rejects_empty_target_mask(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        laser.start_position = [100, 100]
        laser.end_position = [200, 100]
        target = self.make_ship()
        target.position = [150, 100]
        empty_mask = pygame.mask.Mask((20, 20), fill=False)
        target.get_collision_mask = lambda: empty_mask

        self.assertIsNone(laser_hit_info(laser, target))

        full_mask = pygame.mask.Mask((20, 20), fill=True)
        target.get_collision_mask = lambda: full_mask
        self.assertIsNotNone(laser_hit_info(laser, target))

    def test_laser_sampling_keeps_padded_mask_canvas_alignment(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        laser.start_position = [130, 100]
        laser.end_position = [170, 100]
        target = body([150, 100], size=(20, 20))
        padded_mask = pygame.mask.Mask((100, 100), fill=False)
        padded_mask.draw(pygame.mask.Mask((20, 20), fill=True), (40, 40))
        target.get_collision_mask = lambda: padded_mask

        self.assertIsNotNone(laser_hit_info(laser, target))

if __name__ == "__main__":
    unittest.main()
