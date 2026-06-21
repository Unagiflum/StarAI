import unittest
from types import SimpleNamespace

import pygame

import src.const as const
from src.Battle import collision_geometry


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


class CollisionGeometryTests(unittest.TestCase):
    def test_ship_rotation_is_blocked_by_live_overlapping_candidates(self):
        ship = body([100, 100], size=(20, 20))
        live_opponent = body([115, 100], size=(20, 20))
        live_opponent.current_hp = 1
        ship.opponent = live_opponent
        ship.asteroids = []
        ship.planet = None

        self.assertTrue(collision_geometry.ship_rotation_blocked(ship))

    def test_ship_rotation_ignores_dead_and_non_overlapping_candidates(self):
        ship = body([100, 100], size=(20, 20))
        dead_opponent = body([100, 100], size=(20, 20))
        dead_opponent.current_hp = 0
        dead_asteroid = body([100, 100], size=(20, 20))
        dead_asteroid.currently_alive = False
        ship.opponent = dead_opponent
        ship.asteroids = [dead_asteroid]
        ship.planet = body([200, 100], size=(20, 20))

        self.assertFalse(collision_geometry.ship_rotation_blocked(ship))

    def test_contact_info_uses_shortest_toroidal_displacement(self):
        left = body([5, 100], size=(8, 8))
        right = body([const.ARENA_SIZE - 5, 100], size=(8, 8))

        normal, distance, overlap = collision_geometry.collision_info(
            left, right
        )

        self.assertEqual([1.0, 0.0], normal)
        self.assertEqual(10, distance)
        self.assertEqual(-2, overlap)

    def test_empty_masks_reject_a_circular_broadphase_overlap(self):
        empty_mask = pygame.mask.Mask((10, 10), fill=False)
        first = body([100, 100], mask=empty_mask)
        second = body([100, 100], mask=empty_mask)
        _, _, overlap = collision_geometry.collision_info(first, second)

        self.assertFalse(
            collision_geometry.objects_overlap(first, second, overlap)
        )

    def test_swept_impact_finds_contact_between_frame_endpoints(self):
        projectile = body([100, 200], previous_position=[0, 200])
        target = body([50, 200])

        contact, normal = collision_geometry.swept_impact(
            projectile, target
        )

        self.assertIsNotNone(contact)
        self.assertEqual([-1.0, 0.0], normal)
        self.assertAlmostEqual(45.0, contact[0])
        self.assertAlmostEqual(200.0, contact[1])

    def test_segment_circle_intercept_crosses_wrapped_boundary(self):
        start = [const.ARENA_SIZE - 10, 300]
        _, end = collision_geometry.wrapped_segment(start, [10, 300])

        contact = collision_geometry.segment_circle_intercept(
            start, end, [0, 300], 5
        )

        self.assertEqual([const.ARENA_SIZE - 5, 300], contact)


if __name__ == "__main__":
    unittest.main()
