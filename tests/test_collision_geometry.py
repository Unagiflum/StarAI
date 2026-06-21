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
