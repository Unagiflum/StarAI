import unittest
from types import SimpleNamespace

from src.Battle import collision_physics


def body(position, velocity, *, size=(10, 10), mass=1):
    value = SimpleNamespace(
        position=list(position),
        velocity=list(velocity),
        size=list(size),
        mass=mass,
    )
    value.get_collision_mask = lambda: None
    return value


class CollisionPhysicsTests(unittest.TestCase):
    def test_equal_mass_elastic_bounce_exchanges_velocity_and_separates(self):
        first = body([4, 100], [1, 0])
        second = body([12, 100], [-1, 0])

        collision_physics.elastic_bounce(
            first, second, [-1, 0], distance=8, overlap=2
        )

        self.assertEqual([-1.0, 0.0], first.velocity)
        self.assertEqual([1.0, 0.0], second.velocity)
        self.assertEqual([3.0, 100.0], first.position)
        self.assertEqual([13.0, 100.0], second.position)

    def test_static_bounce_reflects_only_normal_velocity(self):
        moving = body([4, 100], [3, 2])
        static = body([12, 100], [0, 0])

        approaching = collision_physics.bounce_off_static_body(
            moving, static, [-1, 0], overlap=2, extra_clearance=1
        )

        self.assertTrue(approaching)
        self.assertEqual([-3, 2], moving.velocity)
        self.assertEqual([1, 100], moving.position)

    def test_static_stop_removes_inward_velocity_and_preserves_tangent(self):
        moving = body([4, 100], [3, 2])
        static = body([12, 100], [0, 0])

        collision_physics.stop_at_static_body(
            moving, static, [-1, 0], overlap=2
        )

        self.assertEqual([0, 2], moving.velocity)
        self.assertEqual([2, 100], moving.position)


if __name__ == "__main__":
    unittest.main()
