import unittest
from types import SimpleNamespace
from unittest import mock

from src.Battle import collision_responses


def body(position, velocity, *, size=(20, 20)):
    value = SimpleNamespace(
        position=list(position),
        velocity=list(velocity),
        size=list(size),
    )
    value.get_collision_mask = lambda: None
    return value


class CollisionResponseTests(unittest.TestCase):
    def test_non_inertial_ship_preserves_approaching_planet_bounce_velocity(self):
        ship = body([100, 100], [1.0, 0.0])
        ship.inertia = False
        ship.collision_velocity = [0.0, 0.0]
        ship.planet_contacts = set()
        ship.current_hp = 10
        planet = body([115, 100], [0.0, 0.0])

        with mock.patch.object(collision_responses.BattleEffect, "play_boom"):
            collision_responses.ship_impacts_planet(ship, planet, [], None)

        self.assertEqual([-1.0, 0.0], ship.velocity)
        self.assertEqual([-1.0, 0.0], ship.collision_velocity)


if __name__ == "__main__":
    unittest.main()
