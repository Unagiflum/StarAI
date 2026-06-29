import unittest
from types import SimpleNamespace
from unittest import mock

from collision_test_support import CollisionTestCase
from src.Battle import collision_responses, collisions
from src.Battle.laser_dispatch import LaserTargetRegistry
from src.collision_capabilities import (
    CollisionCapabilities,
    CollisionRole,
    LaserTargetCapabilities,
)


def target_with_role(role):
    return SimpleNamespace(collision_capabilities=CollisionCapabilities(role))


class LaserTargetRegistryTests(unittest.TestCase):
    def test_registry_routes_eligibility_and_impact(self):
        registry = LaserTargetRegistry()
        is_eligible = mock.Mock(return_value=True)
        apply_impact = mock.Mock()
        registry.register(
            CollisionRole.SHIP,
            is_eligible=is_eligible,
            apply_impact=apply_impact,
        )
        laser = object()
        target = target_with_role(CollisionRole.SHIP)
        effects = []

        self.assertTrue(registry.is_eligible(laser, target, explicit=True))
        registry.apply_impact(target, effects, [1.0, 0.0], 2, [100, 100])

        is_eligible.assert_called_once_with(laser, target, True)
        apply_impact.assert_called_once_with(
            target,
            effects,
            [1.0, 0.0],
            2,
            [100, 100],
        )

    def test_unregistered_role_is_ineligible_and_ignores_impact(self):
        registry = LaserTargetRegistry()
        target = target_with_role(CollisionRole.NONE)

        self.assertFalse(registry.is_eligible(object(), target))
        self.assertIsNone(registry.apply_impact(target, [], None, 1, None))

    def test_duplicate_policy_is_rejected(self):
        registry = LaserTargetRegistry()
        registry.register(
            CollisionRole.SHIP,
            is_eligible=mock.Mock(),
            apply_impact=mock.Mock(),
        )

        with self.assertRaisesRegex(ValueError, "SHIP"):
            registry.register(
                CollisionRole.SHIP,
                is_eligible=mock.Mock(),
                apply_impact=mock.Mock(),
            )

    def test_production_registry_has_policy_for_every_laser_target_role(self):
        expected = {
            CollisionRole.SHIP: (
                collision_responses.ship_is_laser_target,
                collision_responses.apply_ship_laser_impact,
            ),
            CollisionRole.PROJECTILE: (
                collision_responses.projectile_is_laser_target,
                collision_responses.apply_projectile_laser_impact,
            ),
            CollisionRole.SPECIAL_OBJECT: (
                collision_responses.special_object_is_laser_target,
                collision_responses.apply_special_object_laser_impact,
            ),
            CollisionRole.ASTEROID: (
                collision_responses.asteroid_is_laser_target,
                collision_responses.apply_asteroid_laser_impact,
            ),
            CollisionRole.PLANET: (
                collision_responses.planet_is_laser_target,
                collision_responses.apply_planet_laser_impact,
            ),
        }

        for role, (eligibility, impact) in expected.items():
            with self.subTest(role=role):
                policy = collisions.LASER_TARGET_REGISTRY.policy_for(
                    target_with_role(role)
                )
                self.assertIs(policy.is_eligible, eligibility)
                self.assertIs(policy.apply_impact, impact)


class ProductionLaserPolicyTests(CollisionTestCase):
    def test_explicit_laser_can_target_friendly_ship(self):
        parent = self.make_ship()
        parent.player = 1
        laser = self.make_laser(parent)
        friendly = self.make_ship()
        friendly.player = 1

        self.assertFalse(
            collisions.LASER_TARGET_REGISTRY.is_eligible(laser, friendly)
        )
        self.assertTrue(
            collisions.LASER_TARGET_REGISTRY.is_eligible(
                laser,
                friendly,
                explicit=True,
            )
        )

    def test_special_objects_preserve_legacy_targetable_exception(self):
        laser = self.make_laser(self.make_ship())
        special_object = self.make_fighter()
        special_object.laser_target_capabilities = LaserTargetCapabilities(
            targetable=False
        )

        self.assertTrue(
            collisions.LASER_TARGET_REGISTRY.is_eligible(
                laser,
                special_object,
            )
        )

    def test_ordinary_projectile_respects_targetable_capability(self):
        parent = self.make_ship()
        laser = self.make_laser(parent)
        projectile = self.make_projectile(parent)
        projectile.laser_target_capabilities = LaserTargetCapabilities(
            targetable=False
        )

        self.assertFalse(
            collisions.LASER_TARGET_REGISTRY.is_eligible(laser, projectile)
        )


if __name__ == "__main__":
    unittest.main()
