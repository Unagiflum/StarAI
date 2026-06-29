import unittest
from types import SimpleNamespace
from unittest import mock

from collision_test_support import CollisionTestCase
from src.Battle import collision_responses, collisions
from src.Battle.area_dispatch import AreaTargetRegistry
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    DurabilityCapabilities,
)


def target_with_role(role):
    return SimpleNamespace(collision_capabilities=CollisionCapabilities(role))


class AreaTargetRegistryTests(unittest.TestCase):
    def test_registry_routes_eligibility_and_damage(self):
        registry = AreaTargetRegistry()
        is_eligible = mock.Mock(return_value=True)
        apply_damage = mock.Mock(return_value=3)
        registry.register(
            CollisionRole.SHIP,
            is_eligible=is_eligible,
            apply_damage=apply_damage,
        )
        source = object()
        target = target_with_role(CollisionRole.SHIP)
        effects = []

        self.assertTrue(registry.is_eligible(source, target))
        applied = registry.apply_damage(
            source,
            target,
            effects,
            [3.0, 4.0],
            5.0,
            3,
        )

        self.assertEqual(applied, 3)
        is_eligible.assert_called_once_with(source, target)
        apply_damage.assert_called_once_with(
            source,
            target,
            effects,
            [3.0, 4.0],
            5.0,
            3,
        )

    def test_unregistered_role_is_ineligible_and_ignores_damage(self):
        registry = AreaTargetRegistry()
        target = target_with_role(CollisionRole.NONE)

        self.assertFalse(registry.is_eligible(object(), target))
        self.assertEqual(
            registry.apply_damage(object(), target, [], None, 0, 1),
            0,
        )

    def test_duplicate_policy_is_rejected(self):
        registry = AreaTargetRegistry()
        registry.register(
            CollisionRole.SHIP,
            is_eligible=mock.Mock(),
            apply_damage=mock.Mock(),
        )

        with self.assertRaisesRegex(ValueError, "SHIP"):
            registry.register(
                CollisionRole.SHIP,
                is_eligible=mock.Mock(),
                apply_damage=mock.Mock(),
            )

    def test_production_registry_has_policy_for_every_area_target_role(self):
        expected = {
            CollisionRole.SHIP: (
                collision_responses.ship_is_area_target,
                collision_responses.apply_ship_area_damage,
            ),
            CollisionRole.PROJECTILE: (
                collision_responses.projectile_is_area_target,
                collision_responses.apply_projectile_area_damage,
            ),
            CollisionRole.SPECIAL_OBJECT: (
                collision_responses.special_object_is_area_target,
                collision_responses.apply_special_object_area_damage,
            ),
            CollisionRole.ASTEROID: (
                collision_responses.asteroid_is_area_target,
                collision_responses.apply_asteroid_area_damage,
            ),
            CollisionRole.PLANET: (
                collision_responses.planet_is_area_target,
                collision_responses.apply_planet_area_damage,
            ),
        }

        for role, (eligibility, damage) in expected.items():
            with self.subTest(role=role):
                policy = collisions.AREA_TARGET_REGISTRY.policy_for(
                    target_with_role(role)
                )
                self.assertIs(policy.is_eligible, eligibility)
                self.assertIs(policy.apply_damage, damage)


class ProductionAreaPolicyTests(CollisionTestCase):
    def test_psychic_immune_ship_is_ineligible(self):
        source = self.make_area_damage([100, 100], lambda distance: 1)
        source.is_psychic = True
        source.player = 1
        target = self.make_ship()
        target.player = 2
        target.durability_capabilities = DurabilityCapabilities(
            immune_to_psychic=True
        )

        self.assertFalse(
            collisions.AREA_TARGET_REGISTRY.is_eligible(source, target)
        )

    def test_invulnerable_area_capability_is_ineligible(self):
        source = self.make_area_damage([100, 100], lambda distance: 1)
        target = self.make_ship()
        target.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True,
            vulnerable=False,
        )

        self.assertFalse(
            collisions.AREA_TARGET_REGISTRY.is_eligible(source, target)
        )


if __name__ == "__main__":
    unittest.main()
