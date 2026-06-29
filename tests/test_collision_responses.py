import unittest
from unittest import mock

from collision_test_support import CollisionTestCase
from src.Battle import collision_responses, collisions
from src.collision_capabilities import (
    SpecialObjectCollisionCapabilities,
)


class ParentCollisionTests(CollisionTestCase):
    def setUp(self):
        self.parent = self.make_ship()
        self.parent.player = 1
        self.parent.position = [100, 100]
        self.projectile = self.make_projectile(self.parent)
        self.projectile.player = 1
        self.projectile.hit_parent = True
        self.projectile.special_object_collision_capabilities = (
            SpecialObjectCollisionCapabilities()
        )

    def test_parent_hit_is_enabled_when_configured(self):
        self.assertTrue(
            collision_responses.projectile_can_hit_ship(
                self.projectile, self.parent
            )
        )

    def test_parent_hit_stays_disabled_when_not_configured(self):
        self.projectile.hit_parent = False
        self.assertFalse(
            collision_responses.projectile_can_hit_ship(
                self.projectile, self.parent
            )
        )

    def test_parent_recovery_takes_precedence_over_parent_damage(self):
        self.projectile.hit_parent = False
        self.projectile.can_recover_with_parent = lambda: True
        self.assertTrue(
            collision_responses.projectile_can_hit_ship(
                self.projectile, self.parent
            )
        )


class LaserParentTargetTests(CollisionTestCase):
    def test_special_object_laser_does_not_target_its_projectile_parent(self):
        special_object = self.make_special_object()
        laser = self.make_laser(special_object)

        self.assertFalse(
            collisions.LASER_TARGET_REGISTRY.is_eligible(laser, special_object)
        )

    def test_laser_can_target_parent_when_explicitly_enabled(self):
        special_object = self.make_special_object()
        laser = self.make_laser(special_object, hit_parent=True)

        self.assertTrue(
            collisions.LASER_TARGET_REGISTRY.is_eligible(laser, special_object)
        )


class ProjectileDestructionTests(CollisionTestCase):
    def test_destruction_effect_and_callback_are_finalized_once(self):
        projectile = self.make_projectile(self.make_ship())
        projectile.on_destroyed = mock.Mock()
        effects = []

        with mock.patch.object(
            collision_responses.BattleEffect,
            "from_blast",
            return_value=object(),
        ) as from_blast:
            collision_responses.destroy_projectile(
                projectile,
                effects,
                [1.0, 0.0],
                projectile.current_damage,
            )
            collision_responses.destroy_projectile(
                projectile,
                effects,
                [1.0, 0.0],
                projectile.current_damage,
            )

        from_blast.assert_called_once()
        projectile.on_destroyed.assert_called_once_with()
        self.assertEqual(len(effects), 1)

if __name__ == '__main__':
    unittest.main()
