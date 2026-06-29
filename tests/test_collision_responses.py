import unittest
import math
from types import SimpleNamespace
from unittest import mock
import pygame
import src.const as const
from collision_test_support import CollisionTestCase
from src.Battle import collision_responses
from src.Battle.collision_geometry import laser_hit_info
from src.toroidal import wrapped_delta
from src.collision_capabilities import (
    CollisionRole,
    ShipImpactContext,
    ShipImpactResult,
    SpecialObjectCollisionCapabilities,
)
from src.Objects.Ships.Druuge.A1.DruugeA1 import DruugeA1
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2


def apply_area_damage_response(source, target, effects, policy):
    delta = wrapped_delta(source.position, target.position)
    distance = math.hypot(*delta)
    policy(target, effects, delta, distance, source.damage_at_distance(distance))


def apply_laser_response(laser, target, effects, policy):
    hit_info = laser_hit_info(laser, target)
    collision_responses.resolve_laser_hit(
        laser, target, effects, hit_info["normal"], hit_info["contact"],
        lambda target, effects, normal, damage, contact: policy(
            target, effects, normal, damage, contact
        ),
    )


def laser_targets_from_policies(laser, ships, projectiles, special_objects, asteroids, planets):
    policies = {
        CollisionRole.SHIP: collision_responses.generic_is_laser_target,
        CollisionRole.PROJECTILE: collision_responses.generic_is_laser_target,
        CollisionRole.SPECIAL_OBJECT: collision_responses.generic_is_laser_target,
        CollisionRole.ASTEROID: collision_responses.asteroid_is_laser_target,
        CollisionRole.PLANET: collision_responses.planet_is_laser_target,
    }
    candidates = (
        [laser.target, *(special_object for special_object in special_objects if special_object is not laser.target)]
        if laser.target is not None
        else [*ships, *projectiles, *special_objects, *asteroids, *planets]
    )
    return [
        target for target in candidates
        if policies[target.collision_capabilities.role](
            laser, target, target is laser.target
        )
    ]


def body(position, velocity, *, size=(20, 20)):
    value = SimpleNamespace(position=list(position), velocity=list(velocity), size=list(size))
    value.get_collision_mask = lambda: None
    return value


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
    def test_fighter_laser_does_not_target_its_projectile_parent(self):
        fighter = self.make_fighter()
        laser = self.make_laser(fighter)

        self.assertFalse(
            collision_responses.generic_is_laser_target(
                laser, fighter, explicit=False
            )
        )

    def test_laser_can_target_parent_when_explicitly_enabled(self):
        fighter = self.make_fighter()
        laser = self.make_laser(fighter, hit_parent=True)

        self.assertTrue(
            collision_responses.generic_is_laser_target(
                laser, fighter, explicit=False
            )
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
