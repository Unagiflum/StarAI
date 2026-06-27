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
from src.collision_capabilities import CollisionRole, ShipImpactContext, ShipImpactResult
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

if __name__ == '__main__':
    unittest.main()
