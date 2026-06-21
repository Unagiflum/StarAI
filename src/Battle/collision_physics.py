"""Physical collision resolution operations."""

import math

import src.const as const
from src.Battle.collision_geometry import (
    collision_info,
    get_collision_mask,
    mask_broadphase_overlap,
    mask_radius,
    objects_overlap,
    radius,
)
from src.Objects.Ships.space_ship import SpaceShip


def mass(obj):
    return max(0.1, getattr(obj, "mass", radius(obj) / 50))


def elastic_bounce(obj, other, normal, distance, overlap):
    obj_mass = mass(obj)
    other_mass = mass(other)
    relative_velocity = [
        obj.velocity[0] - other.velocity[0],
        obj.velocity[1] - other.velocity[1],
    ]
    velocity_along_normal = dot(relative_velocity, normal)

    if velocity_along_normal < 0:
        impulse = -(1 + 1.0) * velocity_along_normal
        impulse /= (1 / obj_mass) + (1 / other_mass)

        obj.velocity[0] += impulse * normal[0] / obj_mass
        obj.velocity[1] += impulse * normal[1] / obj_mass
        other.velocity[0] -= impulse * normal[0] / other_mass
        other.velocity[1] -= impulse * normal[1] / other_mass

    separate_dynamic_bodies(
        obj, other, normal, overlap, obj_mass, other_mass
    )


def bounce_off_static_body(
    obj, static_body, normal, overlap, extra_clearance=0.0
):
    velocity_along_normal = dot(obj.velocity, normal)
    if velocity_along_normal < 0:
        impulse = [
            -2 * velocity_along_normal * normal[0],
            -2 * velocity_along_normal * normal[1],
        ]
        obj.velocity[0] += impulse[0]
        obj.velocity[1] += impulse[1]
        if isinstance(obj, SpaceShip) and not obj.inertia:
            obj.collision_velocity = obj.velocity.copy()
        collided_while_approaching = True
    else:
        collided_while_approaching = False

    separate_from_static_body(
        obj, static_body, normal, overlap, extra_clearance
    )
    return collided_while_approaching


def stop_at_static_body(obj, static_body, normal, overlap):
    velocity_along_normal = dot(obj.velocity, normal)
    if velocity_along_normal < 0:
        obj.velocity[0] -= velocity_along_normal * normal[0]
        obj.velocity[1] -= velocity_along_normal * normal[1]

    separate_from_static_body(obj, static_body, normal, overlap)


def separate_from_static_body(
    obj, static_body, normal, overlap, extra_clearance=0.0
):
    if overlap <= 0 and not mask_broadphase_overlap(obj, static_body):
        return

    if (
        get_collision_mask(obj) is None
        or get_collision_mask(static_body) is None
    ):
        if overlap <= 0:
            return
        separation = overlap + extra_clearance
        obj.position[0] = (
            obj.position[0] + normal[0] * separation
        ) % const.ARENA_SIZE
        obj.position[1] = (
            obj.position[1] + normal[1] * separation
        ) % const.ARENA_SIZE
        return

    max_separation = (
        int(math.ceil(mask_radius(obj) + mask_radius(static_body))) + 1
    )
    moved = 0
    step = 2
    while moved <= max_separation:
        _, _, current_overlap = collision_info(obj, static_body)
        if not objects_overlap(obj, static_body, current_overlap):
            if extra_clearance > 0:
                obj.position[0] = (
                    obj.position[0] + normal[0] * extra_clearance
                ) % const.ARENA_SIZE
                obj.position[1] = (
                    obj.position[1] + normal[1] * extra_clearance
                ) % const.ARENA_SIZE
            return

        obj.position[0] = (
            obj.position[0] + normal[0] * step
        ) % const.ARENA_SIZE
        obj.position[1] = (
            obj.position[1] + normal[1] * step
        ) % const.ARENA_SIZE
        moved += step

    if extra_clearance > 0:
        obj.position[0] = (
            obj.position[0] + normal[0] * extra_clearance
        ) % const.ARENA_SIZE
        obj.position[1] = (
            obj.position[1] + normal[1] * extra_clearance
        ) % const.ARENA_SIZE


def separate_dynamic_bodies(
    obj, other, normal, overlap, obj_mass, other_mass
):
    if overlap <= 0:
        return

    total_mass = obj_mass + other_mass
    obj_push = overlap * (other_mass / total_mass)
    other_push = overlap * (obj_mass / total_mass)

    obj.position[0] = (
        obj.position[0] + normal[0] * obj_push
    ) % const.ARENA_SIZE
    obj.position[1] = (
        obj.position[1] + normal[1] * obj_push
    ) % const.ARENA_SIZE
    other.position[0] = (
        other.position[0] - normal[0] * other_push
    ) % const.ARENA_SIZE
    other.position[1] = (
        other.position[1] - normal[1] * other_push
    ) % const.ARENA_SIZE


def dot(vector, other):
    return vector[0] * other[0] + vector[1] * other[1]
