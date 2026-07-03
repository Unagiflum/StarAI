"""Collision-only brute-force versus spatial broad-phase benchmark.

Rendering, resource loading, audio, and scene construction are outside timed
regions.  Each case is warmed up and then rebuilt for multiple timed samples
because collision handling mutates object state.

Example:
    python benchmark_collision.py --counts 50,150,300 --iterations 7
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import time
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

from src import const
from src.Battle import collisions
from src.Battle.effects import BattleEffect
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    LaserTargetCapabilities,
    PhysicalCollisionCapabilities,
    SpecialObjectCollisionCapabilities,
)
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


def make_ship(player=1, position=(0.0, 0.0)):
    ship = SpaceShip.__new__(SpaceShip)
    ship.name = "BenchmarkShip"
    ship.player = player
    ship.position = list(position)
    ship.previous_position = ship.position.copy()
    ship.size = [28, 28]
    ship.velocity = [0.0, 0.0]
    ship.mass = 1.0
    ship.can_move = True
    ship.can_collide = True
    ship.inertia = True
    ship.collision_velocity = [0.0, 0.0]
    ship.planet_contacts = set()
    ship.current_hp = 100000
    ship.currently_alive = True
    ship.collision_capabilities = CollisionCapabilities(CollisionRole.SHIP)
    ship.laser_target_capabilities = LaserTargetCapabilities()
    ship.area_damage_capabilities = AreaDamageCapabilities(targetable=True)
    ship.physical_collision_capabilities = PhysicalCollisionCapabilities(
        is_solid=True,
        bounces_on_immovable=True,
    )
    return ship


def make_projectile(player, position, previous, *, collidable=True):
    parent = make_ship(player)
    projectile = Ability.__new__(Ability)
    projectile.name = projectile.projectile_name = f"BenchmarkP{player}"
    projectile.type = "projectile"
    projectile.player = player
    projectile.parent = parent
    projectile.position = list(position)
    projectile.previous_position = list(previous)
    projectile.size = [12, 12]
    projectile.masks = None
    projectile.heading = 0
    projectile.frames = 1
    projectile.can_move = True
    projectile.can_collide = collidable
    projectile.currently_alive = True
    projectile.current_hp = 100000
    projectile.current_damage = 1
    projectile.hit_parent = False
    projectile.hit_self = False
    projectile.death_animation = []
    projectile.velocity = [0.0, 0.0]
    projectile.collision_capabilities = CollisionCapabilities(
        CollisionRole.PROJECTILE
    )
    projectile.laser_target_capabilities = LaserTargetCapabilities()
    projectile.area_damage_capabilities = AreaDamageCapabilities(targetable=True)
    projectile.physical_collision_capabilities = PhysicalCollisionCapabilities(
        is_solid=False,
        is_projectile=True,
    )
    return projectile


def make_special(player, position, previous):
    special = make_projectile(player, position, previous)
    special.name = special.projectile_name = f"BenchmarkS{player}"
    special.type = "special_object"
    special.collision_capabilities = CollisionCapabilities(
        CollisionRole.SPECIAL_OBJECT
    )
    special.special_object_collision_capabilities = (
        SpecialObjectCollisionCapabilities()
    )
    return special


def make_laser(index, position, length=1200.0):
    parent = make_ship(1, position)
    laser = Ability.__new__(Ability)
    laser.name = laser.projectile_name = f"BenchmarkLaser{index}"
    laser.type = "laser"
    laser.player = 1
    laser.parent = parent
    laser.position = list(position)
    laser.previous_position = laser.position.copy()
    laser.start_position = laser.position.copy()
    laser.end_position = [position[0] + length, position[1]]
    laser.size = [1, 1]
    laser.masks = None
    laser.heading = 0
    laser.frames = 1
    laser.can_move = False
    laser.can_collide = True
    laser.currently_alive = True
    laser.current_hp = 1
    laser.current_damage = 1
    laser.hit_parent = False
    laser.hit_self = False
    laser.target = None
    laser.LASER_WIDTH = 6
    laser.collision_capabilities = CollisionCapabilities(CollisionRole.LASER)
    laser.laser_target_capabilities = LaserTargetCapabilities(targetable=False)
    laser.area_damage_capabilities = AreaDamageCapabilities()
    laser.physical_collision_capabilities = PhysicalCollisionCapabilities(
        is_intangible=True
    )
    return laser


def make_laser_area_target(position):
    target = Ability.__new__(Ability)
    target.name = target.projectile_name = "BenchmarkLaserTarget"
    target.type = "area"
    target.player = 2
    target.position = list(position)
    target.previous_position = target.position.copy()
    target.size = [18, 18]
    target.masks = None
    target.heading = 0
    target.frames = 1
    target.can_move = False
    target.can_collide = False
    target.currently_alive = True
    target.current_hp = 100000
    target.collision_capabilities = CollisionCapabilities(CollisionRole.AREA)
    target.laser_target_capabilities = LaserTargetCapabilities(
        targetable=True,
        vulnerable=False,
        blocks_lasers=True,
    )
    target.area_damage_capabilities = AreaDamageCapabilities()
    target.physical_collision_capabilities = PhysicalCollisionCapabilities(
        is_solid=False
    )
    return target


def make_area_emitter(index, position, radius=220.0):
    emitter = make_laser_area_target(position)
    emitter.name = f"BenchmarkArea{index}"
    emitter.player = 1
    emitter.area_damage_pending = True
    emitter.area_damage_capabilities = AreaDamageCapabilities(emits=True)
    emitter.special_object_collision_capabilities = (
        SpecialObjectCollisionCapabilities(
            collides_with_friendly_ships=True,
            collides_with_enemy_ships=True,
        )
    )
    emitter.damage_at_distance = lambda distance: 0
    emitter.maximum_area_damage_radius = lambda: radius
    return emitter


def positions(count, distribution, rng, *, margin=80.0):
    if distribution == "sparse":
        columns = max(1, int(math.sqrt(count)))
        spacing = min(300.0, (const.ARENA_SIZE - margin * 2) / max(1, columns))
        return [
            (
                margin + (index % columns) * spacing,
                margin + (index // columns) * spacing,
            )
            for index in range(count)
        ]
    return [
        (4000.0 + rng.uniform(-180, 180), 4000.0 + rng.uniform(-180, 180))
        for _ in range(count)
    ]


def build_scene(scenario, count, distribution, seed):
    rng = random.Random(seed)
    scene_positions = positions(count, distribution, rng)
    if scenario in ("projectiles", "special_objects"):
        factory = make_projectile if scenario == "projectiles" else make_special
        return [
            factory(
                1 + index % 2,
                position,
                (
                    (position[0] + rng.uniform(-90, 90)) % const.ARENA_SIZE,
                    (position[1] + rng.uniform(-90, 90)) % const.ARENA_SIZE,
                ),
            )
            for index, position in enumerate(scene_positions)
        ]

    if scenario == "lasers":
        targets = [make_laser_area_target(position) for position in scene_positions]
        laser_count = max(1, count // 8)
        if distribution == "sparse":
            laser_positions = [
                (80.0, 80.0 + index * 37.0) for index in range(laser_count)
            ]
        else:
            laser_positions = [
                (3400.0, 4000.0 + rng.uniform(-180, 180))
                for _ in range(laser_count)
            ]
        return [
            *(make_laser(index, position) for index, position in enumerate(laser_positions)),
            *targets,
        ]

    if scenario == "area_damage":
        targets = [
            make_projectile(2, position, position, collidable=False)
            for position in scene_positions
        ]
        emitter_count = max(1, count // 12)
        if distribution == "sparse":
            emitter_positions = [
                (100.0 + index * 211.0, 100.0 + index * 173.0)
                for index in range(emitter_count)
            ]
        else:
            emitter_positions = [
                (4000.0 + rng.uniform(-100, 100), 4000.0 + rng.uniform(-100, 100))
                for _ in range(emitter_count)
            ]
        return [
            *(
                make_area_emitter(index, position)
                for index, position in enumerate(emitter_positions)
            ),
            *targets,
        ]
    raise ValueError(scenario)


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def benchmark_case(args, scenario, count, distribution, mode):
    for warmup in range(args.warmups):
        collisions.handle_collisions(
            build_scene(scenario, count, distribution, args.seed + warmup),
            broad_phase=mode,
        )

    elapsed = []
    for iteration in range(args.iterations):
        objects = build_scene(
            scenario,
            count,
            distribution,
            args.seed + args.warmups + iteration,
        )
        start = time.perf_counter()
        collisions.handle_collisions(
            objects,
            broad_phase=mode,
        )
        elapsed.append((time.perf_counter() - start) * 1000.0)

    # Collect deterministic counters in a separate untimed pass so per-pair
    # diagnostics do not distort the brute-force reference timing.
    metrics = collisions.CollisionMetrics()
    collisions.handle_collisions(
        build_scene(scenario, count, distribution, args.seed + 10000),
        broad_phase=mode,
        metrics=metrics,
    )

    return {
        "median_ms": statistics.median(elapsed),
        "p95_ms": percentile(elapsed, 0.95),
        "possible": metrics.possible_physical_pairs,
        "physical": metrics.physical_candidate_pairs,
        "possible_laser": metrics.possible_laser_targets,
        "laser": metrics.laser_candidates,
        "possible_area": metrics.possible_area_targets,
        "area": metrics.area_candidates,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="50,150,300")
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("projectiles", "special_objects", "lasers", "area_damage"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    counts = [int(value) for value in args.counts.split(",")]
    scenarios = args.scenario or [
        "projectiles",
        "special_objects",
        "lasers",
        "area_damage",
    ]
    original_blast = BattleEffect.__dict__["from_blast"]
    original_boom = BattleEffect.__dict__["play_boom"]
    BattleEffect.from_blast = staticmethod(
        lambda *args, **kwargs: SimpleNamespace(benchmark_effect=True)
    )
    BattleEffect.play_boom = staticmethod(lambda damage: None)
    try:
        print(
            "scenario,distribution,count,mode,possible_pairs,"
            "physical_candidates,possible_laser_targets,laser_candidates,"
            "possible_area_targets,area_candidates,"
            "median_ms,p95_ms"
        )
        for scenario in scenarios:
            for distribution in ("sparse", "dense"):
                for count in counts:
                    for mode in (
                        collisions.BROAD_PHASE_BRUTE_FORCE,
                        collisions.BROAD_PHASE_SPATIAL,
                    ):
                        result = benchmark_case(
                            args,
                            scenario,
                            count,
                            distribution,
                            mode,
                        )
                        print(
                            f"{scenario},{distribution},{count},{mode},"
                            f"{result['possible']},{result['physical']},"
                            f"{result['possible_laser']},{result['laser']},"
                            f"{result['possible_area']},{result['area']},"
                            f"{result['median_ms']:.3f},{result['p95_ms']:.3f}"
                        )
    finally:
        BattleEffect.from_blast = original_blast
        BattleEffect.play_boom = original_boom


if __name__ == "__main__":
    main()
