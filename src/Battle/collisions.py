"""Ordered collision pipeline."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain

from src.Battle import collision_responses as responses
from src.Battle.area_dispatch import AreaTargetRegistry
from src.Battle.collision_contract import (
    CollisionContext,
    CollisionEnvironment,
    CollisionOutcome,
    collision_context,
)
from src.Battle.collision_dispatch import CollisionPairRegistry
from src.Battle.collision_geometry import (
    geometry_cache_active,
    invalidate_geometry,
    laser_hit_info,
    objects_overlap_during_frame,
    pair_geometry_cache,
)
from src.Battle.collision_spatial_index import (
    DEFAULT_CELL_SIZE,
    SpatialIndexMetrics,
    ToroidalSpatialIndex,
)
from src.Battle.laser_dispatch import LaserTargetRegistry

# BattleEffect remains exposed here for existing callers and test patches.
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.collision_capabilities import CollisionRole
from src.Objects.Space.space_obj import Asteroid
from src.toroidal import (
    wrapped_delta as _wrapped_delta,
)


BROAD_PHASE_SPATIAL = "spatial"
BROAD_PHASE_BRUTE_FORCE = "brute_force"

# Exact scans beat grid query setup for the small laser/area target sets seen in
# ordinary battles.  Physical pair handling still uses the grid at all sizes.
_LASER_SPATIAL_QUERY_MIN_TARGETS = 32
_AREA_SPATIAL_QUERY_MIN_TARGETS = 256


@dataclass
class CollisionMetrics:
    """Optional counters for collision diagnostics and benchmarks."""

    possible_physical_pairs: int = 0
    physical_candidate_pairs: int = 0
    physical_dispatched_pairs: int = 0
    possible_laser_targets: int = 0
    laser_candidates: int = 0
    possible_area_targets: int = 0
    area_candidates: int = 0
    area_full_scan_fallbacks: int = 0
    spatial: SpatialIndexMetrics = field(default_factory=SpatialIndexMetrics)


def _dispatch_collision_pairs(
    first_objects,
    second_objects,
    effects,
    *,
    stop_after_handled=True,
    environment=None,
    spatial_index=None,
    second_category=None,
    shadow_validate=False,
    metrics=None,
    positions_by_id=None,
    intangible_indices=None,
):
    """Dispatch ordered object pairs by their explicit collision roles."""
    if environment is None:
        environment = CollisionEnvironment()
    context = CollisionContext(effects, environment)
    if spatial_index is not None:
        _dispatch_spatial_collision_pairs(
            first_objects,
            second_objects,
            context,
            spatial_index,
            second_category,
            stop_after_handled=stop_after_handled,
            shadow_validate=shadow_validate,
            metrics=metrics,
            positions_by_id=positions_by_id,
            intangible_indices=intangible_indices,
        )
        return

    for first in first_objects:
        for second in second_objects:
            if metrics is not None:
                metrics.physical_candidate_pairs += 1
                metrics.physical_dispatched_pairs += 1
            outcome = _dispatch_collision_pair(first, second, context)
            if outcome.handled and stop_after_handled:
                break


def _dispatch_unique_collision_pairs(
    objects,
    effects,
    first_is_active,
    environment=None,
    spatial_index=None,
    category=None,
    shadow_validate=False,
    metrics=None,
    positions_by_id=None,
    intangible_indices=None,
):
    """Dispatch each unordered pair while preserving outer-loop activity rules."""
    if environment is None:
        environment = CollisionEnvironment()
    context = CollisionContext(effects, environment)
    if spatial_index is not None:
        _dispatch_spatial_unique_collision_pairs(
            objects,
            context,
            first_is_active,
            spatial_index,
            category,
            shadow_validate=shadow_validate,
            metrics=metrics,
            positions_by_id=positions_by_id,
            intangible_indices=intangible_indices,
        )
        return

    for index, first in enumerate(objects):
        if not first_is_active(first):
            continue
        # Index iteration avoids allocating a tail slice for every outer body.
        for second_index in range(index + 1, len(objects)):
            second = objects[second_index]
            if metrics is not None:
                metrics.physical_candidate_pairs += 1
                metrics.physical_dispatched_pairs += 1
            _dispatch_collision_pair(first, second, context)


def _dispatch_spatial_collision_pairs(
    first_objects,
    second_objects,
    context,
    spatial_index,
    second_category,
    *,
    stop_after_handled,
    shadow_validate,
    metrics,
    positions_by_id,
    intangible_indices,
):
    positions_by_id = positions_by_id or _group_positions(second_objects)
    if intangible_indices is None:
        intangible_indices = _intangible_indices(second_objects)
    for first in first_objects:
        start_index = 0
        while start_index < len(second_objects):
            candidates = _ordered_spatial_candidates(
                spatial_index,
                first,
                second_objects,
                second_category,
                start_index,
                positions_by_id,
                intangible_indices,
            )
            if shadow_validate:
                _assert_no_omitted_collision(
                    first,
                    second_objects,
                    start_index,
                    candidates,
                )
            if not candidates:
                break

            must_requery = False
            handled = False
            for second_index, second in candidates:
                if metrics is not None:
                    metrics.physical_candidate_pairs += 1
                    metrics.physical_dispatched_pairs += 1
                outcome, first_membership_changed = _dispatch_and_refresh(
                    first,
                    second,
                    context,
                    spatial_index,
                )
                start_index = second_index + 1
                if outcome.handled and stop_after_handled:
                    handled = True
                    break
                if first_membership_changed:
                    must_requery = True
                    break

            if handled:
                break
            if not must_requery:
                break


def _dispatch_spatial_unique_collision_pairs(
    objects,
    context,
    first_is_active,
    spatial_index,
    category,
    *,
    shadow_validate,
    metrics,
    positions_by_id,
    intangible_indices,
):
    positions_by_id = positions_by_id or _group_positions(objects)
    if intangible_indices is None:
        intangible_indices = _intangible_indices(objects)
    for first_index, first in enumerate(objects):
        if not first_is_active(first):
            continue
        start_index = first_index + 1
        while start_index < len(objects):
            candidates = _ordered_spatial_candidates(
                spatial_index,
                first,
                objects,
                category,
                start_index,
                positions_by_id,
                intangible_indices,
            )
            if shadow_validate:
                _assert_no_omitted_collision(
                    first,
                    objects,
                    start_index,
                    candidates,
                )
            if not candidates:
                break

            must_requery = False
            for second_index, second in candidates:
                if metrics is not None:
                    metrics.physical_candidate_pairs += 1
                    metrics.physical_dispatched_pairs += 1
                _, first_membership_changed = _dispatch_and_refresh(
                    first,
                    second,
                    context,
                    spatial_index,
                )
                start_index = second_index + 1
                if first_membership_changed:
                    must_requery = True
                    break
            if not must_requery:
                break


def _ordered_spatial_candidates(
    spatial_index,
    first,
    second_objects,
    second_category,
    start_index,
    positions_by_id,
    intangible_indices,
):
    if not spatial_index.contains(first) or _is_intangible(first):
        return [
            (index, second_objects[index])
            for index in range(start_index, len(second_objects))
        ]

    nearby_indices = set()
    for obj in spatial_index.candidates_for(
        first,
        categories=(second_category,) if second_category else None,
        include_self=True,
    ):
        nearby_indices.update(
            index
            for index in positions_by_id.get(id(obj), ())
            if index >= start_index
        )
    # Intangible pair dispatch is currently authoritative even at a distance;
    # retain those entries until collision policies replace that legacy rule.
    nearby_indices.update(
        index for index in intangible_indices if index >= start_index
    )
    return [
        (index, second_objects[index])
        for index in sorted(nearby_indices)
    ]


def _group_positions(objects):
    positions = {}
    for index, obj in enumerate(objects):
        positions.setdefault(id(obj), []).append(index)
    return positions


def _intangible_indices(objects):
    return tuple(
        index for index, obj in enumerate(objects) if _is_intangible(obj)
    )


def _dispatch_and_refresh(first, second, context, spatial_index):
    first_position = _position_tuple(first)
    second_position = _position_tuple(second)
    outcome = _dispatch_collision_pair(first, second, context)
    first_moved = first_position != _position_tuple(first)
    second_moved = second_position != _position_tuple(second)

    first_changed = False
    if (outcome.handled or first_moved) and not outcome.first_consumed:
        first_changed = spatial_index.update(first)
    if (outcome.handled or second_moved) and not outcome.second_consumed:
        spatial_index.update(second)
    return outcome, first_changed


def _position_tuple(obj):
    position = getattr(obj, "position", None)
    return tuple(position) if position is not None else None


def _assert_no_omitted_collision(first, second_objects, start_index, candidates):
    candidate_ids = {id(second) for _, second in candidates}
    for second_index in range(start_index, len(second_objects)):
        second = second_objects[second_index]
        if id(second) in candidate_ids:
            continue
        with pair_geometry_cache():
            collides = objects_overlap_during_frame(first, second)
        if collides:
            raise AssertionError(
                "spatial broad phase omitted an exact swept collision: "
                f"{first!r} x {second!r}"
            )


def _dispatch_collision_pair(first, second, context_or_effects, environment=None):
    context = collision_context(context_or_effects, environment)

    phys_first = getattr(first, "physical_collision_capabilities", None)
    phys_second = getattr(second, "physical_collision_capabilities", None)

    if (phys_first and phys_first.is_intangible) or (phys_second and phys_second.is_intangible):
        # Preserve the existing stop-after-handled behavior until pair policies
        # explicitly define how intangible objects affect candidate scanning.
        return CollisionOutcome.RESOLVED

    if geometry_cache_active():
        outcome = COLLISION_PAIR_REGISTRY.dispatch(first, second, context)
    else:
        with pair_geometry_cache():
            outcome = COLLISION_PAIR_REGISTRY.dispatch(first, second, context)
    if outcome.handled:
        invalidate_geometry(first)
        invalidate_geometry(second)
    return outcome


def _create_collision_pair_registry():
    registry = CollisionPairRegistry()
    registry.register(
        CollisionRole.SHIP,
        CollisionRole.SHIP,
        responses.resolve_ship_ship_collision,
    )
    registry.register(
        CollisionRole.ASTEROID,
        CollisionRole.ASTEROID,
        responses.resolve_asteroid_asteroid_collision,
    )
    registry.register(
        CollisionRole.SHIP,
        CollisionRole.ASTEROID,
        responses.resolve_ship_asteroid_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.ASTEROID,
        CollisionRole.PLANET,
        responses.resolve_asteroid_planet_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SHIP,
        CollisionRole.PLANET,
        responses.resolve_ship_planet_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.PROJECTILE,
        responses.resolve_projectile_projectile_collision,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.SHIP,
        responses.resolve_projectile_ship_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.ASTEROID,
        responses.resolve_projectile_asteroid_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        CollisionRole.PLANET,
        responses.resolve_projectile_planet_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.SPECIAL_OBJECT,
        responses.resolve_projectile_projectile_collision,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.PROJECTILE,
        responses.resolve_projectile_projectile_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.SHIP,
        responses.resolve_projectile_ship_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.ASTEROID,
        responses.resolve_projectile_asteroid_collision,
        canonical_order=True,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        CollisionRole.PLANET,
        responses.resolve_projectile_planet_collision,
        canonical_order=True,
    )
    return registry


COLLISION_PAIR_REGISTRY = _create_collision_pair_registry()


def _create_area_target_registry():
    registry = AreaTargetRegistry()
    registry.register(
        CollisionRole.SHIP,
        is_eligible=responses.ship_is_area_target,
        apply_damage=responses.apply_ship_area_damage,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        is_eligible=responses.projectile_is_area_target,
        apply_damage=responses.apply_projectile_area_damage,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        is_eligible=responses.special_object_is_area_target,
        apply_damage=responses.apply_special_object_area_damage,
    )
    registry.register(
        CollisionRole.ASTEROID,
        is_eligible=responses.asteroid_is_area_target,
        apply_damage=responses.apply_asteroid_area_damage,
    )
    registry.register(
        CollisionRole.PLANET,
        is_eligible=responses.planet_is_area_target,
        apply_damage=responses.apply_planet_area_damage,
    )
    return registry


AREA_TARGET_REGISTRY = _create_area_target_registry()


def _create_laser_target_registry():
    registry = LaserTargetRegistry()
    registry.register(
        CollisionRole.SHIP,
        is_eligible=responses.ship_is_laser_target,
        apply_impact=responses.apply_ship_laser_impact,
    )
    registry.register(
        CollisionRole.PROJECTILE,
        is_eligible=responses.projectile_is_laser_target,
        apply_impact=responses.apply_projectile_laser_impact,
    )
    registry.register(
        CollisionRole.SPECIAL_OBJECT,
        is_eligible=responses.special_object_is_laser_target,
        apply_impact=responses.apply_special_object_laser_impact,
    )
    registry.register(
        CollisionRole.ASTEROID,
        is_eligible=responses.asteroid_is_laser_target,
        apply_impact=responses.apply_asteroid_laser_impact,
    )
    registry.register(
        CollisionRole.PLANET,
        is_eligible=responses.planet_is_laser_target,
        apply_impact=responses.apply_planet_laser_impact,
    )
    registry.register(
        CollisionRole.AREA,
        is_eligible=responses.area_is_laser_target,
        apply_impact=responses.apply_area_laser_impact,
    )
    return registry


LASER_TARGET_REGISTRY = _create_laser_target_registry()


@dataclass(frozen=True)
class _CollisionPhase:
    first_group: str
    second_group: str | None = None
    unique_pairs: bool = False
    first_is_active: Callable[[object], bool] | None = None
    stop_after_handled: bool = True
    collects_effects: bool = True
    includes_ship_environment: bool = False


def _always_active(obj):
    return True


def _currently_alive(obj):
    return obj.currently_alive


def _is_intangible(obj):
    physics = getattr(obj, "physical_collision_capabilities", None)
    return bool(physics and physics.is_intangible)


COLLISION_PHASES = (
    _CollisionPhase("ships", unique_pairs=True),
    _CollisionPhase("ships", "asteroids", stop_after_handled=False),
    _CollisionPhase(
        "asteroids",
        unique_pairs=True,
        first_is_active=_currently_alive,
    ),
    _CollisionPhase(
        "ships",
        "planets",
        stop_after_handled=False,
        collects_effects=False,
    ),
    _CollisionPhase(
        "asteroids",
        "planets",
        includes_ship_environment=True,
    ),
    _CollisionPhase(
        "projectiles",
        unique_pairs=True,
        first_is_active=responses.is_live_projectile_like,
    ),
    _CollisionPhase("projectiles", "ships"),
    _CollisionPhase("projectiles", "asteroids"),
    _CollisionPhase("projectiles", "planets"),
    _CollisionPhase(
        "special_objects",
        unique_pairs=True,
        first_is_active=responses.is_live_special_object,
    ),
    _CollisionPhase("special_objects", "projectiles"),
    _CollisionPhase("special_objects", "ships"),
    _CollisionPhase("special_objects", "asteroids"),
    _CollisionPhase(
        "special_objects",
        "planets",
        collects_effects=False,
    ),
)


def handle_collisions(
    game_objects,
    *,
    rng=None,
    resources=None,
    excluded_objects=(),
    broad_phase=BROAD_PHASE_SPATIAL,
    shadow_validate=False,
    cell_size=DEFAULT_CELL_SIZE,
    metrics=None,
    visual_effects_enabled=True,
):
    """Handle one collision frame with a shared, invalidatable geometry cache."""
    with pair_geometry_cache():
        return _handle_collision_frame(
            game_objects,
            rng=rng,
            resources=resources,
            excluded_objects=excluded_objects,
            broad_phase=broad_phase,
            shadow_validate=shadow_validate,
            cell_size=cell_size,
            metrics=metrics,
            visual_effects_enabled=visual_effects_enabled,
        )


def _handle_collision_frame(
    game_objects,
    *,
    rng=None,
    resources=None,
    excluded_objects=(),
    broad_phase=BROAD_PHASE_SPATIAL,
    shadow_validate=False,
    cell_size=DEFAULT_CELL_SIZE,
    metrics=None,
    visual_effects_enabled=True,
):
    if broad_phase not in (BROAD_PHASE_SPATIAL, BROAD_PHASE_BRUTE_FORCE):
        raise ValueError(f"unknown collision broad phase: {broad_phase!r}")

    world = World.coerce(game_objects)
    excluded_ids = {id(obj) for obj in excluded_objects}
    effects = []
    frame = world.collision_snapshot()
    all_asteroids = frame.asteroids
    spatial_index = (
        ToroidalSpatialIndex(
            frame.objects,
            categories=frame.spatial_categories,
            cell_size=cell_size,
            metrics=metrics.spatial if metrics is not None else None,
        )
        if broad_phase == BROAD_PHASE_SPATIAL
        else None
    )
    _handle_area_damage(
        world,
        effects,
        excluded_ids,
        frame=frame,
        spatial_index=spatial_index,
        shadow_validate=shadow_validate,
        metrics=metrics,
        use_spatial_queries=(
            shadow_validate
            or len(frame.objects) >= _AREA_SPATIAL_QUERY_MIN_TARGETS
        ),
    )

    ships = [
        ship
        for ship in frame.ships
        if (
            world.is_alive(ship)
            and id(ship) not in excluded_ids
            and not _is_intangible(ship)
        )
    ]
    projectiles = [
        ability
        for ability in frame.projectiles
        if world.is_colliding_ability_kind(ability, "projectile")
    ]
    special_objects = [
        ability
        for ability in frame.special_objects
        if world.is_colliding_ability_kind(ability, "special_object")
    ]
    lasers = [
        ability
        for ability in frame.lasers
        if world.is_colliding_ability_kind(ability, "laser")
    ]
    area_abilities = frame.area_abilities
    asteroids = [
        asteroid for asteroid in frame.asteroids if asteroid.currently_alive
    ]
    planets = frame.planets

    _handle_laser_collisions(
        lasers,
        ships,
        projectiles,
        special_objects,
        asteroids,
        planets,
        effects,
        excluded_ids,
        area_abilities=area_abilities,
        spatial_index=spatial_index,
        shadow_validate=shadow_validate,
        metrics=metrics,
        use_spatial_queries=(
            shadow_validate
            or sum(
                len(group)
                for group in (
                    ships,
                    projectiles,
                    special_objects,
                    area_abilities,
                    asteroids,
                    planets,
                )
            )
            >= _LASER_SPATIAL_QUERY_MIN_TARGETS
        ),
    )
    _run_collision_phases(
        {
            "ships": ships,
            "asteroids": asteroids,
            "projectiles": projectiles,
            "special_objects": special_objects,
            "planets": planets,
        },
        effects,
        spatial_index=spatial_index,
        shadow_validate=shadow_validate,
        metrics=metrics,
    )
    _spawn_replacement_asteroids(
        world,
        all_asteroids,
        ships,
        planets,
        rng=rng,
        resources=resources,
    )

    if visual_effects_enabled:
        world.add_all(effects)
    else:
        world.add_all(
            effect for effect in effects if not isinstance(effect, BattleEffect)
        )
    world.finalize_collision_frame()


def _handle_area_damage(
    game_objects,
    effects,
    excluded_ids=frozenset(),
    *,
    frame=None,
    spatial_index=None,
    shadow_validate=False,
    metrics=None,
    use_spatial_queries=True,
):
    world = World.coerce(game_objects)
    frame = frame or world.collision_snapshot()
    area_abilities = frame.pending_area_damage
    frame_positions = _group_positions(frame.objects)

    for ability in area_abilities:
        ability.area_damage_pending = False
        if metrics is not None:
            metrics.possible_area_targets += len(frame.objects)

        maximum_radius = _maximum_area_damage_radius(ability)
        bounded_spatial_query = (
            use_spatial_queries
            and spatial_index is not None
            and maximum_radius is not None
        )
        if (
            spatial_index is not None
            and maximum_radius is None
            and not bounded_spatial_query
            and metrics is not None
        ):
            metrics.area_full_scan_fallbacks += 1

        start_index = 0
        while start_index < len(frame.objects):
            if bounded_spatial_query:
                targets = _ordered_area_candidates(
                    spatial_index,
                    frame.objects,
                    ability.position,
                    maximum_radius,
                    start_index,
                    frame_positions,
                )
            else:
                targets = [
                    (index, frame.objects[index])
                    for index in range(start_index, len(frame.objects))
                ]

            if shadow_validate and bounded_spatial_query:
                _assert_no_omitted_area_target(
                    ability,
                    frame.objects,
                    targets,
                    excluded_ids,
                    start_index,
                )

            if metrics is not None:
                metrics.area_candidates += len(targets)
            if not targets:
                break

            must_requery = False
            for target_index, target in targets:
                start_index = target_index + 1
                if id(target) in excluded_ids:
                    continue
                if not AREA_TARGET_REGISTRY.is_eligible(ability, target):
                    continue

                delta = _wrapped_delta(ability.position, target.position)
                distance = math.hypot(delta[0], delta[1])
                damage = ability.area_damage_for_target(target, distance)
                if damage <= 0:
                    continue

                previous_ability_position = _position_tuple(ability)
                applied_damage = AREA_TARGET_REGISTRY.apply_damage(
                    ability,
                    target,
                    effects,
                    delta,
                    distance,
                    damage,
                )
                ability.on_area_damage_hit(target, applied_damage)
                append_effects = getattr(ability, "append_area_hit_effects", None)
                if append_effects is not None:
                    append_effects(target, effects, delta, distance, damage)
                if ability.area_damage_capabilities.plays_impact_sound:
                    BattleEffect.play_boom(damage)
                invalidate_geometry(ability)
                invalidate_geometry(target)

                ability_cells_changed = False
                if spatial_index is not None:
                    if getattr(ability, "currently_alive", True):
                        ability_cells_changed = spatial_index.update(ability)
                    if getattr(target, "currently_alive", True) and getattr(
                        target, "current_hp", 1
                    ) > 0:
                        spatial_index.update(target)
                if not ability.currently_alive:
                    break

                if bounded_spatial_query:
                    next_maximum_radius = _maximum_area_damage_radius(ability)
                    if next_maximum_radius is None:
                        # A formerly bounded effect became undiscoverable; the
                        # unvisited suffix must safely fall back to a full scan.
                        bounded_spatial_query = False
                        if metrics is not None:
                            metrics.area_full_scan_fallbacks += 1
                        must_requery = True
                    elif (
                        ability_cells_changed
                        or previous_ability_position != _position_tuple(ability)
                        or next_maximum_radius != maximum_radius
                    ):
                        maximum_radius = next_maximum_radius
                        must_requery = True
                    if must_requery:
                        break

            if not ability.currently_alive or not must_requery:
                break

        ability.area_damage_pending = bool(
            ability.area_damage_capabilities.persistent and ability.currently_alive
        )


def _ordered_area_candidates(
    spatial_index,
    all_targets,
    position,
    maximum_radius,
    start_index,
    positions_by_id,
):
    nearby_indices = set()
    for target in spatial_index.query_radius(
        position,
        maximum_radius,
        categories=("area_targets",),
    ):
        nearby_indices.update(
            index
            for index in positions_by_id.get(id(target), ())
            if index >= start_index
        )
    return [
        (index, all_targets[index])
        for index in sorted(nearby_indices)
    ]


def _maximum_area_damage_radius(ability):
    provider = getattr(ability, "maximum_area_damage_radius", None)
    if provider is None:
        return None
    maximum_radius = provider()
    if (
        not isinstance(maximum_radius, (int, float))
        or not math.isfinite(maximum_radius)
        or maximum_radius < 0
    ):
        return None
    return float(maximum_radius)


def _assert_no_omitted_area_target(
    ability,
    all_targets,
    candidates,
    excluded_ids,
    start_index,
):
    candidate_ids = {id(target) for _, target in candidates}
    for target_index in range(start_index, len(all_targets)):
        target = all_targets[target_index]
        if id(target) in candidate_ids or id(target) in excluded_ids:
            continue
        if not AREA_TARGET_REGISTRY.is_eligible(ability, target):
            continue
        delta = _wrapped_delta(ability.position, target.position)
        distance = math.hypot(delta[0], delta[1])
        if ability.area_damage_for_target(target, distance) > 0:
            raise AssertionError(
                "spatial broad phase omitted an area-damage hit: "
                f"{ability!r} -> {target!r}"
            )


def _run_collision_phases(
    groups,
    effects,
    *,
    spatial_index=None,
    shadow_validate=False,
    metrics=None,
):
    """Run physical contact phases in their gameplay-significant order."""
    spatial_ordering = (
        {
            name: (_group_positions(objects), _intangible_indices(objects))
            for name, objects in groups.items()
        }
        if spatial_index is not None
        else {}
    )
    for phase in COLLISION_PHASES:
        phase_effects = effects if phase.collects_effects else []
        environment = (
            CollisionEnvironment(ships=tuple(groups["ships"]))
            if phase.includes_ship_environment
            else None
        )
        first_objects = groups[phase.first_group]

        if metrics is not None:
            if phase.unique_pairs:
                active = phase.first_is_active or _always_active
                metrics.possible_physical_pairs += sum(
                    len(first_objects) - index - 1
                    for index, first in enumerate(first_objects)
                    if active(first)
                )
            else:
                metrics.possible_physical_pairs += len(first_objects) * len(
                    groups[phase.second_group]
                )

        if phase.unique_pairs:
            positions_by_id, intangible_indices = spatial_ordering.get(
                phase.first_group,
                (None, None),
            )
            _dispatch_unique_collision_pairs(
                first_objects,
                phase_effects,
                phase.first_is_active or _always_active,
                environment,
                spatial_index=spatial_index,
                category=phase.first_group,
                shadow_validate=shadow_validate,
                metrics=metrics,
                positions_by_id=positions_by_id,
                intangible_indices=intangible_indices,
            )
            continue

        positions_by_id, intangible_indices = spatial_ordering.get(
            phase.second_group,
            (None, None),
        )
        _dispatch_collision_pairs(
            first_objects,
            groups[phase.second_group],
            phase_effects,
            stop_after_handled=phase.stop_after_handled,
            environment=environment,
            spatial_index=spatial_index,
            second_category=phase.second_group,
            shadow_validate=shadow_validate,
            metrics=metrics,
            positions_by_id=positions_by_id,
            intangible_indices=intangible_indices,
        )


def _handle_laser_collisions(
    lasers,
    ships,
    projectiles,
    special_objects,
    asteroids,
    planets,
    effects,
    excluded_ids=frozenset(),
    area_abilities=(),
    spatial_index=None,
    shadow_validate=False,
    metrics=None,
    use_spatial_queries=True,
):
    target_groups = (
        ships,
        projectiles,
        special_objects,
        area_abilities,
        asteroids,
        planets,
    )
    target_group_positions = tuple(
        _group_positions(group) for group in target_groups
    )
    for laser in lasers:
        if not responses.is_live_laser(laser):
            continue

        laser.position = laser.parent.position.copy()
        laser.calculate_end_position()
        if laser.target is not None:
            target_delta = _wrapped_delta(laser.position, laser.target.position)
            laser.end_position = [
                laser.position[0] + target_delta[0],
                laser.position[1] + target_delta[1],
            ]

        segments = _laser_collision_segments(laser)
        spatial_candidates = None
        if spatial_index is not None and use_spatial_queries:
            spatial_candidates = spatial_index.query_segments(
                segments,
                width=getattr(laser, "LASER_WIDTH", 1),
                categories=("laser_targets",),
            )

        if metrics is not None:
            metrics.possible_laser_targets += _possible_laser_target_count(
                laser,
                ships,
                projectiles,
                special_objects,
                area_abilities,
                asteroids,
                planets,
            )
        targets = _laser_targets(
            laser,
            ships,
            projectiles,
            special_objects,
            area_abilities,
            asteroids,
            planets,
            excluded_ids,
            spatial_candidates=spatial_candidates,
            spatial_group_positions=target_group_positions,
        )
        if metrics is not None:
            metrics.laser_candidates += len(targets)

        if shadow_validate and spatial_candidates is not None:
            brute_targets = _laser_targets(
                laser,
                ships,
                projectiles,
                special_objects,
                area_abilities,
                asteroids,
                planets,
                excluded_ids,
            )
            _assert_no_omitted_laser_hit(
                laser,
                brute_targets,
                targets,
                segments,
            )

        hit_infos = []
        for target in targets:
            with pair_geometry_cache():
                hit_info = laser_hit_info(
                    laser,
                    target,
                    raw_segments=segments,
                )
            if hit_info is not None:
                hit_infos.append(hit_info)
        if not hit_infos:
            continue

        for hit_info in sorted(hit_infos, key=lambda info: info["distance"]):
            target = hit_info["target"]
            responses.resolve_laser_hit(
                laser,
                target,
                effects,
                hit_info["normal"],
                hit_info["contact"],
                lambda target, effects, normal, damage, contact, laser=laser: (
                    _apply_laser_impact(
                        laser,
                        target,
                        effects,
                        normal,
                        damage,
                        contact,
                    )
                ),
                segment_index=hit_info.get("segment_index"),
            )
            invalidate_geometry(target)
            if (
                spatial_index is not None
                and getattr(target, "currently_alive", True)
                and getattr(target, "current_hp", 1) > 0
            ):
                spatial_index.update(target)
            target_capabilities = getattr(
                target, "laser_target_capabilities", None
            )
            if (
                target_capabilities is None
                or getattr(target_capabilities, "blocks_lasers", True)
            ):
                break


def _laser_collision_segments(laser):
    provider = getattr(laser, "collision_segments", None)
    if provider is not None:
        return tuple(provider())
    start = getattr(laser, "start_position", laser.parent.position)
    end = getattr(laser, "end_position", laser.position)
    return ((start, end),)


def _possible_laser_target_count(
    laser,
    ships,
    projectiles,
    special_objects,
    area_abilities,
    asteroids,
    planets,
):
    if laser.target is not None:
        return 1 + len(special_objects) + len(area_abilities)
    return sum(
        len(group)
        for group in (
            ships,
            projectiles,
            special_objects,
            area_abilities,
            asteroids,
            planets,
        )
    )


def _assert_no_omitted_laser_hit(
    laser,
    brute_targets,
    spatial_targets,
    segments,
):
    candidate_ids = {id(target) for target in spatial_targets}
    for target in brute_targets:
        if id(target) in candidate_ids:
            continue
        with pair_geometry_cache():
            hit = laser_hit_info(laser, target, raw_segments=segments)
        if hit is not None:
            raise AssertionError(
                "spatial broad phase omitted an exact laser hit: "
                f"{laser!r} -> {target!r}"
            )


def _apply_laser_impact(laser, target, effects, normal, damage, contact):
    LASER_TARGET_REGISTRY.apply_impact(
        target,
        effects,
        normal,
        damage,
        contact,
        source=laser,
    )


def _laser_targets(
    laser,
    ships,
    projectiles,
    special_objects,
    area_abilities,
    asteroids,
    planets,
    excluded_ids=frozenset(),
    spatial_candidates=None,
    spatial_group_positions=None,
):
    groups = (
        ships,
        projectiles,
        special_objects,
        area_abilities,
        asteroids,
        planets,
    )
    explicit_target = laser.target
    if explicit_target is not None:
        targets = (
            [explicit_target]
            if (
                id(explicit_target) not in excluded_ids
                and _laser_target_is_eligible(laser, explicit_target, explicit=True)
            )
            else []
        )
        blockers = (
            chain(special_objects, area_abilities)
            if spatial_candidates is None
            else _ordered_laser_spatial_targets(
                spatial_candidates,
                groups,
                spatial_group_positions,
                group_indices=(2, 3),
            )
        )
        targets.extend(
            blocker
            for blocker in blockers
            if (
                blocker is not explicit_target
                and _laser_target_is_eligible(laser, blocker)
            )
        )
        return targets

    target_pool = (
        chain(*groups)
        if spatial_candidates is None
        else _ordered_laser_spatial_targets(
            spatial_candidates,
            groups,
            spatial_group_positions,
        )
    )
    targets = [
        target
        for target in target_pool
        if (
            id(target) not in excluded_ids
            and _laser_target_is_eligible(laser, target)
        )
    ]
    return targets


def _ordered_laser_spatial_targets(
    spatial_candidates,
    groups,
    group_positions,
    *,
    group_indices=None,
):
    if group_positions is None:
        group_positions = tuple(_group_positions(group) for group in groups)
    if group_indices is None:
        group_indices = range(len(groups))

    selected = []
    for group_index in group_indices:
        positions = []
        for target in spatial_candidates:
            positions.extend(
                group_positions[group_index].get(id(target), ())
            )
        if any(
            positions[index - 1] > positions[index]
            for index in range(1, len(positions))
        ):
            # This only occurs for duplicate references separated in the
            # authoritative group.  Ordinary world-ordered groups stay on the
            # allocation-light fast path.
            positions.sort()
        selected.extend(groups[group_index][position] for position in positions)
    return selected


def _laser_target_is_eligible(laser, target, explicit=False):
    target_filter = getattr(laser, "should_consider_laser_target", None)
    if target_filter is not None and not target_filter(target):
        return False
    return LASER_TARGET_REGISTRY.is_eligible(
        laser,
        target,
        explicit=explicit,
    )


def _spawn_replacement_asteroids(
    game_objects,
    asteroids,
    ships,
    planets,
    *,
    rng=None,
    resources=None,
):
    world = World.coerce(game_objects)
    if not planets:
        return

    dead_count = sum(1 for asteroid in asteroids if not asteroid.currently_alive)
    if dead_count <= 0:
        return

    planet = planets[0]
    avoid_bodies = world.asteroid_spawn_avoid_bodies

    for _ in range(dead_count):
        if resources is None and rng is None:
            asteroid = Asteroid()
        else:
            asteroid = Asteroid(resources=resources, rng=rng)
        asteroid.set_planet(planet)
        asteroid.position = asteroid.get_respawn_position(planet, ships, avoid_bodies)
        asteroid.previous_position = asteroid.position.copy()
        avoid_bodies.append(asteroid)
        world.add(asteroid)
