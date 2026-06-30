"""Gameplay policies applied after collision geometry identifies contact."""

import math

import src.const as const
from src.Battle.collision_contract import (
    CollisionOutcome,
    collision_context,
)
from src.Battle.collision_geometry import (
    collision_info,
    collision_size,
    projectile_impact,
    radius,
    solid_sweep_overlap,
    swept_overlap_positions,
)
from src.Battle.collision_physics import (
    bounce_off_static_body,
    elastic_bounce,
    stop_at_static_body,
)
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.Objects.Ships.ability import Ability
from src.toroidal import view_center_and_size, wrapped_delta

PLANET_CONTACT_EXIT_MARGIN = 4.0


def damage_ship(ship, damage, *, shieldable=True, non_lethal=False):
    """Route combat damage through a ship's defenses.

    The fallback keeps lightweight collision test doubles compatible.
    """
    take_damage = getattr(ship, "take_damage", None)
    if take_damage is not None:
        return take_damage(damage, shieldable=shieldable, non_lethal=non_lethal)
    previous_hp = ship.current_hp
    min_hp = 1 if non_lethal else 0
    ship.current_hp = max(min_hp, ship.current_hp - max(0, damage))
    return previous_hp - ship.current_hp


def ship_is_area_target(source, ship):
    if not _area_target_is_active(source, ship):
        return False
    if ship is getattr(source, "parent", None):
        return getattr(source, "hit_parent", False)
    if getattr(source, "is_psychic", False):
        durability = getattr(ship, "durability_capabilities", None)
        if durability and durability.immune_to_psychic:
            return False
    capabilities = source.special_object_collision_capabilities
    if ship.player == source.player:
        return capabilities.collides_with_friendly_ships
    return capabilities.collides_with_enemy_ships


def projectile_is_area_target(source, projectile):
    if not _area_target_is_active(source, projectile):
        return False
    return source.special_object_collision_capabilities.collides_with_projectiles


def special_object_is_area_target(source, special_object):
    return projectile_is_area_target(source, special_object)


def asteroid_is_area_target(source, asteroid):
    if not _area_target_is_active(source, asteroid):
        return False
    return source.special_object_collision_capabilities.collides_with_asteroids


def planet_is_area_target(source, planet):
    if not _area_target_is_active(source, planet):
        return False
    return source.special_object_collision_capabilities.collides_with_planets


def _area_target_is_active(source, target):
    if target is source or not getattr(target, "currently_alive", True):
        return False
    target_capabilities = getattr(target, "area_damage_capabilities", None)
    if not target_capabilities or not target_capabilities.vulnerable:
        return False
    physics = getattr(target, "physical_collision_capabilities", None)
    if physics is None or physics.is_intangible:
        return False
    return getattr(source, "special_object_collision_capabilities", None) is not None


def apply_ship_area_damage(source, ship, effects, delta, distance, damage):
    shieldable = not getattr(source, "ignores_shields", False)
    non_lethal = getattr(source, "is_psychic", False)
    return damage_ship(
        ship,
        damage,
        shieldable=shieldable,
        non_lethal=non_lethal,
    )


def apply_projectile_area_damage(
    source,
    projectile,
    effects,
    delta,
    distance,
    damage,
):
    previous_hp = projectile.current_hp
    remaining_hp = projectile.current_hp - damage
    if remaining_hp <= 0:
        direction = (
            [delta[0] / distance, delta[1] / distance]
            if distance > 0
            else [0, -1]
        )
        destroy_projectile(projectile, effects, direction, damage)
    else:
        set_projectile_hp(projectile, remaining_hp)
    return previous_hp - projectile.current_hp


def apply_special_object_area_damage(
    source,
    special_object,
    effects,
    delta,
    distance,
    damage,
):
    return apply_projectile_area_damage(
        source,
        special_object,
        effects,
        delta,
        distance,
        damage,
    )


def apply_asteroid_area_damage(
    source,
    asteroid,
    effects,
    delta,
    distance,
    damage,
):
    was_alive = getattr(asteroid, "currently_alive", True)
    destroy_asteroid(asteroid, effects)
    return damage if was_alive and not asteroid.currently_alive else 0


def apply_planet_area_damage(source, planet, effects, delta, distance, damage):
    handler = getattr(source, "on_planet_area_hit", None)
    if handler is not None:
        handler(planet, effects, delta, distance, damage)
    return 0


def _resolve_mobile_solid_contact(
    first,
    second,
):
    """Apply shared overlap and bounce behavior for movable solid bodies."""
    phys_first = getattr(first, "physical_collision_capabilities", None)
    phys_second = getattr(second, "physical_collision_capabilities", None)
    if (
        not phys_first
        or not phys_second
        or not phys_first.is_solid
        or not phys_second.is_solid
        or phys_first.is_immovable
        or phys_second.is_immovable
    ):
        return False

    if not getattr(first, "currently_alive", True) or not getattr(
        second, "currently_alive", True
    ):
        return False

    if not solid_sweep_overlap(first, second):
        return False

    normal, distance, overlap = collision_info(first, second)

    elastic_bounce(first, second, normal, distance, overlap)
    for obj, other in ((first, second), (second, first)):
        on_elastic_bounce = getattr(obj, "on_elastic_bounce", None)
        if on_elastic_bounce is not None:
            on_elastic_bounce(other)

    return True


def _apply_ramming_damage_to_ship(source, ship):
    impact = getattr(source, "impact_capabilities", None)
    if impact and impact.ramming_damage > 0:
        damage_ship(ship, impact.ramming_damage)
        BattleEffect.play_boom(impact.ramming_damage)


def _apply_ramming_damage_to_asteroid(source, asteroid, effects):
    impact = getattr(source, "impact_capabilities", None)
    if impact and impact.ramming_damage > 0:
        destroy_asteroid(asteroid, effects)
        BattleEffect.play_boom(impact.ramming_damage)


def resolve_ship_ship_collision(
    first_ship,
    second_ship,
    context_or_effects,
    environment=None,
):
    """Bounce two ships and exchange ship-specific ramming payloads."""
    if not _resolve_mobile_solid_contact(first_ship, second_ship):
        return CollisionOutcome.IGNORED

    _apply_ramming_damage_to_ship(first_ship, second_ship)
    _apply_ramming_damage_to_ship(second_ship, first_ship)
    return CollisionOutcome.RESOLVED


def resolve_asteroid_asteroid_collision(
    first_asteroid,
    second_asteroid,
    context_or_effects,
    environment=None,
):
    """Bounce two asteroids and exchange asteroid-specific ramming payloads."""
    context = collision_context(context_or_effects, environment)
    if not _resolve_mobile_solid_contact(first_asteroid, second_asteroid):
        return CollisionOutcome.IGNORED

    _apply_ramming_damage_to_asteroid(
        first_asteroid,
        second_asteroid,
        context.effects,
    )
    _apply_ramming_damage_to_asteroid(
        second_asteroid,
        first_asteroid,
        context.effects,
    )
    return CollisionOutcome.RESOLVED


def resolve_ship_asteroid_collision(
    ship,
    asteroid,
    context_or_effects,
    environment=None,
):
    """Bounce a ship and asteroid and apply payloads to explicit target roles."""
    context = collision_context(context_or_effects, environment)
    if not _resolve_mobile_solid_contact(ship, asteroid):
        return CollisionOutcome.IGNORED

    _apply_ramming_damage_to_asteroid(ship, asteroid, context.effects)
    _apply_ramming_damage_to_ship(asteroid, ship)
    return CollisionOutcome.RESOLVED


def resolve_asteroid_planet_collision(
    asteroid,
    planet,
    context_or_effects,
    environment=None,
):
    """Destroy an asteroid that contacts a planet."""
    context = collision_context(context_or_effects, environment)
    asteroid_physics = getattr(asteroid, "physical_collision_capabilities", None)
    planet_physics = getattr(planet, "physical_collision_capabilities", None)
    if (
        not asteroid_physics
        or not planet_physics
        or not asteroid_physics.fragile_to_immovable
        or not planet_physics.is_immovable
        or not getattr(asteroid, "currently_alive", True)
    ):
        return CollisionOutcome.IGNORED

    if not solid_sweep_overlap(asteroid, planet):
        return CollisionOutcome.IGNORED

    on_screen = context.object_on_screen_policy or object_on_screen
    if on_screen(asteroid, context.environment.ships):
        BattleEffect.play_boom(1)

    destroy_asteroid(asteroid, context.effects)
    return CollisionOutcome.CONSUMED_FIRST


def resolve_ship_planet_collision(
    ship,
    planet,
    context_or_effects,
    environment=None,
):
    """Bounce a ship from a planet and exchange configured impact payloads."""
    context = collision_context(context_or_effects, environment)
    ship_physics = getattr(ship, "physical_collision_capabilities", None)
    planet_physics = getattr(planet, "physical_collision_capabilities", None)
    if (
        not ship_physics
        or not planet_physics
        or not ship_physics.bounces_on_immovable
        or not planet_physics.is_immovable
    ):
        return CollisionOutcome.IGNORED

    overlaps = solid_sweep_overlap(ship, planet)
    normal, distance, overlap = collision_info(ship, planet)
    contact_id = id(planet)

    contacts_set = getattr(ship, "planet_contacts", None)
    if contacts_set is None:
        contacts_set = set()
        setattr(ship, "planet_contacts", contacts_set)

    if contact_id in contacts_set and not overlaps:
        contact_distance = radius(ship) + radius(planet)
        if distance > contact_distance + PLANET_CONTACT_EXIT_MARGIN:
            contacts_set.remove(contact_id)

    if not overlaps:
        return CollisionOutcome.IGNORED

    new_contact = contact_id not in contacts_set
    contacts_set.add(contact_id)

    if new_contact:
        collided_while_approaching = bounce_off_static_body(
            ship,
            planet,
            normal,
            overlap,
        )
        if collided_while_approaching and not getattr(ship, "inertia", True):
            ship.collision_velocity = ship.velocity.copy()
    else:
        collided_while_approaching = False
        stop_at_static_body(ship, planet, normal, overlap)

    if new_contact and collided_while_approaching:
        planet_impact = getattr(planet, "impact_capabilities", None)
        if planet_impact and planet_impact.impact_damage_percent > 0:
            if getattr(ship, "current_hp", 0) > 0:
                damage = max(
                    1,
                    math.ceil(
                        ship.current_hp * planet_impact.impact_damage_percent
                    ),
                )
                take_planet_damage = getattr(
                    ship, "take_planet_impact_damage", None
                )
                if take_planet_damage is None:
                    damage_ship(ship, damage)
                else:
                    take_planet_damage(damage)
                BattleEffect.play_boom(damage)

        ship_impact = getattr(ship, "impact_capabilities", None)
        if ship_impact and ship_impact.ramming_damage > 0:
            BattleEffect.play_boom(ship_impact.ramming_damage)

    return CollisionOutcome.RESOLVED


def resolve_projectile_projectile_collision(
    first,
    second,
    context_or_effects,
    environment=None,
):
    """Resolve contact between two projectile-like objects."""
    context = collision_context(context_or_effects, environment)
    effects = context.effects
    if not is_live_projectile_like(first) or not is_live_projectile_like(second):
        return CollisionOutcome.IGNORED
    if not projectile_like_objects_can_hit_each_other(first, second):
        return CollisionOutcome.IGNORED

    _, _, overlap = collision_info(first, second)
    contact, impact_normal = projectile_impact(first, second, overlap)
    if contact is None:
        return CollisionOutcome.IGNORED

    first_handler = getattr(first, "handle_projectile_contact", None)
    second_handler = getattr(second, "handle_projectile_contact", None)
    first_handled = bool(first_handler and first_handler(second))
    second_handled = bool(second_handler and second_handler(first))

    if first_handled or second_handled:
        BattleEffect.play_boom(max(first.current_damage, second.current_damage))
        if first.current_hp <= 0:
            destroy_projectile(
                first,
                effects,
                impact_normal,
                first.current_damage,
                contact,
            )
        if second.current_hp <= 0:
            destroy_projectile(
                second,
                effects,
                [-impact_normal[0], -impact_normal[1]],
                second.current_damage,
                contact,
            )
        return _consumption_outcome(first, second)

    if first.projectile_name == second.projectile_name:
        first_caps = getattr(first, "special_object_collision_capabilities", None)
        second_caps = getattr(second, "special_object_collision_capabilities", None)
        if (
            first_caps
            and first_caps.bounces_off_same_type
            and second_caps
            and second_caps.bounces_off_same_type
        ):
            normal, distance, actual_overlap = collision_info(first, second)
            elastic_bounce(first, second, normal, distance, actual_overlap)
            return CollisionOutcome.RESOLVED

        BattleEffect.play_boom(max(first.current_damage, second.current_damage))
        destroy_projectile(
            first,
            effects,
            impact_normal,
            first.current_damage,
            contact,
        )
        destroy_projectile(
            second,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            second.current_damage,
            contact,
        )
        return CollisionOutcome.CONSUMED_BOTH

    first_caps = getattr(first, "special_object_collision_capabilities", None)
    second_caps = getattr(second, "special_object_collision_capabilities", None)
    first_damage = (
        first.current_damage
        if not first_caps or first_caps.damages_projectiles
        else 0
    )
    second_damage = (
        second.current_damage
        if not second_caps or second_caps.damages_projectiles
        else 0
    )

    first_hp = first.current_hp - second_damage
    second_hp = second.current_hp - first_damage

    BattleEffect.play_boom(max(first.current_damage, second.current_damage))

    if first_hp <= 0 and second_hp <= 0:
        destroy_projectile(
            first,
            effects,
            impact_normal,
            first.current_damage,
            contact,
        )
        destroy_projectile(
            second,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            second.current_damage,
            contact,
        )
    elif first_hp > 0 and first_hp > second_hp:
        set_projectile_hp(first, first_hp)
        destroy_projectile(
            second,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            second.current_damage,
            contact,
        )
    elif second_hp > 0 and second_hp > first_hp:
        destroy_projectile(
            first,
            effects,
            impact_normal,
            first.current_damage,
            contact,
        )
        set_projectile_hp(second, second_hp)
    else:
        destroy_projectile(
            first,
            effects,
            impact_normal,
            first.current_damage,
            contact,
        )
        destroy_projectile(
            second,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            second.current_damage,
            contact,
        )
    return _consumption_outcome(first, second)


def resolve_projectile_planet_collision(
    projectile,
    planet,
    context_or_effects,
    environment=None,
):
    """Resolve projectile contact with an immovable planet."""
    context = collision_context(context_or_effects, environment)
    if not is_live_projectile_like(projectile):
        return CollisionOutcome.IGNORED

    capabilities = getattr(
        projectile,
        "special_object_collision_capabilities",
        None,
    )
    if capabilities and not capabilities.collides_with_planets:
        return CollisionOutcome.IGNORED

    _, _, overlap = collision_info(projectile, planet)
    contact, impact_normal = projectile_impact(projectile, planet, overlap)
    if contact is None:
        return CollisionOutcome.IGNORED

    handler = getattr(projectile, "handle_planet_contact", None)
    if handler and handler(planet, impact_normal, overlap):
        return _consumption_outcome(projectile, planet)

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    destroy_projectile(
        projectile,
        context.effects,
        impact_normal,
        damage,
        contact,
        planet,
    )
    return CollisionOutcome.CONSUMED_FIRST


def resolve_projectile_ship_collision(
    projectile,
    ship,
    context_or_effects,
    environment=None,
):
    """Resolve projectile contact with a ship."""
    context = collision_context(context_or_effects, environment)
    if (
        not is_live_projectile_like(projectile)
        or not getattr(ship, "currently_alive", True)
        or getattr(ship, "current_hp", 1) <= 0
        or not projectile_can_hit_ship(projectile, ship)
    ):
        return CollisionOutcome.IGNORED

    _, _, overlap = collision_info(projectile, ship)
    contact, impact_normal = projectile_impact(projectile, ship, overlap)
    if contact is None:
        return CollisionOutcome.IGNORED

    capabilities = getattr(
        projectile,
        "special_object_collision_capabilities",
        None,
    )
    incoming_handler = getattr(
        ship,
        "handle_incoming_special_object_contact",
        None,
    )
    if capabilities and incoming_handler and incoming_handler(
        projectile,
        impact_normal,
    ):
        return _consumption_outcome(projectile, ship)

    if ship is projectile.parent:
        can_recover = getattr(projectile, "can_recover_with_parent", None)
        if can_recover and can_recover():
            recover = getattr(projectile, "recover_with_parent", None)
            if recover:
                recover()
                return _consumption_outcome(projectile, ship)

    handler = getattr(projectile, "handle_ship_contact", None)
    if handler and handler(ship, impact_normal):
        return _consumption_outcome(projectile, ship)

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    damage_ship(ship, damage)
    projectile.on_ship_impact(ship)
    motion_handler = getattr(ship, "on_projectile_motion_contact", None)
    if motion_handler is not None:
        _, ship_contact_position = swept_overlap_positions(projectile, ship)
        motion_handler(projectile, ship_contact_position)
    attached = ship if ship.current_hp > 0 else None
    destroy_projectile(
        projectile,
        context.effects,
        impact_normal,
        damage,
        contact,
        attached,
    )
    return _consumption_outcome(projectile, ship)


def resolve_projectile_asteroid_collision(
    projectile,
    asteroid,
    context_or_effects,
    environment=None,
):
    """Resolve projectile contact with an asteroid."""
    context = collision_context(context_or_effects, environment)
    if (
        not is_live_projectile_like(projectile)
        or not getattr(asteroid, "currently_alive", True)
        or getattr(asteroid, "current_hp", 1) <= 0
    ):
        return CollisionOutcome.IGNORED

    capabilities = getattr(
        projectile,
        "special_object_collision_capabilities",
        None,
    )
    if capabilities and not capabilities.collides_with_asteroids:
        return CollisionOutcome.IGNORED

    _, _, overlap = collision_info(projectile, asteroid)
    contact, impact_normal = projectile_impact(projectile, asteroid, overlap)
    if contact is None:
        return CollisionOutcome.IGNORED

    handler = getattr(projectile, "handle_asteroid_contact", None)
    if handler and handler(asteroid, impact_normal):
        return _consumption_outcome(projectile, asteroid)

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    attached = (
        asteroid
        if capabilities and not capabilities.damages_asteroids
        else None
    )
    destroy_projectile(
        projectile,
        context.effects,
        impact_normal,
        damage,
        contact,
        attached,
    )
    if not capabilities or capabilities.damages_asteroids:
        destroy_asteroid(asteroid, context.effects)
    return _consumption_outcome(projectile, asteroid)


def _consumption_outcome(first, second):
    first_consumed = (
        not getattr(first, "currently_alive", True)
        or getattr(first, "current_hp", 1) <= 0
    )
    second_consumed = (
        not getattr(second, "currently_alive", True)
        or getattr(second, "current_hp", 1) <= 0
    )
    if first_consumed and second_consumed:
        return CollisionOutcome.CONSUMED_BOTH
    if first_consumed:
        return CollisionOutcome.CONSUMED_FIRST
    if second_consumed:
        return CollisionOutcome.CONSUMED_SECOND
    return CollisionOutcome.RESOLVED

def projectile_like_objects_can_hit_each_other(first, second):
    first_is_special_object = is_live_special_object(first)
    second_is_special_object = is_live_special_object(second)
    first_capabilities = getattr(
        first,
        "special_object_collision_capabilities",
        None,
    )
    second_capabilities = getattr(
        second,
        "special_object_collision_capabilities",
        None,
    )

    if first_is_special_object and second_is_special_object:
        first_hits = first_capabilities.collides_with_fighters
        second_hits = second_capabilities.collides_with_fighters
        if not first_hits and not second_hits:
            return False
        return True

    if first_is_special_object and not second_is_special_object:
        if not first_capabilities.collides_with_projectiles:
            return False
        return True

    if not first_is_special_object and second_is_special_object:
        if not second_capabilities.collides_with_projectiles:
            return False
        return True

    if first.player != second.player:
        return True

    if first.projectile_name == second.projectile_name:
        return getattr(first, "hit_self", False) and getattr(
            second,
            "hit_self",
            False,
        )

    return getattr(first, "hit_team", False) and getattr(
        second,
        "hit_team",
        False,
    )


def is_live_projectile_like(obj):
    return (
        World.is_colliding_ability_kind(obj, "projectile")
        or World.is_colliding_ability_kind(obj, "special_object")
    )


def is_live_special_object(obj):
    return World.is_colliding_ability_kind(obj, "special_object")


def is_live_laser(obj):
    return World.is_colliding_ability_kind(obj, "laser")


def projectile_can_hit_ship(projectile, ship):
    collision_filter = getattr(projectile, "should_collide_with_ship", None)
    if collision_filter is not None:
        return collision_filter(ship)

    if ship is projectile.parent:
        can_recover = getattr(projectile, "can_recover_with_parent", None)
        if can_recover is not None and can_recover():
            return True
        return projectile.hit_parent

    special_object_capabilities = getattr(
        projectile,
        "special_object_collision_capabilities",
        None,
    )
    if special_object_capabilities:
        if ship.player == projectile.player:
            return special_object_capabilities.collides_with_friendly_ships
        return special_object_capabilities.collides_with_enemy_ships

    if ship.player != projectile.player:
        return True

    return False


def ship_is_laser_target(laser, ship, explicit=False):
    if not _laser_target_is_active(laser, ship, require_targetable=True):
        return False
    if getattr(ship, "current_hp", 1) <= 0:
        return False
    return explicit or ship.player != laser.player


def projectile_is_laser_target(laser, projectile, explicit=False):
    if not _laser_target_is_active(laser, projectile, require_targetable=True):
        return False
    return (
        getattr(projectile, "can_collide", True)
        and getattr(projectile, "current_hp", 1) > 0
    )


def special_object_is_laser_target(laser, special_object, explicit=False):
    if not _laser_target_is_active(
        laser,
        special_object,
        require_targetable=False,
    ):
        return False
    return (
        getattr(special_object, "can_collide", True)
        and getattr(special_object, "current_hp", 1) > 0
    )


def asteroid_is_laser_target(laser, asteroid, explicit=False):
    return _laser_target_is_active(laser, asteroid, require_targetable=True)


def planet_is_laser_target(laser, planet, explicit=False):
    return _laser_target_is_active(laser, planet, require_targetable=True)


def area_is_laser_target(laser, area, explicit=False):
    return _laser_target_is_active(laser, area, require_targetable=True)


def _laser_target_is_active(laser, target, *, require_targetable):
    if not getattr(target, "currently_alive", True):
        return False
    if target is laser.parent and not laser.hit_parent:
        return False
    physics = getattr(target, "physical_collision_capabilities", None)
    if physics is not None and physics.is_intangible:
        return False
    if require_targetable:
        capabilities = getattr(target, "laser_target_capabilities", None)
        if capabilities is None or not capabilities.targetable:
            return False
    return True


def apply_ship_laser_impact(ship, effects, normal, damage, contact):
    damage_ship(ship, damage)


def apply_projectile_laser_impact(projectile, effects, normal, damage, contact):
    set_projectile_hp(projectile, projectile.current_hp - damage)
    if projectile.current_hp <= 0:
        destroy_projectile(projectile, effects, normal, damage, contact)


def apply_special_object_laser_impact(
    special_object,
    effects,
    normal,
    damage,
    contact,
):
    apply_projectile_laser_impact(
        special_object,
        effects,
        normal,
        damage,
        contact,
    )


def apply_asteroid_laser_impact(asteroid, effects, normal, damage, contact):
    destroy_asteroid(asteroid, effects)


def apply_planet_laser_impact(planet, effects, normal, damage, contact):
    return None


def apply_area_laser_impact(area, effects, normal, damage, contact):
    area.set_hp(area.current_hp - damage)


def resolve_laser_hit(
    laser,
    target,
    effects,
    normal,
    contact,
    apply_impact,
    *,
    segment_index=None,
):
    damage = laser.current_damage
    target_capabilities = getattr(target, "laser_target_capabilities", None)
    blocks_laser = (
        target_capabilities is None
        or getattr(target_capabilities, "blocks_lasers", True)
    )
    if blocks_laser:
        laser.end_position = [
            contact[0] % const.ARENA_SIZE,
            contact[1] % const.ARENA_SIZE,
        ]
        laser.intercepted = True
    on_laser_hit = getattr(laser, "on_laser_hit", None)
    if on_laser_hit is not None:
        on_laser_hit(target, contact, segment_index)
    
    should_damage = getattr(laser, "should_damage_target", None)
    if should_damage is None or should_damage(target):
        apply_impact(target, effects, normal, damage, contact)

    attached = target if (
        blocks_laser
        and getattr(target, 'current_hp', 1) > 0
        and getattr(target, 'currently_alive', True)
        and not is_live_projectile_like(target)
    ) else None
    
    laser.attached_target = attached
    if attached:
        laser.target_contact_offset = wrapped_delta(attached.position, contact)
        laser.initial_target_heading = getattr(attached, "heading", 0)

    effects.append(BattleEffect.from_blast(contact, normal, damage, attached_target=attached))
    BattleEffect.play_boom(damage)


def destroy_projectile(projectile, effects, direction, damage, contact_position=None, attached_target=None):
    if getattr(projectile, "_destruction_finalized", False):
        return
    if not projectile.currently_alive and getattr(projectile, "current_hp", 0) > 0:
        return
    projectile._destruction_finalized = True

    effect_position = (
        contact_position if contact_position is not None else projectile.position
    )
    animation = getattr(projectile, "death_animation", None)
    if animation:
        effects.append(
            BattleEffect.from_animation(
                effect_position,
                animation,
                direction_vector=direction,
                align_edge=contact_position is not None,
                attached_target=attached_target,
                video_multiplier=const.VIDEO_FPS_MULTIPLIER
            )
        )
    elif damage > 0:
        effects.append(
            BattleEffect.from_blast(
                effect_position,
                direction,
                damage,
                attached_target=attached_target,
            )
        )

    projectile.current_hp = 0
    projectile.currently_alive = False
    on_destroyed = getattr(projectile, "on_destroyed", None)
    if on_destroyed is not None:
        on_destroyed()


def destroy_asteroid(asteroid, effects):
    if not asteroid.currently_alive:
        return

    if asteroid.death_animation:
        effects.append(
            BattleEffect.from_animation(
                asteroid.position, 
                asteroid.death_animation,
                video_multiplier=const.VIDEO_FPS_MULTIPLIER
            )
        )
    asteroid.currently_alive = False


def object_on_screen(obj, ships):
    if len(ships) != 2:
        return True

    view_center, view_size = view_center_and_size([ship.position for ship in ships])
    delta = wrapped_delta(view_center, obj.position)
    margin = max(collision_size(obj)) / 2
    return (
        abs(delta[0]) <= view_size / 2 + margin
        and abs(delta[1]) <= view_size / 2 + margin
    )


def set_projectile_hp(projectile, hp):
    hp = max(0, hp)
    if isinstance(projectile, Ability):
        projectile.set_hp(hp)
        return

    # Compatibility boundary for collision test doubles.
    set_hp = getattr(projectile, "set_hp", None)
    if set_hp is not None:
        set_hp(hp)
        return
    projectile.current_hp = hp
