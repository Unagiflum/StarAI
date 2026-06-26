"""Gameplay policies applied after collision geometry identifies contact."""

import math

import src.const as const
from src.Battle.collision_geometry import (
    collision_info,
    collision_size,
    objects_overlap,
    projectile_impact,
    radius,
)
from src.Battle.collision_physics import (
    bounce_off_static_body,
    dot,
    elastic_bounce,
    separate_from_static_body,
    stop_at_static_body,
)
from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.collision_capabilities import ShipImpactContext
from src.Objects.Ships.ability import Ability
from src.toroidal import view_center_and_size, wrapped_delta

PLANET_CONTACT_EXIT_MARGIN = 4.0


def damage_ship(ship, damage, *, shieldable=True):
    """Route combat damage through a ship's defenses.

    The fallback keeps lightweight collision test doubles compatible.
    """
    take_damage = getattr(ship, "take_damage", None)
    if take_damage is not None:
        return take_damage(damage, shieldable=shieldable)
    previous_hp = ship.current_hp
    ship.current_hp = max(0, ship.current_hp - max(0, damage))
    return previous_hp - ship.current_hp


def area_damage_target_is_eligible(source, target, impact_policies):
    capabilities = target.area_damage_capabilities
    return (
        target is not source
        and capabilities.targetable
        and capabilities.vulnerable
        and target.currently_alive
        and target.collision_capabilities.role in impact_policies
    )


def area_damage_impacts_ship(target, effects, delta, distance, damage):
    if target.current_hp <= 0:
        return
    damage_ship(target, damage)


def area_damage_impacts_ability(target, effects, delta, distance, damage):
    if target.current_hp <= 0:
        return
    remaining_hp = target.current_hp - damage
    if remaining_hp <= 0:
        direction = (
            [delta[0] / distance, delta[1] / distance] if distance > 0 else [0, -1]
        )
        destroy_projectile(target, effects, direction, damage)
    else:
        set_projectile_hp(target, remaining_hp)


def area_damage_impacts_asteroid(target, effects, delta, distance, damage):
    destroy_asteroid(target, effects)


def ship_impacts_ship(ship, other, effects, environment):
    normal, distance, overlap = collision_info(ship, other)
    if not objects_overlap(ship, other, overlap):
        return False

    closing_speed = max(
        0.0,
        -dot(
            [
                ship.velocity[0] - other.velocity[0],
                ship.velocity[1] - other.velocity[1],
            ],
            normal,
        ),
    )
    impact = ShipImpactContext(
        normal=(normal[0], normal[1]),
        distance=distance,
        overlap=overlap,
        closing_speed=closing_speed,
    )
    other_impact = ShipImpactContext(
        normal=(-normal[0], -normal[1]),
        distance=distance,
        overlap=overlap,
        closing_speed=closing_speed,
    )

    elastic_bounce(ship, other, normal, distance, overlap)

    ship_result = ship.on_ship_impact(other, impact)
    other_result = other.on_ship_impact(ship, other_impact)
    apply_ship_impact_damage(other, ship_result.damage_to_other)
    apply_ship_impact_damage(ship, other_result.damage_to_other)
    return True


def apply_ship_impact_damage(ship, damage):
    damage = max(0.0, damage)
    if damage <= 0 or ship.current_hp <= 0:
        return
    damage_ship(ship, damage)
    BattleEffect.play_boom(damage)


def ship_impacts_asteroid(ship, asteroid, effects, environment):
    if not asteroid.currently_alive:
        return False

    normal, distance, overlap = collision_info(ship, asteroid)
    if not objects_overlap(ship, asteroid, overlap):
        return False

    elastic_bounce(ship, asteroid, normal, distance, overlap)
    return True


def ship_impacts_planet(ship, planet, effects, environment):
    normal, distance, overlap = collision_info(ship, planet)
    contact_id = id(planet)
    overlaps = objects_overlap(ship, planet, overlap)
    if (
        contact_id in ship.planet_contacts
        and not overlaps
        and planet_contact_has_ended(ship, planet, distance)
    ):
        ship.planet_contacts.remove(contact_id)

    if not overlaps:
        return False

    new_contact = contact_id not in ship.planet_contacts
    ship.planet_contacts.add(contact_id)
    if new_contact:
        collided_while_approaching = bounce_off_static_body(
            ship, planet, normal, overlap
        )
        if collided_while_approaching and not ship.inertia:
            ship.collision_velocity = ship.velocity.copy()
    else:
        collided_while_approaching = False
        stop_at_static_body(ship, planet, normal, overlap)
    if new_contact and collided_while_approaching and ship.current_hp > 0:
        damage = max(1, math.ceil(ship.current_hp * 0.15))
        damage_ship(ship, damage)
        BattleEffect.play_boom(damage)
    return True


def planet_contact_has_ended(
    ship, planet, distance, exit_margin=PLANET_CONTACT_EXIT_MARGIN
):
    contact_distance = radius(ship) + radius(planet)
    return distance > contact_distance + exit_margin


def asteroid_impacts_planet(
    asteroid,
    planet,
    effects,
    environment,
    *,
    object_on_screen_policy=None,
):
    if not asteroid.currently_alive:
        return False

    _, _, overlap = collision_info(asteroid, planet)
    if not objects_overlap(asteroid, planet, overlap):
        return False

    on_screen = object_on_screen_policy or object_on_screen
    if on_screen(asteroid, environment.ships):
        BattleEffect.play_boom(1)
    destroy_asteroid(asteroid, effects)
    return True


def projectile_impacts_projectile(projectile, other, effects, environment):
    if not is_live_projectile(other):
        return False

    _, _, overlap = collision_info(projectile, other)
    if not projectiles_can_hit_each_other(projectile, other):
        return False

    contact, impact_normal = projectile_impact(projectile, other, overlap)
    if contact is None:
        return False

    if projectile.projectile_name == other.projectile_name:
        BattleEffect.play_boom(max(projectile.current_damage, other.current_damage))
        destroy_projectile(
            projectile,
            effects,
            impact_normal,
            projectile.current_damage,
            contact,
        )
        destroy_projectile(
            other,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            other.current_damage,
            contact,
        )
        return True

    projectile_damage = projectile.current_damage
    other_damage = other.current_damage
    projectile_hp = projectile.current_hp - other_damage
    other_hp = other.current_hp - projectile_damage

    BattleEffect.play_boom(max(projectile_damage, other_damage))

    if projectile_hp <= 0 and other_hp <= 0:
        destroy_projectile(
            projectile, effects, impact_normal, projectile_damage, contact
        )
        destroy_projectile(
            other,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            other_damage,
            contact,
        )
    elif projectile_hp > 0 and projectile_hp > other_hp:
        set_projectile_hp(projectile, projectile_hp)
        destroy_projectile(
            other,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            other_damage,
            contact,
        )
    elif other_hp > 0 and other_hp > projectile_hp:
        destroy_projectile(
            projectile, effects, impact_normal, projectile_damage, contact
        )
        set_projectile_hp(other, other_hp)
    else:
        destroy_projectile(
            projectile, effects, impact_normal, projectile_damage, contact
        )
        destroy_projectile(
            other,
            effects,
            [-impact_normal[0], -impact_normal[1]],
            other_damage,
            contact,
        )
    return True


def projectiles_can_hit_each_other(projectile, other):
    if projectile.player != other.player:
        return True

    return (
        projectile.projectile_name == other.projectile_name
        and projectile.hit_self
        and other.hit_self
    )


def projectile_impacts_ship(projectile, ship, effects, environment):
    if (
        not is_live_projectile(projectile)
        or ship.current_hp <= 0
        or not projectile_can_hit_ship(projectile, ship)
    ):
        return False

    _, _, overlap = collision_info(projectile, ship)
    contact, impact_normal = projectile_impact(projectile, ship, overlap)
    if contact is None:
        return False

    damage = projectile.current_damage
    damage_ship(ship, damage)
    projectile.on_ship_impact(ship)
    BattleEffect.play_boom(damage)
    attached = ship if ship.current_hp > 0 else None
    destroy_projectile(projectile, effects, impact_normal, damage, contact, attached)
    return True


def projectile_impacts_asteroid(projectile, asteroid, effects, environment):
    if not is_live_projectile(projectile) or not asteroid.currently_alive:
        return False

    _, _, overlap = collision_info(projectile, asteroid)
    contact, impact_normal = projectile_impact(projectile, asteroid, overlap)
    if contact is None:
        return False

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    destroy_projectile(projectile, effects, impact_normal, damage, contact)
    destroy_asteroid(asteroid, effects)
    return True


def projectile_impacts_planet(projectile, planet, effects, environment):
    if not is_live_projectile(projectile):
        return False

    _, _, overlap = collision_info(projectile, planet)
    contact, impact_normal = projectile_impact(projectile, planet, overlap)
    if contact is None:
        return False

    damage = projectile.current_damage
    BattleEffect.play_boom(damage)
    destroy_projectile(projectile, effects, impact_normal, damage, contact, planet)
    return True


def fighter_impacts_fighter(fighter, other, effects, environment):
    if not is_live_fighter(other):
        return False

    fighter_hits = fighter.fighter_collision_capabilities.collides_with_fighters
    other_hits = other.fighter_collision_capabilities.collides_with_fighters
    if not fighter_hits and not other_hits:
        return False

    _, _, overlap = collision_info(fighter, other)
    contact, normal = projectile_impact(fighter, other, overlap)
    if contact is None:
        return False

    if fighter_hits:
        other.current_hp = max(0, other.current_hp - fighter.current_damage)
    if other_hits:
        fighter.current_hp = max(0, fighter.current_hp - other.current_damage)
    BattleEffect.play_boom(max(fighter.current_damage, other.current_damage))
    if fighter.current_hp <= 0:
        destroy_projectile(fighter, effects, normal, fighter.current_damage, contact)
    if other.current_hp <= 0:
        destroy_projectile(
            other,
            effects,
            [-normal[0], -normal[1]],
            other.current_damage,
            contact,
        )
    return True


def fighter_impacts_projectile(fighter, projectile, effects, environment):
    if (
        not is_live_fighter(fighter)
        or not fighter.fighter_collision_capabilities.collides_with_projectiles
        or not is_live_projectile(projectile)
    ):
        return False

    _, _, overlap = collision_info(fighter, projectile)
    contact, normal = projectile_impact(fighter, projectile, overlap)
    if contact is None:
        return False

    initial_proj_hp = projectile.current_hp

    if fighter.fighter_collision_capabilities.damages_projectiles:
        projectile_hp = projectile.current_hp - fighter.current_damage
        set_projectile_hp(projectile, projectile_hp)

    BattleEffect.play_boom(fighter.current_damage)
    contact_handler = getattr(fighter, "handle_projectile_contact", None)
    contact_handled = bool(contact_handler is not None and contact_handler(projectile))

    if initial_proj_hp > 0 and projectile.current_hp <= 0:
        projectile.currently_alive = True  # Resurrect so destroy_projectile plays animation
        attached = fighter if fighter.current_hp > 0 else None
        destroy_projectile(
            projectile,
            effects,
            [-normal[0], -normal[1]],
            projectile.current_damage,
            contact,
            attached_target=attached,
        )

    if not contact_handled or fighter.current_hp <= 0:
        destroy_projectile(fighter, effects, normal, fighter.current_damage, contact)
    return True


def fighter_impacts_ship(fighter, ship, effects, environment):
    if not is_live_fighter(fighter) or ship.current_hp <= 0:
        return False

    if ship is fighter.parent:
        if not fighter.can_recover_with_parent():
            return False
    elif ship.player == fighter.player:
        if not (fighter.fighter_collision_capabilities.collides_with_friendly_ships):
            return False
    elif not (fighter.fighter_collision_capabilities.collides_with_enemy_ships):
        return False

    _, _, overlap = collision_info(fighter, ship)
    contact, normal = projectile_impact(fighter, ship, overlap)
    if contact is None:
        return False

    if ship is fighter.parent:
        fighter.recover_with_parent()
    else:
        contact_handler = getattr(fighter, "handle_ship_contact", None)
        if contact_handler is not None and contact_handler(ship, normal):
            return True
        damage = fighter.current_damage
        damage_ship(ship, damage)
        BattleEffect.play_boom(damage)
        attached = ship if ship.current_hp > 0 else None
        destroy_projectile(fighter, effects, normal, damage, contact, attached)
    return True


def fighter_impacts_asteroid(fighter, asteroid, effects, environment):
    if (
        not is_live_fighter(fighter)
        or not fighter.fighter_collision_capabilities.collides_with_asteroids
        or not asteroid.currently_alive
    ):
        return False

    _, _, overlap = collision_info(fighter, asteroid)
    contact, normal = projectile_impact(fighter, asteroid, overlap)
    if contact is None:
        return False
    contact_handler = getattr(fighter, "handle_asteroid_contact", None)
    if contact_handler is not None and contact_handler(asteroid, normal):
        return True

    BattleEffect.play_boom(fighter.current_damage)
    attached = None
    if not fighter.fighter_collision_capabilities.damages_asteroids:
        attached = asteroid
    destroy_projectile(fighter, effects, normal, fighter.current_damage, contact, attached)
    if fighter.fighter_collision_capabilities.damages_asteroids:
        destroy_asteroid(asteroid, effects)
    return True


def fighter_impacts_planet(fighter, planet, effects, environment):
    if (
        not is_live_fighter(fighter)
        or not fighter.fighter_collision_capabilities.collides_with_planets
    ):
        return False

    normal, _, overlap = collision_info(fighter, planet)
    contact, _ = projectile_impact(fighter, planet, overlap)
    if contact is None:
        return False

    separate_from_static_body(fighter, planet, normal, overlap, extra_clearance=1.0)
    fighter.begin_planet_avoidance(planet, normal)
    return True


def is_live_projectile(obj):
    return World.is_colliding_ability_kind(obj, "projectile")


def is_live_fighter(obj):
    return World.is_colliding_ability_kind(obj, "fighter")


def is_live_laser(obj):
    return World.is_colliding_ability_kind(obj, "laser")


def projectile_can_hit_ship(projectile, ship):
    if ship.player != projectile.player:
        return True

    if ship != projectile.parent or not projectile.hit_parent:
        return False

    if projectile.has_left_parent:
        return True

    _, _, overlap = collision_info(projectile, ship)
    if objects_overlap(projectile, ship, overlap):
        return False

    projectile.has_left_parent = True
    return False


def resolve_laser_hit(laser, target, effects, normal, contact, apply_impact):
    damage = laser.current_damage
    laser.end_position = [
        contact[0] % const.ARENA_SIZE,
        contact[1] % const.ARENA_SIZE,
    ]
    laser.intercepted = True
    
    apply_impact(target, effects, normal, damage, contact)

    attached = target if (getattr(target, 'current_hp', 1) > 0 and getattr(target, 'currently_alive', True) and not is_live_projectile(target)) else None
    
    laser.attached_target = attached
    if attached:
        laser.target_contact_offset = wrapped_delta(attached.position, contact)
        laser.initial_target_heading = getattr(attached, "heading", 0)

    effects.append(BattleEffect.from_blast(contact, normal, damage, attached_target=attached))
    BattleEffect.play_boom(damage)


def laser_impacts_ship(target, effects, normal, damage, contact):
    damage_ship(target, damage)


def laser_impacts_ability(target, effects, normal, damage, contact):
    target.current_hp = max(0, target.current_hp - damage)
    if target.current_hp <= 0:
        destroy_projectile(target, effects, normal, damage, contact)


def laser_impacts_asteroid(target, effects, normal, damage, contact):
    destroy_asteroid(target, effects)


def laser_impacts_planet(target, effects, normal, damage, contact):
    pass


def ship_is_laser_target(laser, target, explicit):
    if target.current_hp <= 0:
        return False
    if explicit:
        return True
    return target.player != laser.player or (
        target is laser.parent and laser.hit_parent
    )


def projectile_is_laser_target(laser, target, explicit):
    if not (target.can_collide and target.currently_alive and target.current_hp > 0):
        return False
    return explicit or target.player != laser.player or laser.hit_self


def fighter_is_laser_target(laser, target, explicit):
    if not (target.can_collide and target.currently_alive and target.current_hp > 0):
        return False
    if explicit:
        return True
    return target is not laser.parent and target.laser_target_capabilities.vulnerable


def asteroid_is_laser_target(laser, target, explicit):
    return target.currently_alive


def planet_is_laser_target(laser, target, explicit):
    return True


def generic_is_laser_target(laser, target, explicit):
    return explicit and target.currently_alive


def destroy_projectile(projectile, effects, direction, damage, contact_position=None, attached_target=None):
    if not projectile.currently_alive:
        return

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
            )
        )
    else:
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
            BattleEffect.from_animation(asteroid.position, asteroid.death_animation)
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
