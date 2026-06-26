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





def resolve_generic_collision(
    first,
    second,
    effects,
    environment,
    *,
    object_on_screen_policy=None,
):
    phys_first = getattr(first, "physical_collision_capabilities", None)
    phys_second = getattr(second, "physical_collision_capabilities", None)

    if not phys_first or not phys_second:
        return False

    is_first_fragile = phys_first.fragile_to_immovable and phys_second.is_immovable
    is_second_fragile = phys_second.fragile_to_immovable and phys_first.is_immovable

    if is_first_fragile or is_second_fragile:
        fragile_obj = first if is_first_fragile else second
        immovable_obj = second if is_first_fragile else first

        if not getattr(fragile_obj, "currently_alive", True):
            return False

        _, _, overlap = collision_info(fragile_obj, immovable_obj)
        if not objects_overlap(fragile_obj, immovable_obj, overlap):
            return False

        on_screen = object_on_screen_policy or object_on_screen
        if on_screen(fragile_obj, environment.ships):
            BattleEffect.play_boom(1)
        
        # Currently, asteroids use destroy_asteroid.
        destroy_asteroid(fragile_obj, effects)
        return True

    is_first_bouncing = phys_first.bounces_on_immovable and phys_second.is_immovable
    is_second_bouncing = phys_second.bounces_on_immovable and phys_first.is_immovable

    if is_first_bouncing or is_second_bouncing:
        bouncing_obj = first if is_first_bouncing else second
        immovable_obj = second if is_first_bouncing else first

        normal, distance, overlap = collision_info(bouncing_obj, immovable_obj)
        contact_id = id(immovable_obj)
        overlaps = objects_overlap(bouncing_obj, immovable_obj, overlap)

        # Anti-stutter logic
        contacts_set = getattr(bouncing_obj, "planet_contacts", None)
        if contacts_set is None:
            contacts_set = set()
            setattr(bouncing_obj, "planet_contacts", contacts_set)

        if contact_id in contacts_set and not overlaps:
            contact_distance = radius(bouncing_obj) + radius(immovable_obj)
            if distance > contact_distance + PLANET_CONTACT_EXIT_MARGIN:
                contacts_set.remove(contact_id)

        if not overlaps:
            return False

        new_contact = contact_id not in contacts_set
        contacts_set.add(contact_id)
        
        if new_contact:
            collided_while_approaching = bounce_off_static_body(
                bouncing_obj, immovable_obj, normal, overlap
            )
            if collided_while_approaching and not getattr(bouncing_obj, "inertia", True):
                bouncing_obj.collision_velocity = bouncing_obj.velocity.copy()
        else:
            collided_while_approaching = False
            stop_at_static_body(bouncing_obj, immovable_obj, normal, overlap)

        # Payload Exchange
        if new_contact and collided_while_approaching:
            durability = getattr(immovable_obj, "durability_capabilities", None)
            is_invulnerable = durability and durability.is_invulnerable
            
            # Immovable Payload -> Bouncing Obj
            imm_impact = getattr(immovable_obj, "impact_capabilities", None)
            if imm_impact and imm_impact.impact_damage_percent > 0:
                if getattr(bouncing_obj, "current_hp", 0) > 0:
                    damage = max(1, math.ceil(bouncing_obj.current_hp * imm_impact.impact_damage_percent))
                    damage_ship(bouncing_obj, damage)
                    BattleEffect.play_boom(damage)
            
            # Bouncing Payload -> Immovable Obj
            bounce_impact = getattr(bouncing_obj, "impact_capabilities", None)
            if bounce_impact and bounce_impact.ramming_damage > 0:
                BattleEffect.play_boom(bounce_impact.ramming_damage)
                # Invulnerable skips HP reduction, but we played the boom.

        return True

    is_first_proj = phys_first.is_projectile
    is_second_proj = phys_second.is_projectile

    if is_first_proj and is_second_proj:
        if not is_live_projectile(first) or not is_live_projectile(second):
            return False
        if not projectiles_can_hit_each_other(first, second):
            return False
            
        _, _, overlap = collision_info(first, second)
        contact, impact_normal = projectile_impact(first, second, overlap)
        if contact is None:
            return False

        if first.projectile_name == second.projectile_name:
            BattleEffect.play_boom(max(first.current_damage, second.current_damage))
            destroy_projectile(first, effects, impact_normal, first.current_damage, contact)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], second.current_damage, contact)
            return True

        proj_dmg = first.current_damage
        other_dmg = second.current_damage
        proj_hp = first.current_hp - other_dmg
        other_hp = second.current_hp - proj_dmg

        BattleEffect.play_boom(max(proj_dmg, other_dmg))

        if proj_hp <= 0 and other_hp <= 0:
            destroy_projectile(first, effects, impact_normal, proj_dmg, contact)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], other_dmg, contact)
        elif proj_hp > 0 and proj_hp > other_hp:
            set_projectile_hp(first, proj_hp)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], other_dmg, contact)
        elif other_hp > 0 and other_hp > proj_hp:
            destroy_projectile(first, effects, impact_normal, proj_dmg, contact)
            set_projectile_hp(second, other_hp)
        else:
            destroy_projectile(first, effects, impact_normal, proj_dmg, contact)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], other_dmg, contact)
        return True

    elif is_first_proj or is_second_proj:
        projectile = first if is_first_proj else second
        other = second if is_first_proj else first
        phys_other = phys_second if is_first_proj else phys_first

        if phys_other.is_immovable:
            if not is_live_projectile(projectile):
                return False
            _, _, overlap = collision_info(projectile, other)
            contact, impact_normal = projectile_impact(projectile, other, overlap)
            if contact is None:
                return False

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            destroy_projectile(projectile, effects, impact_normal, damage, contact, other)
            return True

        elif phys_other.is_solid:
            if not is_live_projectile(projectile) or not getattr(other, "currently_alive", True) or getattr(other, "current_hp", 1) <= 0:
                return False
                
            if hasattr(other, "player") and not projectile_can_hit_ship(projectile, other):
                return False

            _, _, overlap = collision_info(projectile, other)
            contact, impact_normal = projectile_impact(projectile, other, overlap)
            if contact is None:
                return False

            damage = projectile.current_damage
            
            if hasattr(other, "player"):
                damage_ship(other, damage)
                projectile.on_ship_impact(other)
            else:
                destroy_asteroid(other, effects)
                
            BattleEffect.play_boom(damage)
            attached = other if (hasattr(other, "player") and other.current_hp > 0) else None
            destroy_projectile(projectile, effects, impact_normal, damage, contact, attached)
            return True
            
        return False

    is_first_solid = phys_first.is_solid and not phys_first.is_immovable
    is_second_solid = phys_second.is_solid and not phys_second.is_immovable

    if is_first_solid and is_second_solid:
        if not getattr(first, "currently_alive", True) or not getattr(second, "currently_alive", True):
            return False

        normal, distance, overlap = collision_info(first, second)
        if not objects_overlap(first, second, overlap):
            return False

        elastic_bounce(first, second, normal, distance, overlap)

        first_impact = getattr(first, "impact_capabilities", None)
        second_impact = getattr(second, "impact_capabilities", None)

        if first_impact and first_impact.ramming_damage > 0:
            damage_ship(second, first_impact.ramming_damage)
            BattleEffect.play_boom(first_impact.ramming_damage)

        if second_impact and second_impact.ramming_damage > 0:
            damage_ship(first, second_impact.ramming_damage)
            BattleEffect.play_boom(second_impact.ramming_damage)

        return True

    return False



def projectiles_can_hit_each_other(projectile, other):
    if projectile.player != other.player:
        return True

    if projectile.projectile_name == other.projectile_name:
        return getattr(projectile, "hit_self", False) and getattr(other, "hit_self", False)

    return getattr(projectile, "hit_team", False) and getattr(other, "hit_team", False)




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


def apply_generic_laser_impact(target, effects, normal, damage, contact):
    phys = getattr(target, "physical_collision_capabilities", None)
    if not phys:
        target.current_hp = max(0, target.current_hp - damage)
        if target.current_hp <= 0:
            destroy_projectile(target, effects, normal, damage, contact)
        return
        
    if phys.is_projectile:
        target.current_hp = max(0, target.current_hp - damage)
        if target.current_hp <= 0:
            destroy_projectile(target, effects, normal, damage, contact)
    elif phys.is_immovable:
        pass
    elif phys.is_solid:
        if hasattr(target, "player"):
            damage_ship(target, damage)
        else:
            destroy_asteroid(target, effects)


def generic_is_laser_target(laser, target, explicit):
    if not getattr(target, "currently_alive", True):
        return False
        
    phys = getattr(target, "physical_collision_capabilities", None)
    if not phys:
        if not getattr(target, "can_collide", True) or getattr(target, "current_hp", 1) <= 0:
            return False
        if explicit:
            return True
        laser_target = getattr(target, "laser_target_capabilities", None)
        if laser_target and laser_target.vulnerable:
            return target is not laser.parent
        return explicit

    if phys.is_intangible:
        return False
        
    if phys.is_projectile:
        return getattr(target, "can_collide", True) and getattr(target, "current_hp", 1) > 0
        
    if phys.is_immovable:
        return True
        
    if phys.is_solid:
        if not hasattr(target, "player"):
            return True
        else:
            if getattr(target, "current_hp", 1) <= 0:
                return False
            if explicit:
                return True
            return target.player != laser.player
        
    return explicit


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
