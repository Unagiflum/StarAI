"""Gameplay policies applied after collision geometry identifies contact."""

import math

import src.const as const
from src.Battle.collision_geometry import (
    collision_info,
    collision_size,
    objects_overlap,
    solid_sweep_overlap,
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


def apply_generic_area_damage(source_ability, target, effects, delta, distance, damage):
    """Apply area damage and return the amount actually removed from the target."""
    phys = getattr(target, "physical_collision_capabilities", None)
    if not phys:
        return 0

    if phys.is_immovable:
        # Planets take no damage, but the ability can spawn effects via on_area_damage_hit
        return 0
        
    if phys.is_solid:
        if hasattr(target, "player"):
            shieldable = not getattr(source_ability, "ignores_shields", False)
            non_lethal = getattr(source_ability, "is_psychic", False)
            return damage_ship(
                target, damage, shieldable=shieldable, non_lethal=non_lethal
            )
        else:
            was_alive = getattr(target, "currently_alive", True)
            destroy_asteroid(target, effects)
            return damage if was_alive and not target.currently_alive else 0
            
    elif phys.is_projectile:
        previous_hp = target.current_hp
        remaining_hp = target.current_hp - damage
        if remaining_hp <= 0:
            direction = (
                [delta[0] / distance, delta[1] / distance]
                if distance > 0
                else [0, -1]
            )
            destroy_projectile(target, effects, direction, damage)
        else:
            set_projectile_hp(target, remaining_hp)
        return previous_hp - target.current_hp

    return 0


def generic_area_damage_target_is_eligible(source, target):
    if target is source or not getattr(target, 'currently_alive', True):
        return False
        
    # Must have vulnerability flag active
    target_area_cap = getattr(target, "area_damage_capabilities", None)
    if not target_area_cap or not target_area_cap.vulnerable:
        return False
        
    phys = getattr(target, "physical_collision_capabilities", None)
    if not phys:
        return False
        
    if phys.is_intangible:
        return False
        
    source_cap = getattr(source, "special_object_collision_capabilities", None)
    if not source_cap:
        # Fallback if source doesn't have collision properties initialized
        return False
        
    if phys.is_immovable:
        return source_cap.collides_with_planets
        
    if phys.is_solid:
        if not hasattr(target, "player"):
            return source_cap.collides_with_asteroids
        else:
            if getattr(source, "is_psychic", False):
                durability = getattr(target, "durability_capabilities", None)
                if durability and durability.immune_to_psychic:
                    return False
            if target.player != source.player:
                return source_cap.collides_with_enemy_ships
            else:
                return source_cap.collides_with_friendly_ships
                
    if phys.is_projectile:
        return source_cap.collides_with_projectiles
        
    return False





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

        if not solid_sweep_overlap(fragile_obj, immovable_obj):
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

        overlaps = solid_sweep_overlap(bouncing_obj, immovable_obj)
        normal, distance, overlap = collision_info(bouncing_obj, immovable_obj)
        contact_id = id(immovable_obj)

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

        handled1 = False
        handled2 = False
        h1 = getattr(first, "handle_projectile_contact", None)
        if h1: handled1 = h1(second)
        h2 = getattr(second, "handle_projectile_contact", None)
        if h2: handled2 = h2(first)

        if handled1 or handled2:
            BattleEffect.play_boom(max(first.current_damage, second.current_damage))
            if first.current_hp <= 0:
                destroy_projectile(first, effects, impact_normal, first.current_damage, contact)
            if second.current_hp <= 0:
                destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], second.current_damage, contact)
            return True

        if first.projectile_name == second.projectile_name:
            f1_caps = getattr(first, "special_object_collision_capabilities", None)
            f2_caps = getattr(second, "special_object_collision_capabilities", None)
            if f1_caps and f1_caps.bounces_off_same_type and f2_caps and f2_caps.bounces_off_same_type:
                normal, distance, overlap_real = collision_info(first, second)
                elastic_bounce(first, second, normal, distance, overlap_real)
                return True

            BattleEffect.play_boom(max(first.current_damage, second.current_damage))
            destroy_projectile(first, effects, impact_normal, first.current_damage, contact)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], second.current_damage, contact)
            return True

        f1_caps = getattr(first, "special_object_collision_capabilities", None)
        f2_caps = getattr(second, "special_object_collision_capabilities", None)
        f1_dmg = first.current_damage if not f1_caps or f1_caps.damages_projectiles else 0
        f2_dmg = second.current_damage if not f2_caps or f2_caps.damages_projectiles else 0

        proj_hp = first.current_hp - f2_dmg
        other_hp = second.current_hp - f1_dmg

        BattleEffect.play_boom(max(first.current_damage, second.current_damage))

        if proj_hp <= 0 and other_hp <= 0:
            destroy_projectile(first, effects, impact_normal, first.current_damage, contact)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], second.current_damage, contact)
        elif proj_hp > 0 and proj_hp > other_hp:
            set_projectile_hp(first, proj_hp)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], second.current_damage, contact)
        elif other_hp > 0 and other_hp > proj_hp:
            destroy_projectile(first, effects, impact_normal, first.current_damage, contact)
            set_projectile_hp(second, other_hp)
        else:
            destroy_projectile(first, effects, impact_normal, first.current_damage, contact)
            destroy_projectile(second, effects, [-impact_normal[0], -impact_normal[1]], second.current_damage, contact)
        return True

    elif is_first_proj or is_second_proj:
        projectile = first if is_first_proj else second
        other = second if is_first_proj else first
        phys_other = phys_second if is_first_proj else phys_first

        if phys_other.is_immovable:
            if not is_live_projectile(projectile):
                return False
                
            fighter_caps = getattr(projectile, "special_object_collision_capabilities", None)
            if fighter_caps and not fighter_caps.collides_with_planets:
                return False
                
            _, _, overlap = collision_info(projectile, other)
            contact, impact_normal = projectile_impact(projectile, other, overlap)
            if contact is None:
                return False

            handler = getattr(projectile, "handle_planet_contact", None)
            if handler and handler(other, impact_normal, overlap):
                return True

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            destroy_projectile(projectile, effects, impact_normal, damage, contact, other)
            return True

        elif phys_other.is_solid:
            if not is_live_projectile(projectile) or not getattr(other, "currently_alive", True) or getattr(other, "current_hp", 1) <= 0:
                return False
                
            is_ship = hasattr(other, "player")
            fighter_caps = getattr(projectile, "special_object_collision_capabilities", None)
            
            if is_ship:
                if not projectile_can_hit_ship(projectile, other):
                    return False
            else:
                if fighter_caps and not fighter_caps.collides_with_asteroids:
                    return False

            _, _, overlap = collision_info(projectile, other)
            contact, impact_normal = projectile_impact(projectile, other, overlap)
            if contact is None:
                return False

            if is_ship:
                if other is projectile.parent:
                    can_recover = getattr(projectile, "can_recover_with_parent", None)
                    if can_recover and can_recover():
                        recover_fn = getattr(projectile, "recover_with_parent", None)
                        if recover_fn:
                            recover_fn()
                            return True
                            
                handler = getattr(projectile, "handle_ship_contact", None)
                if handler and handler(other, impact_normal):
                    return True
            else:
                handler = getattr(projectile, "handle_asteroid_contact", None)
                if handler and handler(other, impact_normal):
                    return True

            damage = projectile.current_damage
            BattleEffect.play_boom(damage)
            
            if is_ship:
                damage_ship(other, damage)
                projectile.on_ship_impact(other)
                attached = other if (hasattr(other, "player") and other.current_hp > 0) else None
                destroy_projectile(projectile, effects, impact_normal, damage, contact, attached)
            else:
                attached = None if not fighter_caps or fighter_caps.damages_asteroids else other
                destroy_projectile(projectile, effects, impact_normal, damage, contact, attached)
                if not fighter_caps or fighter_caps.damages_asteroids:
                    destroy_asteroid(other, effects)
            return True
            
        return False

    is_first_solid = phys_first.is_solid and not phys_first.is_immovable
    is_second_solid = phys_second.is_solid and not phys_second.is_immovable

    if is_first_solid and is_second_solid:
        if not getattr(first, "currently_alive", True) or not getattr(second, "currently_alive", True):
            return False

        if not solid_sweep_overlap(first, second):
            return False
            
        normal, distance, overlap = collision_info(first, second)

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



def projectiles_can_hit_each_other(projectile, other):
    is_fighter1 = is_live_fighter(projectile)
    is_fighter2 = is_live_fighter(other)

    if is_fighter1 and is_fighter2:
        f1_hits = projectile.special_object_collision_capabilities.collides_with_fighters
        f2_hits = other.special_object_collision_capabilities.collides_with_fighters
        if not f1_hits and not f2_hits:
            return False
        return True

    if is_fighter1 and not is_fighter2:
        if not projectile.special_object_collision_capabilities.collides_with_projectiles:
            return False
        return True

    if not is_fighter1 and is_fighter2:
        if not other.special_object_collision_capabilities.collides_with_projectiles:
            return False
        return True

    if projectile.player != other.player:
        return True

    if projectile.projectile_name == other.projectile_name:
        return getattr(projectile, "hit_self", False) and getattr(other, "hit_self", False)

    return getattr(projectile, "hit_team", False) and getattr(other, "hit_team", False)


def is_live_projectile(obj):
    return World.is_colliding_ability_kind(obj, "projectile") or World.is_colliding_ability_kind(obj, "special_object")


def is_live_fighter(obj):
    return World.is_colliding_ability_kind(obj, "special_object")


def is_live_laser(obj):
    return World.is_colliding_ability_kind(obj, "laser")


def projectile_can_hit_ship(projectile, ship):
    fighter_caps = getattr(projectile, "special_object_collision_capabilities", None)
    if fighter_caps:
        if ship is projectile.parent:
            recover_fn = getattr(projectile, "can_recover_with_parent", None)
            return recover_fn is not None and recover_fn()
        if ship.player == projectile.player:
            return fighter_caps.collides_with_friendly_ships
        return fighter_caps.collides_with_enemy_ships

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
        set_projectile_hp(target, target.current_hp - damage)
        if target.current_hp <= 0:
            destroy_projectile(target, effects, normal, damage, contact)
        return
        
    if phys.is_projectile:
        set_projectile_hp(target, target.current_hp - damage)
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
