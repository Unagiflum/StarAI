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

class CollisionResponseSmokeTests(unittest.TestCase):

    def test_non_inertial_ship_preserves_approaching_planet_bounce_velocity(self):
        ship = body([100, 100], [1.0, 0.0])
        ship.inertia = False
        ship.collision_velocity = [0.0, 0.0]
        ship.planet_contacts = set()
        ship.current_hp = 10
        planet = body([115, 100], [0.0, 0.0])
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual([-1.0, 0.0], ship.velocity)
        self.assertEqual([-1.0, 0.0], ship.collision_velocity)

class CollisionResponsePolicyTests(CollisionTestCase):

    def test_area_damage_clamps_ship_hp_to_zero(self):
        area = self.make_area_damage([100, 100], lambda distance: 12)
        ship = self.make_ship()
        ship.position = [100, 100]
        apply_area_damage_response(area, ship, [], collision_responses.area_damage_impacts_ship)
        self.assertEqual(ship.current_hp, 0)

    def test_area_damage_uses_projectile_hp_hook_when_target_survives(self):
        area = self.make_area_damage([100, 100], lambda distance: 2)
        projectile = self.make_projectile(self.make_ship())
        projectile.position = [110, 100]
        projectile.current_hp = 5
        projectile.set_hp = mock.Mock()
        apply_area_damage_response(area, projectile, [], collision_responses.area_damage_impacts_ability)
        projectile.set_hp.assert_called_once_with(3)
        self.assertTrue(projectile.currently_alive)

    def test_area_damage_destroys_ability_with_outward_effect_direction(self):
        area = self.make_area_damage([100, 100], lambda distance: 2)
        special_object = self.make_fighter()
        special_object.position = [100, 110]
        blast = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=blast) as from_blast:
            effects = []
            apply_area_damage_response(area, special_object, effects, collision_responses.area_damage_impacts_ability)
        self.assertFalse(special_object.currently_alive)
        self.assertEqual(special_object.current_hp, 0)
        self.assertEqual(effects, [blast])
        from_blast.assert_called_once_with(special_object.position, [0.0, 1.0], 2)

    def test_area_damage_destroys_asteroid_without_hp_state(self):
        area = self.make_area_damage([100, 100], lambda distance: 1)
        asteroid = self.make_asteroid([120, 100])
        apply_area_damage_response(area, asteroid, [], collision_responses.area_damage_impacts_asteroid)
        self.assertFalse(asteroid.currently_alive)

    def test_equal_mass_ships_exchange_velocity_and_separate(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [115, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]
        self.resolve_collision(first, second, [], ships=[first, second])
        self.assertEqual(first.velocity, [-1.0, 0.0])
        self.assertEqual(second.velocity, [1.0, 0.0])
        self.assertEqual(first.position, [97.5, 100.0])
        self.assertEqual(second.position, [117.5, 100.0])
        self.assertEqual((first.current_hp, second.current_hp), (10, 10))

    def test_unequal_mass_ship_bounce_uses_mass_weighted_separation(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [115, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]
        first.mass = 1.0
        second.mass = 3.0
        self.resolve_collision(first, second, [], ships=[first, second])
        self.assertEqual(first.velocity, [-2.0, 0.0])
        self.assertEqual(second.velocity, [0.0, 0.0])
        self.assertEqual(first.position, [96.25, 100.0])
        self.assertEqual(second.position, [116.25, 100.0])

    def test_ship_ship_bounce_uses_wrapped_boundary_geometry(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [5, 100]
        second.position = [const.ARENA_SIZE - 5, 100]
        first.velocity = [-1.0, 0.0]
        second.velocity = [1.0, 0.0]
        self.resolve_collision(first, second, [], ships=[first, second])
        self.assertEqual(first.velocity, [1.0, 0.0])
        self.assertEqual(second.velocity, [-1.0, 0.0])
        self.assertEqual(first.position, [10.0, 100.0])
        self.assertEqual(second.position, [const.ARENA_SIZE - 10.0, 100.0])

    def test_non_overlapping_ships_keep_velocity_and_position(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [200, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]
        self.resolve_collision(first, second, [], ships=[first, second])
        self.assertEqual(first.velocity, [1.0, 0.0])
        self.assertEqual(second.velocity, [-1.0, 0.0])
        self.assertEqual(first.position, [100, 100])
        self.assertEqual(second.position, [200, 100])

    def test_ship_impact_hook_can_add_damage_without_changing_dispatch(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [115, 100]
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]
        first.on_ship_impact = mock.Mock(return_value=ShipImpactResult(damage_to_other=3))
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertEqual(first.current_hp, 10)
        self.assertEqual(second.current_hp, 7)
        play_boom.assert_called_once_with(3)
        impact = first.on_ship_impact.call_args.args[1]
        self.assertIsInstance(impact, ShipImpactContext)
        self.assertEqual(impact.normal, (-1.0, 0.0))
        self.assertEqual(impact.distance, 15.0)
        self.assertEqual(impact.overlap, 5.0)
        self.assertEqual(impact.closing_speed, 2.0)

    def test_ship_asteroid_bounce_uses_asteroid_radius_mass_fallback(self):
        ship = self.make_ship()
        asteroid = self.make_asteroid([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        ship.mass = 1.0
        asteroid.velocity = [-1.0, 0.0]
        self.resolve_collision(ship, asteroid, [], ships=[ship, asteroid])
        self.assertAlmostEqual(ship.velocity[0], 1 / 3)
        self.assertAlmostEqual(asteroid.velocity[0], 7 / 3)
        self.assertAlmostEqual(ship.position[0], 100 - 5 / 6)
        self.assertAlmostEqual(asteroid.position[0], 115 + 25 / 6)
        self.assertEqual(ship.current_hp, 10)

    def test_dead_asteroid_does_not_collide_with_ship(self):
        ship = self.make_ship()
        asteroid = self.make_asteroid([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        asteroid.velocity = [-1.0, 0.0]
        asteroid.currently_alive = False
        self.resolve_collision(ship, asteroid, [], ships=[ship, asteroid])
        self.assertEqual(ship.velocity, [1.0, 0.0])
        self.assertEqual(asteroid.velocity, [-1.0, 0.0])
        self.assertEqual(ship.position, [100, 100])
        self.assertEqual(asteroid.position, [115, 100])

    def test_first_approaching_planet_contact_bounces_and_damages_ship(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(ship.position, [95.0, 100.0])
        self.assertEqual(ship.current_hp, 8)
        self.assertEqual(ship.planet_contacts, {id(planet)})
        play_boom.assert_called_once_with(2)

    def test_persistent_planet_contact_stops_ship_without_repeated_damage(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
            ship.position = [100, 100]
            ship.velocity = [1.0, 0.0]
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.velocity, [0.0, 0.0])
        self.assertEqual(ship.position, [95.0, 100.0])
        self.assertEqual(ship.current_hp, 8)
        play_boom.assert_called_once_with(2)

    def test_separating_first_planet_contact_does_not_damage_ship(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [-1.0, 0.0]
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(ship.current_hp, 10)
        self.assertEqual(ship.planet_contacts, {id(planet)})
        play_boom.assert_not_called()

    def test_planet_contact_clears_only_beyond_exit_margin(self):
        ship = self.make_ship()
        planet = self.make_planet([100, 100])
        ship.planet_contacts.add(id(planet))
        ship.position = [123, 100]
        self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.planet_contacts, {id(planet)})
        ship.position = [125, 100]
        self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.planet_contacts, set())

    def test_planet_contact_uses_wrapped_geometry(self):
        ship = self.make_ship()
        planet = self.make_planet([const.ARENA_SIZE - 5, 100])
        ship.position = [5, 100]
        ship.velocity = [-1.0, 0.0]
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.velocity, [1.0, 0.0])
        self.assertEqual(ship.position, [15.0, 100.0])
        self.assertEqual(ship.current_hp, 8)

    def test_planet_contact_respects_collision_masks(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        ship_mask = pygame.mask.Mask((20, 20), fill=False)
        planet.mask = pygame.mask.Mask((20, 20), fill=True)
        ship.get_collision_mask = lambda: ship_mask
        self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.velocity, [1.0, 0.0])
        self.assertEqual(ship.current_hp, 10)
        self.assertEqual(ship.planet_contacts, set())

    def test_non_inertial_ship_keeps_planet_bounce_collision_velocity(self):
        ship = self.make_ship()
        planet = self.make_planet([115, 100])
        ship.position = [100, 100]
        ship.velocity = [1.0, 0.0]
        ship.inertia = False
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(ship, planet, [], ships=[ship, planet])
        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(ship.collision_velocity, [-1.0, 0.0])

    def test_asteroid_planet_contact_destroys_asteroid_with_animation(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([108, 100])
        asteroid.death_animation = [object()]
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_animation', return_value=sentinel_effect) as from_animation, mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            collision_responses.asteroid_impacts_planet(asteroid, planet, effects, SimpleNamespace(ships=tuple([])))
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        from_animation.assert_called_once_with(asteroid.position, asteroid.death_animation)
        play_boom.assert_called_once_with(1, attached_target=mock.ANY)

    def test_offscreen_asteroid_planet_contact_is_silent(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([108, 100])
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            collision_responses.asteroid_impacts_planet(
                asteroid,
                planet,
                [],
                SimpleNamespace(ships=tuple([self.make_ship(), self.make_ship()])),
                object_on_screen_policy=lambda asteroid, ships: False,
            )
        self.assertFalse(asteroid.currently_alive)
        play_boom.assert_not_called()

    def test_asteroid_planet_contact_uses_wrapped_geometry(self):
        asteroid = self.make_asteroid([5, 100])
        planet = self.make_planet([const.ARENA_SIZE - 5, 100])
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            collision_responses.asteroid_impacts_planet(asteroid, planet, [], SimpleNamespace(ships=tuple([])))
        self.assertFalse(asteroid.currently_alive)

    def test_dead_asteroid_is_ignored_by_planet_collision(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([108, 100])
        asteroid.currently_alive = False
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            collision_responses.asteroid_impacts_planet(asteroid, planet, [], SimpleNamespace(ships=tuple([])))
        play_boom.assert_not_called()

    def test_fighter_planet_contact_separates_and_begins_avoidance(self):
        special_object = self.make_fighter()
        planet = self.make_planet([115, 100])
        special_object.begin_planet_avoidance = mock.Mock()
        self.resolve_collision(special_object, planet, [], ships=[special_object, planet])
        self.assertEqual(special_object.position, [94.0, 100.0])
        self.assertTrue(special_object.currently_alive)
        special_object.begin_planet_avoidance.assert_called_once_with(planet, [-1.0, 0.0])

    def test_fighter_with_planet_collision_disabled_is_ignored(self):
        special_object = self.make_fighter(collides_with_planets=False)
        planet = self.make_planet([115, 100])
        special_object.begin_planet_avoidance = mock.Mock()
        self.resolve_collision(special_object, planet, [], ships=[special_object, planet])
        self.assertEqual(special_object.position, [100, 100])
        special_object.begin_planet_avoidance.assert_not_called()

    def test_fighter_planet_collision_defaults_to_enabled(self):
        special_object = self.make_fighter()
        planet = self.make_planet([115, 100])
        self.resolve_collision(special_object, planet, [], ships=[special_object, planet])
        self.assertEqual(special_object.position, [94.0, 100.0])

    def test_swept_fighter_planet_contact_begins_avoidance(self):
        special_object = self.make_fighter()
        planet = self.make_planet([150, 100])
        special_object.size = [10, 10]
        special_object.previous_position = [100, 100]
        special_object.position = [200, 100]
        special_object.begin_planet_avoidance = mock.Mock()
        self.resolve_collision(special_object, planet, [], ships=[special_object, planet])
        self.assertEqual(special_object.position, [200, 100])
        special_object.begin_planet_avoidance.assert_called_once_with(planet, [1.0, 0.0])

    def test_fighter_planet_contact_uses_wrapped_geometry(self):
        special_object = self.make_fighter()
        planet = self.make_planet([const.ARENA_SIZE - 5, 100])
        special_object.position = [5, 100]
        special_object.previous_position = special_object.position.copy()
        self.resolve_collision(special_object, planet, [], ships=[special_object, planet])
        self.assertEqual(special_object.position, [16.0, 100.0])

    def test_fighter_asteroid_contact_destroys_both_by_default(self):
        special_object = self.make_fighter()
        asteroid = self.make_asteroid([108, 100])
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(special_object, asteroid, effects, ships=[special_object, asteroid])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(1, attached_target=mock.ANY)

    def test_fighter_can_collide_without_damaging_asteroid(self):
        special_object = self.make_fighter(damages_asteroids=False)
        asteroid = self.make_asteroid([108, 100])
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, asteroid, [], ships=[special_object, asteroid])
        self.assertFalse(special_object.currently_alive)
        self.assertTrue(asteroid.currently_alive)

    def test_fighter_with_asteroid_collision_disabled_is_ignored(self):
        special_object = self.make_fighter(collides_with_asteroids=False)
        asteroid = self.make_asteroid([108, 100])
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(special_object, asteroid, [], ships=[special_object, asteroid])
        self.assertTrue(special_object.currently_alive)
        self.assertTrue(asteroid.currently_alive)
        play_boom.assert_not_called()

    def test_swept_fighter_asteroid_impact_is_not_tunneled(self):
        special_object = self.make_fighter()
        asteroid = self.make_asteroid([150, 100])
        special_object.size = [10, 10]
        asteroid.size = [10, 10]
        special_object.previous_position = [100, 100]
        special_object.position = [200, 100]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, asteroid, [], ships=[special_object, asteroid])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(asteroid.currently_alive)

    def test_fighter_projectile_contact_destroys_both_by_default(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(special_object, projectile, effects, ships=[special_object, projectile])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(effects, [sentinel_effect, sentinel_effect])
        play_boom.assert_called_once_with(1, attached_target=mock.ANY)

    def test_fighter_can_collide_without_damaging_projectile(self):
        special_object = self.make_fighter(damages_projectiles=False)
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, projectile, [], ships=[special_object, projectile])
        self.assertFalse(special_object.currently_alive)
        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 1)

    def test_fighter_with_projectile_collision_disabled_is_ignored(self):
        special_object = self.make_fighter(collides_with_projectiles=False)
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(special_object, projectile, [], ships=[special_object, projectile])
        self.assertTrue(special_object.currently_alive)
        self.assertTrue(projectile.currently_alive)
        play_boom.assert_not_called()

    def test_projectile_with_remaining_hp_survives_fighter_impact(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [108, 100]
        projectile.previous_position = projectile.position.copy()
        projectile.current_hp = 3
        projectile.hp_array = [3]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, projectile, [], ships=[special_object, projectile])
        self.assertFalse(special_object.currently_alive)
        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 2)

    def test_swept_fighter_projectile_impact_is_not_tunneled(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        special_object.size = [10, 10]
        projectile.size = [10, 10]
        special_object.previous_position = [100, 100]
        special_object.position = [200, 100]
        projectile.previous_position = [150, 100]
        projectile.position = [150, 100]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, projectile, [], ships=[special_object, projectile])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(projectile.currently_alive)

    def test_fighter_enemy_ship_contact_damages_ship_and_destroys_fighter(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        target = self.make_ship()
        target.player = 2
        target.position = [108, 100]
        special_object.parent = parent
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(special_object, target, effects, ships=[special_object, target])
        self.assertEqual(target.current_hp, 9)
        self.assertFalse(special_object.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(1, attached_target=mock.ANY)

    def test_fighter_ignores_friendly_ship_by_default(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        friendly = self.make_ship()
        friendly.player = 1
        friendly.position = [108, 100]
        special_object.parent = parent
        self.resolve_collision(special_object, friendly, [], ships=[special_object, friendly])
        self.assertEqual(friendly.current_hp, 10)
        self.assertTrue(special_object.currently_alive)

    def test_fighter_can_collide_with_friendly_ship(self):
        special_object = self.make_fighter(collides_with_friendly_ships=True)
        parent = self.make_ship()
        parent.player = 1
        friendly = self.make_ship()
        friendly.player = 1
        friendly.position = [108, 100]
        special_object.parent = parent
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, friendly, [], ships=[special_object, friendly])
        self.assertEqual(friendly.current_hp, 9)
        self.assertFalse(special_object.currently_alive)

    def test_fighter_with_enemy_ship_collision_disabled_is_ignored(self):
        special_object = self.make_fighter(collides_with_enemy_ships=False)
        parent = self.make_ship()
        parent.player = 1
        enemy = self.make_ship()
        enemy.player = 2
        enemy.position = [108, 100]
        special_object.parent = parent
        self.resolve_collision(special_object, enemy, [], ships=[special_object, enemy])
        self.assertEqual(enemy.current_hp, 10)
        self.assertTrue(special_object.currently_alive)

    def test_fighter_cannot_recover_with_parent_until_returning(self):
        special_object = self.make_fighter(fighter_class=KzerZaA2)
        parent = self.make_ship()
        parent.player = 1
        parent.position = [108, 100]
        parent.max_hp = 10
        parent.current_hp = 5
        special_object.parent = parent
        special_object.mode = special_object.ATTACKING
        special_object.return_sound = None
        self.resolve_collision(special_object, parent, [], ships=[special_object, parent])
        self.assertEqual(parent.current_hp, 5)
        self.assertTrue(special_object.currently_alive)

    def test_returning_fighter_recovers_with_parent(self):
        special_object = self.make_fighter(fighter_class=KzerZaA2)
        parent = self.make_ship()
        parent.player = 1
        parent.position = [108, 100]
        parent.max_hp = 10
        parent.current_hp = 5
        special_object.parent = parent
        special_object.mode = special_object.RETURNING
        special_object.return_sound = None
        self.resolve_collision(special_object, parent, [], ships=[special_object, parent])
        self.assertEqual(parent.current_hp, 6)
        self.assertFalse(special_object.currently_alive)
        self.assertEqual(special_object.current_hp, 0)

    def test_swept_fighter_enemy_ship_impact_is_not_tunneled(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        special_object.parent = parent
        enemy = self.make_ship()
        enemy.player = 2
        special_object.size = [10, 10]
        enemy.size = [10, 10]
        special_object.previous_position = [100, 100]
        special_object.position = [200, 100]
        enemy.previous_position = [150, 100]
        enemy.position = [150, 100]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(special_object, enemy, [], ships=[special_object, enemy])
        self.assertEqual(enemy.current_hp, 9)
        self.assertFalse(special_object.currently_alive)

    def test_fighter_fighter_contact_destroys_both_by_default(self):
        first = self.make_fighter()
        second = self.make_fighter()
        second.position = [108, 100]
        second.previous_position = second.position.copy()
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(first, second, effects, ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertEqual(effects, [sentinel_effect, sentinel_effect])
        play_boom.assert_called_once_with(1, attached_target=mock.ANY)

    def test_one_sided_fighter_collision_only_damages_disabled_fighter(self):
        first = self.make_fighter()
        second = self.make_fighter(collides_with_fighters=False)
        first.current_hp = 3
        second.current_damage = 5
        second.position = [108, 100]
        second.previous_position = second.position.copy()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertTrue(first.currently_alive)
        self.assertEqual(first.current_hp, 3)
        self.assertFalse(second.currently_alive)
        play_boom.assert_called_once_with(5)

    def test_two_fighters_with_collision_disabled_ignore_each_other(self):
        first = self.make_fighter(collides_with_fighters=False)
        second = self.make_fighter(collides_with_fighters=False)
        second.position = [108, 100]
        second.previous_position = second.position.copy()
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertTrue(first.currently_alive)
        self.assertTrue(second.currently_alive)
        play_boom.assert_not_called()

    def test_fighters_with_remaining_hp_survive_mutual_impact(self):
        first = self.make_fighter()
        second = self.make_fighter()
        first.current_hp = 3
        second.current_hp = 3
        second.position = [108, 100]
        second.previous_position = second.position.copy()
        with mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertEqual(first.current_hp, 2)
        self.assertEqual(second.current_hp, 2)
        self.assertTrue(first.currently_alive)
        self.assertTrue(second.currently_alive)

    def test_swept_fighter_fighter_impact_is_not_tunneled(self):
        first = self.make_fighter()
        second = self.make_fighter()
        first.size = [10, 10]
        second.size = [10, 10]
        first.previous_position = [100, 100]
        first.position = [200, 100]
        second.previous_position = [150, 100]
        second.position = [150, 100]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_laser_targets_apply_role_and_ownership_rules(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        enemy_ship = self.make_ship()
        enemy_ship.player = 2
        enemy_ship.position = [180, 100]
        friendly_ship = self.make_ship()
        friendly_ship.player = 1
        friendly_ship.position = [190, 100]
        enemy_projectile = self.make_projectile(parent)
        enemy_projectile.player = 2
        enemy_projectile.position = [200, 100]
        friendly_projectile = self.make_projectile(parent)
        friendly_projectile.player = 1
        friendly_projectile.position = [210, 100]
        enemy_fighter = self.make_fighter()
        enemy_fighter.player = 2
        enemy_fighter.position = [220, 100]
        friendly_fighter = self.make_fighter()
        friendly_fighter.player = 1
        friendly_fighter.position = [230, 100]
        invulnerable_fighter = self.make_fighter(laser_vulnerable=False)
        invulnerable_fighter.player = 2
        invulnerable_fighter.position = [240, 100]
        asteroid = self.make_asteroid([250, 100])
        dead_asteroid = self.make_asteroid([260, 100])
        dead_asteroid.currently_alive = False
        planet = self.make_planet([270, 100])
        targets = laser_targets_from_policies(laser, [parent, enemy_ship, friendly_ship], [enemy_projectile, friendly_projectile], [enemy_fighter, friendly_fighter, invulnerable_fighter], [asteroid, dead_asteroid], [planet])
        for target in (enemy_ship, enemy_projectile, enemy_fighter, friendly_fighter, asteroid, planet):
            self.assertIn(target, targets)
        for target in (parent, friendly_ship, friendly_projectile, invulnerable_fighter, dead_asteroid):
            self.assertNotIn(target, targets)

    def test_laser_hit_flags_enable_parent_and_friendly_projectiles_only(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        friendly_ship = self.make_ship()
        friendly_ship.player = 1
        friendly_ship.position = [180, 100]
        friendly_projectile = self.make_projectile(parent)
        friendly_projectile.player = 1
        friendly_projectile.position = [190, 100]
        laser = self.make_laser(parent, hit_parent=True, hit_self=True)
        targets = laser_targets_from_policies(laser, [parent, friendly_ship], [friendly_projectile], [], [], [])
        self.assertIn(parent, targets)
        self.assertIn(friendly_projectile, targets)
        self.assertNotIn(friendly_ship, targets)

    def test_explicit_laser_target_overrides_normal_eligibility(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        explicit = self.make_fighter(laser_vulnerable=False)
        explicit.player = 1
        explicit.position = [180, 100]
        interceptor = self.make_fighter()
        interceptor.player = 2
        interceptor.position = [160, 100]
        unrelated_ship = self.make_ship()
        unrelated_ship.player = 2
        unrelated_ship.position = [150, 100]
        laser = self.make_laser(parent, target=explicit)
        targets = laser_targets_from_policies(laser, [unrelated_ship], [], [explicit, interceptor], [], [])
        self.assertEqual(targets, [explicit, interceptor])

    def test_laser_ship_impact_clips_endpoint_and_applies_effect(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        target = self.make_ship()
        target.player = 2
        target.position = [150, 100]
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect) as from_blast, mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(laser, target, effects, ships=[target])
        self.assertEqual(target.current_hp, 8)
        self.assertEqual(laser.end_position, [140.0, 100.0])
        self.assertTrue(laser.intercepted)
        self.assertEqual(effects, [sentinel_effect])
        from_blast.assert_called_once_with([140.0, 100.0], [-1.0, 0.0], 2)
        play_boom.assert_called_once_with(2)

    def test_planet_absorbs_laser_without_damage_state(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        planet = self.make_planet([150, 100])
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(laser, planet, [], ships=[])
        self.assertFalse(hasattr(planet, 'current_hp'))
        self.assertTrue(laser.intercepted)

    def test_laser_destroys_projectile_at_zero_hp(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        projectile = self.make_projectile(parent)
        projectile.player = 2
        projectile.position = [150, 100]
        projectile.previous_position = projectile.position.copy()
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(laser, projectile, effects, ships=[projectile])
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 0)
        self.assertEqual(effects, [sentinel_effect, sentinel_effect])

    def test_laser_destroys_fighter_at_zero_hp(self):
        parent = self.make_ship()
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        special_object = self.make_fighter()
        special_object.position = [150, 100]
        special_object.previous_position = special_object.position.copy()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(laser, special_object, [], ships=[special_object])
        self.assertFalse(special_object.currently_alive)
        self.assertEqual(special_object.current_hp, 0)

    def test_laser_directly_reduces_surviving_projectile_hp(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        projectile = self.make_projectile(parent)
        projectile.player = 2
        projectile.position = [150, 100]
        projectile.previous_position = projectile.position.copy()
        projectile.current_hp = 3
        projectile.set_hp = mock.Mock()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(laser, projectile, [], ships=[projectile])
        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 1)
        projectile.set_hp.assert_not_called()

    def test_enemy_projectile_with_greater_remaining_hp_survives(self):
        first, second = self.make_projectile_pair(first_hp=10, second_hp=3, first_damage=4, second_damage=2)
        effects = []
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(first, second, effects, ships=[first, second])
        self.assertTrue(first.currently_alive)
        self.assertEqual(first.current_hp, 8)
        self.assertFalse(second.currently_alive)
        self.assertEqual(len(effects), 1)
        play_boom.assert_called_once_with(4)

    def test_second_enemy_projectile_with_greater_remaining_hp_survives(self):
        first, second = self.make_projectile_pair(first_hp=3, second_hp=10, first_damage=2, second_damage=4)
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertTrue(second.currently_alive)
        self.assertEqual(second.current_hp, 8)

    def test_equal_positive_projectile_hp_after_impact_destroys_both(self):
        first, second = self.make_projectile_pair(first_hp=5, second_hp=5, first_damage=2, second_damage=2)
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_same_name_enemy_projectiles_destroy_each_other_regardless_of_hp(self):
        first, second = self.make_projectile_pair(first_name='MatchingProjectile', second_name='MatchingProjectile', first_hp=10, second_hp=10, first_damage=1, second_damage=1)
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_same_player_projectiles_require_matching_names_and_mutual_self_hit(self):
        cases = (('First', 'Second', True, True), ('Matching', 'Matching', True, False), ('Matching', 'Matching', False, True))
        for first_name, second_name, first_hit_self, second_hit_self in cases:
            with self.subTest(names=(first_name, second_name), hit_self=(first_hit_self, second_hit_self)):
                first, second = self.make_projectile_pair(first_name=first_name, second_name=second_name, first_player=1, second_player=1)
                first.hit_self = first_hit_self
                second.hit_self = second_hit_self
                with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
                    self.resolve_collision(first, second, [], ships=[first, second])
                self.assertTrue(first.currently_alive)
                self.assertTrue(second.currently_alive)

    def test_matching_same_player_projectiles_with_self_hit_destroy_each_other(self):
        first, second = self.make_projectile_pair(first_name='Matching', second_name='Matching', first_player=1, second_player=1)
        first.hit_self = True
        second.hit_self = True
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_swept_projectile_projectile_impact_is_not_tunneled(self):
        first, second = self.make_projectile_pair()
        first.size = [10, 10]
        second.size = [10, 10]
        first.previous_position = [100, 100]
        first.position = [200, 100]
        second.previous_position = [150, 100]
        second.position = [150, 100]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(first, second, [], ships=[first, second])
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)

    def test_projectile_damages_ship_across_wrapped_boundary(self):
        parent = self.make_ship()
        parent.player = 1
        target = self.make_ship()
        projectile = self.make_projectile(parent)
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(projectile, target, effects, ships=[projectile, target])
        self.assertEqual(target.current_hp, 6)
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(effects, [sentinel_effect])

    def test_generic_projectile_ship_impact_does_not_add_momentum(self):
        parent = self.make_ship()
        parent.player = 1
        target = self.make_ship()
        target.add_impulse = mock.Mock()
        projectile = self.make_projectile(parent)
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(projectile, target, [], ships=[projectile, target])
        target.add_impulse.assert_not_called()

    def test_druuge_projectile_ship_impact_adds_projectile_momentum(self):
        parent = self.make_ship()
        parent.player = 1
        parent.mass = 10
        target = self.make_ship()
        target.mass = 20
        target.add_impulse = mock.Mock()
        projectile = self.make_projectile(parent, DruugeA1)
        projectile.projectile_name = 'DruugeA1'
        projectile.RECOIL_INCREMENT = 24
        projectile.velocity = [3, 4]
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast'), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(projectile, target, [], ships=[projectile, target])
        target.add_impulse.assert_called_once_with(7.2, 9.6)

    def test_projectile_impact_with_planet_destroys_projectile_at_contact(self):
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        planet = self.make_planet([108, 100])
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect) as from_blast, mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(projectile, planet, effects, ships=[projectile, planet])
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 0)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(4)
        from_blast.assert_called_once_with([98.0, 100.0], [-1.0, 0.0], 4, attached_target=mock.ANY)

    def test_swept_projectile_impact_with_planet_is_not_tunneled(self):
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.size = [10, 10]
        projectile.previous_position = [100, 100]
        projectile.position = [200, 100]
        planet = self.make_planet([150, 100])
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collision_responses.BattleEffect, 'play_boom'):
            self.resolve_collision(projectile, planet, effects, ships=[projectile, planet])
        self.assertFalse(projectile.currently_alive)
        self.assertEqual(effects, [sentinel_effect])

    def test_projectile_impact_with_asteroid_destroys_both_objects(self):
        parent = self.make_ship()
        projectile = self.make_projectile(parent)
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        asteroid = self.make_asteroid([108, 100])
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collision_responses.BattleEffect, 'from_blast', return_value=sentinel_effect) as from_blast, mock.patch.object(collision_responses.BattleEffect, 'play_boom') as play_boom:
            self.resolve_collision(projectile, asteroid, effects, ships=[projectile, asteroid])
        self.assertFalse(projectile.currently_alive)
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(effects, [sentinel_effect])
        play_boom.assert_called_once_with(4)
        from_blast.assert_called_once_with([98.0, 100.0], [-1.0, 0.0], 4, attached_target=mock.ANY)
if __name__ == '__main__':
    unittest.main()
