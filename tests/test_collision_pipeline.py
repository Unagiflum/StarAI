import unittest
from types import SimpleNamespace
from unittest import mock
import src.const as const
from collision_test_support import CollisionTestCase
from src.Battle import collisions
from src.Battle.world import World
from src.collision_capabilities import AreaDamageCapabilities, CollisionCapabilities
from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.ability import Ability


def run_collision_pipeline(game_objects, effects=None):
    original_ids = {id(obj) for obj in game_objects}
    collisions.handle_collisions(game_objects)
    if effects is not None:
        effects.extend(obj for obj in game_objects if id(obj) not in original_ids)


class CollisionPipelineTests(CollisionTestCase):

    def test_excluded_ship_is_not_hit_as_an_explicit_laser_target(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        target = self.make_ship()
        target.position = [150, 100]
        laser = self.make_laser(parent, target=target)
        laser.calculate_end_position = mock.Mock()

        collisions.handle_collisions(
            [parent, target, laser],
            excluded_objects=(target,),
        )

        self.assertEqual(target.current_hp, 10)

    def test_area_damage_is_consumed_once_and_uses_wrapped_distance(self):
        area = self.make_area_damage([const.ARENA_SIZE - 5, 100], lambda distance: 4 if distance == 10 else 0)
        ship = self.make_ship()
        ship.position = [5, 100]
        run_collision_pipeline([area, ship], [])
        run_collision_pipeline([area, ship], [])
        self.assertFalse(area.area_damage_pending)
        self.assertEqual(ship.current_hp, 6)

    def test_area_damage_excludes_planets_lasers_dead_and_invulnerable_targets(self):
        area = self.make_area_damage([100, 100], lambda distance: 5)
        planet = self.make_planet([100, 100])
        parent = self.make_ship()
        laser = self.make_laser(parent)
        laser.position = [100, 100]
        laser.can_collide = False
        dead_projectile = self.make_projectile(parent)
        dead_projectile.position = [100, 100]
        dead_projectile.currently_alive = False
        invulnerable_ship = self.make_ship()
        invulnerable_ship.position = [100, 100]
        invulnerable_ship.area_damage_capabilities = AreaDamageCapabilities(targetable=True, vulnerable=False)
        unrelated = SimpleNamespace(position=[100, 100], current_hp=10, currently_alive=True, area_damage_capabilities=AreaDamageCapabilities(), collision_capabilities=CollisionCapabilities())
        run_collision_pipeline([area, planet, laser, dead_projectile, invulnerable_ship, unrelated], [])
        self.assertEqual(laser.current_hp, 1)
        self.assertEqual(dead_projectile.current_hp, 1)
        self.assertEqual(invulnerable_ship.current_hp, 10)
        self.assertEqual(unrelated.current_hp, 10)

    def test_dead_asteroids_are_replaced_with_incremental_avoidance(self):

        class ReplacementAsteroid:
            created = []

            def __init__(self):
                self.currently_alive = True
                self.can_collide = True
                self.planet = None
                self.spawn_args = None
                self.position = None
                self.created.append(self)

            def set_planet(self, planet):
                self.planet = planet

            def get_respawn_position(self, planet, ships, avoid_bodies):
                self.spawn_args = (planet, tuple(ships), tuple(avoid_bodies))
                return [300 + len(self.created), 400]
        planet = self.make_planet([100, 100])
        ship = self.make_ship()
        alive = self.make_asteroid([200, 200])
        first_dead = self.make_asteroid([300, 300])
        second_dead = self.make_asteroid([400, 400])
        first_dead.currently_alive = False
        second_dead.currently_alive = False
        game_objects = [planet, ship, alive, first_dead, second_dead]
        with mock.patch.object(collisions, 'Asteroid', ReplacementAsteroid):
            collisions.handle_collisions(game_objects)
        self.assertEqual(len(ReplacementAsteroid.created), 2)
        first, second = ReplacementAsteroid.created
        self.assertIs(first.planet, planet)
        self.assertIs(second.planet, planet)
        self.assertEqual(first.spawn_args[1], (ship,))
        self.assertEqual(first.spawn_args[2], (ship, alive))
        self.assertEqual(second.spawn_args[2], (ship, alive, first))
        self.assertEqual(game_objects[-2:], [first, second])

    def test_asteroids_are_not_replaced_without_a_planet(self):
        asteroid = self.make_asteroid([100, 100])
        asteroid.currently_alive = False
        replacement_factory = mock.Mock()
        with mock.patch.object(collisions, 'Asteroid', replacement_factory):
            collisions.handle_collisions([asteroid])
        replacement_factory.assert_not_called()

    def test_fighter_stops_after_first_planet_contact(self):
        special_object = self.make_fighter()
        first = self.make_planet([115, 100])
        second = self.make_planet([85, 100])
        special_object.begin_planet_avoidance = mock.Mock()
        run_collision_pipeline([*[special_object], *[first, second]], None)
        special_object.begin_planet_avoidance.assert_called_once()
        self.assertIs(special_object.begin_planet_avoidance.call_args.args[0], first)

    def test_fighter_ignores_dead_asteroid_and_hits_next_live_target(self):
        special_object = self.make_fighter()
        dead = self.make_asteroid([108, 100])
        live = self.make_asteroid([108, 100])
        dead.currently_alive = False
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[special_object], *[dead, live]], [])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_fighter_ignores_dead_projectile_and_hits_next_live_target(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        dead = self.make_projectile(parent)
        live = self.make_projectile(parent)
        for projectile in (dead, live):
            projectile.position = [108, 100]
            projectile.previous_position = projectile.position.copy()
        dead.currently_alive = False
        dead.current_hp = 0
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[special_object], *[dead, live]], [])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_fighter_skips_dead_ship_and_hits_next_live_target(self):
        special_object = self.make_fighter()
        parent = self.make_ship()
        parent.player = 1
        special_object.parent = parent
        dead = self.make_ship()
        dead.player = 2
        dead.position = [108, 100]
        dead.current_hp = 0
        live = self.make_ship()
        live.player = 2
        live.position = [108, 100]
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[special_object], *[dead, live]], [])
        self.assertEqual(live.current_hp, 9)
        self.assertFalse(special_object.currently_alive)

    def test_fighter_ignores_dead_fighter_and_hits_next_live_target(self):
        first = self.make_fighter()
        dead = self.make_fighter()
        live = self.make_fighter()
        for special_object in (dead, live):
            special_object.position = [108, 100]
            special_object.previous_position = special_object.position.copy()
        dead.current_hp = 0
        dead.currently_alive = False
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[first, dead, live], *[]], None)
        self.assertFalse(first.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_destroyed_fighter_finishes_remaining_pairs_in_same_frame(self):
        first = self.make_fighter()
        second = self.make_fighter()
        third = self.make_fighter()
        for special_object in (second, third):
            special_object.position = [108, 100]
            special_object.previous_position = special_object.position.copy()
        effects = []
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom') as play_boom:
            run_collision_pipeline([first, second, third], effects)
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertFalse(third.currently_alive)
        self.assertEqual(len(effects), 3)
        self.assertEqual(play_boom.call_count, 2)

    def test_laser_selects_nearest_intercept_across_target_roles(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        enemy_ship = self.make_ship()
        enemy_ship.player = 2
        enemy_ship.position = [240, 100]
        asteroid = self.make_asteroid([170, 100])
        effects = []
        sentinel_effect = object()
        with mock.patch.object(collisions.BattleEffect, 'from_blast', return_value=sentinel_effect), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[laser], *[enemy_ship], *[], *[], *[asteroid], *[]], effects)
        self.assertFalse(asteroid.currently_alive)
        self.assertEqual(enemy_ship.current_hp, 10)
        self.assertTrue(laser.intercepted)
        self.assertLess(laser.end_position[0], enemy_ship.position[0])

    def test_destroyed_projectile_finishes_remaining_pairs_in_same_frame(self):
        first, second = self.make_projectile_pair(first_name='Matching', second_name='Matching')
        third_parent = self.make_ship()
        third_parent.player = 2
        third = self.make_projectile(third_parent)
        third.name = third.projectile_name = 'Matching'
        third.player = 2
        third.position = [108, 100]
        third.previous_position = third.position.copy()
        effects = []
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom') as play_boom:
            run_collision_pipeline([first, second, third], effects)
        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertFalse(third.currently_alive)
        self.assertEqual(len(effects), 3)
        self.assertEqual(play_boom.call_count, 2)

    def test_collision_cleanup_preserves_survivor_order_and_list_identity(self):
        first = object()
        live_ability = Ability.__new__(Ability)
        live_ability.currently_alive = True
        dead_ability = Ability.__new__(Ability)
        dead_ability.currently_alive = False
        live_asteroid = Asteroid.__new__(Asteroid)
        live_asteroid.currently_alive = True
        dead_asteroid = Asteroid.__new__(Asteroid)
        dead_asteroid.currently_alive = False
        last = object()
        game_objects = [first, dead_ability, live_ability, dead_asteroid, live_asteroid, last]
        authoritative = game_objects
        World(game_objects).remove_dead_collision_objects()
        self.assertIs(game_objects, authoritative)
        self.assertEqual(game_objects, [first, live_ability, live_asteroid, last])
if __name__ == '__main__':
    unittest.main()
