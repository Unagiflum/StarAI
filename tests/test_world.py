import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()

from src.Battle.effects import BattleEffect
from src.Battle.world import World
from src.Battle import battle_init
from src.Battle.battle import stop_tracking_projectiles, update_preserved_abilities
from src.Objects.object import Object, ThrustMarker
from src.Objects.Space.space_obj import Asteroid, Planet, Star
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip


class WorldCharacterizationTests(unittest.TestCase):
    @staticmethod
    def ability(kind, *, alive=True, hp=1, can_collide=True):
        ability = Ability.__new__(Ability)
        ability.type = kind
        ability.currently_alive = alive
        ability.current_hp = hp
        ability.can_collide = can_collide
        return ability

    def test_wraps_one_authoritative_list_and_mutates_it_in_place(self):
        first = object()
        second = object()
        objects = [first]
        world = World(objects)

        world.add(second)
        world.remove(first)

        self.assertIs(world.objects, objects)
        self.assertEqual(objects, [second])

    def test_base_object_supplies_hp_free_lifecycle_defaults(self):
        obj = Object("plain", None, [10, 10])

        self.assertTrue(obj.is_alive())
        self.assertEqual(obj.drain_spawned_objects(), [])
        self.assertIsNone(obj.get_collision_mask())
        self.assertFalse(hasattr(obj, "current_hp"))

        obj.currently_alive = False
        self.assertFalse(World.is_alive(obj))

    def test_ability_supplies_default_round_and_laser_hooks(self):
        ability = Ability.__new__(Ability)
        ability.currently_alive = True
        ability.current_hp = 1

        self.assertTrue(ability.is_alive())
        self.assertIsNone(ability.stop_and_track())
        self.assertIsNone(ability.calculate_end_position())

        ability.current_hp = 0
        self.assertFalse(World.is_alive(ability))

    def test_update_contract_keeps_simple_double_compatibility(self):
        class SimpleDouble:
            def update(self):
                return True

        obj = SimpleDouble()
        objects = [obj]

        World(objects).update_objects()

        self.assertEqual(objects, [obj])

    def test_round_transition_calls_tracking_hook_only_for_live_abilities(self):
        live = self.ability("projectile")
        dead = self.ability("projectile", alive=False)
        live.stop_and_track = mock.Mock()
        dead.stop_and_track = mock.Mock()

        stop_tracking_projectiles([live, dead])

        live.stop_and_track.assert_called_once_with()
        dead.stop_and_track.assert_not_called()

    def test_preserved_ability_retargets_dead_opponent_and_keeps_planet(self):
        player1 = SpaceShip.__new__(SpaceShip)
        player1.player = 1
        player2 = SpaceShip.__new__(SpaceShip)
        player2.player = 2
        old_target = SpaceShip.__new__(SpaceShip)
        old_target.currently_alive = False
        old_target.current_hp = 1
        planet = object()
        ability = self.ability("projectile")
        ability.player = 1
        ability.target = old_target
        ability.stop_and_track = mock.Mock()

        update_preserved_abilities([ability], player1, player2, planet)

        ability.stop_and_track.assert_called_once_with()
        self.assertIs(ability.opponent, player2)
        self.assertIs(ability.target, player2)
        self.assertIs(ability.planet, planet)

    def test_typed_snapshots_follow_authoritative_order_after_mutation(self):
        ship = SpaceShip.__new__(SpaceShip)
        projectile = self.ability("projectile")
        special_object = self.ability("special_object")
        laser = self.ability("laser")
        asteroid = Asteroid.__new__(Asteroid)
        planet = Planet.__new__(Planet)
        star = Star.__new__(Star)
        effect = BattleEffect.__new__(BattleEffect)
        thrust = ThrustMarker.__new__(ThrustMarker)
        world = World([
            star,
            projectile,
            ship,
            asteroid,
            special_object,
            planet,
            thrust,
            laser,
            effect,
        ])

        self.assertEqual(world.ships, [ship])
        self.assertEqual(world.abilities, [projectile, special_object, laser])
        self.assertEqual(world.projectiles, [projectile])
        self.assertEqual(world.special_objects, [special_object])
        self.assertEqual(world.lasers, [laser])
        self.assertEqual(world.asteroids, [asteroid])
        self.assertEqual(world.planets, [planet])
        self.assertEqual(world.stars, [star])
        self.assertEqual(world.effects, [effect])
        self.assertEqual(world.thrust_markers, [thrust])

        world.remove(projectile)
        self.assertEqual(world.abilities, [special_object, laser])

    def test_live_and_collision_queries_preserve_order_and_rules(self):
        live_ship = SpaceShip.__new__(SpaceShip)
        live_ship.currently_alive = True
        live_ship.current_hp = 1
        dead_ship = SpaceShip.__new__(SpaceShip)
        dead_ship.currently_alive = False
        dead_ship.current_hp = 1
        live_projectile = self.ability("projectile")
        disabled_projectile = self.ability("projectile", can_collide=False)
        zero_hp_fighter = self.ability("special_object", hp=0)
        live_laser = self.ability("laser")
        live_asteroid = Asteroid.__new__(Asteroid)
        live_asteroid.currently_alive = True
        dead_asteroid = Asteroid.__new__(Asteroid)
        dead_asteroid.currently_alive = False
        world = World([
            dead_ship,
            live_projectile,
            live_ship,
            disabled_projectile,
            live_asteroid,
            zero_hp_fighter,
            dead_asteroid,
            live_laser,
        ])

        self.assertEqual(world.live_ships, [live_ship])
        self.assertEqual(world.colliding_projectiles, [live_projectile])
        self.assertEqual(world.colliding_fighters, [])
        self.assertEqual(world.colliding_lasers, [live_laser])
        self.assertEqual(world.live_asteroids, [live_asteroid])

    def test_battle_initialization_preserves_legacy_object_order(self):
        class Ship:
            def __init__(self):
                self.name = "TestShip"
                self.battles_fought = 0
                
            def initialize_in_battle(self, position, rotation):
                self.position = position
                self.rotation = rotation

            def set_planet(self, planet):
                self.planet = planet

        class ReplacementAsteroid:
            created = []

            def __init__(self):
                self.created.append(self)

            def set_planet(self, planet):
                self.planet = planet

            def get_valid_asteroid_position(self, planet, ships, objects):
                return [len(self.created), 0]

        stars = [object(), object()]
        player1 = Ship()
        player2 = Ship()
        planet = object()

        with (
            mock.patch.object(
                battle_init.Star, "create_random_stars", return_value=stars
            ),
            mock.patch.object(
                battle_init, "get_valid_ship_positions", return_value=([1, 2], [3, 4])
            ),
            mock.patch.object(
                battle_init.Planet, "create_center", return_value=planet
            ),
            mock.patch.object(battle_init, "Asteroid", ReplacementAsteroid),
            mock.patch.object(battle_init.const, "ASTEROID_COUNT", 2),
        ):
            state = battle_init.initialize_battle(None, player1, player2)

        self.assertEqual(
            state["world"].objects,
            stars + [player1, player2, planet] + ReplacementAsteroid.created,
        )
        self.assertIs(state["game_objects"], state["world"].objects)


if __name__ == "__main__":
    unittest.main()
