import unittest

import src.const as const
from src.collision_capabilities import (
    PhysicalCollisionCapabilities,
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    SpecialObjectCollisionCapabilities,
    LaserTargetCapabilities,
)
from src.Objects.Space.space_obj import Asteroid, Planet
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.space_ship import SpaceShip
from src.Battle import collision_responses
from unittest import mock


class CollisionTestCase(unittest.TestCase):
    @staticmethod
    def make_ship():
        ship = SpaceShip.__new__(SpaceShip)
        ship.name = "Target"
        ship.player = 2
        ship.position = [5, 100]
        ship.previous_position = ship.position.copy()
        ship.size = [20, 20]
        ship.velocity = [0.0, 0.0]
        ship.mass = 1.0
        ship.can_move = True
        ship.inertia = True
        ship.collision_velocity = [0.0, 0.0]
        ship.planet_contacts = set()
        ship.current_hp = 10
        ship.currently_alive = True
        ship.collision_capabilities = CollisionCapabilities(CollisionRole.SHIP)
        ship.laser_target_capabilities = LaserTargetCapabilities()
        ship.area_damage_capabilities = AreaDamageCapabilities(targetable=True)
        ship.physical_collision_capabilities = PhysicalCollisionCapabilities(is_solid=True, bounces_on_immovable=True)
        return ship

    @staticmethod
    def make_projectile(parent, projectile_class=Ability):
        projectile = projectile_class.__new__(projectile_class)
        projectile.name = "TestProjectile"
        projectile.projectile_name = "TestProjectile"
        projectile.type = "projectile"
        projectile.player = 1
        projectile.parent = parent
        projectile.position = [const.ARENA_SIZE - 5, 100]
        projectile.previous_position = projectile.position.copy()
        projectile.size = [20, 20]
        projectile.masks = None
        projectile.heading = 0
        projectile.frames = 1
        projectile.can_collide = True
        projectile.currently_alive = True
        projectile.current_hp = 1
        projectile.current_damage = 4
        projectile.hit_parent = False
        projectile.hit_self = False
        projectile.death_animation = []
        projectile.velocity = [1, 0]
        projectile.collision_capabilities = CollisionCapabilities(
            CollisionRole.PROJECTILE
        )
        projectile.laser_target_capabilities = LaserTargetCapabilities()
        projectile.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        projectile.physical_collision_capabilities = PhysicalCollisionCapabilities(is_solid=True, is_projectile=True)
        return projectile

    def make_projectile_pair(
        self,
        *,
        first_name="FirstProjectile",
        second_name="SecondProjectile",
        first_player=1,
        second_player=2,
        first_hp=1,
        second_hp=1,
        first_damage=4,
        second_damage=4,
    ):
        first_parent = self.make_ship()
        first_parent.player = first_player
        second_parent = self.make_ship()
        second_parent.player = second_player
        first = self.make_projectile(first_parent)
        second = self.make_projectile(second_parent)

        first.name = first.projectile_name = first_name
        second.name = second.projectile_name = second_name
        first.player = first_player
        second.player = second_player
        first.current_hp = first_hp
        second.current_hp = second_hp
        first.hp_array = [first_hp]
        second.hp_array = [second_hp]
        first.current_damage = first_damage
        second.current_damage = second_damage
        first.position = [100, 100]
        second.position = [108, 100]
        first.previous_position = first.position.copy()
        second.previous_position = second.position.copy()
        return first, second

    @staticmethod
    def make_fighter(
        *,
        collides_with_planets=True,
        collides_with_asteroids=True,
        damages_asteroids=True,
        collides_with_projectiles=True,
        damages_projectiles=True,
        collides_with_enemy_ships=True,
        collides_with_friendly_ships=False,
        collides_with_fighters=True,
        laser_vulnerable=True,
        fighter_class=Ability,
    ):
        special_object = fighter_class.__new__(fighter_class)
        special_object.name = "TestFighter"
        special_object.projectile_name = "TestFighter"
        special_object.type = "special_object"
        special_object.player = 1
        special_object.position = [100, 100]
        special_object.previous_position = special_object.position.copy()
        special_object.size = [20, 20]
        special_object.masks = None
        special_object.heading = 0
        special_object.frames = 1
        special_object.can_move = True
        special_object.can_collide = True
        special_object.currently_alive = True
        special_object.current_hp = 1
        special_object.current_damage = 1
        special_object.death_animation = []
        special_object.velocity = [1.0, 0.0]
        special_object.collision_capabilities = CollisionCapabilities(
            CollisionRole.SPECIAL_OBJECT
        )
        special_object.laser_target_capabilities = LaserTargetCapabilities(
            vulnerable=laser_vulnerable
        )
        special_object.special_object_collision_capabilities = SpecialObjectCollisionCapabilities(
            collides_with_planets=collides_with_planets,
            collides_with_asteroids=collides_with_asteroids,
            damages_asteroids=damages_asteroids,
            collides_with_projectiles=collides_with_projectiles,
            damages_projectiles=damages_projectiles,
            collides_with_enemy_ships=collides_with_enemy_ships,
            collides_with_friendly_ships=collides_with_friendly_ships,
            collides_with_fighters=collides_with_fighters,
        )
        special_object.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        special_object.physical_collision_capabilities = PhysicalCollisionCapabilities(is_solid=True, is_projectile=True)
        return special_object

    @staticmethod
    def make_laser(parent, *, hit_parent=False, hit_self=False, target=None):
        laser = Ability.__new__(Ability)
        laser.name = "TestLaser"
        laser.projectile_name = "TestLaser"
        laser.type = "laser"
        laser.player = parent.player
        laser.parent = parent
        laser.position = parent.position.copy()
        laser.previous_position = laser.position.copy()
        laser.start_position = laser.position.copy()
        laser.end_position = [laser.position[0] + 300, laser.position[1]]
        laser.size = [1, 1]
        laser.masks = None
        laser.heading = 0
        laser.frames = 1
        laser.can_collide = True
        laser.currently_alive = True
        laser.current_hp = 1
        laser.current_damage = 2
        laser.hit_parent = hit_parent
        laser.hit_self = hit_self
        laser.target = target
        laser.collision_capabilities = CollisionCapabilities(CollisionRole.LASER)
        laser.laser_target_capabilities = LaserTargetCapabilities(targetable=False)
        laser.area_damage_capabilities = AreaDamageCapabilities()
        laser.physical_collision_capabilities = PhysicalCollisionCapabilities(is_intangible=True)
        return laser

    @staticmethod
    def make_planet(position):
        planet = Planet.__new__(Planet)
        planet.position = list(position)
        planet.previous_position = planet.position.copy()
        planet.diameter = 20
        planet.size = [20, 20]
        planet.mask = None
        planet.can_move = False
        planet.collision_capabilities = CollisionCapabilities(CollisionRole.PLANET)
        planet.laser_target_capabilities = LaserTargetCapabilities()
        planet.area_damage_capabilities = AreaDamageCapabilities()
        planet.physical_collision_capabilities = PhysicalCollisionCapabilities(is_immovable=True)
        return planet

    @staticmethod
    def make_asteroid(position):
        asteroid = Asteroid.__new__(Asteroid)
        asteroid.position = list(position)
        asteroid.previous_position = asteroid.position.copy()
        asteroid.size = [20, 20]
        asteroid.masks = [None]
        asteroid.current_sprite = 0
        asteroid.currently_alive = True
        asteroid.death_animation = []
        asteroid.can_move = True
        asteroid.can_collide = True
        asteroid.velocity = [0.0, 0.0]
        asteroid.collision_capabilities = CollisionCapabilities(
            CollisionRole.ASTEROID
        )
        asteroid.laser_target_capabilities = LaserTargetCapabilities()
        asteroid.area_damage_capabilities = AreaDamageCapabilities(
            targetable=True
        )
        asteroid.physical_collision_capabilities = PhysicalCollisionCapabilities(fragile_to_immovable=True, is_solid=True)
        return asteroid

    @staticmethod
    def make_area_damage(position, damage_at_distance):
        ability = Ability.__new__(Ability)
        ability.name = "TestAreaDamage"
        ability.type = "area"
        ability.position = list(position)
        ability.previous_position = ability.position.copy()
        ability.currently_alive = True
        ability.area_damage_pending = True
        ability.area_damage_capabilities = AreaDamageCapabilities(
            emits=True,
            targetable=True,
            vulnerable=False,
        )
        ability.current_hp = 100
        ability.damage_at_distance = damage_at_distance
        ability.collision_capabilities = CollisionCapabilities(
            CollisionRole.NONE
        )
        from src.Objects.Ships.special_object_properties import SpecialObjectCollisionCapabilities
        ability.special_object_collision_capabilities = SpecialObjectCollisionCapabilities(collides_with_friendly_ships=True, collides_with_enemy_ships=True)
        ability.physical_collision_capabilities = PhysicalCollisionCapabilities(is_intangible=True)
        return ability


    def resolve_collision(self, first, second, effects, ships=None):
        class MockEnvironment:
            def __init__(self, ships):
                self.ships = ships or []
        env = MockEnvironment(ships)
        return collision_responses.resolve_generic_collision(first, second, effects, env)
