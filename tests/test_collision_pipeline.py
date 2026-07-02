import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock
from collision_test_support import CollisionTestCase
from src.Battle import collisions
from src.Battle.world import World
from src.collision_capabilities import (
    ImpactCapabilities,
    LaserTargetCapabilities,
    ProjectileContactPolicy,
)
from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.Chenjesu.A2.ChenjesuA2 import ChenjesuA2
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
from src.Objects.Ships.Orz.A3.OrzA3 import OrzA3
from src.Objects.Ships.Syreen.A2.SyreenCrew import SyreenCrew
from src.Objects.Ships.Vux.A2.VuxA2 import VuxA2


def run_collision_pipeline(game_objects, effects=None):
    original_ids = {id(obj) for obj in game_objects}
    collisions.handle_collisions(game_objects)
    if effects is not None:
        effects.extend(obj for obj in game_objects if id(obj) not in original_ids)


class CollisionPipelineTests(CollisionTestCase):

    def make_named_special_object(
        self,
        special_object_class,
        name,
        *,
        hp=1,
        collides_with_fighters=True,
    ):
        special_object = self.make_special_object(
            special_object_class=special_object_class,
            collides_with_fighters=collides_with_fighters,
        )
        special_object.name = special_object.projectile_name = name
        special_object.current_hp = hp
        special_object.hp_array = [hp]
        return special_object

    def make_orz_marine_projectile_contact(self, death_animation):
        marine = self.make_special_object(special_object_class=OrzA3)
        marine.name = marine.projectile_name = "OrzA3"
        marine.special_object_collision_capabilities = replace(
            marine.special_object_collision_capabilities,
            projectile_contact_policy=(
                ProjectileContactPolicy.TAKE_DAMAGE_AND_DESTROY_PROJECTILE
            ),
        )
        marine.mode = OrzA3.OUTBOUND
        marine.get_collision_mask = lambda: None
        marine.boarded_ship = None
        marine._death_sound_played = False
        marine.die_sound = None

        projectile = self.make_projectile(self.make_ship())
        projectile.position = marine.position.copy()
        projectile.previous_position = projectile.position.copy()
        projectile.death_animation = death_animation
        return marine, projectile

    def make_kzerza_parent_contact(self, mode):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        parent.previous_position = parent.position.copy()
        parent.current_hp = 5
        parent.max_hp = 10

        fighter = self.make_special_object(
            special_object_class=KzerZaA2,
            collides_with_friendly_ships=True,
            collides_with_fighters=False,
        )
        fighter.name = fighter.projectile_name = "KzerZaA2"
        fighter.parent = parent
        fighter.hit_parent = False
        fighter.mode = mode
        fighter.return_sound = None
        fighter.position = parent.position.copy()
        fighter.previous_position = fighter.position.copy()
        return parent, fighter

    def test_kzerza_fighter_ignores_parent_before_returning(self):
        for mode in (KzerZaA2.LAUNCHING, KzerZaA2.ATTACKING):
            with self.subTest(mode=mode):
                parent, fighter = self.make_kzerza_parent_contact(mode)
                game_objects = [parent, fighter]

                collisions.handle_collisions(game_objects)

                self.assertEqual(parent.current_hp, 5)
                self.assertTrue(fighter.currently_alive)
                self.assertEqual(fighter.current_hp, 1)
                self.assertIn(fighter, game_objects)

    def test_returning_kzerza_fighter_is_recovered_by_parent(self):
        parent, fighter = self.make_kzerza_parent_contact(KzerZaA2.RETURNING)
        game_objects = [parent, fighter]

        collisions.handle_collisions(game_objects)

        self.assertEqual(parent.current_hp, 6)
        self.assertFalse(fighter.currently_alive)
        self.assertEqual(fighter.current_hp, 0)
        self.assertNotIn(fighter, game_objects)

    def test_ships_collide_elastically_without_damage(self):
        first = self.make_ship()
        second = self.make_ship()
        first.position = [100, 100]
        second.position = [118, 100]
        first.previous_position = first.position.copy()
        second.previous_position = second.position.copy()
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]

        collisions.handle_collisions([first, second])

        self.assertEqual(first.velocity, [-1.0, 0.0])
        self.assertEqual(second.velocity, [1.0, 0.0])
        self.assertEqual(first.current_hp, 10)
        self.assertEqual(second.current_hp, 10)

    def test_ship_ramming_damage_uses_projectile_contact_policy(self):
        ship = self.make_ship()
        ship.player = 1
        ship.position = [100, 100]
        ship.previous_position = ship.position.copy()
        ship.impact_capabilities = ImpactCapabilities(ramming_damage=3)
        special_object = self.make_special_object()
        special_object.player = 2
        special_object.parent = self.make_ship()
        special_object.hit_parent = False
        special_object.current_hp = 5
        special_object.hp_array = [5]
        special_object.handle_ship_contact = mock.Mock(return_value=True)
        special_object.special_object_collision_capabilities = replace(
            special_object.special_object_collision_capabilities,
            projectile_contact_policy=ProjectileContactPolicy.TAKE_DAMAGE,
        )

        collisions.handle_collisions([ship, special_object])

        self.assertTrue(special_object.currently_alive)
        self.assertEqual(special_object.current_hp, 2)
        special_object.handle_ship_contact.assert_called_once()

    def test_ship_ramming_damage_ignores_non_projectile_vulnerable_special_object(self):
        ship = self.make_ship()
        ship.player = 1
        ship.position = [100, 100]
        ship.previous_position = ship.position.copy()
        ship.impact_capabilities = ImpactCapabilities(ramming_damage=3)
        special_object = self.make_special_object()
        special_object.player = 2
        special_object.parent = self.make_ship()
        special_object.hit_parent = False
        special_object.current_hp = 5
        special_object.hp_array = [5]
        special_object.handle_ship_contact = mock.Mock(return_value=True)

        collisions.handle_collisions([ship, special_object])

        self.assertEqual(special_object.current_hp, 5)
        special_object.handle_ship_contact.assert_called_once()

    def test_asteroids_collide_elastically_without_damage(self):
        first = self.make_asteroid([100, 100])
        second = self.make_asteroid([118, 100])
        first.velocity = [1.0, 0.0]
        second.velocity = [-1.0, 0.0]

        collisions.handle_collisions([first, second])

        self.assertEqual(first.velocity, [-1.0, 0.0])
        self.assertEqual(second.velocity, [1.0, 0.0])
        self.assertTrue(first.currently_alive)
        self.assertTrue(second.currently_alive)

    def test_ship_and_asteroid_collide_elastically_without_damage(self):
        ship = self.make_ship()
        ship.position = [100, 100]
        ship.previous_position = ship.position.copy()
        ship.velocity = [1.0, 0.0]
        asteroid = self.make_asteroid([118, 100])
        asteroid.mass = 1.0
        asteroid.velocity = [-1.0, 0.0]

        collisions.handle_collisions([ship, asteroid])

        self.assertEqual(ship.velocity, [-1.0, 0.0])
        self.assertEqual(asteroid.velocity, [1.0, 0.0])
        self.assertEqual(ship.current_hp, 10)
        self.assertTrue(asteroid.currently_alive)

    def test_asteroid_is_consumed_by_planet_contact(self):
        asteroid = self.make_asteroid([100, 100])
        planet = self.make_planet([100, 100])
        game_objects = [asteroid, planet]

        with (
            mock.patch.object(collisions, "_spawn_replacement_asteroids"),
            mock.patch.object(collisions.BattleEffect, "play_boom") as play_boom,
        ):
            collisions.handle_collisions(game_objects)

        self.assertFalse(asteroid.currently_alive)
        self.assertNotIn(asteroid, game_objects)
        play_boom.assert_called_once_with(1)

    def test_ship_bounces_from_planet_and_tracks_contact(self):
        ship = self.make_ship()
        ship.position = [100, 100]
        ship.previous_position = ship.position.copy()
        ship.velocity = [3.0, 2.0]
        ship.current_hp = 11
        planet = self.make_planet([118, 100])
        planet.impact_capabilities = ImpactCapabilities(
            impact_damage_percent=0.25
        )

        with mock.patch.object(
            collisions.BattleEffect,
            "play_boom",
        ) as play_boom:
            collisions.handle_collisions([ship, planet])

        self.assertEqual(ship.velocity, [-3.0, 2.0])
        self.assertIn(id(planet), ship.planet_contacts)
        self.assertEqual(ship.current_hp, 9)
        play_boom.assert_called_once_with(2)

    def test_planet_impact_uses_ship_specific_damage_hook(self):
        ship = self.make_ship()
        ship.position = [100, 100]
        ship.previous_position = ship.position.copy()
        ship.velocity = [3.0, 0.0]
        ship.take_planet_impact_damage = mock.Mock(return_value=0)
        planet = self.make_planet([118, 100])
        planet.impact_capabilities = ImpactCapabilities(
            impact_damage_percent=0.25
        )

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([ship, planet])

        ship.take_planet_impact_damage.assert_called_once_with(2)
        self.assertEqual(ship.current_hp, 10)

    def test_projectile_is_consumed_by_planet_contact(self):
        projectile = self.make_projectile(self.make_ship())
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        planet = self.make_planet([100, 100])
        game_objects = [projectile, planet]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions(game_objects)

        self.assertFalse(projectile.currently_alive)
        self.assertNotIn(projectile, game_objects)

    def test_projectile_and_asteroid_consume_each_other(self):
        projectile = self.make_projectile(self.make_ship())
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        asteroid = self.make_asteroid([100, 100])
        game_objects = [projectile, asteroid]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions(game_objects)

        self.assertFalse(projectile.currently_alive)
        self.assertFalse(asteroid.currently_alive)
        self.assertNotIn(projectile, game_objects)
        self.assertNotIn(asteroid, game_objects)

    def test_projectile_damages_ship_and_is_consumed(self):
        projectile = self.make_projectile(self.make_ship())
        projectile.position = [100, 100]
        projectile.previous_position = projectile.position.copy()
        ship = self.make_ship()
        ship.position = [100, 100]
        ship.previous_position = ship.position.copy()
        game_objects = [projectile, ship]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions(game_objects)

        self.assertEqual(ship.current_hp, 6)
        self.assertFalse(projectile.currently_alive)
        self.assertNotIn(projectile, game_objects)

    def test_enemy_projectiles_consume_each_other(self):
        first, second = self.make_projectile_pair()
        game_objects = [first, second]

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions(game_objects)

        self.assertFalse(first.currently_alive)
        self.assertFalse(second.currently_alive)
        self.assertNotIn(first, game_objects)
        self.assertNotIn(second, game_objects)

    def test_each_asteroid_pair_is_dispatched_once(self):
        asteroids = [
            self.make_asteroid([100, 100]),
            self.make_asteroid([200, 100]),
            self.make_asteroid([300, 100]),
        ]

        with mock.patch.object(collisions, "_dispatch_collision_pair") as dispatch:
            collisions._run_collision_phases(
                {
                    "ships": [],
                    "asteroids": asteroids,
                    "projectiles": [],
                    "special_objects": [],
                    "planets": [],
                },
                [],
            )

        self.assertEqual(dispatch.call_count, 3)
        dispatched_pairs = {
            frozenset((id(call.args[0]), id(call.args[1])))
            for call in dispatch.call_args_list
        }
        self.assertEqual(
            dispatched_pairs,
            {
                frozenset((id(asteroids[0]), id(asteroids[1]))),
                frozenset((id(asteroids[0]), id(asteroids[2]))),
                frozenset((id(asteroids[1]), id(asteroids[2]))),
            },
        )

    def test_physical_collision_phase_order_is_explicit(self):
        self.assertEqual(
            [
                (phase.first_group, phase.second_group)
                for phase in collisions.COLLISION_PHASES
            ],
            [
                ("ships", None),
                ("ships", "asteroids"),
                ("asteroids", None),
                ("ships", "planets"),
                ("asteroids", "planets"),
                ("projectiles", None),
                ("projectiles", "ships"),
                ("projectiles", "asteroids"),
                ("projectiles", "planets"),
                ("special_objects", None),
                ("special_objects", "projectiles"),
                ("special_objects", "ships"),
                ("special_objects", "asteroids"),
                ("special_objects", "planets"),
            ],
        )

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

    def test_fighter_ignores_dead_asteroid_and_hits_next_live_target(self):
        special_object = self.make_special_object()
        dead = self.make_asteroid([108, 100])
        live = self.make_asteroid([108, 100])
        dead.currently_alive = False
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[special_object], *[dead, live]], [])
        self.assertFalse(special_object.currently_alive)
        self.assertFalse(live.currently_alive)

    def test_fighter_ignores_dead_projectile_and_hits_next_live_target(self):
        special_object = self.make_special_object()
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

    def test_fragile_special_objects_die_to_projectiles_without_harming_them(self):
        for special_object_class, name in (
            (VuxA2, "VuxA2"),
            (KzerZaA2, "KzerZaA2"),
            (SyreenCrew, "SyreenCrew"),
        ):
            with self.subTest(name=name):
                fighter = self.make_special_object(
                    special_object_class=special_object_class
                )
                fighter.name = fighter.projectile_name = name
                fighter.special_object_collision_capabilities = replace(
                    fighter.special_object_collision_capabilities,
                    projectile_contact_policy=ProjectileContactPolicy.FRAGILE,
                )
                fighter.physical_collision_capabilities = replace(
                    fighter.physical_collision_capabilities,
                    is_fragile=True,
                )
                projectile = self.make_projectile(self.make_ship())
                projectile.current_damage = 0
                projectile.current_hp = 3
                projectile.position = fighter.position.copy()
                projectile.previous_position = projectile.position.copy()

                with mock.patch.object(
                    collisions.BattleEffect, "play_boom"
                ):
                    collisions.handle_collisions([fighter, projectile])

                self.assertFalse(fighter.currently_alive)
                self.assertTrue(projectile.currently_alive)
                self.assertEqual(projectile.current_hp, 3)

    def test_chenjesu_a2_destroys_selected_special_objects_without_damage(self):
        for target_class, target_name in (
            (VuxA2, "VuxA2"),
            (SyreenCrew, "SyreenCrew"),
            (KzerZaA2, "KzerZaA2"),
        ):
            with self.subTest(target=target_name):
                cloud = self.make_named_special_object(
                    ChenjesuA2,
                    "ChenjesuA2",
                    hp=3,
                )
                cloud.mass = 4
                target = self.make_named_special_object(
                    target_class,
                    target_name,
                    collides_with_fighters=target_name != "KzerZaA2",
                )
                target.position = cloud.position.copy()
                target.previous_position = target.position.copy()

                with mock.patch.object(collisions.BattleEffect, "play_boom"):
                    collisions.handle_collisions([cloud, target])

                self.assertTrue(cloud.currently_alive)
                self.assertEqual(cloud.current_hp, 3)
                self.assertFalse(target.currently_alive)

    def test_orz_marine_bounces_from_chenjesu_without_deflecting_it(self):
        cloud = self.make_named_special_object(
            ChenjesuA2,
            "ChenjesuA2",
            hp=3,
        )
        cloud.mass = 4
        cloud.velocity = [1.0, 0.0]
        marine = self.make_named_special_object(OrzA3, "OrzA3", hp=3)
        marine.mode = OrzA3.OUTBOUND
        marine.get_collision_mask = lambda: None
        marine.velocity = [-1.0, 0.0]
        marine.position = [108, 100]
        marine.previous_position = marine.position.copy()

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([cloud, marine])

        self.assertTrue(cloud.currently_alive)
        self.assertTrue(marine.currently_alive)
        self.assertEqual(cloud.current_hp, 3)
        self.assertEqual(marine.current_hp, 3)
        self.assertEqual(cloud.velocity, [1.0, 0.0])
        self.assertEqual(cloud.position, [100, 100])
        self.assertGreater(marine.velocity[0], 0.0)

    def test_chenjesu_a2_bounces_from_asteroids_and_planets(self):
        cloud = self.make_named_special_object(
            ChenjesuA2,
            "ChenjesuA2",
            hp=3,
        )
        cloud.mass = 4
        cloud.velocity = [1.0, 0.0]
        asteroid = self.make_asteroid([108, 100])

        collisions.handle_collisions([cloud, asteroid])

        self.assertTrue(cloud.currently_alive)
        self.assertTrue(asteroid.currently_alive)
        self.assertLess(cloud.velocity[0], 1.0)
        self.assertGreater(asteroid.velocity[0], 0.0)

        cloud.position = [100, 100]
        cloud.previous_position = cloud.position.copy()
        cloud.velocity = [1.0, 0.0]
        planet = self.make_planet([108, 100])

        collisions.handle_collisions([cloud, planet])

        self.assertTrue(cloud.currently_alive)
        self.assertEqual(cloud.current_hp, 2)
        self.assertLess(cloud.velocity[0], 0.0)
        self.assertEqual(cloud.collision_wait_timer, 6)

    def test_kzerza_fighter_contact_does_not_damage_enemy_ship(self):
        parent = self.make_ship()
        parent.player = 1
        fighter = self.make_named_special_object(
            KzerZaA2,
            "KzerZaA2",
            collides_with_fighters=False,
        )
        fighter.parent = parent
        fighter.hit_parent = False
        target = self.make_ship()
        target.player = 2
        target.position = fighter.position.copy()
        target.previous_position = target.position.copy()
        starting_hp = target.current_hp

        collisions.handle_collisions([fighter, target])

        self.assertFalse(fighter.currently_alive)
        self.assertEqual(target.current_hp, starting_hp)

    def test_kzerza_fighter_contact_leaves_projectile_and_asteroid_intact(self):
        fighter = self.make_named_special_object(
            KzerZaA2,
            "KzerZaA2",
            collides_with_fighters=False,
        )
        fighter.special_object_collision_capabilities = type(
            fighter.special_object_collision_capabilities
        )(
            collides_with_projectiles=True,
            damages_projectiles=False,
            collides_with_asteroids=True,
            damages_asteroids=False,
            collides_with_fighters=False,
        )
        projectile = self.make_projectile(self.make_ship())
        projectile.current_damage = 2
        projectile.position = fighter.position.copy()
        projectile.previous_position = projectile.position.copy()

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([fighter, projectile])

        self.assertFalse(fighter.currently_alive)
        self.assertTrue(projectile.currently_alive)
        self.assertEqual(projectile.current_hp, 1)

        fighter = self.make_named_special_object(
            KzerZaA2,
            "KzerZaA2",
            collides_with_fighters=False,
        )
        fighter.special_object_collision_capabilities = type(
            fighter.special_object_collision_capabilities
        )(
            collides_with_asteroids=True,
            damages_asteroids=False,
            collides_with_fighters=False,
        )
        asteroid = self.make_asteroid(fighter.position)

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([fighter, asteroid])

        self.assertFalse(fighter.currently_alive)
        self.assertTrue(asteroid.currently_alive)

    def test_orz_marine_destroys_kzerza_fighter_without_damage(self):
        fighter = self.make_named_special_object(
            KzerZaA2,
            "KzerZaA2",
            collides_with_fighters=False,
        )
        marine = self.make_named_special_object(OrzA3, "OrzA3", hp=3)
        marine.mode = OrzA3.OUTBOUND
        marine.get_collision_mask = lambda: None
        marine.position = fighter.position.copy()
        marine.previous_position = marine.position.copy()

        with mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([fighter, marine])

        self.assertFalse(fighter.currently_alive)
        self.assertTrue(marine.currently_alive)
        self.assertEqual(marine.current_hp, 3)

    def test_kzerza_fighter_only_collides_with_approved_special_objects(self):
        fighter = self.make_named_special_object(KzerZaA2, "KzerZaA2")

        for name in ("VuxA2", "SyreenCrew", "KzerZaA2", "OtherSpecial"):
            with self.subTest(name=name):
                other = SimpleNamespace(type="special_object", name=name)
                self.assertFalse(
                    fighter.should_collide_with_projectile_like(other)
                )

        for name in ("ChenjesuA2", "OrzA3"):
            with self.subTest(name=name):
                other = SimpleNamespace(type="special_object", name=name)
                self.assertTrue(fighter.should_collide_with_projectile_like(other))

    def test_projectile_death_animation_is_emitted_when_killing_orz_marine(self):
        animation = [object()]
        marine, projectile = self.make_orz_marine_projectile_contact(animation)

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_animation",
                return_value=object(),
            ) as from_animation,
            mock.patch.object(collisions.BattleEffect, "from_blast"),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions([marine, projectile])

        self.assertTrue(
            any(call.args[1] is animation for call in from_animation.call_args_list)
        )

    def test_projectile_blast_is_emitted_when_killing_orz_marine(self):
        marine, projectile = self.make_orz_marine_projectile_contact([])

        with (
            mock.patch.object(
                collisions.BattleEffect,
                "from_blast",
                return_value=object(),
            ) as from_blast,
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions([marine, projectile])

        self.assertTrue(
            any(
                call.args[2] == projectile.current_damage
                for call in from_blast.call_args_list
            )
        )

    def test_kzerza_fighter_survives_planet_contact_and_begins_avoidance(self):
        fighter = self.make_special_object(special_object_class=KzerZaA2)
        fighter.name = fighter.projectile_name = "KzerZaA2"
        fighter.position = [100, 100]
        fighter.previous_position = fighter.position.copy()
        planet = self.make_planet([100, 100])
        game_objects = [fighter, planet]

        collisions.handle_collisions(game_objects)

        self.assertTrue(fighter.currently_alive)
        self.assertIn(fighter, game_objects)
        self.assertEqual(fighter.planet_avoidance, (planet, None))

    def test_fighter_skips_dead_ship_and_hits_next_live_target(self):
        special_object = self.make_special_object()
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
        first = self.make_special_object()
        dead = self.make_special_object()
        live = self.make_special_object()
        for special_object in (dead, live):
            special_object.position = [108, 100]
            special_object.previous_position = special_object.position.copy()
        dead.current_hp = 0
        dead.currently_alive = False
        with mock.patch.object(collisions.BattleEffect, 'from_blast'), mock.patch.object(collisions.BattleEffect, 'play_boom'):
            run_collision_pipeline([*[first, dead, live], *[]], None)
        self.assertFalse(first.currently_alive)
        self.assertFalse(live.currently_alive)

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

    def test_laser_damages_nonblocking_target_then_hits_blocker_behind_it(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        laser = self.make_laser(parent)
        fighter = self.make_special_object()
        fighter.position = [170, 100]
        fighter.laser_target_capabilities = LaserTargetCapabilities(
            blocks_lasers=False
        )
        enemy_ship = self.make_ship()
        enemy_ship.player = 2
        enemy_ship.position = [240, 100]

        with mock.patch.object(
            collisions.BattleEffect, "from_blast", return_value=object()
        ), mock.patch.object(collisions.BattleEffect, "play_boom"):
            collisions.handle_collisions([parent, enemy_ship, laser, fighter])

        self.assertFalse(fighter.currently_alive)
        self.assertEqual(enemy_ship.current_hp, 8)
        self.assertTrue(laser.intercepted)
        self.assertGreater(laser.end_position[0], fighter.position[0])

    def test_exact_laser_stops_at_explicit_nonblocking_target(self):
        parent = self.make_ship()
        parent.player = 1
        parent.position = [100, 100]
        intended = self.make_special_object()
        intended.parent = parent
        intended.hit_parent = False
        intended.position = [170, 100]
        intended.laser_target_capabilities = LaserTargetCapabilities(
            blocks_lasers=False
        )
        behind = self.make_special_object()
        behind.parent = parent
        behind.hit_parent = False
        behind.position = [220, 100]
        behind.laser_target_capabilities = LaserTargetCapabilities(
            blocks_lasers=False
        )
        laser = self.make_laser(parent, target=intended)
        laser.calculate_end_position = mock.Mock()

        with (
            mock.patch.object(collisions.BattleEffect, "from_blast", return_value=object()),
            mock.patch.object(collisions.BattleEffect, "play_boom"),
        ):
            collisions.handle_collisions([parent, laser, intended, behind])

        self.assertFalse(intended.currently_alive)
        self.assertTrue(behind.currently_alive)
        self.assertEqual(laser.end_position, intended.position)

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
