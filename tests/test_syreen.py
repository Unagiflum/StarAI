import math
import unittest
from types import SimpleNamespace

import src.const as const
from src.Battle import collisions
from src.Battle.collision_geometry import collision_info, objects_overlap
from src.Battle.collision_responses import generic_area_damage_target_is_eligible
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.Syreen.A2.SyreenA2 import SyreenA2
from src.Objects.Ships.Syreen.A2.SyreenCrew import SyreenCrew
from src.collision_capabilities import (
    AreaDamageCapabilities,
    DurabilityCapabilities,
    PhysicalCollisionCapabilities,
)
from src.toroidal import wrapped_delta


class SyreenCrewMotionTests(unittest.TestCase):
    def setUp(self):
        self.parent = create_ship("Syreen", 1)
        self.parent.position = [500.0, 300.0]
        self.parent.velocity = [0.0, 0.0]
        self.parent.planet = None
        self.planet = SimpleNamespace(
            position=[600.0, 500.0], diameter=100.0, gravity=1.0
        )

    def make_crew(self):
        crew = SyreenCrew(self.parent, [500.0, 500.0])
        crew.planet = self.planet
        return crew

    def test_default_constructor_uses_parent_position_and_configured_radius(self):
        crew = SyreenCrew(self.parent)

        self.assertEqual(crew.position, self.parent.position)
        self.assertIsNot(crew.position, self.parent.position)
        self.assertEqual(crew.size, [8, 8])
        self.assertEqual(crew.get_sprite().get_size(), (8, 8))

    def test_prediction_uses_the_same_motion_step_as_runtime(self):
        crew = self.make_crew()

        predicted = crew.predict_unhindered_trajectory(6)
        actual = []
        for _ in range(6):
            crew.update_physics()
            actual.append(list(crew.position))

        self.assertNotEqual(predicted[0], [500.0, 500.0])
        for predicted_position, actual_position in zip(predicted, actual):
            self.assertAlmostEqual(predicted_position[0], actual_position[0])
            self.assertAlmostEqual(predicted_position[1], actual_position[1])

    def test_parent_death_does_not_reapply_the_whole_gravity_velocity(self):
        crew = self.make_crew()
        crew.update_physics()
        self.parent.currently_alive = False
        crew.planet = None

        crew.update_physics()
        first_velocity = list(crew.velocity)
        crew.update_physics()

        self.assertEqual(crew.velocity, first_velocity)
        self.assertEqual(crew.velocity, crew.gravity_velocity)

    def test_persistent_gravity_component_respects_speed_limit(self):
        crew = self.make_crew()
        crew.planet = None
        self.parent.currently_alive = False
        crew.gravity_velocity = [const.SPEED_LIMIT * 2, 0.0]
        start = list(crew.position)

        crew.update_physics()

        self.assertLessEqual(math.hypot(*crew.velocity), const.SPEED_LIMIT)
        delta = wrapped_delta(start, crew.position)
        self.assertLessEqual(math.hypot(*delta), const.SPEED_LIMIT * const.SPEED_SCALE)


class SyreenSongTests(unittest.TestCase):
    def setUp(self):
        self.parent = create_ship("Syreen", 1)
        self.parent.position = [500.0, 500.0]
        self.parent.in_battle = True
        self.parent.planet = None

    def test_catalog_flags_limit_song_to_enemy_ships(self):
        definition = ABILITY_DEFINITIONS["SyreenA2"]
        song = SyreenA2(self.parent)
        projectile = SimpleNamespace(
            currently_alive=True,
            player=2,
            area_damage_capabilities=AreaDamageCapabilities(targetable=True),
            physical_collision_capabilities=PhysicalCollisionCapabilities(
                is_solid=False, is_projectile=True
            ),
        )

        self.assertTrue(definition.is_psychic)
        self.assertTrue(definition.ignores_shields)
        self.assertFalse(definition.collide_planets)
        self.assertFalse(definition.collide_asteroids)
        self.assertFalse(definition.collide_projectiles)
        self.assertFalse(definition.collide_fighters)
        self.assertFalse(generic_area_damage_target_is_eligible(song, projectile))

    def test_psychic_immunity_uses_durability_capability(self):
        song = SyreenA2(self.parent)
        target = create_ship("Earthling", 2)
        target.durability_capabilities = DurabilityCapabilities(
            immune_to_psychic=True
        )

        self.assertFalse(generic_area_damage_target_is_eligible(song, target))
        self.assertEqual(song.area_damage_for_target(target, 0), 0)

    def test_song_penetrates_yehat_and_utwig_shields(self):
        for ship_name in ("Yehat", "Utwig"):
            with self.subTest(ship=ship_name):
                song = SyreenA2(self.parent)
                target = create_ship(ship_name, 2)
                target.position = [510.0, 500.0]
                target.current_hp = 5
                shield = target.perform_action2()
                energy_after_activation = target.current_energy

                self.assertIsNotNone(shield)
                self.assertTrue(target.damage_shield_is_active())

                collisions._handle_area_damage([song, target], [])

                self.assertEqual(target.current_hp, 1)
                self.assertEqual(target.current_energy, energy_after_activation)
                self.assertEqual(len(song.drain_spawned_objects()), 4)

    def test_spawn_count_matches_actual_nonlethal_damage(self):
        song = SyreenA2(self.parent)
        target = create_ship("Earthling", 2)
        target.position = [510.0, 500.0]
        target.current_hp = 5

        collisions._handle_area_damage([song, target], [])

        self.assertEqual(target.current_hp, 1)
        self.assertEqual(len(song.drain_spawned_objects()), 4)

    def test_song_tracks_parent_before_area_collision_processing(self):
        song = SyreenA2(self.parent)
        self.parent.position = [750.0, 900.0]

        song.update()

        self.assertEqual(song.position, self.parent.position)
        self.assertEqual(song.range, ABILITY_DEFINITIONS["SyreenA2"].effect_range)

    def test_crew_spawn_outside_cloaked_ilwrath_hull_and_move_away(self):
        target = create_ship("Ilwrath", 2)
        target.position = [620.0, 500.0]
        target.current_hp = 5
        target.cloak()
        target.fade_timer = target.FADE_DURATION
        song = SyreenA2(self.parent)

        collisions._handle_area_damage([song, target], [])
        crews = song.drain_spawned_objects()

        self.assertEqual(len(crews), 4)
        for crew in crews:
            _, _, overlap = collision_info(crew, target)
            self.assertFalse(objects_overlap(crew, target, overlap))
            crew.update_physics()
            _, _, overlap = collision_info(crew, target)
            self.assertFalse(objects_overlap(crew, target, overlap))


if __name__ == "__main__":
    unittest.main()
