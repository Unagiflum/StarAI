import math
import unittest
from types import SimpleNamespace

import src.const as const
from src.Battle import collisions
from src.Battle.battle_aftermath import hide_dead_ship
from src.Battle.collision_geometry import collision_info, objects_overlap
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.Objects.Ships.Syreen.A2.SyreenA2 import (
    SyreenA2,
    SyreenSongEffect,
)
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

    def test_free_crew_detaches_and_survives_parent_cleanup(self):
        crew = self.make_crew()
        game_objects = [self.parent, crew]

        hide_dead_ship(self.parent, game_objects)

        self.assertTrue(crew.currently_alive)
        self.assertIsNone(crew.parent)
        self.assertIn(crew, game_objects)

    def test_default_constructor_uses_parent_position_and_configured_radius(self):
        crew = SyreenCrew(self.parent)

        self.assertEqual(crew.position, self.parent.position)
        self.assertIsNot(crew.position, self.parent.position)
        self.assertEqual(crew.expiration_timer, 300)
        self.assertEqual(crew.size, [6, 6])
        self.assertEqual(crew.get_sprite().get_size(), (6, 6))

    def test_crew_immune_slylandro_consumes_crew_without_recovering_it(self):
        crew = SyreenCrew(self.parent)
        slylandro = create_ship("Slylandro", 2)
        slylandro.current_hp -= 1
        starting_hp = slylandro.current_hp

        crew.handle_ship_contact(slylandro, None)

        self.assertEqual(slylandro.current_hp, starting_hp)
        self.assertFalse(crew.currently_alive)

    def test_configured_colors_cycle_each_physics_frame(self):
        crew = SyreenCrew(self.parent)
        crew.colors = ((10, 20, 30, 40), (50, 60, 70, 80))

        self.assertEqual(crew.get_sprite().get_at((4, 4)), (10, 20, 30, 40))
        crew.update_physics()
        self.assertEqual(crew.get_sprite().get_at((4, 4)), (50, 60, 70, 80))
        crew.update_physics()
        self.assertEqual(crew.get_sprite().get_at((4, 4)), (10, 20, 30, 40))

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

    def test_tracks_origin_ship_after_calling_syreen_dies(self):
        origin = create_ship("Earthling", 2)
        origin.position = [700.0, 500.0]
        origin.velocity = [0.0, 0.0]
        crew = SyreenCrew(
            self.parent, [500.0, 500.0], origin_ship=origin
        )
        crew.planet = None

        crew.update_physics()
        self.assertLess(crew.velocity[1], 0.0)
        self.assertAlmostEqual(crew.velocity[0], 0.0)

        self.parent.currently_alive = False
        crew.update_physics()

        self.assertGreater(crew.velocity[0], 0.0)
        self.assertAlmostEqual(crew.velocity[1], 0.0)

    def test_dead_origin_ship_leaves_crew_moving_only_by_gravity(self):
        origin = create_ship("Earthling", 2)
        origin.position = [700.0, 500.0]
        origin.currently_alive = False
        self.parent.currently_alive = False
        crew = SyreenCrew(
            self.parent, [500.0, 500.0], origin_ship=origin
        )
        crew.planet = None
        crew.gravity_velocity = [3.0, -2.0]

        crew.update_physics()

        self.assertEqual(crew.velocity, [3.0, -2.0])

    def test_cloaked_origin_ship_leaves_crew_moving_only_by_gravity(self):
        origin = create_ship("Ilwrath", 2)
        origin.position = [700.0, 500.0]
        origin.cloaked = True
        self.parent.currently_alive = False
        crew = SyreenCrew(
            self.parent, [500.0, 500.0], origin_ship=origin
        )
        crew.planet = None

        crew.update_physics()

        self.assertEqual(crew.velocity, [0.0, 0.0])

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

    @staticmethod
    def drain_crews(song):
        return [
            spawned
            for spawned in song.drain_spawned_objects()
            if isinstance(spawned, SyreenCrew)
        ]

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
        self.assertFalse(
            collisions.AREA_TARGET_REGISTRY.is_eligible(song, projectile)
        )

    def test_song_uses_uqm_crew_loss_bands(self):
        song = SyreenA2(self.parent)
        target = create_ship("Earthling", 2)
        target.current_hp = target.max_hp = 20

        self.assertEqual(song.area_damage_for_target(target, 0), 9)
        self.assertEqual(song.area_damage_for_target(target, song.range / 2), 5)
        self.assertEqual(song.area_damage_for_target(target, song.range), 1)
        self.assertEqual(song.area_damage_for_target(target, song.range + 1), 0)

    def test_psychic_immunity_uses_durability_capability(self):
        song = SyreenA2(self.parent)
        target = create_ship("Earthling", 2)
        target.durability_capabilities = DurabilityCapabilities(
            immune_to_psychic=True
        )

        self.assertFalse(collisions.AREA_TARGET_REGISTRY.is_eligible(song, target))
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
                self.assertEqual(len(self.drain_crews(song)), 4)

    def test_spawn_count_matches_actual_nonlethal_damage(self):
        song = SyreenA2(self.parent)
        target = create_ship("Earthling", 2)
        target.position = [510.0, 500.0]
        target.current_hp = 5

        collisions._handle_area_damage([song, target], [])

        self.assertEqual(target.current_hp, 1)
        crews = self.drain_crews(song)
        self.assertEqual(len(crews), 4)
        self.assertTrue(all(crew.origin_ship is target for crew in crews))

    def test_song_tracks_parent_before_area_collision_processing(self):
        song = SyreenA2(self.parent)
        self.parent.position = [750.0, 900.0]

        song.update()

        self.assertEqual(song.position, song.configured_gun_position())
        self.assertEqual(song.range, ABILITY_DEFINITIONS["SyreenA2"].range)

    def test_crew_spawn_outside_cloaked_ilwrath_hull_and_move_away(self):
        target = create_ship("Ilwrath", 2)
        target.position = [620.0, 500.0]
        target.current_hp = 5
        target.cloak()
        target.fade_timer = target.FADE_DURATION
        song = SyreenA2(self.parent)

        collisions._handle_area_damage([song, target], [])
        crews = self.drain_crews(song)

        self.assertEqual(len(crews), 4)
        for crew in crews:
            _, _, overlap = collision_info(crew, target)
            self.assertFalse(objects_overlap(crew, target, overlap))
            crew.update_physics()
            _, _, overlap = collision_info(crew, target)
            self.assertFalse(objects_overlap(crew, target, overlap))

    def test_song_spawns_a_fixed_origin_battle_effect(self):
        song = SyreenA2(self.parent)
        effect = next(
            spawned
            for spawned in song.drain_spawned_objects()
            if isinstance(spawned, SyreenSongEffect)
        )
        origin = list(effect.position)

        self.parent.position = [750.0, 900.0]
        song.update()

        self.assertEqual(effect.position, origin)
        self.assertEqual(effect.render_layer, "after_lasers")

    def test_song_effect_uses_full_range_and_anim_length(self):
        song = SyreenA2(self.parent)
        effect = next(
            spawned
            for spawned in song.drain_spawned_objects()
            if isinstance(spawned, SyreenSongEffect)
        )
        definition = ABILITY_DEFINITIONS["SyreenA2"]

        self.assertEqual(effect.radius, definition.range)
        self.assertEqual(effect.total_frames, definition.anim_length)

    def test_ring_radius_is_fixed_and_color_pulses(self):
        effect = SyreenSongEffect(
            position=[100.0, 100.0],
            radius=80.0,
            thickness=8,
            colors=((200, 100, 50, 240), (100, 50, 0, 40)),
            total_frames=5,
        )

        samples = []
        for frame in range(5):
            effect.current_frame = frame
            samples.append(effect.radius_and_color())

        self.assertEqual([sample[0] for sample in samples], [80.0] * 5)
        self.assertEqual(samples[0][1], (200, 100, 50, 240))
        self.assertEqual(samples[2][1], (100, 50, 0, 40))
        self.assertEqual(samples[-1][1], (200, 100, 50, 240))

    def test_ring_alpha_is_full_at_centerline_and_fades_at_edges(self):
        import pygame

        effect = SyreenSongEffect(
            position=[100.0, 100.0],
            radius=50.0,
            thickness=8,
            colors=((255, 100, 255, 200), (255, 100, 255, 200)),
            total_frames=2,
        )
        surface = pygame.Surface(
            (const.SCREEN_WIDTH, const.SCREEN_HEIGHT), pygame.SRCALPHA
        )
        translation = [
            const.SCREEN_HEIGHT / 2 - effect.position[0],
            const.SCREEN_HEIGHT / 2 - effect.position[1],
        ]

        effect.draw(surface, 1.0, translation)

        center_x = const.SCREEN_LEFT + const.SCREEN_HEIGHT // 2
        center_y = const.SCREEN_HEIGHT // 2
        middle_alpha = surface.get_at((center_x + 50, center_y)).a
        near_edge_alpha = surface.get_at((center_x + 53, center_y)).a
        outside_alpha = surface.get_at((center_x + 55, center_y)).a
        self.assertEqual(middle_alpha, 200)
        self.assertEqual(near_edge_alpha, 100)
        self.assertEqual(outside_alpha, 0)

    def test_ring_surrounding_view_is_culled_without_changing_viewport_path(self):
        import pygame

        effect = SyreenSongEffect(
            position=[50.0, 50.0],
            radius=200.0,
            thickness=8,
            colors=((255, 100, 255, 200), (255, 100, 255, 200)),
            total_frames=2,
        )
        surface = pygame.Surface((100, 100), pygame.SRCALPHA)

        effect.draw(surface, 1.0, [0.0, 0.0])

        self.assertEqual(surface.get_bounding_rect(), pygame.Rect(0, 0, 0, 0))

    def test_ring_overlay_cache_is_bounded_to_main_and_viewport_sizes(self):
        SyreenSongEffect._overlay_cache.clear()

        for size in ((960, 960), (200, 200), (320, 240)):
            SyreenSongEffect._overlay(size)

        self.assertEqual(len(SyreenSongEffect._overlay_cache), 2)


if __name__ == "__main__":
    unittest.main()
