import math
import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle.collisions import handle_collisions
from src.Battle.effects import BattleEffect
from src.Objects.Space.space_obj import Asteroid
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ability, create_ship
from src.resources import AssetManager


class AndrosynthTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.resources = AssetManager()
        self.ship = create_ship("Androsynth", 1, resources=self.resources)
        self.ship.position = [1000.0, 1000.0]
        self.ship.previous_position = self.ship.position.copy()

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_catalog_and_assets_define_base_and_blazer_forms(self):
        definition = SHIP_DEFINITIONS["Androsynth"]

        self.assertEqual(definition.ship_type, "Guardian")
        self.assertEqual(tuple(definition.forms), ("Base", "A2"))
        self.assertTrue(definition.forms["Base"].inertia)
        self.assertFalse(definition.forms["A2"].inertia)
        self.assertEqual(definition.forms["A2"].energy_wait, 8)
        self.assertEqual(definition.forms["A2"].collision_damage, 3)
        self.assertEqual(ABILITY_DEFINITIONS["AndrosynthA1"].frames, 2)
        self.assertEqual(len(self.resources.ability("AndrosynthA1").sprites[0]), 2)
        self.assertEqual(len(self.resources.ship_form("Androsynth", "A2").sprites), 80)
        self.assertFalse(self.resources._asset_errors)

    def test_bubble_uses_uqm_facing_bias_toward_target(self):
        target = create_ship("Earthling", 2, resources=self.resources)
        target.position = [1100.0, 1000.0]
        self.ship.opponent = target
        self.ship.rng = mock.Mock()
        self.ship.rng.randrange.return_value = 3
        bubble = create_ability("AndrosynthA1", self.ship)

        bubble.update_heading()

        # Starting north with a target to the east, TrackShip first turns one
        # facing clockwise, then applies the selected three-facing offset.
        self.assertEqual(bubble.rotation, 4 * 22.5)
        self.assertAlmostEqual(math.hypot(*bubble.velocity), 32.0)
        self.ship.rng.randrange.assert_called_once_with(8)

    def test_bubble_chooses_a_random_facing_without_a_visible_target(self):
        self.ship.rng = mock.Mock()
        self.ship.rng.randrange.return_value = 11
        bubble = create_ability("AndrosynthA1", self.ship)

        bubble.update_heading()

        self.assertEqual(bubble.rotation, 247.5)
        self.ship.rng.randrange.assert_called_once_with(16)

    def test_bubble_steers_on_first_update_then_every_three_updates(self):
        self.ship.rng = mock.Mock()
        self.ship.rng.randrange.return_value = 0
        bubble = create_ability("AndrosynthA1", self.ship)

        bubble.update_heading()
        self.assertEqual(self.ship.rng.randrange.call_count, 1)
        self.assertEqual(bubble.turn_timer, 3)

        for _ in range(3):
            bubble.update_heading()
        self.assertEqual(self.ship.rng.randrange.call_count, 1)

        bubble.update_heading()
        self.assertEqual(self.ship.rng.randrange.call_count, 2)

    def test_bubble_animation_uses_uqm_random_zero_to_three_frame_wait(self):
        self.ship.rng = mock.Mock()
        self.ship.rng.randrange.side_effect = [0, 3]
        bubble = create_ability("AndrosynthA1", self.ship)
        starting_frame = bubble.current_frame

        bubble.update()

        self.assertNotEqual(bubble.current_frame, starting_frame)
        self.assertEqual(bubble.frame_timer, 3)
        self.ship.rng.randrange.assert_has_calls([mock.call(16), mock.call(4)])

    def test_bubble_animation_tolerates_single_size_entry(self):
        self.ship.rng = mock.Mock()
        self.ship.rng.randrange.side_effect = [0, 0]
        bubble = create_ability("AndrosynthA1", self.ship)
        bubble.sizes = (bubble.sizes[0],)

        bubble.update()

        self.assertEqual(bubble.current_frame, 1)
        self.assertEqual(bubble.size, list(bubble.sizes[0]))

    def test_transform_preserves_shared_state_then_forces_forward_thrust(self):
        self.ship.velocity = [12.0, -7.0]
        self.ship.heading = 9
        self.ship.rotation = self.ship.heading * 22.5
        self.ship.current_hp = 13
        self.ship.current_energy = 10

        result = self.ship.commit_action(self.ship.plan_action2())

        self.assertTrue(result.valid)
        self.assertTrue(self.ship.is_blazer)
        self.assertEqual(self.ship.current_energy, 10)
        self.assertEqual(self.ship.current_hp, 13)
        self.assertEqual(self.ship.heading, 9)
        self.assertAlmostEqual(math.hypot(*self.ship.velocity), math.hypot(12, -7))
        heading_angle = math.radians(self.ship.rotation)
        self.assertAlmostEqual(
            self.ship.velocity[0], math.sin(heading_angle) * math.hypot(12, -7)
        )
        self.assertAlmostEqual(
            self.ship.velocity[1], -math.cos(heading_angle) * math.hypot(12, -7)
        )
        self.assertFalse(self.ship.inertia)
        self.assertEqual(self.ship.mass, 1)

        self.assertEqual(self.ship.process_controls(frame_id=1), [])
        self.ship.update_physics()
        self.assertAlmostEqual(math.hypot(*self.ship.velocity), math.hypot(12, -7))
        self.assertEqual(self.ship.process_controls(frame_id=2), [])
        self.ship.update_physics()
        self.assertAlmostEqual(math.hypot(*self.ship.velocity), 60.0)

    def test_blocked_transform_commits_wait_without_spending_energy(self):
        target = create_ship("Earthling", 2, resources=self.resources)
        target.position = self.ship.position.copy()
        target.previous_position = target.position.copy()
        self.ship.opponent = target
        self.ship.current_energy = 10

        result = self.ship.commit_action(self.ship.plan_action2())

        self.assertTrue(result.valid)
        self.assertEqual(self.ship.current_energy, 10)
        self.assertEqual(self.ship.action2_timer, 1)
        self.assertFalse(self.ship.is_blazer)

    def test_transform_requires_two_energy_without_spending_it(self):
        self.ship.current_energy = 1
        self.assertFalse(self.ship.plan_action2().valid)

        self.ship.current_energy = 2
        result = self.ship.commit_action(self.ship.plan_action2())

        self.assertTrue(result.valid)
        self.assertTrue(self.ship.is_blazer)
        self.assertEqual(self.ship.current_energy, 2)

    def test_blazer_drains_one_energy_every_nine_updates(self):
        self.ship.current_energy = 10
        result = self.ship.commit_action(self.ship.plan_action2())
        self.assertTrue(result.valid)

        for _ in range(8):
            self.ship.update_timers()
        self.assertEqual(self.ship.current_energy, 10)

        self.ship.update_timers()
        self.assertEqual(self.ship.current_energy, 9)

    def test_zero_energy_immediately_forces_unblocked_return_to_base(self):
        self.assertTrue(self.ship._try_transform())
        self.ship.current_energy = 1
        self.ship.energy_timer = self.ship.energy_wait

        self.ship.update_timers()

        self.assertEqual(self.ship.current_energy, 0)
        self.assertFalse(self.ship.is_blazer)
        self.assertTrue(self.ship.inertia)

    def test_limpets_are_form_scoped(self):
        self.ship.attach_limpet()
        self.assertEqual(self.ship.limpets_attached, 1)

        self.assertTrue(self.ship._try_transform())
        self.assertEqual(self.ship.limpets_attached, 0)
        self.ship.attach_limpet()
        self.ship.attach_limpet()
        self.assertEqual(self.ship.limpets_attached, 2)

        self.ship._activate_form(self.ship.BASE)
        self.assertEqual(self.ship.limpets_attached, 1)

    def test_blazer_destroys_limpets_and_marines_before_they_attach(self):
        self.assertTrue(self.ship._try_transform())

        vux = create_ship("Vux", 2, resources=self.resources)
        vux.position = [2000.0, 2000.0]
        limpet = create_ability("VuxA2", vux)
        limpet.position = self.ship.position.copy()
        limpet.previous_position = limpet.position.copy()
        handle_collisions([self.ship, vux, limpet])
        self.assertFalse(limpet.currently_alive)
        self.assertEqual(self.ship.limpets_attached, 0)

        orz = create_ship("Orz", 2, resources=self.resources)
        orz.position = [2000.0, 2000.0]
        orz.previous_position = orz.position.copy()
        orz.opponent = self.ship
        marine = create_ability("OrzA3", orz)
        marine.mode = marine.OUTBOUND
        marine.position = self.ship.position.copy()
        marine.previous_position = marine.position.copy()
        handle_collisions([self.ship, orz, marine])
        self.assertFalse(marine.currently_alive)
        self.assertEqual(self.ship.boarded_marines, [])

    def test_blazer_marine_kill_plays_impact_and_marine_death_sounds(self):
        self.assertTrue(self.ship._try_transform())
        orz = create_ship("Orz", 2, resources=self.resources)
        orz.position = [2000.0, 2000.0]
        orz.previous_position = orz.position.copy()
        orz.opponent = self.ship
        marine = create_ability("OrzA3", orz)
        marine.mode = marine.OUTBOUND
        marine.position = self.ship.position.copy()
        marine.previous_position = marine.position.copy()
        marine.die_sound = mock.Mock()

        with mock.patch.object(BattleEffect, "play_boom") as play_boom:
            handle_collisions([self.ship, orz, marine])

        play_boom.assert_called_once_with(3)
        marine.die_sound.play.assert_called_once_with()

    def test_surviving_marine_bounces_as_if_it_hit_a_shield(self):
        self.assertTrue(self.ship._try_transform())
        orz = create_ship("Orz", 2, resources=self.resources)
        orz.position = [2000.0, 2000.0]
        orz.opponent = self.ship
        marine = create_ability("OrzA3", orz)
        marine.mode = marine.OUTBOUND
        marine.current_hp = 4
        marine.position = self.ship.position.copy()
        marine.previous_position = [self.ship.position[0], self.ship.position[1] - 20]
        marine.velocity = [0.0, 10.0]

        handle_collisions([self.ship, orz, marine])

        self.assertTrue(marine.currently_alive)
        self.assertEqual(marine.current_hp, 1)
        self.assertEqual(marine.position, marine.previous_position)
        self.assertGreater(marine.shield_bounce_timer, 0)
        self.assertEqual(self.ship.boarded_marines, [])

    def test_blazer_destroys_syreen_crew_instead_of_recovering_it(self):
        self.assertTrue(self.ship._try_transform())
        syreen = create_ship("Syreen", 2, resources=self.resources)
        syreen.position = [2000.0, 2000.0]
        crew = create_ability("SyreenCrew", syreen)
        crew.position = self.ship.position.copy()
        crew.previous_position = crew.position.copy()
        starting_hp = self.ship.current_hp

        handle_collisions([self.ship, syreen, crew])

        self.assertFalse(crew.currently_alive)
        self.assertEqual(self.ship.current_hp, starting_hp)

    def test_blazer_damages_dogi_through_central_collision_handling(self):
        self.assertTrue(self.ship._try_transform())
        chenjesu = create_ship("Chenjesu", 2, resources=self.resources)
        chenjesu.position = [2000.0, 2000.0]
        chenjesu.previous_position = chenjesu.position.copy()
        chenjesu.opponent = self.ship
        dogi = create_ability("ChenjesuA2", chenjesu)
        dogi.position = self.ship.position.copy()
        dogi.previous_position = dogi.position.copy()
        starting_energy = self.ship.current_energy

        handle_collisions([self.ship, chenjesu, dogi])

        self.assertFalse(dogi.currently_alive)
        self.assertEqual(dogi.current_hp, 0)
        self.assertEqual(self.ship.current_energy, starting_energy - dogi.drain)

    def test_blazer_rams_ships_and_destroys_asteroids(self):
        self.assertTrue(self.ship._try_transform())
        target = create_ship("Earthling", 2, resources=self.resources)
        target.position = self.ship.position.copy()
        target.previous_position = target.position.copy()
        starting_hp = target.current_hp

        handle_collisions([self.ship, target])

        self.assertEqual(target.current_hp, starting_hp - 3)

        asteroid = Asteroid(resources=self.resources)
        asteroid.position = self.ship.position.copy()
        asteroid.previous_position = asteroid.position.copy()
        objects = [self.ship, asteroid]
        handle_collisions(objects)

        self.assertFalse(asteroid.currently_alive)
        self.assertNotIn(asteroid, objects)


if __name__ == "__main__":
    unittest.main()
