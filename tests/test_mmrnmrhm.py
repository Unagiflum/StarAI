import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

from src.Battle.collisions import handle_collisions
from src.Battle.effects import BattleEffect
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS
from src.Objects.Ships.registry import create_ship
from src.resources import AssetManager


class MmrnmrhmTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        self.resources = AssetManager()
        self.ship = create_ship("Mmrnmrhm", 1, resources=self.resources)
        self.ship.position = [1000.0, 1000.0]
        self.ship.previous_position = self.ship.position.copy()

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled

    def test_catalog_and_assets_define_two_forms(self):
        definition = SHIP_DEFINITIONS["Mmrnmrhm"]

        self.assertEqual(definition.default_form, "XForm")
        self.assertEqual(tuple(definition.forms), ("XForm", "YWing"))
        self.assertEqual(definition.forms["XForm"].energy_regen, 2)
        self.assertEqual(definition.forms["YWing"].max_thrust, 50)
        self.assertEqual(self.ship.form, "XForm")
        self.assertEqual(self.ship.size, [104, 60])
        self.assertFalse(self.resources._asset_errors)

    def test_xform_primary_fires_converging_beams_from_both_guns(self):
        plan = self.ship.plan_action1()

        self.assertTrue(plan.valid)
        self.assertEqual(len(plan.spawned_objects), 2)
        left, right = plan.spawned_objects
        self.assertEqual(left.start_position, [951.0, 985.0])
        self.assertEqual(right.start_position, [1048.0, 985.0])
        self.assertEqual(left.end_position, [1000.0, 436.0])
        self.assertEqual(right.end_position, [1000.0, 436.0])

    def test_ywing_primary_fires_tracking_projectiles_at_configured_angles(self):
        self.assertTrue(self.ship._try_transform())
        self.ship.current_energy = self.ship.max_energy

        plan = self.ship.plan_action1()

        self.assertTrue(plan.valid)
        self.assertEqual(len(plan.spawned_objects), 2)
        left, right = plan.spawned_objects
        self.assertEqual(left.position, [988.0, 983.0])
        self.assertEqual(right.position, [1010.0, 983.0])
        self.assertEqual(left.rotation, 337.5)
        self.assertEqual(right.rotation, 22.5)
        self.assertTrue(left.tracking)
        self.assertTrue(right.tracking)

    def test_transform_changes_form_stats_without_changing_shared_motion(self):
        self.ship.velocity = [12.0, -7.0]
        self.ship.heading = 9
        self.ship.current_hp = 13
        self.ship.current_energy = 10

        result = self.ship.commit_action(self.ship.plan_action2())

        self.assertTrue(result.valid)
        self.assertEqual(self.ship.form, "YWing")
        self.assertEqual(self.ship.current_energy, 0)
        self.assertEqual(self.ship.current_hp, 13)
        self.assertEqual(self.ship.heading, 9)
        self.assertEqual(self.ship.velocity, [12.0, -7.0])
        self.assertEqual(self.ship.energy_regen, 1)
        self.assertEqual(self.ship.max_thrust, 50)
        self.assertEqual(self.ship.turn_wait, 14)
        self.assertEqual(self.ship.a1_wait, 20)
        self.assertEqual(self.ship.size, [77, 79])

    def test_blocked_transform_spends_energy_but_keeps_current_form(self):
        target = create_ship("Earthling", 2, resources=self.resources)
        target.position = self.ship.position.copy()
        target.previous_position = target.position.copy()
        self.ship.opponent = target
        self.ship.current_energy = 10

        result = self.ship.commit_action(self.ship.plan_action2())

        self.assertTrue(result.valid)
        self.assertEqual(self.ship.current_energy, 0)
        self.assertEqual(self.ship.form, "XForm")

    def test_limpets_are_form_scoped_while_marines_are_ship_scoped(self):
        marine = object()
        self.ship.boarded_marines.append(marine)
        self.ship.attach_limpet()
        x_sprites = self.ship.sprites

        self.assertEqual(self.ship.limpets_attached, 1)
        self.assertEqual(self.ship.turn_wait, 3)
        self.assertEqual(self.ship.max_thrust, 16)
        self.assertTrue(self.ship._try_transform())
        self.assertEqual(self.ship.limpets_attached, 0)
        self.assertEqual(self.ship.turn_wait, 14)
        self.assertEqual(self.ship.max_thrust, 50)
        self.assertEqual(self.ship.boarded_marines, [marine])

        self.ship.attach_limpet()
        self.ship.attach_limpet()
        self.assertEqual(self.ship.limpets_attached, 2)
        self.assertEqual(self.ship.turn_wait, 16)
        self.assertEqual(self.ship.max_thrust, 40)

        self.assertTrue(self.ship._try_transform())
        self.assertEqual(self.ship.limpets_attached, 1)
        self.assertEqual(self.ship.turn_wait, 3)
        self.assertEqual(self.ship.max_thrust, 16)
        self.assertIs(self.ship.sprites, x_sprites)
        self.assertEqual(self.ship.boarded_marines, [marine])

    def test_both_xform_beams_damage_one_target_only_once(self):
        target = create_ship("Earthling", 2, resources=self.resources)
        target.position = [1000.0, 700.0]
        target.previous_position = target.position.copy()
        self.ship.opponent = target
        target.opponent = self.ship
        beams = list(self.ship.plan_action1().spawned_objects)
        objects = [self.ship, target, *beams]
        starting_hp = target.current_hp

        handle_collisions(objects)

        self.assertEqual(target.current_hp, starting_hp - 2)
        self.assertEqual(
            len([obj for obj in objects if isinstance(obj, BattleEffect)]),
            2,
        )

    def test_ywing_gun_directions_are_typed_catalog_data(self):
        definition = ABILITY_DEFINITIONS["MmrnmrhmYWingA1"]
        self.assertEqual(definition.gun_directions, (337.5, 22.5))


if __name__ == "__main__":
    unittest.main()
