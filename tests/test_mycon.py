import os
import unittest


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.registry import create_ability, create_ship
from src.resources import AssetManager


class MyconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sound_enabled = Ability.sound_enabled
        Ability.sound_enabled = False
        cls.resources = AssetManager()

    @classmethod
    def tearDownClass(cls):
        Ability.sound_enabled = cls.sound_enabled

    def make_plasma(self):
        ship = create_ship("Mycon", 1, resources=self.resources)
        ship.initialize_in_battle([500, 500], 0)
        return create_ability("MyconA1", ship)

    def test_assets_define_eleven_plasma_and_eight_blast_frames(self):
        definition = ABILITY_DEFINITIONS["MyconA1"]
        plasma = self.make_plasma()

        self.assertEqual(definition.frames, 11)
        self.assertEqual(len(plasma.sprites[0]), 11)
        self.assertEqual(
            len(plasma.death_animation),
            8 * const.VIDEO_FPS_MULTIPLIER,
        )
        self.assertFalse(self.resources._asset_errors)

    def test_plasma_strength_and_frame_evolve_from_remaining_lifetime(self):
        plasma = self.make_plasma()

        plasma.expiration_timer = 130
        plasma._evolve_plasma()
        self.assertEqual((plasma.current_frame, plasma.current_hp), (1, 10))
        self.assertEqual(plasma.current_damage, 10)

        plasma.expiration_timer = 128
        plasma._evolve_plasma()
        self.assertEqual((plasma.current_frame, plasma.current_hp), (1, 9))
        self.assertEqual(plasma.current_damage, 9)

        plasma.expiration_timer = 13
        plasma._evolve_plasma()
        self.assertEqual((plasma.current_frame, plasma.current_hp), (10, 1))
        self.assertEqual(plasma.current_damage, 1)

    def test_surviving_damage_shortens_lifetime_and_advances_frame(self):
        plasma = self.make_plasma()

        plasma.set_hp(9)
        plasma._evolve_plasma()

        self.assertEqual(plasma.expiration_timer, 9 * 13)
        self.assertEqual(plasma.current_frame, 2)
        self.assertEqual((plasma.current_hp, plasma.current_damage), (9, 9))


if __name__ == "__main__":
    unittest.main()
