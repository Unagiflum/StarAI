import os
import unittest
from unittest import mock


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

pygame.init()
pygame.display.set_mode((1, 1))

import src.const as const
from src.Battle.effects import BattleEffect
from src.Objects.Space.space_obj import Asteroid, Planet, Star
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.registry import create_ability, create_ship
from src.Objects.Ships.space_ship import SpaceShip
from src.UI.ui import SoundManager
from src.resources import AssetManager


def surface_bytes(surface):
    return pygame.image.tobytes(surface, "RGBA")


class ResourceCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        self.effect_sound_enabled = BattleEffect.sound_enabled
        Ability.sound_enabled = False

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled
        BattleEffect.sound_enabled = self.effect_sound_enabled

    def test_ship_instances_share_scaled_sprites_and_masks_but_not_state(self):
        SpaceShip._shared_sprites.pop("Earthling", None)
        SpaceShip._shared_masks.pop("Earthling", None)

        first = create_ship("Earthling", 1)
        second = create_ship("Earthling", 2)

        self.assertIs(first.sprites, second.sprites)
        self.assertIs(first.sprites[0], second.sprites[0])
        self.assertIs(first.get_collision_mask(), second.get_collision_mask())
        self.assertEqual(first.size, list(first.sprites[0].get_size()))
        self.assertEqual(first.get_collision_mask().get_size(), tuple(first.size))

        first.current_hp -= 1
        first.heading = 3
        self.assertNotEqual(first.current_hp, second.current_hp)
        self.assertNotEqual(first.heading, second.heading)

    def test_ability_instances_share_frames_masks_and_end_animation_in_order(self):
        for cache in (Ability._sprites, Ability._masks, Ability._end_anims, Ability._sizes):
            cache.pop("MyconA1", None)

        parent = create_ship("Mycon", 1)
        first = create_ability("MyconA1", parent)
        second = create_ability("MyconA1", parent)

        self.assertIs(first.sprites, second.sprites)
        self.assertIs(first.masks, second.masks)
        self.assertIs(first.death_animation, second.death_animation)
        self.assertIs(first.sprites[0][0], second.sprites[0][0])
        self.assertIs(first.masks[0], second.masks[0])
        self.assertEqual(first.size, list(first.sprites[0][0].get_size()))
        self.assertEqual(first.masks[0].get_size(), tuple(first.size))

        resource_dir = const.source_path("Objects/Ships/Mycon/A1")
        expected_first = pygame.transform.smoothscale_by(
            pygame.image.load(str(resource_dir / "MyconA100_00.png")).convert_alpha(),
            first.sprite_scale,
        )
        expected_last_end = pygame.transform.smoothscale_by(
            pygame.image.load(str(resource_dir / "MyconA1end07.png")).convert_alpha(),
            first.sprite_scale,
        )
        self.assertEqual(surface_bytes(first.sprites[0][0]), surface_bytes(expected_first))
        self.assertEqual(surface_bytes(first.death_animation[-1]), surface_bytes(expected_last_end))

        first.current_frame = 2
        first.frame_timer = 0
        self.assertEqual(second.current_frame, 0)
        self.assertNotEqual(first.frame_timer, second.frame_timer)

    def test_asteroids_share_ordered_visual_resources_but_not_animation_state(self):
        Asteroid.shared_sprites = None
        Asteroid.shared_masks = None
        Asteroid.shared_death_animation = None

        first = Asteroid()
        second = Asteroid()

        self.assertIs(first.sprites, second.sprites)
        self.assertIs(first.masks, second.masks)
        self.assertIs(first.death_animation, second.death_animation)
        self.assertEqual(len(first.sprites), 30)
        self.assertEqual(len(first.death_animation), 4)
        self.assertEqual(first.masks[0].get_size(), first.sprites[0].get_size())
        expected = pygame.image.load(
            str(const.ASTEROID_PATH / "asteroidend03.png")
        ).convert_alpha()
        self.assertEqual(surface_bytes(first.death_animation[-1]), surface_bytes(expected))

        first.current_sprite = 0
        second.current_sprite = 1
        first.rotation_timer = 3
        self.assertEqual(second.rotation_timer, 0)

    def test_planet_dimensions_and_mask_follow_catalog_definition(self):
        with mock.patch("src.Objects.Space.space_obj.random.choices", return_value=["Gas01"]):
            planet = Planet()

        self.assertEqual(planet.image.get_size(), (planet.diameter, planet.diameter))
        self.assertEqual(planet.mask.get_size(), (planet.diameter, planet.diameter))

    def test_battle_animation_frames_follow_numeric_filename_order(self):
        BattleEffect._blast_sprites = None
        BattleEffect._ship_explosion_sprites = None

        blast = BattleEffect.from_blast([0, 0], (1, 0), 1)
        explosion = BattleEffect.ship_explosion([0, 0])

        expected_blast = pygame.image.load(
            str(const.source_path("Objects/Battle/blast-002.png"))
        ).convert_alpha()
        self.assertEqual(surface_bytes(blast.frames[0]), surface_bytes(expected_blast))
        self.assertEqual(len(explosion.frames), 8)
        expected_last = pygame.image.load(
            str(const.source_path("Objects/Battle/explosion-007.png"))
        ).convert_alpha()
        self.assertEqual(surface_bytes(explosion.frames[-1]), surface_bytes(expected_last))

    def test_disabled_audio_never_attempts_to_load_ability_or_battle_sounds(self):
        Ability._launch_sounds.pop("EarthlingA1", None)
        Ability._sound_load_attempted.discard("EarthlingA1")
        BattleEffect._ship_death_sound = None
        BattleEffect._boom_sounds.clear()
        BattleEffect.sound_enabled = False

        with mock.patch("pygame.mixer.Sound") as sound:
            parent = create_ship("Earthling", 1)
            ability = create_ability("EarthlingA1", parent)
            death_length = BattleEffect.play_ship_death()
            BattleEffect.play_boom(2)

        self.assertIsNone(ability.launch_sound)
        self.assertEqual(death_length, 0)
        sound.assert_not_called()

    def test_explicit_factory_provider_is_inherited_by_abilities(self):
        resources = AssetManager()
        ship = create_ship("Earthling", 1, resources=resources)
        ability = create_ability("EarthlingA1", ship)

        self.assertIs(ship.resources, resources)
        self.assertIs(ability.resources, resources)
        self.assertIs(ship.sprites, resources.ship("Earthling").sprites)
        self.assertIs(ability.sprites, resources.ability("EarthlingA1").sprites)

    def test_planets_stars_and_menu_images_reuse_cached_surfaces(self):
        resources = AssetManager()
        planet_name = next(iter(Planet._planet_data))
        star_name = next(iter(Star._star_data))
        with mock.patch(
            "src.Objects.Space.space_obj.random.choices",
            side_effect=[[planet_name], [planet_name], [star_name], [star_name]],
        ):
            first_planet = Planet(resources)
            second_planet = Planet(resources)
            first_star = Star(resources)
            second_star = Star(resources)

        self.assertIs(first_planet.image, second_planet.image)
        self.assertIs(first_planet.mask, second_planet.mask)
        self.assertIs(first_star.image, second_star.image)
        first_background = resources.background(const.MENU_BG_PATH, (320, 200))
        second_background = resources.background(const.MENU_BG_PATH, (320, 200))
        self.assertIs(first_background, second_background)

    def test_disabled_menu_audio_does_not_initialize_or_load_mixer(self):
        resources = AssetManager()
        with (
            mock.patch("pygame.mixer.init") as mixer_init,
            mock.patch("pygame.mixer.Sound") as sound,
        ):
            manager = SoundManager(enabled=False, resources=resources)
            manager.load_sounds()
            manager.play_sound("menu")

        mixer_init.assert_not_called()
        sound.assert_not_called()


if __name__ == "__main__":
    unittest.main()
