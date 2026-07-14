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
from src.Objects.Ships.Ilwrath.Ilwrath import Ilwrath
from src.Objects.Ships.KzerZa.A2.KzerZaA2 import KzerZaA2
from src.Objects.Ships.Pkunk.A2.PkunkA2 import PkunkA2
from src.Objects.Ships.registry import create_ability, create_ship
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS
from src.Objects.Ships.space_ship import SpaceShip
from src.UI.ui import SoundManager
from src.resources import AssetManager, _interpolate_frames


def surface_bytes(surface):
    return pygame.image.tobytes(surface, "RGBA")


def centered_on_surface(source, destination):
    centered = pygame.Surface(destination.get_size(), pygame.SRCALPHA)
    centered.blit(source, source.get_rect(center=centered.get_rect().center))
    return centered


def opaque_size(mask):
    bounds = mask.get_bounding_rects()
    left = min(rect.left for rect in bounds)
    top = min(rect.top for rect in bounds)
    right = max(rect.right for rect in bounds)
    bottom = max(rect.bottom for rect in bounds)
    return [right - left, bottom - top]


class ResourceCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.sound_enabled = Ability.sound_enabled
        self.effect_sound_enabled = BattleEffect.sound_enabled
        Ability.sound_enabled = False

    def tearDown(self):
        Ability.sound_enabled = self.sound_enabled
        BattleEffect.sound_enabled = self.effect_sound_enabled

    def test_ship_logical_size_excludes_transparent_canvas_padding(self):
        mask = pygame.mask.Mask((100, 100), fill=False)
        mask.draw(pygame.mask.Mask((30, 20), fill=True), (35, 40))

        self.assertEqual(AssetManager._opaque_size(mask), (30, 20))

    def test_ship_instances_share_scaled_sprites_and_masks_but_not_state(self):
        first = create_ship("Earthling", 1)
        second = create_ship("Earthling", 2)

        self.assertIs(first.sprites, second.sprites)
        self.assertIs(first.sprites[0], second.sprites[0])
        self.assertIs(first.get_collision_mask(), second.get_collision_mask())
        self.assertEqual(first.size, opaque_size(first.get_collision_mask()))
        self.assertEqual(
            first.get_collision_mask().get_size(),
            first.sprites[0].get_size(),
        )

        first.current_hp -= 1
        first.heading = 3
        self.assertNotEqual(first.current_hp, second.current_hp)
        self.assertNotEqual(first.heading, second.heading)

    def test_ability_instances_share_frames_masks_and_end_animation_in_order(self):
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
        last_base_frame = 7 * const.VIDEO_FPS_MULTIPLIER
        expected_last_end = centered_on_surface(
            expected_last_end,
            first.death_animation[last_base_frame],
        )
        self.assertEqual(
            surface_bytes(first.death_animation[last_base_frame]),
            surface_bytes(expected_last_end),
        )

        first.current_frame = 2
        first.frame_timer = 0
        self.assertEqual(second.current_frame, 0)
        self.assertNotEqual(first.frame_timer, second.frame_timer)

    def test_asteroids_share_ordered_visual_resources_but_not_animation_state(self):
        first = Asteroid()
        second = Asteroid()

        self.assertIs(first.sprites, second.sprites)
        self.assertIs(first.masks, second.masks)
        self.assertIs(first.death_animation, second.death_animation)
        self.assertEqual(len(first.sprites), 29)
        self.assertEqual(len(first.death_animation), 5 * const.VIDEO_FPS_MULTIPLIER)
        self.assertEqual(first.masks[0].get_size(), first.sprites[0].get_size())
        self.assertEqual(first.mass, const.ASTEROID_MASS)
        speed = (first.velocity[0] ** 2 + first.velocity[1] ** 2) ** 0.5
        allowed_speeds = range(
            const.ASTEROID_MIN_SPEED,
            const.ASTEROID_MAX_SPEED + const.ASTEROID_SPEED_STEP,
            const.ASTEROID_SPEED_STEP,
        )
        self.assertTrue(
            any(abs(speed - allowed_speed) < 1e-9 for allowed_speed in allowed_speeds)
        )
        expected = pygame.image.load(
            str(const.ASTEROID_PATH / "asteroidend04.png")
        ).convert_alpha()
        last_base_frame = 4 * const.VIDEO_FPS_MULTIPLIER
        self.assertEqual(
            surface_bytes(first.death_animation[last_base_frame]),
            surface_bytes(expected),
        )

        first.current_sprite = 0
        second.current_sprite = 1
        first.rotation_timer = 3
        self.assertEqual(second.rotation_timer, 0)

    def test_planet_dimensions_and_mask_follow_catalog_definition(self):
        with mock.patch("src.Objects.Space.space_obj.random.choices", return_value=["Gas01"]):
            planet = Planet()

        self.assertEqual(planet.image.get_size(), (planet.diameter, planet.diameter))
        self.assertEqual(planet.mask.get_size(), (planet.diameter, planet.diameter))
        self.assertEqual(
            planet.impact_capabilities.impact_damage_percent,
            const.PLANET_IMPACT_DAMAGE_PERCENT,
        )

    def test_battle_animation_frames_follow_numeric_filename_order(self):
        blast = BattleEffect.from_blast([0, 0], (1, 0), 1)
        explosion = BattleEffect.ship_explosion([0, 0])

        expected_blast = pygame.image.load(
            str(const.source_path("Objects/Battle/blast-002.png"))
        ).convert_alpha()
        self.assertEqual(surface_bytes(blast.frames[0]), surface_bytes(expected_blast))
        self.assertEqual(len(explosion.frames), 8 * const.VIDEO_FPS_MULTIPLIER)
        expected_last = pygame.image.load(
            str(const.source_path("Objects/Battle/explosion-007.png"))
        ).convert_alpha()
        last_base_frame = 7 * const.VIDEO_FPS_MULTIPLIER
        expected_last = centered_on_surface(
            expected_last,
            explosion.frames[last_base_frame],
        )
        self.assertEqual(
            surface_bytes(explosion.frames[last_base_frame]),
            surface_bytes(expected_last),
        )

    def test_crossfade_preserves_opacity_where_both_frames_are_opaque(self):
        red = pygame.Surface((1, 1), pygame.SRCALPHA)
        red.fill((255, 0, 0, 255))
        blue = pygame.Surface((1, 1), pygame.SRCALPHA)
        blue.fill((0, 0, 255, 255))

        frames = _interpolate_frames((red, blue), 5, loop=False)

        self.assertEqual(
            [frame.get_at((0, 0)).a for frame in frames[:5]],
            [255, 255, 255, 255, 255],
        )
        self.assertEqual(frames[2].get_at((0, 0)), pygame.Color(153, 0, 102, 255))

    def test_fade_to_transparent_omits_only_the_invisible_endpoint(self):
        frame = pygame.Surface((1, 1), pygame.SRCALPHA)
        frame.fill((255, 255, 255, 255))

        frames = _interpolate_frames(
            (frame,), 5, loop=False, fade_to_transparent=True
        )

        self.assertEqual(
            [value.get_at((0, 0)).a for value in frames],
            [255, 204, 153, 102, 51],
        )

    def test_evolving_ability_visual_does_not_overshoot_its_next_base_frame(self):
        parent = create_ship("Mycon", 1)
        ability = create_ability("MyconA1", parent)
        interpolated = ability.resources.ability("MyconA1").interpolated_sprites[0]

        ability.current_frame = 0
        ability.frame_timer = 1
        self.assertIs(
            ability.get_sprite(0.99),
            interpolated[const.VIDEO_FPS_MULTIPLIER - 1],
        )

        ability.current_frame = 1
        ability.frame_timer = ability.frame_delay
        self.assertIs(
            ability.get_sprite(0.0),
            interpolated[const.VIDEO_FPS_MULTIPLIER],
        )

    def test_asteroid_visual_reaches_last_subframe_before_base_frame_advances(self):
        asteroid = Asteroid()
        asteroid.current_sprite = 0
        asteroid.rotation_delay = 1
        asteroid.rotation_timer = 0
        asteroid.position = [100, 100]
        screen = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))

        asteroid.draw(screen, 1.0, [0, 0], interp_t=0.99)

        self.assertIs(
            asteroid.image,
            asteroid.resources.asteroid().interpolated_sprites[
                const.VIDEO_FPS_MULTIPLIER - 1
            ],
        )

    def test_impact_effect_aligns_its_opaque_edge_to_contact(self):
        sprite = pygame.Surface((20, 20), pygame.SRCALPHA)
        sprite.fill((255, 255, 255, 255), pygame.Rect(5, 8, 10, 4))

        position = BattleEffect._edge_aligned_position(
            [100, 100], sprite, 2.0, [1, 0]
        )

        self.assertEqual(position, [110.0, 100.0])

    def test_single_frame_blast_is_centered_at_contact(self):
        effect = BattleEffect.from_blast([100, 100], [1, 0], 1)

        self.assertEqual(effect.position, [100, 100])

    def test_disabled_audio_never_attempts_to_load_ability_or_battle_sounds(self):
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

    def test_placeholder_multiframe_ability_has_size_for_each_frame(self):
        resources = AssetManager()
        definition = ABILITY_DEFINITIONS["IlwrathA1"]

        with mock.patch.object(
            resources,
            "_image",
            side_effect=FileNotFoundError("missing test asset"),
        ):
            assets = resources.ability("IlwrathA1")

        self.assertEqual(definition.frames, 8)
        self.assertEqual(len(assets.sizes), definition.frames)

    def test_assets_are_shared_only_within_their_resource_provider(self):
        first_resources = AssetManager()
        second_resources = AssetManager()

        first_ship = create_ship("Earthling", 1, resources=first_resources)
        same_manager_ship = create_ship("Earthling", 2, resources=first_resources)
        second_ship = create_ship("Earthling", 2, resources=second_resources)
        first_ability = create_ability("EarthlingA1", first_ship)
        second_ability = create_ability("EarthlingA1", second_ship)
        first_asteroid = Asteroid(first_resources)
        second_asteroid = Asteroid(second_resources)
        first_ilwrath = create_ship("Ilwrath", 1, resources=first_resources)
        second_ilwrath = create_ship("Ilwrath", 2, resources=second_resources)

        self.assertIs(first_ship.sprites, same_manager_ship.sprites)
        self.assertIs(first_ship.masks, same_manager_ship.masks)
        self.assertIs(first_ship.sprites, first_resources.ship("Earthling").sprites)
        self.assertIs(first_ship.masks, first_resources.ship("Earthling").masks)
        self.assertIsNot(first_ship.sprites, second_ship.sprites)
        self.assertIsNot(first_ship.masks, second_ship.masks)
        self.assertIs(first_ship.get_collision_mask(), first_ship.masks[0])

        self.assertIs(first_ability.sprites, first_resources.ability("EarthlingA1").sprites)
        self.assertIs(first_ability.masks, first_resources.ability("EarthlingA1").masks)
        self.assertIsNot(first_ability.sprites, second_ability.sprites)
        self.assertIsNot(first_ability.masks, second_ability.masks)

        self.assertIs(first_asteroid.sprites, first_resources.asteroid().sprites)
        self.assertIs(first_asteroid.masks, first_resources.asteroid().masks)
        self.assertIsNot(first_asteroid.sprites, second_asteroid.sprites)
        self.assertIsNot(first_asteroid.masks, second_asteroid.masks)

        self.assertIs(
            first_ilwrath.black_sprites,
            first_resources.black_ship_sprites("Ilwrath"),
        )
        self.assertIsNot(first_ilwrath.black_sprites, second_ilwrath.black_sprites)

    def test_gameplay_classes_do_not_own_immutable_asset_caches(self):
        for owner, attributes in (
            (SpaceShip, ("_shared_sprites", "_shared_masks")),
            (Ability, (
                "_sprites", "_masks", "_end_anims", "_sizes",
                "_launch_sounds", "_sound_load_attempted",
            )),
            (Asteroid, ("shared_sprites", "shared_masks", "shared_death_animation")),
            (BattleEffect, (
                "_blast_sprites", "_ship_explosion_sprites",
                "_ship_death_sound", "_boom_sounds",
            )),
            (Ilwrath, ("_shared_sprites_black", "_uncloak_sound")),
            (KzerZaA2, ("_fighter_sounds",)),
            (PkunkA2, ("_insults",)),
        ):
            for attribute in attributes:
                self.assertFalse(hasattr(owner, attribute), f"{owner.__name__}.{attribute}")

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
