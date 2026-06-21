"""Central loading and caching of immutable Pygame resources."""

from dataclasses import dataclass
from pathlib import Path

import pygame

import src.const as const
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS


@dataclass(frozen=True)
class ShipAssets:
    sprites: tuple
    masks: tuple
    size: tuple
    ditty_path: Path


@dataclass(frozen=True)
class AbilityAssets:
    sprites: object
    masks: tuple | None
    end_animation: tuple
    sizes: tuple


@dataclass(frozen=True)
class AsteroidAssets:
    sprites: tuple
    masks: tuple
    death_animation: tuple


@dataclass(frozen=True)
class ImageAssets:
    image: pygame.Surface
    mask: pygame.mask.Mask | None = None


class AssetManager:
    """Load immutable assets once without owning gameplay or animation state."""

    def __init__(self):
        self._ships = {}
        self._abilities = {}
        self._asteroids = None
        self._images = {}
        self._sounds = {}
        self._sound_load_attempted = set()
        self._animations = {}
        self._backgrounds = {}
        self._menu_ship_sprites = {}
        self._ship_variants = {}

    @staticmethod
    def _image(path, convert_alpha=True):
        path = const.source_path(path)
        image = pygame.image.load(str(path))
        return image.convert_alpha() if convert_alpha else image

    def ship(self, ship_name):
        if ship_name not in self._ships:
            definition = SHIP_DEFINITIONS[ship_name]
            resource_dir = const.source_path(definition.sprite_path)
            scale = definition.sprite_scale
            sprites = tuple(
                pygame.transform.smoothscale_by(
                    self._image(resource_dir / f"{ship_name}{index:02d}.png"),
                    scale,
                )
                for index in range(const.SHIP_DIRECTIONS)
            )
            masks = tuple(pygame.mask.from_surface(sprite) for sprite in sprites)
            self._ships[ship_name] = ShipAssets(
                sprites=sprites,
                masks=masks,
                size=sprites[0].get_size(),
                ditty_path=resource_dir / f"{ship_name}-ditty.mp3",
            )
        return self._ships[ship_name]

    def ability(self, ability_name):
        if ability_name in self._abilities:
            return self._abilities[ability_name]

        definition = ABILITY_DEFINITIONS[ability_name]
        resource_dir = const.source_path(definition.file_path)
        scale = definition.sprite_scale
        sprites = []
        masks = []
        sizes = []

        if definition.has_sprites:
            if definition.omnidirectional and definition.frames > 1:
                frames = []
                for frame in range(definition.frames):
                    path = resource_dir / f"{ability_name}00_{frame:02d}.png"
                    if not path.exists():
                        path = resource_dir / f"{ability_name}{frame:02d}.png"
                    sprite = pygame.transform.smoothscale_by(self._image(path), scale)
                    frames.append(sprite)
                    masks.append(pygame.mask.from_surface(sprite))
                    sizes.append(sprite.get_size())
                sprites.append(tuple(frames))
            else:
                directions = 1 if definition.omnidirectional else const.SHIP_DIRECTIONS
                for index in range(directions):
                    path = resource_dir / f"{ability_name}{index:02d}.png"
                    sprite = pygame.transform.smoothscale_by(self._image(path), scale)
                    sprites.append(sprite)
                    masks.append(pygame.mask.from_surface(sprite))
                    if index == 0:
                        sizes.append(sprite.get_size())
            sprite_assets = tuple(sprites)
            mask_assets = tuple(masks)
        else:
            sprite_assets = None
            mask_assets = None

        end_animation = tuple(
            pygame.transform.smoothscale_by(
                self._image(resource_dir / f"{ability_name}end{index:02d}.png"),
                scale,
            )
            for index in range(definition.end_anim)
        )
        assets = AbilityAssets(
            sprites=sprite_assets,
            masks=mask_assets,
            end_animation=end_animation,
            sizes=tuple(sizes),
        )
        self._abilities[ability_name] = assets
        return assets

    def ability_sound(self, ability_name, enabled=True):
        definition = ABILITY_DEFINITIONS[ability_name]
        if not definition.has_sound:
            return None
        path = const.source_path(definition.file_path) / f"{ability_name}.wav"
        return self.sound(path, const.SOUND_EFFECT_VOLUME, enabled)

    def black_ship_sprites(self, ship_name):
        key = (ship_name, "black")
        if key not in self._ship_variants:
            variants = []
            for sprite in self.ship(ship_name).sprites:
                black_sprite = sprite.copy()
                black_pixels = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
                black_pixels.fill((0, 0, 0, 255))
                black_sprite.blit(
                    black_pixels, (0, 0), special_flags=pygame.BLEND_RGBA_MULT
                )
                variants.append(black_sprite)
            self._ship_variants[key] = tuple(variants)
        return self._ship_variants[key]

    def asteroid(self):
        if self._asteroids is None:
            sprites = tuple(
                self._image(const.ASTEROID_PATH / f"asteroid{index:02d}.png")
                for index in range(30)
            )
            self._asteroids = AsteroidAssets(
                sprites=sprites,
                masks=tuple(pygame.mask.from_surface(sprite) for sprite in sprites),
                death_animation=tuple(
                    self._image(const.ASTEROID_PATH / f"asteroidend{index:02d}.png")
                    for index in range(4)
                ),
            )
        return self._asteroids

    def image(self, path, size=None, with_mask=False):
        path = const.source_path(path)
        key = (path, tuple(size) if size else None, with_mask)
        if key not in self._images:
            image = self._image(path)
            if size and image.get_size() != tuple(size):
                image = pygame.transform.smoothscale(image, size)
            mask = pygame.mask.from_surface(image) if with_mask else None
            self._images[key] = ImageAssets(image, mask)
        return self._images[key]

    def animation(self, key, paths):
        if key not in self._animations:
            self._animations[key] = tuple(self._image(path) for path in paths)
        return self._animations[key]

    def sound(self, path, volume=1.0, enabled=True):
        if not enabled:
            return None
        path = const.source_path(path)
        key = (path, volume)
        if key not in self._sound_load_attempted:
            self._sound_load_attempted.add(key)
            try:
                sound = pygame.mixer.Sound(str(path))
                sound.set_volume(volume)
                self._sounds[key] = sound
            except (pygame.error, FileNotFoundError):
                self._sounds[key] = None
        return self._sounds.get(key)

    @staticmethod
    def play_music(path, volume, loops=0):
        """Stream music through Pygame; callers remain responsible for enablement."""
        pygame.mixer.music.load(str(const.source_path(path)))
        pygame.mixer.music.play(loops)
        pygame.mixer.music.set_volume(volume)

    def background(self, path, size):
        path = const.source_path(path)
        key = (path, tuple(size))
        if key not in self._backgrounds:
            self._backgrounds[key] = pygame.transform.scale(
                self._image(path, convert_alpha=False), size
            )
        return self._backgrounds[key]

    def menu_ship_sprite(self, ship_name):
        if ship_name not in self._menu_ship_sprites:
            resource_dir = const.source_path(SHIP_DEFINITIONS[ship_name].sprite_path)
            self._menu_ship_sprites[ship_name] = self._image(
                resource_dir / f"{ship_name}00.png"
            )
        return self._menu_ship_sprites[ship_name]


_default_assets = AssetManager()


def default_assets():
    """Return the narrowly scoped default provider used by legacy constructors."""
    return _default_assets
