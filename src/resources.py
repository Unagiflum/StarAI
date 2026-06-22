"""Central loading and caching of immutable Pygame resources."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import math

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
class DirectionalRetractionAssets:
    sprites: tuple
    masks: tuple
    projection_bounds: tuple


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
        self._ability_retractions = {}
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
                    if not definition.omnidirectional:
                        sprite = _scale_directional_sprite(
                            sprite,
                            index,
                            definition.sprite_scale_x,
                            definition.sprite_scale_y,
                        )
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

    def ability_retraction(self, ability_name, frame_count):
        """Return cached directional sprites clipped toward their rear edge."""
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        key = (ability_name, frame_count)
        if key in self._ability_retractions:
            return self._ability_retractions[key]

        source = self.ability(ability_name)
        if source.sprites is None or source.masks is None:
            raise ValueError(f"Ability '{ability_name}' has no directional sprites")
        if len(source.sprites) != const.SHIP_DIRECTIONS:
            raise ValueError(f"Ability '{ability_name}' is not directional")

        sprites = []
        masks = []
        bounds = []
        for heading, (source_sprite, source_mask) in enumerate(
            zip(source.sprites, source.masks)
        ):
            projection_bounds = _projection_bounds(source_mask, heading)
            heading_visuals = tuple(
                _retracted_visual(
                    source_sprite,
                    source_mask,
                    heading,
                    1.0 - frame / frame_count,
                    projection_bounds,
                )
                for frame in range(frame_count)
            )
            sprites.append(tuple(visual[0] for visual in heading_visuals))
            masks.append(tuple(visual[1] for visual in heading_visuals))
            bounds.append(projection_bounds)

        assets = DirectionalRetractionAssets(
            sprites=tuple(sprites),
            masks=tuple(masks),
            projection_bounds=tuple(bounds),
        )
        self._ability_retractions[key] = assets
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


def _scale_directional_sprite(sprite, heading, scale_x, scale_y):
    if scale_x == scale_y == 1.0:
        return sprite
    if scale_x == scale_y:
        return pygame.transform.smoothscale_by(sprite, scale_x)

    angle = heading * const.TURN_ANGLE
    local_sprite = pygame.transform.rotate(sprite, angle)
    local_bounds = local_sprite.get_bounding_rect(min_alpha=1)
    if local_bounds.width and local_bounds.height:
        local_sprite = local_sprite.subsurface(local_bounds).copy()
    local_size = (
        max(1, round(local_sprite.get_width() * scale_x)),
        max(1, round(local_sprite.get_height() * scale_y)),
    )
    local_sprite = pygame.transform.smoothscale(local_sprite, local_size)
    scaled_sprite = pygame.transform.rotate(local_sprite, -angle)
    scaled_bounds = scaled_sprite.get_bounding_rect(min_alpha=1)
    if scaled_bounds.width and scaled_bounds.height:
        scaled_sprite = scaled_sprite.subsurface(scaled_bounds).copy()
    return scaled_sprite


def _projection_bounds(mask, heading):
    width, height = mask.get_size()
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    angle = math.radians(heading * const.TURN_ANGLE)
    forward_x = math.sin(angle)
    forward_y = -math.cos(angle)
    projections = (
        (x - center_x) * forward_x + (y - center_y) * forward_y
        for y in range(height)
        for x in range(width)
        if mask.get_at((x, y))
    )
    projections = tuple(projections)
    return (min(projections), max(projections)) if projections else (0.0, 0.0)


def _retracted_visual(
    source_sprite,
    source_mask,
    heading,
    ratio,
    projection_bounds,
):
    if ratio >= 1.0:
        return source_sprite, source_mask

    visible_mask = source_mask.copy()
    minimum, maximum = projection_bounds
    cutoff = minimum + (maximum - minimum) * max(0.0, ratio)
    width, height = source_mask.get_size()
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    angle = math.radians(heading * const.TURN_ANGLE)
    forward_x = math.sin(angle)
    forward_y = -math.cos(angle)
    for y in range(height):
        for x in range(width):
            projection = (
                (x - center_x) * forward_x
                + (y - center_y) * forward_y
            )
            if projection > cutoff:
                visible_mask.set_at((x, y), 0)

    sprite = source_sprite.copy()
    alpha_mask = visible_mask.to_surface(
        setcolor=(255, 255, 255, 255),
        unsetcolor=(255, 255, 255, 0),
    )
    sprite.blit(alpha_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return sprite, visible_mask


class HeadlessAssetManager(AssetManager):
    """Load collision assets without requiring an initialized display.

    Pygame can decode images and construct masks before ``pygame.init()``.
    The display-dependent operation is ``Surface.convert_alpha()``, so the
    headless adapter deliberately keeps each image in its decoded format.
    """

    @staticmethod
    def _image(path, convert_alpha=True):
        path = const.source_path(path)
        return pygame.image.load(str(path))


_default_assets = AssetManager()
_active_assets = ContextVar("starai_active_assets", default=None)


def default_assets():
    """Return the narrowly scoped default provider used by legacy constructors."""
    return _default_assets


@contextmanager
def use_asset_manager(resources):
    """Scope legacy effect factories to one simulation's asset provider."""
    token = _active_assets.set(resources)
    try:
        yield
    finally:
        _active_assets.reset(token)


def active_asset_manager():
    return _active_assets.get()
