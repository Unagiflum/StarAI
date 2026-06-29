"""Central loading and caching of immutable Pygame resources."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import json
import math

import pygame

import src.const as const
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS, SHIP_DEFINITIONS

_PLACEHOLDER_COLOR = (255, 0, 255, 200)


@dataclass(frozen=True)
class AssetError:
    """Record of a single asset that failed to load."""

    category: str
    name: str
    path: str
    message: str


@dataclass(frozen=True)
class ShipAssets:
    sprites: tuple
    masks: tuple
    # Opaque bounds from heading 00, in scaled gameplay pixels.
    size: tuple
    ditty_path: Path


@dataclass(frozen=True)
class AbilityAssets:
    sprites: object
    masks: tuple | None
    end_animation: tuple
    sizes: tuple
    interpolated_sprites: object = None


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
    interpolated_sprites: tuple


@dataclass(frozen=True)
class ImageAssets:
    image: pygame.Surface
    mask: pygame.mask.Mask | None = None


def centered_overlay(base, overlay):
    """Return a copy of ``base`` with ``overlay`` aligned by center."""
    composite = base.copy()
    composite.blit(overlay, overlay.get_rect(center=composite.get_rect().center))
    return composite


def _make_rectangle_surface(width, height, color=_PLACEHOLDER_COLOR):
    """Create a transparent-background surface with a centered colored rectangle."""
    canvas = max(width, height) + 4
    surface = pygame.Surface((canvas, canvas), pygame.SRCALPHA)
    rect = pygame.Rect(
        (canvas - width) // 2,
        (canvas - height) // 2,
        width,
        height,
    )
    pygame.draw.rect(surface, color, rect)
    return surface


def _make_circle_surface(diameter, color=_PLACEHOLDER_COLOR):
    """Create a transparent-background surface with a centered colored circle."""
    diameter = max(4, diameter)
    surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(surface, color, (diameter // 2, diameter // 2), diameter // 2)
    return surface


def _placeholder_ship_sprites(scale=1.0):
    """Return rotated rectangle sprites as a ship placeholder."""
    base_w = max(4, int(30 * scale))
    base_h = max(4, int(50 * scale))
    base = _make_rectangle_surface(base_w, base_h)
    sprites = []
    asset_turn = 360 / const.ASSET_SPRITE_DIRECTIONS
    for heading in range(const.ASSET_SPRITE_DIRECTIONS):
        angle = -(heading * asset_turn)
        rotated = pygame.transform.rotate(base, angle)
        sprites.append(rotated)
    return tuple(sprites)


def _placeholder_ability_sprites(definition):
    """Return placeholder sprites appropriate for an ability definition."""
    diameter = max(8, int(32 * definition.sprite_scale))
    if definition.omnidirectional:
        sprite = _make_circle_surface(diameter)
        if definition.frames > 1:
            frames = tuple(sprite.copy() for _ in range(definition.frames))
            return ([frames], None)
        return ([sprite], None)
    else:
        sprites = []
        for heading in range(const.ASSET_SPRITE_DIRECTIONS):
            sprite = _make_circle_surface(diameter)
            sprites.append(sprite)
        return (sprites, None)


class AssetManager:
    """Load immutable assets once without owning gameplay or animation state."""

    def __init__(self):
        self._ships = {}
        self._ship_forms = {}
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
        self._asset_errors = []

    @staticmethod
    def _image(path, convert_alpha=True):
        path = const.source_path(path)
        image = pygame.image.load(str(path))
        return image.convert_alpha() if convert_alpha else image

    @staticmethod
    def _opaque_size(mask):
        """Return the combined opaque-pixel bounds of a sprite mask."""
        bounds = mask.get_bounding_rects()
        if not bounds:
            return (0, 0)

        left = min(rect.left for rect in bounds)
        top = min(rect.top for rect in bounds)
        right = max(rect.right for rect in bounds)
        bottom = max(rect.bottom for rect in bounds)
        return (right - left, bottom - top)

    def ship(self, ship_name):
        if ship_name not in self._ships:
            definition = SHIP_DEFINITIONS[ship_name]
            if definition.forms:
                self._ships[ship_name] = self.ship_form(
                    ship_name, definition.default_form
                )
            else:
                resource_dir = const.source_path(definition.sprite_path)
                self._ships[ship_name] = self._load_ship_assets(
                    cache_name=ship_name,
                    resource_dir=resource_dir,
                    sprite_prefix=ship_name,
                    scale=definition.sprite_scale,
                    ditty_path=resource_dir / f"{ship_name}-ditty.mp3",
                )
        return self._ships[ship_name]

    def ship_form(self, ship_name, form_name):
        """Return directional assets for one configured runtime ship form."""
        key = (ship_name, form_name)
        if key not in self._ship_forms:
            definition = SHIP_DEFINITIONS[ship_name]
            try:
                form = definition.forms[form_name]
            except KeyError:
                raise KeyError(f"Unknown form for {ship_name}: {form_name}") from None
            resource_dir = const.source_path(form.sprite_path)
            ditty_dir = const.source_path(definition.sprite_path)
            self._ship_forms[key] = self._load_ship_assets(
                cache_name=f"{ship_name}.{form_name}",
                resource_dir=resource_dir,
                sprite_prefix=form.sprite_prefix or f"{ship_name}{form_name}",
                scale=form.sprite_scale,
                ditty_path=ditty_dir / f"{ship_name}-ditty.mp3",
            )
        return self._ship_forms[key]

    def _load_ship_assets(
        self, *, cache_name, resource_dir, sprite_prefix, scale, ditty_path
    ):
        try:
            base_sprites = tuple(
                pygame.transform.smoothscale_by(
                    self._image(resource_dir / f"{sprite_prefix}{index:02d}.png"),
                    scale,
                )
                for index in range(const.ASSET_SPRITE_DIRECTIONS)
            )
        except (pygame.error, FileNotFoundError, OSError) as error:
            self._asset_errors.append(
                AssetError("ship", cache_name, str(resource_dir), str(error))
            )
            base_sprites = _placeholder_ship_sprites(scale)

        sprites, masks = _expand_directional_sprites(base_sprites)
        return ShipAssets(
            sprites=sprites,
            masks=masks,
            size=self._opaque_size(masks[0]),
            ditty_path=ditty_path,
        )

    def ability(self, ability_name):
        if ability_name in self._abilities:
            return self._abilities[ability_name]

        definition = ABILITY_DEFINITIONS[ability_name]
        resource_dir = const.source_path(definition.file_path)
        scale = definition.sprite_scale
        sprites = []
        masks = []
        sizes = []
        used_placeholder = False

        if definition.has_sprites:
            try:
                if not definition.omnidirectional and definition.frames > 1:
                    directional_sprites = []
                    directional_masks = []
                    for frame in range(definition.frames):
                        base_sprites = []
                        for index in range(const.ASSET_SPRITE_DIRECTIONS):
                            path = resource_dir / (
                                f"{ability_name}_{frame + 1}{index:02d}.png"
                            )
                            sprite = pygame.transform.smoothscale_by(
                                self._image(path), scale
                            )
                            sprite = _scale_directional_sprite(
                                sprite,
                                index,
                                definition.sprite_scale_x,
                                definition.sprite_scale_y,
                            )
                            if definition.excluded_radius is not None:
                                parent_scale = SHIP_DEFINITIONS[
                                    definition.ship_name
                                ].sprite_scale
                                sprite = _exclude_center_circle(
                                    sprite,
                                    round(definition.excluded_radius * parent_scale),
                                )
                            base_sprites.append(sprite)
                        frame_sprites, frame_masks = _expand_directional_sprites(
                            base_sprites
                        )
                        directional_sprites.append(frame_sprites)
                        directional_masks.append(frame_masks)
                        sizes.append(base_sprites[0].get_size())
                    sprites = directional_sprites
                    masks = directional_masks
                elif definition.omnidirectional and definition.frames > 1:
                    frames = []
                    for frame in range(definition.frames):
                        path = resource_dir / f"{ability_name}00_{frame:02d}.png"
                        if not path.exists():
                            path = resource_dir / f"{ability_name}{frame:02d}.png"
                        sprite = pygame.transform.smoothscale_by(
                            self._image(path), scale
                        )
                        frames.append(sprite)
                        masks.append(pygame.mask.from_surface(sprite))
                        sizes.append(sprite.get_size())
                    sprites.append(tuple(frames))
                else:
                    directions = (
                        1
                        if definition.omnidirectional
                        else const.ASSET_SPRITE_DIRECTIONS
                    )
                    for index in range(directions):
                        path = resource_dir / f"{ability_name}{index:02d}.png"
                        if definition.omnidirectional and not path.exists():
                            path = resource_dir / f"{ability_name}.png"
                        sprite = pygame.transform.smoothscale_by(
                            self._image(path), scale
                        )
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
            except (pygame.error, FileNotFoundError, OSError) as error:
                self._asset_errors.append(
                    AssetError(
                        "ability",
                        ability_name,
                        str(resource_dir),
                        str(error),
                    )
                )
                placeholder_sprites, _ = _placeholder_ability_sprites(definition)
                sprites = placeholder_sprites
                masks = [
                    pygame.mask.from_surface(s)
                    for group in sprites
                    for s in (group if isinstance(group, tuple) else (group,))
                ]
                diameter = max(8, int(32 * scale))
                sizes = [(diameter, diameter)]
                used_placeholder = True

                if not definition.omnidirectional and definition.frames > 1:
                    frame_sprites, frame_masks = _expand_directional_sprites(sprites)
                    sprites = [frame_sprites for _ in range(definition.frames)]
                    masks = [frame_masks for _ in range(definition.frames)]
                    sizes = [(diameter, diameter) for _ in range(definition.frames)]

            if not definition.omnidirectional:
                if definition.frames > 1:
                    sprite_assets = tuple(sprites)
                    mask_assets = tuple(masks)
                else:
                    sprite_assets, mask_assets = _expand_directional_sprites(sprites)
                interpolated_sprites = None
            else:
                sprite_assets = tuple(sprites)
                mask_assets = tuple(masks)
                
                if const.VIDEO_FPS_MULTIPLIER > 1 and definition.frames > 1:
                    interpolated_frames = _interpolate_frames(sprites[0], const.VIDEO_FPS_MULTIPLIER, loop=False, fade_to_transparent=False)
                    interpolated_sprites = (interpolated_frames,)
                else:
                    interpolated_sprites = None
        else:
            sprite_assets = None
            mask_assets = None
            interpolated_sprites = None

        try:
            end_animation = tuple(
                pygame.transform.smoothscale_by(
                    self._image(resource_dir / f"{ability_name}end{index:02d}.png"),
                    scale,
                )
                for index in range(definition.end_anim)
            )
        except (pygame.error, FileNotFoundError, OSError) as error:
            if not used_placeholder:
                self._asset_errors.append(
                    AssetError(
                        "ability_end_anim",
                        ability_name,
                        str(resource_dir),
                        str(error),
                    )
                )
            end_animation = ()
            
        if const.VIDEO_FPS_MULTIPLIER > 1 and end_animation:
            end_animation = _interpolate_frames(end_animation, const.VIDEO_FPS_MULTIPLIER, loop=False, fade_to_transparent=True)

        assets = AbilityAssets(
            sprites=sprite_assets,
            masks=mask_assets,
            end_animation=end_animation,
            sizes=tuple(sizes),
            interpolated_sprites=interpolated_sprites,
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
        if len(source.sprites) != const.TOTAL_SPRITE_DIRECTIONS:
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
            try:
                sprites = tuple(
                    self._image(const.ASTEROID_PATH / f"asteroid{index:02d}.png")
                    for index in range(29)
                )
            except (pygame.error, FileNotFoundError, OSError) as error:
                self._asset_errors.append(
                    AssetError(
                        "asteroid",
                        "asteroid",
                        str(const.ASTEROID_PATH),
                        str(error),
                    )
                )
                placeholder = _make_circle_surface(40)
                sprites = tuple(placeholder.copy() for _ in range(29))
            try:
                base_death_animation = tuple(
                    self._image(const.ASTEROID_PATH / f"asteroidend{index:02d}.png")
                    for index in range(5)
                )
            except (pygame.error, FileNotFoundError, OSError) as error:
                self._asset_errors.append(
                    AssetError(
                        "asteroid_death",
                        "asteroid",
                        str(const.ASTEROID_PATH),
                        str(error),
                    )
                )
                base_death_animation = ()
                
            if const.VIDEO_FPS_MULTIPLIER > 1:
                interpolated_sprites = _interpolate_frames(sprites, const.VIDEO_FPS_MULTIPLIER, loop=True, fade_to_transparent=False)
                if base_death_animation:
                    death_animation = _interpolate_frames(base_death_animation, const.VIDEO_FPS_MULTIPLIER, loop=False, fade_to_transparent=True)
                else:
                    death_animation = ()
            else:
                interpolated_sprites = sprites
                death_animation = base_death_animation
                
            self._asteroids = AsteroidAssets(
                sprites=sprites,
                masks=tuple(pygame.mask.from_surface(sprite) for sprite in sprites),
                death_animation=death_animation,
                interpolated_sprites=interpolated_sprites,
            )
        return self._asteroids

    def image(self, path, size=None, with_mask=False):
        path = const.source_path(path)
        key = (path, tuple(size) if size else None, with_mask)
        if key not in self._images:
            try:
                image = self._image(path)
            except (pygame.error, FileNotFoundError, OSError) as error:
                self._asset_errors.append(
                    AssetError(
                        "image",
                        str(path),
                        str(path),
                        str(error),
                    )
                )
                diameter = max(size) if size else 32
                image = _make_circle_surface(diameter)
            if size and image.get_size() != tuple(size):
                image = pygame.transform.smoothscale(image, size)
            mask = pygame.mask.from_surface(image) if with_mask else None
            self._images[key] = ImageAssets(image, mask)
        return self._images[key]

    def animation(self, key, paths, interpolated=False, fade_to_transparent=False):
        if key not in self._animations:
            frames = []
            for path in paths:
                try:
                    frames.append(self._image(path))
                except (pygame.error, FileNotFoundError, OSError) as error:
                    self._asset_errors.append(
                        AssetError(
                            "animation",
                            key,
                            str(path),
                            str(error),
                        )
                    )
                    frames.append(_make_circle_surface(32))
                    
            if interpolated and const.VIDEO_FPS_MULTIPLIER > 1:
                frames = _interpolate_frames(frames, const.VIDEO_FPS_MULTIPLIER, loop=False, fade_to_transparent=fade_to_transparent)
                
            self._animations[key] = tuple(frames)
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
        try:
            pygame.mixer.music.load(str(const.source_path(path)))
            pygame.mixer.music.play(loops)
            pygame.mixer.music.set_volume(volume)
        except (pygame.error, FileNotFoundError, OSError):
            pass

    def background(self, path, size):
        path = const.source_path(path)
        key = (path, tuple(size))
        if key not in self._backgrounds:
            try:
                self._backgrounds[key] = pygame.transform.scale(
                    self._image(path, convert_alpha=False), size
                )
            except (pygame.error, FileNotFoundError, OSError) as error:
                self._asset_errors.append(
                    AssetError(
                        "background",
                        str(path),
                        str(path),
                        str(error),
                    )
                )
                fallback = pygame.Surface(size)
                fallback.fill((0, 0, 20))
                self._backgrounds[key] = fallback
        return self._backgrounds[key]

    def menu_ship_sprite(self, ship_name):
        if ship_name not in self._menu_ship_sprites:
            definition = SHIP_DEFINITIONS[ship_name]
            resource_dir = const.source_path(definition.sprite_path)
            try:
                if definition.forms:
                    form = definition.forms[definition.default_form]
                    form_dir = const.source_path(form.sprite_path)
                    sprite_prefix = (
                        form.sprite_prefix
                        or f"{ship_name}{definition.default_form}"
                    )
                    sprite = self._image(
                        form_dir / f"{sprite_prefix}00.png"
                    )
                else:
                    sprite = self._image(resource_dir / f"{ship_name}00.png")
                if definition.menu_overlay_path is not None:
                    overlay = self._image(definition.menu_overlay_path)
                    sprite = centered_overlay(sprite, overlay)
            except (pygame.error, FileNotFoundError, OSError) as error:
                self._asset_errors.append(
                    AssetError(
                        "menu_sprite",
                        ship_name,
                        str(resource_dir),
                        str(error),
                    )
                )
                scale = definition.sprite_scale
                sprite = _make_rectangle_surface(
                    max(4, int(30 * scale)),
                    max(4, int(50 * scale)),
                )
            self._menu_ship_sprites[ship_name] = sprite
        return self._menu_ship_sprites[ship_name]

    def preload_all(self):
        """Eagerly load all assets referenced by the catalogs.

        Returns a list of :class:`AssetError` for every asset that could not
        be loaded.  Failed assets are replaced by colored placeholders so the
        game can continue.
        """
        self._asset_errors = []

        # Ships and their menu sprites.
        for ship_name in SHIP_DEFINITIONS:
            self.ship(ship_name)
            self.menu_ship_sprite(ship_name)
            definition = SHIP_DEFINITIONS[ship_name]
            for form_name in definition.forms:
                self.ship_form(ship_name, form_name)

        # Abilities, their sounds, and retraction assets.
        for ability_name, definition in ABILITY_DEFINITIONS.items():
            self.ability(ability_name)
            self.ability_sound(ability_name)
            if (
                definition.retraction_frames is not None
                and definition.retraction_frames > 0
            ):
                try:
                    self.ability_retraction(ability_name, definition.retraction_frames)
                except Exception as error:
                    self._asset_errors.append(
                        AssetError(
                            "ability_retraction",
                            ability_name,
                            str(const.source_path(definition.file_path)),
                            str(error),
                        )
                    )

        # Asteroids.
        self.asteroid()

        # Planets — iterate the JSON to preload every image.
        try:
            with open(const.PLANETS_JSON_PATH, "r") as fh:
                planet_data = json.load(fh)
            for planet_name, planet_info in planet_data.items():
                diameter = planet_info.get("Diameter", 300)
                self.image(
                    planet_info["Image"],
                    (diameter, diameter),
                    with_mask=True,
                )
        except (OSError, json.JSONDecodeError, KeyError) as error:
            self._asset_errors.append(
                AssetError(
                    "planet_catalog",
                    "planets.json",
                    str(const.PLANETS_JSON_PATH),
                    str(error),
                )
            )

        # Stars — iterate the JSON to preload every image.
        try:
            with open(const.STARS_JSON_PATH, "r") as fh:
                star_data = json.load(fh)
            for star_name, star_info in star_data.items():
                self.image(star_info["Image"])
        except (OSError, json.JSONDecodeError, KeyError) as error:
            self._asset_errors.append(
                AssetError(
                    "star_catalog",
                    "stars.json",
                    str(const.STARS_JSON_PATH),
                    str(error),
                )
            )

        # Battle effect sprites (explosions, blasts).
        battle_path = const.source_path("Objects/Battle")
        self.animation(
            "ship-explosions",
            tuple(battle_path / f"explosion-{i:03d}.png" for i in range(8)),
            interpolated=True,
            fade_to_transparent=True
        )
        self.animation(
            "battle-blasts",
            tuple(battle_path / f"blast-{i:03d}.png" for i in range(8)),
            interpolated=False
        )

        # Battle sounds.
        for sound_name in (
            "boom1.wav",
            "boom2.wav",
            "boom4.wav",
            "boom6.wav",
            "shipdies.wav",
        ):
            self.sound(battle_path / sound_name, const.SOUND_EFFECT_VOLUME)

        # Backgrounds.
        screen_size = (const.SCREEN_WIDTH, const.SCREEN_HEIGHT)
        self.background(const.MAIN_BG_PATH, screen_size)
        self.background(const.MENU_BG_PATH, screen_size)

        # Battle music — verify file exists (streamed, not decoded).
        battle_music = const.source_path(const.BATTLE_MUSIC_PATH)
        if not Path(battle_music).exists():
            self._asset_errors.append(
                AssetError(
                    "music",
                    "battle.ogg",
                    str(battle_music),
                    "file not found",
                )
            )

        # Victory ditties — verify each file exists.
        for ship_name in SHIP_DEFINITIONS:
            ditty_path = self.ship(ship_name).ditty_path
            if not Path(ditty_path).exists():
                self._asset_errors.append(
                    AssetError(
                        "ditty",
                        ship_name,
                        str(ditty_path),
                        "file not found",
                    )
                )

        # Menu sound.
        self.sound(const.MENU_WAV_PATH, 1.0)

        return list(self._asset_errors)


def _interpolate_frames(frames, multiplier, loop=True, fade_to_transparent=False):
    """Generate intermediate frames using alpha-blended crossfades."""
    if not frames or multiplier <= 1:
        return frames

    interpolated = []
    num_frames = len(frames)
    
    max_w = max(frame.get_width() for frame in frames)
    max_h = max(frame.get_height() for frame in frames)
    max_size = (max_w, max_h)
    
    empty_surface = pygame.Surface(max_size, pygame.SRCALPHA)
    empty_surface.fill((0, 0, 0, 0))

    for i in range(num_frames):
        current_frame = frames[i]
        
        if i == num_frames - 1:
            if fade_to_transparent:
                next_frame = empty_surface
            elif loop:
                next_frame = frames[0]
            else:
                next_frame = current_frame
        else:
            next_frame = frames[i + 1]

        for step in range(multiplier):
            ratio = step / multiplier
            
            new_frame = pygame.Surface(max_size, pygame.SRCALPHA)
            new_frame.fill((0, 0, 0, 0))
            
            if ratio == 0.0:
                c_x = (max_w - current_frame.get_width()) // 2
                c_y = (max_h - current_frame.get_height()) // 2
                new_frame.blit(current_frame, (c_x, c_y))
            else:
                a_copy = current_frame.copy()
                a_weight = round((1.0 - ratio) * 255)
                a_copy.fill(
                    (a_weight, a_weight, a_weight, a_weight),
                    special_flags=pygame.BLEND_RGBA_MULT,
                )
                a_x = (max_w - a_copy.get_width()) // 2
                a_y = (max_h - a_copy.get_height()) // 2
                new_frame.blit(
                    a_copy,
                    (a_x, a_y),
                    special_flags=pygame.BLEND_RGBA_ADD,
                )
                
                if next_frame is not empty_surface:
                    b_copy = next_frame.copy()
                    b_weight = 255 - a_weight
                    b_copy.fill(
                        (b_weight, b_weight, b_weight, b_weight),
                        special_flags=pygame.BLEND_RGBA_MULT,
                    )
                    b_x = (max_w - b_copy.get_width()) // 2
                    b_y = (max_h - b_copy.get_height()) // 2
                    new_frame.blit(
                        b_copy,
                        (b_x, b_y),
                        special_flags=pygame.BLEND_RGBA_ADD,
                    )
                
            interpolated.append(new_frame)
                
    return tuple(interpolated)


def _expand_directional_sprites(base_sprites, base_masks=None):
    """Expand A base directional sprites to TOTAL_SPRITE_DIRECTIONS sprites.

    Uses pygame.transform.rotozoom to create intermediate rotation sprites
    from the nearest base asset. Returns (sprites_tuple, masks_tuple).
    """
    total = const.TOTAL_SPRITE_DIRECTIONS
    base_count = const.ASSET_SPRITE_DIRECTIONS
    if total == base_count:
        if base_masks is None:
            base_masks = tuple(pygame.mask.from_surface(s) for s in base_sprites)
        return tuple(base_sprites), tuple(base_masks)

    asset_step = 360.0 / base_count
    total_step = const.TOTAL_SPRITE_STEP
    sprites = []
    masks = []

    for i in range(total):
        target_angle = i * total_step
        nearest_idx = round(target_angle / asset_step) % base_count
        nearest_angle = nearest_idx * asset_step

        delta = nearest_angle - target_angle
        delta = (delta + 180) % 360 - 180

        if abs(delta) < 0.001:
            sprite = base_sprites[nearest_idx]
        else:
            sprite = pygame.transform.rotozoom(base_sprites[nearest_idx], delta, 1.0)

        sprites.append(sprite)
        masks.append(pygame.mask.from_surface(sprite))

    return tuple(sprites), tuple(masks)


def _exclude_center_circle(sprite, radius):
    """Return a sprite with a transparent circular center exclusion."""
    excluded = sprite.copy()
    pygame.draw.circle(
        excluded,
        (0, 0, 0, 0),
        excluded.get_rect().center,
        max(0, radius),
    )
    return excluded


def _scale_directional_sprite(sprite, heading, scale_x, scale_y):
    if scale_x == scale_y == 1.0:
        return sprite
    if scale_x == scale_y:
        return pygame.transform.smoothscale_by(sprite, scale_x)

    angle = heading * (360 / const.ASSET_SPRITE_DIRECTIONS)
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
    angle = math.radians(heading * const.TOTAL_SPRITE_STEP)
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
    angle = math.radians(heading * const.TOTAL_SPRITE_STEP)
    forward_x = math.sin(angle)
    forward_y = -math.cos(angle)
    for y in range(height):
        for x in range(width):
            projection = (x - center_x) * forward_x + (y - center_y) * forward_y
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
