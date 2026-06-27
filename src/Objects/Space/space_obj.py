import math
from src.Objects.object import Object
import src.const as Const
import pygame
import json
import random
from src.collision_capabilities import (
    AreaDamageCapabilities,
    CollisionCapabilities,
    CollisionRole,
    PhysicalCollisionCapabilities,
    DurabilityCapabilities,
    ImpactCapabilities,
)
from src.toroidal import view_center_and_size, wrapped_delta, wrapped_distance
from src.resources import default_assets


class Planet(Object):
    # Load planet data once at module level
    with open(Const.PLANETS_JSON_PATH, "r") as f:
        _planet_data = json.load(f)

    def __init__(self, resources=None, rng=None):
        self.resources = resources or default_assets()
        self.rng = rng or random
        weights = {
            name: (
                Const.PLANET_WEIGHTS[0]
                if "Gas" in name
                else (
                    Const.PLANET_WEIGHTS[1]
                    if "Ice" in name
                    else (
                        Const.PLANET_WEIGHTS[2]
                        if "Life" in name
                        else Const.PLANET_WEIGHTS[3] if "Rocky" in name else 0
                    )
                )
            )
            for name in Planet._planet_data.keys()
        }
        planet_name = self.rng.choices(
            list(Planet._planet_data.keys()), weights=list(weights.values()), k=1
        )[0]
        planet_data = Planet._planet_data[planet_name]

        self.gravity = planet_data["Gravity"]
        self.diameter = planet_data["Diameter"]

        super().__init__(
            name=planet_name, sprite_location=None, size=[self.diameter, self.diameter]
        )

        assets = self.resources.image(
            planet_data["Image"], (self.diameter, self.diameter), with_mask=True
        )
        self.image = assets.image
        self.mask = assets.mask
        self.collision_capabilities = CollisionCapabilities(CollisionRole.PLANET)
        self.physical_collision_capabilities = PhysicalCollisionCapabilities(is_immovable=True)
        self.durability_capabilities = DurabilityCapabilities(is_invulnerable=True)
        self.impact_capabilities = ImpactCapabilities(impact_damage_percent=0.15)
        self.can_move = False
        self.can_die = False

    def update(self):
        return True

    def get_collision_mask(self):
        return self.mask

    @staticmethod
    def create_center(resources=None, rng=None):
        planet = Planet(resources, rng=rng)
        planet.position = Const.PLANET_POSITION
        planet.previous_position = planet.position.copy()
        return planet

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        # Draw gravity range circle
        range_color = (255, 255, 255, 8)  # Light blue, semi-transparent
        border_color = (255, 0, 0, 150)  # Red, semi-transparent
        gravity_range_surface = pygame.Surface(
            (Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT), pygame.SRCALPHA
        )

        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)
        screen_x = int((pos[0] + translation[0]) * scale_factor)
        screen_y = int((pos[1] + translation[1]) * scale_factor)
        range_radius = int(Const.GRAVITY_RANGE * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                # Draw dashed outer circle
                num_segments = 64
                for i in range(0, num_segments, 2):
                    start_angle = i * (2 * math.pi / num_segments)
                    end_angle = (i + 0.2) * (2 * math.pi / num_segments)
                    pygame.draw.arc(
                        gravity_range_surface,
                        border_color,
                        (
                            Const.SCREEN_LEFT + pos_x - range_radius,
                            pos_y - range_radius,
                            range_radius * 2,
                            range_radius * 2,
                        ),
                        start_angle,
                        end_angle,
                        int(20 * scale_factor),
                    )

        screen.blit(gravity_range_surface, (0, 0))

        # Draw planet sprite
        scaled_image = pygame.transform.smoothscale_by(self.image, scale_factor)
        planet_size = scaled_image.get_width()

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (
                    -planet_size <= pos_x <= Const.SCREEN_HEIGHT + planet_size
                    and -planet_size <= pos_y <= Const.SCREEN_HEIGHT + planet_size
                ):
                    screen.blit(
                        scaled_image,
                        (
                            Const.SCREEN_LEFT + pos_x - planet_size // 2,
                            pos_y - planet_size // 2,
                        ),
                    )


class Star(Object):
    # Load star data once at module level
    with open(Const.STARS_JSON_PATH, "r") as f:
        _star_data = json.load(f)

    def __init__(self, resources=None, rng=None):
        self.resources = resources or default_assets()
        self.rng = rng or random
        weights = {
            name: (
                Const.STAR_WEIGHTS[0]
                if "e" in name
                else (
                    Const.STAR_WEIGHTS[1]
                    if "d" in name
                    else (
                        Const.STAR_WEIGHTS[2]
                        if "c" in name
                        else (
                            Const.STAR_WEIGHTS[3]
                            if "b" in name
                            else Const.STAR_WEIGHTS[4] if "a" in name else 0
                        )
                    )
                )
            )
            for name in Star._star_data.keys()
        }
        star_name = self.rng.choices(
            list(Star._star_data.keys()), weights=list(weights.values()), k=1
        )[0]
        star_data = Star._star_data[star_name]

        self.diameter = star_data["Diameter"]
        self.depth = self.rng.randint(0, Const.STAR_DEPTHS - 1)

        super().__init__(
            name=star_name, sprite_location=None, size=[self.diameter, self.diameter]
        )

        self.image = self.resources.image(star_data["Image"]).image
        self.physical_collision_capabilities = PhysicalCollisionCapabilities(is_intangible=True)
        self.durability_capabilities = DurabilityCapabilities(is_invulnerable=True)
        self.can_move = False
        self.can_die = False
        self.can_collide = False

    @staticmethod
    def create_random_stars(count, resources=None, rng=None):
        explicit_rng = rng is not None
        rng = rng or random
        stars = []
        for _ in range(count):
            star = Star(resources, rng=rng) if explicit_rng else Star(resources)
            star.position = [
                rng.randint(0, Const.ARENA_SIZE),
                rng.randint(0, Const.ARENA_SIZE),
            ]
            stars.append(star)
        return stars

    def update(self):
        return True


class Asteroid(Object):
    def __init__(self, resources=None, rng=None):
        self.resources = resources or default_assets()
        self.rng = rng or random
        super().__init__(name="Asteroid", sprite_location=None, size=[0, 0])
        assets = self.resources.asteroid()

        # Randomly rotate sprites for this instance
        if self.rng.random() < 0.0:  # if 0 then no rotation will be applied
            sprite_rot = self.rng.random() * 360
            self.sprites = [
                pygame.transform.rotate(sprite, sprite_rot) for sprite in assets.sprites
            ]
            self.masks = [pygame.mask.from_surface(sprite) for sprite in self.sprites]
        else:
            self.sprites = assets.sprites
            self.masks = assets.masks

        self.death_animation = assets.death_animation
        self.collision_capabilities = CollisionCapabilities(CollisionRole.ASTEROID)
        self.area_damage_capabilities = AreaDamageCapabilities(targetable=True)
        self.physical_collision_capabilities = PhysicalCollisionCapabilities(fragile_to_immovable=True)

        self.current_sprite = self.rng.randint(0, 29)
        self.size = [
            self.sprites[self.current_sprite].get_width(),
            self.sprites[self.current_sprite].get_height(),
        ]

        self.rotation_delay = self.rng.randint(0, 3)
        self.rotation_timer = 0

        speed = self.rng.uniform(Const.ASTEROID_SPEED / 2, Const.ASTEROID_SPEED)
        angle = self.rng.uniform(0, 2 * math.pi)
        self.velocity = [speed * math.cos(angle), speed * math.sin(angle)]

        self.planet = None
        self.ships = []
        self.asteroids = []
        self.can_move = True
        self.can_die = True
        self.can_collide = True
        self.can_expire = False

    def set_planet(self, planet):
        self.planet = planet

    def planet_distance(self):
        dx, dy = wrapped_delta(self.position, self.planet.position)
        distance = math.sqrt(dx * dx + dy * dy)
        return [dx, dy], distance

    def get_valid_asteroid_position(self, planet, view_bodies, avoid_bodies):
        return self.get_respawn_position(planet, view_bodies, avoid_bodies)

    def get_respawn_position(self, planet, view_bodies, avoid_bodies):
        spawn_rules = [
            {"avoid_gravity": True, "avoid_bodies": True},
            {"avoid_gravity": True, "avoid_bodies": False},
            {"avoid_gravity": False, "avoid_bodies": True, "only_view_bodies": True},
            {"avoid_gravity": False, "avoid_bodies": False},
        ]
        for rules in spawn_rules:
            position = self._find_spawn_position(
                planet, view_bodies, avoid_bodies, rules
            )
            if position is not None:
                return position

        return [
            self.rng.randint(0, Const.ARENA_SIZE),
            self.rng.randint(0, Const.ARENA_SIZE),
        ]

    def _find_spawn_position(self, planet, view_bodies, avoid_bodies, rules):
        for _ in range(1000):
            position = [
                self.rng.randint(0, Const.ARENA_SIZE),
                self.rng.randint(0, Const.ARENA_SIZE),
            ]

            if not self._position_is_offscreen(position, view_bodies):
                continue

            if rules["avoid_gravity"] and not self._position_is_outside_gravity(
                position, planet
            ):
                continue

            if rules["avoid_bodies"]:
                bodies = view_bodies if rules.get("only_view_bodies") else avoid_bodies
                if not self._position_is_away_from_bodies(
                    position, bodies, planet.diameter
                ):
                    continue

            return position
        return None

    def _position_is_outside_gravity(self, position, planet):
        asteroid_radius = max(self.size[0], self.size[1]) / 2
        return (
            wrapped_distance(position, planet.position)
            >= Const.GRAVITY_RANGE + asteroid_radius
        )

    def _position_is_away_from_bodies(self, position, bodies, minimum_distance):
        for body in bodies:
            if isinstance(body, Object):
                if not body.is_alive() or not body.can_collide:
                    continue
            else:
                # Compatibility for lightweight positioning test doubles.
                if not getattr(body, "currently_alive", True):
                    continue
                if not getattr(body, "can_collide", True):
                    continue
            if wrapped_distance(position, body.position) < minimum_distance:
                return False
        return True

    def _position_is_offscreen(self, position, view_bodies):
        if len(view_bodies) != 2:
            return True

        view_center, view_size = view_center_and_size(
            [body.position for body in view_bodies]
        )
        dx, dy = wrapped_delta(view_center, position)
        asteroid_radius = max(self.size[0], self.size[1]) / 2
        return (
            abs(dx) > view_size / 2 + asteroid_radius
            or abs(dy) > view_size / 2 + asteroid_radius
        )

    def get_gravity(self):
        if not self.can_move or not self.planet:
            return [0.0, 0.0]
        [dx, dy], distance = self.planet_distance()

        if distance < self.planet.diameter / 2 or distance > Const.GRAVITY_RANGE:
            return [0.0, 0.0]

        gravity_force = Const.GRAVITY_MULTIPLIER * self.planet.gravity
        return [gravity_force * dx / distance, gravity_force * dy / distance]

    def next_sprite(self):
        next_sprite = (self.current_sprite + 1) % 30
        if self.sprite_would_overlap(next_sprite):
            return

        self.current_sprite = next_sprite
        self.image = self.sprites[self.current_sprite]

    def sprite_would_overlap(self, sprite_index):
        candidate_mask = self.masks[sprite_index]
        candidate_size = candidate_mask.get_size()

        for ship in self.ships:
            if self._candidate_overlaps_object(candidate_mask, candidate_size, ship):
                return True

        for asteroid in self.asteroids:
            if asteroid is self or not asteroid.currently_alive:
                continue
            if self._candidate_overlaps_object(
                candidate_mask, candidate_size, asteroid
            ):
                return True

        return False

    def _candidate_overlaps_object(self, candidate_mask, candidate_size, other):
        other_mask = other.get_collision_mask()
        other_size = other_mask.get_size() if other_mask else other.size
        delta = wrapped_delta(other.position, self.position)
        radius_sum = max(candidate_size) / 2 + max(other_size) / 2
        if math.hypot(delta[0], delta[1]) >= radius_sum:
            return False

        if other_mask is None:
            return True

        offset = (
            int(round(-delta[0] + candidate_size[0] / 2 - other_size[0] / 2)),
            int(round(-delta[1] + candidate_size[1] / 2 - other_size[1] / 2)),
        )
        return candidate_mask.overlap(other_mask, offset) is not None

    def update(self):
        self.previous_position = self.position.copy()
        gravity_impulse = self.get_gravity()
        acc0 = [gravity_impulse[0], gravity_impulse[1]]

        self.position[0] = (
            self.position[0] + (self.velocity[0] + 0.5 * acc0[0]) * Const.SPEED_SCALE
        ) % Const.ARENA_SIZE
        self.position[1] = (
            self.position[1] + (self.velocity[1] + 0.5 * acc0[1]) * Const.SPEED_SCALE
        ) % Const.ARENA_SIZE

        gravity_impulse = self.get_gravity()
        acc1 = [gravity_impulse[0], gravity_impulse[1]]
        self.velocity[0] += (acc0[0] + acc1[0]) * 0.5
        self.velocity[1] += (acc0[1] + acc1[1]) * 0.5

        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > Const.SPEED_LIMIT:
            scale = Const.SPEED_LIMIT / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale

        self.rotation_timer += 1
        if self.rotation_timer >= self.rotation_delay:
            self.rotation_timer = 0
            self.next_sprite()

        current_sprite = self.sprites[self.current_sprite]
        self.size = [current_sprite.get_width(), current_sprite.get_height()]

        return self.currently_alive

    def draw(self, screen, scale_factor, translation, interp_t=0.0):
        assets = self.resources.asteroid()
        if Const.VIDEO_FPS_MULTIPLIER > 1 and hasattr(assets, 'interpolated_sprites') and assets.interpolated_sprites:
            fraction = (self.rotation_timer + interp_t) / (self.rotation_delay + 1.0)
            sub_frame_offset = int(fraction * Const.VIDEO_FPS_MULTIPLIER)
            draw_sprite_idx = (self.current_sprite * Const.VIDEO_FPS_MULTIPLIER + sub_frame_offset) % len(assets.interpolated_sprites)
            self.image = assets.interpolated_sprites[draw_sprite_idx]
        else:
            self.image = self.sprites[self.current_sprite]

        scaled_image = pygame.transform.smoothscale_by(self.image, scale_factor)
        size = [scaled_image.get_width(), scaled_image.get_height()]
        from src.Battle.interpolation import interpolated_position

        pos = interpolated_position(self, interp_t)
        screen_x = int((pos[0] + translation[0]) * scale_factor)
        screen_y = int((pos[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (
                    -size[0] <= pos_x <= Const.SCREEN_HEIGHT + size[0]
                    and -size[1] <= pos_y <= Const.SCREEN_HEIGHT + size[1]
                ):
                    screen.blit(
                        scaled_image,
                        (
                            Const.SCREEN_LEFT + pos_x - size[0] // 2,
                            pos_y - size[1] // 2,
                        ),
                    )

    def get_collision_mask(self):
        return self.masks[self.current_sprite]
