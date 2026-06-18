import math
from src.Objects.object import Object
import src.const as Const
import pygame
from pathlib import Path
import json
import random

class Planet(Object):
    # Load planet data once at module level
    with open(Const.PLANETS_JSON_PATH, 'r') as f:
        _planet_data = json.load(f)

    def __init__(self):
        weights = {
            name: Const.PLANET_WEIGHTS[0] if 'Gas' in name
            else Const.PLANET_WEIGHTS[1] if 'Ice' in name
            else Const.PLANET_WEIGHTS[2] if 'Life' in name
            else Const.PLANET_WEIGHTS[3] if 'Rocky' in name
            else 0 for name in Planet._planet_data.keys()
        }
        planet_name = random.choices(list(Planet._planet_data.keys()), weights=list(weights.values()), k=1)[0]
        planet_data = Planet._planet_data[planet_name]

        self.gravity = planet_data['Gravity']
        self.diameter = planet_data['Diameter']

        super().__init__(
            name=planet_name,
            sprite_location=None,
            size=[self.diameter, self.diameter]
        )

        self.image = pygame.image.load(str(Path(planet_data['Image']))).convert_alpha()
        if self.image.get_size() != (self.diameter, self.diameter):
            self.image = pygame.transform.smoothscale(self.image, (self.diameter, self.diameter))
        self.mask = pygame.mask.from_surface(self.image)
        self.can_move = False
        self.can_die = False

    def update(self):
        return True

    def get_collision_mask(self):
        return self.mask

    @staticmethod
    def create_center():
        planet = Planet()
        planet.position = Const.PLANET_POSITION
        planet.previous_position = planet.position.copy()
        return planet

    def draw(self, screen, scale_factor, translation):
        # Draw gravity range circle
        range_color = (255, 255, 255, 8)  # Light blue, semi-transparent
        border_color = (255, 0, 0, 150)  # Red, semi-transparent
        gravity_range_surface = pygame.Surface((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT), pygame.SRCALPHA)

        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)
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
                    pygame.draw.arc(gravity_range_surface, border_color,
                                    (Const.SCREEN_LEFT+pos_x - range_radius, pos_y - range_radius,
                                     range_radius * 2, range_radius * 2), start_angle, end_angle, int(20*scale_factor))

        screen.blit(gravity_range_surface, (0, 0))

        # Draw planet sprite
        scaled_image = pygame.transform.smoothscale_by(self.image, scale_factor)
        planet_size = scaled_image.get_width()

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (-planet_size <= pos_x <= Const.SCREEN_HEIGHT + planet_size and
                        -planet_size <= pos_y <= Const.SCREEN_HEIGHT + planet_size):
                    screen.blit(scaled_image, (
                        Const.SCREEN_LEFT + pos_x - planet_size // 2,
                        pos_y - planet_size // 2
                    ))

class Star(Object):
    # Load star data once at module level
    with open(Const.STARS_JSON_PATH, 'r') as f:
        _star_data = json.load(f)

    depth_surfaces = [pygame.Surface((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT), pygame.SRCALPHA) for _ in range(Const.STAR_DEPTHS)]
    stars_by_depth = [[] for _ in range(Const.STAR_DEPTHS)]

    def __init__(self):
        weights = {
            name: Const.STAR_WEIGHTS[0] if 'e' in name
            else Const.STAR_WEIGHTS[1] if 'd' in name
            else Const.STAR_WEIGHTS[2] if 'c' in name
            else Const.STAR_WEIGHTS[3] if 'b' in name
            else Const.STAR_WEIGHTS[4] if 'a' in name
            else 0 for name in Star._star_data.keys()
        }
        star_name = random.choices(list(Star._star_data.keys()), weights=list(weights.values()), k=1)[0]
        star_data = Star._star_data[star_name]

        self.diameter = star_data['Diameter']
        self.depth = random.randint(0, Const.STAR_DEPTHS-1)

        super().__init__(
            name=star_name,
            sprite_location=None,
            size=[self.diameter, self.diameter]
        )

        self.image = pygame.image.load(str(Path(star_data['Image']))).convert_alpha()
        self.can_move = False
        self.can_die = False
        self.can_collide = False

    @staticmethod
    def create_random_stars(count):
        Star.stars_by_depth = [[] for _ in range(Const.STAR_DEPTHS)]
        stars = []
        for _ in range(count):
            star = Star()
            star.position = [
                random.randint(0, Const.ARENA_SIZE),
                random.randint(0, Const.ARENA_SIZE)
            ]
            Star.stars_by_depth[star.depth].append(star)
            stars.append(star)
        return stars

    @staticmethod
    def update_depth_surface(depth, stars, scale_factor, translation, midpoint, parallax_factor):
        surface = Star.depth_surfaces[depth]
        surface.fill((0, 0, 0, 0))

        for star in Star.stars_by_depth[depth]:
            dx = star.position[0] - midpoint[0]
            dy = star.position[1] - midpoint[1]

            if abs(dx) > Const.ARENA_SIZE / 2:
                dx = dx - Const.ARENA_SIZE if dx > 0 else dx + Const.ARENA_SIZE
            if abs(dy) > Const.ARENA_SIZE / 2:
                dy = dy - Const.ARENA_SIZE if dy > 0 else dy + Const.ARENA_SIZE

            relative_x = midpoint[0] + dx * parallax_factor
            relative_y = midpoint[1] + dy * parallax_factor

            screen_x = int((relative_x + translation[0]) * scale_factor)
            screen_y = int((relative_y + translation[1]) * scale_factor)

            scaled_image = pygame.transform.smoothscale_by(star.image, scale_factor)
            scaled_image.set_alpha(Const.STAR_ALPHA)
            star_size = scaled_image.get_width()

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                    pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                    if (-star_size <= pos_x <= Const.SCREEN_HEIGHT + star_size and
                            -star_size <= pos_y <= Const.SCREEN_HEIGHT + star_size):
                        surface.blit(scaled_image, (
                            Const.SCREEN_LEFT + pos_x - star_size // 2,
                            pos_y - star_size // 2
                        ))

    def update(self):
        return True


class Asteroid(Object):
    shared_sprites = None
    shared_masks = None
    shared_death_animation = None

    def __init__(self):
        super().__init__(
            name="Asteroid",
            sprite_location=None,
            size=[0, 0]
        )
        # Load shared sprites if not already loaded
        if Asteroid.shared_sprites is None:
            Asteroid.shared_sprites = [
                pygame.image.load(str(Const.ASTEROID_PATH / f"asteroid{i:02d}.png")).convert_alpha()
                for i in range(30)]
            Asteroid.shared_masks = [
                pygame.mask.from_surface(sprite)
                for sprite in Asteroid.shared_sprites]

        # Load shared death animation if not already loaded
        if Asteroid.shared_death_animation is None:
            Asteroid.shared_death_animation = [
                pygame.image.load(str(Const.ASTEROID_PATH / f"asteroidend{i:02d}.png")).convert_alpha()
                for i in range(4)]

        # Randomly rotate sprites for this instance
        if random.random() < 0.0: # if 0 then no rotation will be applied
            sprite_rot = random.random()*360
            self.sprites = [pygame.transform.rotate(sprite, sprite_rot) for sprite in Asteroid.shared_sprites]
            self.masks = [pygame.mask.from_surface(sprite) for sprite in self.sprites]
        else:
            self.sprites = Asteroid.shared_sprites
            self.masks = Asteroid.shared_masks

        self.death_animation = Asteroid.shared_death_animation

        self.current_sprite = random.randint(0, 29)
        self.size = [self.sprites[self.current_sprite].get_width(), self.sprites[self.current_sprite].get_height()]

        self.rotation_delay = random.randint(0, 3)
        self.rotation_timer = 0

        speed = random.uniform(Const.ASTEROID_SPEED / 2, Const.ASTEROID_SPEED)
        angle = random.uniform(0, 2*math.pi)
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
        dx = self.planet.position[0] - self.position[0]
        dy = self.planet.position[1] - self.position[1]
        if abs(dx) > Const.ARENA_SIZE / 2:
            dx = dx - Const.ARENA_SIZE if dx > 0 else dx + Const.ARENA_SIZE
        if abs(dy) > Const.ARENA_SIZE / 2:
            dy = dy - Const.ARENA_SIZE if dy > 0 else dy + Const.ARENA_SIZE
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
            position = self._find_spawn_position(planet, view_bodies, avoid_bodies, rules)
            if position is not None:
                return position

        return [
            random.randint(0, Const.ARENA_SIZE),
            random.randint(0, Const.ARENA_SIZE)
        ]

    def _find_spawn_position(self, planet, view_bodies, avoid_bodies, rules):
        for _ in range(1000):
            position = [
                random.randint(0, Const.ARENA_SIZE),
                random.randint(0, Const.ARENA_SIZE)
            ]

            if not self._position_is_offscreen(position, view_bodies):
                continue

            if rules["avoid_gravity"] and not self._position_is_outside_gravity(position, planet):
                continue

            if rules["avoid_bodies"]:
                bodies = view_bodies if rules.get("only_view_bodies") else avoid_bodies
                if not self._position_is_away_from_bodies(position, bodies, planet.diameter):
                    continue

            return position
        return None

    def _position_is_outside_gravity(self, position, planet):
        asteroid_radius = max(self.size[0], self.size[1]) / 2
        return self._distance_between_positions(position, planet.position) >= Const.GRAVITY_RANGE + asteroid_radius

    def _position_is_away_from_bodies(self, position, bodies, minimum_distance):
        for body in bodies:
            if not getattr(body, "currently_alive", True):
                continue
            if not getattr(body, "can_collide", True):
                continue
            if self._distance_between_positions(position, body.position) < minimum_distance:
                return False
        return True

    def _position_is_offscreen(self, position, view_bodies):
        if len(view_bodies) != 2:
            return True

        view_center, view_size = self._view_center_and_size([body.position for body in view_bodies])
        dx = abs(position[0] - view_center[0])
        dy = abs(position[1] - view_center[1])
        dx = min(dx, Const.ARENA_SIZE - dx)
        dy = min(dy, Const.ARENA_SIZE - dy)
        asteroid_radius = max(self.size[0], self.size[1]) / 2
        return dx > view_size / 2 + asteroid_radius or dy > view_size / 2 + asteroid_radius

    def _view_center_and_size(self, positions):
        p1_pos, p2_pos = positions
        dx = p2_pos[0] - p1_pos[0]
        dy = p2_pos[1] - p1_pos[1]

        if abs(dx) > Const.ARENA_SIZE / 2:
            dx = dx - Const.ARENA_SIZE if dx > 0 else dx + Const.ARENA_SIZE
        if abs(dy) > Const.ARENA_SIZE / 2:
            dy = dy - Const.ARENA_SIZE if dy > 0 else dy + Const.ARENA_SIZE

        view_center = [
            (p1_pos[0] + dx / 2) % Const.ARENA_SIZE,
            (p1_pos[1] + dy / 2) % Const.ARENA_SIZE
        ]
        distance = math.sqrt(dx * dx + dy * dy)
        min_view_size = Const.SCREEN_HEIGHT / Const.MAX_ZOOM
        view_size = min(max(distance / 0.8, min_view_size), Const.ARENA_SIZE / 2)
        scale_factor = min(Const.MAX_ZOOM, Const.SCREEN_HEIGHT / view_size)
        return view_center, Const.SCREEN_HEIGHT / scale_factor

    def _distance_between_positions(self, position, other_position):
        dx = abs(position[0] - other_position[0])
        dy = abs(position[1] - other_position[1])
        dx = min(dx, Const.ARENA_SIZE - dx)
        dy = min(dy, Const.ARENA_SIZE - dy)
        return math.sqrt(dx * dx + dy * dy)

    def get_gravity(self):
        if not self.can_move or not self.planet:
            return [0.0, 0.0]
        [dx, dy], distance = self.planet_distance()

        if distance < self.planet.diameter / 2 or distance > Const.GRAVITY_RANGE:
            return [0.0, 0.0]

        gravity_force = Const.GRAVITY_MULTIPLIER * self.planet.gravity
        return [
            gravity_force * dx / distance,
            gravity_force * dy / distance
        ]

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
            if self._candidate_overlaps_object(candidate_mask, candidate_size, asteroid):
                return True

        return False

    def _candidate_overlaps_object(self, candidate_mask, candidate_size, other):
        other_mask = other.get_collision_mask() if hasattr(other, "get_collision_mask") else None
        other_size = other_mask.get_size() if other_mask else other.size
        delta = self._wrapped_delta(other.position, self.position)
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

    @staticmethod
    def _wrapped_delta(from_position, to_position):
        dx = to_position[0] - from_position[0]
        dy = to_position[1] - from_position[1]

        if abs(dx) > Const.ARENA_SIZE / 2:
            dx = dx - Const.ARENA_SIZE if dx > 0 else dx + Const.ARENA_SIZE
        if abs(dy) > Const.ARENA_SIZE / 2:
            dy = dy - Const.ARENA_SIZE if dy > 0 else dy + Const.ARENA_SIZE

        return [dx, dy]

    def update(self):
        self.previous_position = self.position.copy()
        gravity_impulse = self.get_gravity()
        acc0 = [gravity_impulse[0], gravity_impulse[1] ]

        self.position[0] = (self.position[0] +
                            (self.velocity[0] + 0.5 * acc0[0]) * Const.SPEED_SCALE) % Const.ARENA_SIZE
        self.position[1] = (self.position[1] +
                            (self.velocity[1] + 0.5 * acc0[1]) * Const.SPEED_SCALE) % Const.ARENA_SIZE

        gravity_impulse = self.get_gravity()
        acc1 = [gravity_impulse[0], gravity_impulse[1] ]
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

    def draw(self, screen, scale_factor, translation):
        if not self.image:
            self.image = self.sprites[self.current_sprite]

        scaled_image = pygame.transform.smoothscale_by(self.image, scale_factor)
        size = [scaled_image.get_width(),scaled_image.get_height()]
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (-size[0] <= pos_x <= Const.SCREEN_HEIGHT + size[0] and
                        -size[1] <= pos_y <= Const.SCREEN_HEIGHT + size[1]):
                    screen.blit(scaled_image, (
                        Const.SCREEN_LEFT + pos_x - size[0] // 2,
                        pos_y - size[1] // 2
                    ))

    def get_collision_mask(self):
        return self.masks[self.current_sprite]
