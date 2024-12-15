import math
from src.Objects.Object import Object
import src.Const as Const
import pygame
from pathlib import Path
import json
import random

class Planet(Object):
    def __init__(self):
        with open(Const.PLANETS_JSON_PATH, 'r') as f:
            planets = json.load(f)

        weights = {
            name: Const.PLANET_WEIGHTS[0] if 'Gas' in name
            else Const.PLANET_WEIGHTS[1] if 'Ice' in name
            else Const.PLANET_WEIGHTS[2] if 'Life' in name
            else Const.PLANET_WEIGHTS[3] if 'Rocky' in name
            else 0 for name in planets.keys()
        }
        planet_name = random.choices(list(planets.keys()), weights=list(weights.values()), k=1)[0]
        planet_data = planets[planet_name]

        self.gravity = planet_data['Gravity']
        self.diameter = planet_data['Diameter']

        super().__init__(
            name=planet_name,
            sprite_location=None,
            size=[self.diameter, self.diameter]
        )

        self.image = pygame.image.load(str(Path(planet_data['Image']))).convert_alpha()
        self.can_move = False
        self.can_die = False

    def update(self):
        return True

    @staticmethod
    def create_center():
        planet = Planet()
        planet.position = Const.PLANET_POSITION
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
                # Draw solid inner circle
                # pygame.draw.circle(gravity_range_surface, range_color, (pos_x, pos_y), range_radius, 0)

                # Draw dashed outer circle
                num_segments = 64
                for i in range(0, num_segments, 2):
                    start_angle = i * (2 * math.pi / num_segments)
                    end_angle = (i + 0.2) * (2 * math.pi / num_segments)
                    pygame.draw.arc(gravity_range_surface, border_color,
                                    (pos_x - range_radius, pos_y - range_radius,
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
                        pos_x - planet_size // 2,
                        pos_y - planet_size // 2
                    ))

class Star(Object):
    depth_surfaces = [pygame.Surface((Const.SCREEN_WIDTH, Const.SCREEN_HEIGHT), pygame.SRCALPHA) for _ in
                      range(Const.STAR_DEPTHS)]
    stars_by_depth = [[] for _ in range(Const.STAR_DEPTHS)]

    def __init__(self):
        with open(Const.STARS_JSON_PATH, 'r') as f:
            stars = json.load(f)

        weights = {
            name: Const.STAR_WEIGHTS[0] if 'e' in name
            else Const.STAR_WEIGHTS[1] if 'd' in name
            else Const.STAR_WEIGHTS[2] if 'c' in name
            else Const.STAR_WEIGHTS[3] if 'b' in name
            else Const.STAR_WEIGHTS[4] if 'a' in name
            else 0 for name in stars.keys()
        }
        star_name = random.choices(list(stars.keys()), weights=list(weights.values()), k=1)[0]
        star_data = stars[star_name]

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
                            pos_x - star_size // 2,
                            pos_y - star_size // 2
                        ))

    def update(self):
        return True


class Asteroid(Object):
    def __init__(self):
        super().__init__(
            name="Asteroid",
            sprite_location=None,
            size=[0, 0]
        )
        self.sprites = [pygame.image.load(str(Const.ASTEROID_PATH / f"asteroid{i:02d}.png")).convert_alpha()
                       for i in range(30)]
        self.current_sprite = random.randint(0, 29)
        self.size = [self.sprites[self.current_sprite].get_width(), self.sprites[self.current_sprite].get_height()]

        self.death_animation = [pygame.image.load(str(Const.ASTEROID_PATH / f"asteroiddie{i:02d}.png")).convert_alpha()
                       for i in range(4)]

        self.rotation_delay = 2
        self.rotation_timer = 0

        speed = random.uniform(Const.ASTEROID_V / 2, Const.ASTEROID_V)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = [speed * math.cos(angle), speed * math.sin(angle)]

        self.planet = None
        self.can_move = True
        self.can_die = True
        self.can_collide = True
        self.can_expire = False

    def set_planet(self, planet):
        self.planet = planet

    def planet_distance(self):
        dx = self.planet.position[0] - self.position[0]
        dy = self.planet.position[1] - self.position[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return [dx, dy], distance

    def get_valid_asteroid_position(self, planet_pos, ship_positions, existing_asteroid_positions):
        while True:
            x = random.randint(0, Const.ARENA_SIZE)
            y = random.randint(0, Const.ARENA_SIZE)

            # Check planet distance
            dx = x - planet_pos[0]
            dy = y - planet_pos[1]
            if math.sqrt(dx * dx + dy * dy) < planet_pos[1]:
                continue

            # Check ship distances
            too_close = False
            for ship_pos in ship_positions:
                dx = abs(x - ship_pos[0])
                dy = abs(y - ship_pos[1])
                dx = min(dx, Const.ARENA_SIZE - dx)
                dy = min(dy, Const.ARENA_SIZE - dy)
                if math.sqrt(dx * dx + dy * dy) < Const.MIN_SHIP_SEPARATION / 2:
                    too_close = True
                    break

            if too_close:
                continue

            # Check other asteroid distances
            for ast_pos in existing_asteroid_positions:
                dx = abs(x - ast_pos[0])
                dy = abs(y - ast_pos[1])
                dx = min(dx, Const.ARENA_SIZE - dx)
                dy = min(dy, Const.ARENA_SIZE - dy)
                if math.sqrt(dx * dx + dy * dy) < self.size[0]+self.size[1]:
                    too_close = True
                    break

            if not too_close:
                return [x, y]

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
        self.current_sprite = (self.current_sprite + 1) % 30
        self.image = self.sprites[self.current_sprite]

    def update(self):
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
                        pos_x - size[0] // 2,
                        pos_y - size[1] // 2
                    ))