from src.Objects.Object import Object
import src.Const as Const
from src.UI import UI
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
        planet.position = [Const.ARENA_SIZE / 2, Const.ARENA_SIZE / 2]
        return planet

    def draw(self, screen, scale_factor, translation):
        scaled_image = pygame.transform.smoothscale_by(self.image, scale_factor)
        planet_size = scaled_image.get_width()
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (-planet_size <= pos_x <= UI.SCREEN_HEIGHT + planet_size and
                        -planet_size <= pos_y <= UI.SCREEN_HEIGHT + planet_size):
                    screen.blit(scaled_image, (
                        pos_x - planet_size // 2,
                        pos_y - planet_size // 2
                    ))


class Star(Object):
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
        stars = []
        for _ in range(count):
            star = Star()
            star.position = [
                random.randint(0, Const.ARENA_SIZE),
                random.randint(0, Const.ARENA_SIZE)
            ]
            stars.append(star)
        return stars

    def draw(self, screen, scale_factor, translation):
        scaled_image = pygame.transform.smoothscale_by(self.image, scale_factor)
        scaled_image.set_alpha(Const.STAR_ALPHA)
        star_size = scaled_image.get_width()
        screen_x = int((self.position[0] + translation[0]) * scale_factor)
        screen_y = int((self.position[1] + translation[1]) * scale_factor)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos_x = screen_x + dx * Const.ARENA_SIZE * scale_factor
                pos_y = screen_y + dy * Const.ARENA_SIZE * scale_factor

                if (-star_size <= pos_x <= UI.SCREEN_HEIGHT + star_size and
                        -star_size <= pos_y <= UI.SCREEN_HEIGHT + star_size):
                    screen.blit(scaled_image, (
                        pos_x - star_size // 2,
                        pos_y - star_size // 2
                    ))

    def update(self):
        return True