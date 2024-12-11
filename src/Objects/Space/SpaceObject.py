import pygame
import json
import random

from src.Objects.Object import Object
import src.Const as Const

class Planet(Object):
    def __init__(self):
        # Load planet data from json
        with open(Const.PLANETS_JSON_PATH, 'r') as f:
            planets = json.load(f)

        # Select random planet
        planet_name = random.choice(list(planets.keys()))
        planet_data = planets[planet_name]

        # Get planet properties
        self.gravity = planet_data['Gravity']
        self.diameter = planet_data['Diameter']
        self.can_expire = False
        self.expiration_timer = 0
        self.can_move = False
        self.can_die = False

        # Load image
        self.image = pygame.image.load(planet_data['Image']).convert_alpha()

        # Initialize parent PlayerObject
        super().__init__(
            player_num=0,  # Planets don't belong to a player
            max_hp=1,
            start_hp=1,
            inertia=False,
            sprite_location=None,
            sprite_scale=1.0,
            size=[self.diameter, self.diameter]
        )


class Star(Object):
    def __init__(self):
        with open(Const.STARS_JSON_PATH, 'r') as f:
            stars = json.load(f)

        # Weight stars by size class (a=largest=weight 1, e=smallest=weight 5)
        weights = {name: 25 if 'e' in name else 25 if 'd' in name else
        5 if 'c' in name else 0 if 'b' in name else 0
                   for name in stars.keys()}
        star_name = random.choices(list(stars.keys()), weights=list(weights.values()), k=1)[0]
        star_data = stars[star_name]

        self.diameter = star_data['Diameter']
        self.image = pygame.image.load(star_data['Image']).convert_alpha()

        super().__init__(
            player_num=0,
            max_hp=1,
            start_hp=1,
            inertia=False,
            sprite_location=None,
            sprite_scale=1.0,
            size=[self.diameter, self.diameter]
        )
        self.can_expire = False
        self.expiration_timer = 0
        self.can_move = False
        self.can_die = False
        self.can_collide = False