import pygame
import json
import os
import random
from src.GameObject import GameObject

class Planet(GameObject):
    def __init__(self):
        # Load planet data from json
        json_path = os.path.join(os.path.dirname(__file__), 'Resources', 'planets.json')
        with open(json_path, 'r') as f:
            planets = json.load(f)

        # Select random planet
        planet_name = random.choice(list(planets.keys()))
        planet_data = planets[planet_name]

        # Get planet properties
        self.gravity = planet_data['Gravity']
        self.diameter = planet_data['Diameter']

        # Load image
        image_path = os.path.join('Battle', 'Resources', 'Planets', f'{planet_name}.png')
        self.image = pygame.image.load(image_path).convert_alpha()

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


class Star(GameObject):
    def __init__(self):
        with open('Battle/Resources/stars.json', 'r') as f:
            stars = json.load(f)

        # Weight stars by size class (a=largest=weight 1, e=smallest=weight 5)
        weights = {name: 150 if 'e' in name else 90 if 'd' in name else
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

        self.can_collide = False