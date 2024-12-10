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