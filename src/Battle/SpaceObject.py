import pygame
import json
import os
import random


class Planet:
    def __init__(self):
        # Load planet data from json
        json_path = os.path.join(os.path.dirname(__file__), 'Resources', 'planets.json')
        with open(json_path, 'r') as f:
            planets = json.load(f)

        # Select random planet
        planet_name = random.choice(list(planets.keys()))
        planet_data = planets[planet_name]

        # Load properties
        self.gravity = planet_data['Gravity']
        self.diameter = planet_data['Diameter']

        # Load image using planet name
        image_path = os.path.join('Battle', 'Resources', 'Planets', f'{planet_name}.png')
        self.image = pygame.image.load(image_path).convert_alpha()

        # Position will be set by Battle.py
        self.position = [0, 0]