import math

import src.const as const
from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class MmrnmrhmYWingA1(Ability):
    def __init__(self, parent, gun_location=None, relative_direction=None):
        super().__init__("MmrnmrhmYWingA1", parent)
        definition = ABILITY_DEFINITIONS["MmrnmrhmYWingA1"]
        gun_location = gun_location or definition.gun_locations[0]
        if relative_direction is None:
            relative_direction = definition.gun_directions[0]
        self._place_at_gun(gun_location, relative_direction)

    def _place_at_gun(self, gun_location, relative_direction):
        parent_angle = math.radians(self.parent.rotation)
        form_assets = self.parent.resources.ship_form("Mmrnmrhm", "YWing")
        canvas_width, canvas_height = form_assets.sprites[0].get_size()
        x_offset = gun_location[0] - canvas_width / 2
        y_offset = canvas_height / 2 - gun_location[1]
        self.position = [
            self.parent.position[0]
            + math.cos(parent_angle) * x_offset
            + math.sin(parent_angle) * y_offset,
            self.parent.position[1]
            + math.sin(parent_angle) * x_offset
            - math.cos(parent_angle) * y_offset,
        ]
        self.rotation = (self.parent.rotation + relative_direction) % 360
        self.heading = round(self.rotation / const.TURN_ANGLE) % const.SHIP_DIRECTIONS
        angle = math.radians(self.rotation)
        self.velocity = [
            math.sin(angle) * self.speed + self.parent.velocity[0] * self.parent_vel,
            -math.cos(angle) * self.speed + self.parent.velocity[1] * self.parent_vel,
        ]

    @classmethod
    def create_projectiles(cls, ship):
        definition = ABILITY_DEFINITIONS["MmrnmrhmYWingA1"]
        locations = definition.gun_locations or ()
        directions = definition.gun_directions or ()
        return [
            cls(ship, location, direction)
            for location, direction in zip(locations, directions)
        ]
