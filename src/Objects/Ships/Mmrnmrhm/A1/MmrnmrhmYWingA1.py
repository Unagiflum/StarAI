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
        self.launch_from_gun(
            gun_location=gun_location,
            relative_direction=relative_direction,
        )

    @classmethod
    def create_projectiles(cls, ship):
        definition = ABILITY_DEFINITIONS["MmrnmrhmYWingA1"]
        locations = definition.gun_locations or ()
        directions = definition.gun_directions or ()
        return [
            cls(ship, location, direction)
            for location, direction in zip(locations, directions)
        ]
