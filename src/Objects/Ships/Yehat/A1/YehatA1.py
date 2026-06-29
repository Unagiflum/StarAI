from src.Objects.Ships.ability import Ability
from src.Objects.Ships.catalog import ABILITY_DEFINITIONS


class YehatA1(Ability):
    def __init__(self, parent, gun_location=None, relative_direction=None):
        super().__init__("YehatA1", parent)
        definition = ABILITY_DEFINITIONS["YehatA1"]
        location = gun_location or definition.gun_locations[0]
        direction = (
            definition.gun_directions[0]
            if relative_direction is None
            else relative_direction
        )
        self.launch_from_gun(gun_location=location, relative_direction=direction)

    @classmethod
    def create_projectiles(cls, ship):
        definition = ABILITY_DEFINITIONS["YehatA1"]
        return [
            cls(ship, location, direction)
            for location, direction in zip(
                definition.gun_locations or (), definition.gun_directions or ()
            )
        ]
