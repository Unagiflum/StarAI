from src.Objects.Ships.Chmmr.A1.ChmmrA1 import ChmmrA1
from src.Objects.Ships.Chmmr.A2.ChmmrA2 import ChmmrA2
from src.Objects.Ships.Chmmr.A3.ChmmrSatellite import ChmmrSatellite
from src.Objects.Ships.catalog import SHIP_DEFINITIONS
from src.Objects.Ships.space_ship import SpaceShip


class Chmmr(SpaceShip):
    action_factories = {1: ChmmrA1, 2: ChmmrA2}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._satellites_spawned = False
        self._spawned_objects = []

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self._satellites_spawned = False
        self._spawned_objects = []

    def update(self):
        alive = super().update()
        if alive and not self._satellites_spawned:
            count = SHIP_DEFINITIONS[self.name].satellite_count
            self._spawned_objects.extend(
                ChmmrSatellite(self, index) for index in range(count)
            )
            self._satellites_spawned = True
        return alive

    def drain_spawned_objects(self):
        spawned, self._spawned_objects = self._spawned_objects, []
        return spawned
