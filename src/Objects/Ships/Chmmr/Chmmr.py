import math

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
        self.satellite_orbit_direction = None
        self._spawned_objects = []

    def initialize_in_battle(self, position, heading):
        super().initialize_in_battle(position, heading)
        self._satellites_spawned = False
        self._spawned_objects = []

    def update(self):
        alive = super().update()
        if alive and not self._satellites_spawned:
            self.spawn_satellites()
        return alive

    def spawn_satellites(self, *, rng=None, randomized_health=False):
        if self._satellites_spawned:
            return ()
        definition = SHIP_DEFINITIONS[self.name]
        rng = rng or self.rng
        if self.satellite_orbit_direction is None:
            self.satellite_orbit_direction = rng.choice((-1, 1))
        count = int(definition.satellite_count)
        hp_per_satellite = int(definition.satellite_hp)
        orbit_indices = list(range(count))
        if randomized_health:
            rng.shuffle(orbit_indices)
            total_hp = math.floor(
                count * hp_per_satellite * float(rng.random()) + 0.5
            )
        else:
            total_hp = count * hp_per_satellite

        satellites = []
        for orbit_index in orbit_indices:
            satellite_hp = min(hp_per_satellite, total_hp)
            if satellite_hp <= 0:
                break
            satellites.append(
                ChmmrSatellite(
                    self,
                    orbit_index,
                    orbit_direction=self.satellite_orbit_direction,
                    starting_hp=satellite_hp,
                )
            )
            total_hp -= satellite_hp
        self._spawned_objects.extend(satellites)
        self._satellites_spawned = True
        return tuple(satellites)

    def drain_spawned_objects(self):
        spawned, self._spawned_objects = self._spawned_objects, []
        return spawned
