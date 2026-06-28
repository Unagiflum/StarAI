from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Syreen.A1.SyreenA1 import SyreenA1
from src.Objects.Ships.Syreen.A2.SyreenA2 import SyreenA2

class Syreen(SpaceShip):
    action_factories = {
        1: SyreenA1,
        2: SyreenA2,
    }
