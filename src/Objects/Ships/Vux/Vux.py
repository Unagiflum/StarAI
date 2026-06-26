from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Vux.A1.VuxA1 import VuxA1
from src.Objects.Ships.Vux.A2.VuxA2 import VuxA2


class Vux(SpaceShip):
    action_factories = {1: VuxA1, 2: VuxA2}
