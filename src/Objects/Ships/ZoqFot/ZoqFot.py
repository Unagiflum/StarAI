from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ZoqFot.A1.ZoqFotA1 import ZoqFotA1
from src.Objects.Ships.ZoqFot.A2.ZoqFotA2 import ZoqFotA2


class ZoqFot(SpaceShip):
    action_factories = {1: ZoqFotA1, 2: ZoqFotA2}
