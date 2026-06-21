from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.ZoqFot.A1.ZoqFotA1 import ZoqFotA1


class ZoqFot(SpaceShip):
    action_factories = {1: ZoqFotA1}
