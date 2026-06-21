from src.Objects.Ships.space_ship import SpaceShip
from src.Objects.Ships.Spathi.A1.SpathiA1 import SpathiA1
from src.Objects.Ships.Spathi.A2.SpathiA2 import SpathiA2


class Spathi(SpaceShip):
    action_factories = {1: SpathiA1, 2: SpathiA2}
