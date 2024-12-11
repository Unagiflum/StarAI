# Spathi.py
from src.Objects.GameObject import SpaceShip
import src.Objects.Ships.Spathi.SpathiA1 as A1
import src.Objects.Ships.Spathi.SpathiA2 as A2

class Spathi(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        return A1.action(self)

    def perform_action2(self):
        return A2.action(self)
