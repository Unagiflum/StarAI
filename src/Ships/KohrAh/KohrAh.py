# KohrAh.py
from src.Ships.SpaceShip import SpaceShip
import src.Ships.KohrAh.KohrAhA1 as A1
import src.Ships.KohrAh.KohrAhA2 as A2

class KohrAh(SpaceShip):
    def __init__(self, ship_name, player_num):
        super().__init__(ship_name, player_num)

    def perform_action1(self):
        return A1.action(self)

    def perform_action2(self):
        return A2.action(self)
